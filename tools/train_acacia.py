#---------------------------------------for single months-------------
import _init_paths
import natsort
import ctypes
import os
import sys
import cv2
import shutil
import pdb
import pprint
import argparse

import tensorflow as tf
import numpy as np

from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import cfg_parameters_acacia as cfg_parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# create library files
lib = ctypes.cdll.LoadLibrary
lib_so = lib("./preprocess_train/get_cut_acacia_img.so")


# data augmentation setting
DATA_AUG = True

# --------------------------------------First step:GET IMAGE-------------------------------------------------

# get image from the origin
ImgPath = cfg_parameters.img_path_cfg

ImgBeginX = cfg_parameters.ImgBeginX_cfg
ImgBeginY = cfg_parameters.ImgBeginY_cfg
Dst_img_width = cfg_parameters.Dst_img_width_cfg
Dst_img_height = cfg_parameters.Dst_img_height_cfg

ImgPath_split = ImgPath.split('/')
ResultName_ = os.path.join(
    ImgPath_split[1] + '/' + ImgPath_split[3][:-4] + '-x' + str(ImgBeginX) + '-y' + str(ImgBeginY) + '-' +
    str(Dst_img_width) + '-' + str(Dst_img_height) + '.jpg')

get = lib_so.get_img(ImgPath, ResultName_, Dst_img_width, Dst_img_height, ImgBeginX, ImgBeginY)

print('ok,get finished')
# ---------------------------------------Second step:CUT IMAGE----------------------------------------------

palm_file = ResultName_
palmxyfile = cfg_parameters.palmxyfile_cfg
N_small_img = 2000
Size_small_img = 500
palm_box = "./preprocess_train/acacia-train/gt_palm.txt"
img_dir = "./preprocess_train/acacia-train/IMG_Palm"
img_name_ = "./preprocess_train/acacia-train/IMG_Palm/IMG_"
#---------------remove the exist gt_palm.txt & train.txt & IMG_Palm file
if os.path.exists(palm_box):
    os.remove(palm_box)
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.mkdir(img_dir)

cut = lib_so.cut_img(palm_file, palmxyfile, N_small_img, Size_small_img, palm_box, img_name_, ImgBeginX, ImgBeginY)

print('ok,cut finished')
# --------------------------------------Third step:GET TRAIN.TXT CORRESPOND TO THE CUTTED IMAGE-------------

path = './preprocess_train/acacia-train/'
with open(path + 'train.txt', 'w') as wf:
    imfiles = os.listdir(os.path.join(path + 'IMG_Palm'))
    imfiles = natsort.natsorted(imfiles)
    frame_id = [os.path.basename(fi)[:-4] for fi in imfiles]
    for frame in frame_id:
        print('frame:', frame)
        wf.write('%s \n' % (frame))

print('ok,get train.txt finished')
# ---------------------------------------Forth step:data augmentation------------------------------
image_path = path

origin_acacia_path = os.listdir(os.path.join(image_path + 'IMG_Palm'))
origin_acacia_path = natsort.natsorted(origin_acacia_path)

if DATA_AUG == True:

    from datasets.data_augmentation import data_augmentation
# ------------------------------data_load-----------------
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        for i in origin_acacia_path:
            image_i = cv2.imread(os.path.join(image_path + 'IMG_Palm' + '/' + i))
            im = data_augmentation(image_i)
            cv2.imwrite(os.path.join(image_path + 'IMG_Palm' + '/' + i), im)

    print('ok,data augmentation finished')

# ---------------------------------------Fifth step:train the network-----------------------------

from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='gpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default='VGG_CNN_M_1024', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=100000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='palm_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default='palm_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--data_path', dest='data_path',
                        help='the training data',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.weights = "./pretrain_model/VGG_imagenet.npy "
    args.data_path = './preprocess_train/acacia-train'

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    imdb = get_imdb(args.imdb_name, args.data_path)
    print ('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print ('Output will be saved to `{:s}`'.format(output_dir))

    device_name = '/{}:{:d}'.format(args.device, args.device_id)
    print (device_name)

    network = get_network(args.network_name)
    print ('Use network `{:s}` in training'.format(args.network_name))

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)

