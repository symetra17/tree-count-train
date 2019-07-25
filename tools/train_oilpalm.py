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
import cfg_parameters

from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# create library files
lib = ctypes.cdll.LoadLibrary
lib_so = lib("./preprocess_train/get_cut_palm_img.so")

# data augmentation setting
DATA_AUG  = False
skip_crop = False
skip_cut  = False    # it takes quite long

try:
    os.mkdir('temp')
except:
    pass

ResultName_ = os.path.join('temp', 'cropped_inp.jpg')


def crop_image(inpp, outpp, x, y, w, h):
    print 'Cropping image ', ResultName_
    t0 = time.time()
    img = cv2.imread(inpp)
    img = img[y:y+h, x:x+w]
    cv2.imwrite(outpp, img)    
    print 'Crop image done ', int(time.time()-t0), 'sec'
    
if (not skip_crop):
    crop_image(cfg_parameters.img_path_cfg, ResultName_, 
        cfg_parameters.ImgBeginX_cfg, cfg_parameters.ImgBeginY_cfg, 
        cfg_parameters.Dst_img_width_cfg, cfg_parameters.Dst_img_height_cfg)


if (not skip_cut):
    # Second step: slice the cropped images and label file, 
    # generate small images and gt_palm.txt

    palm_file = ResultName_
    print 'palm_file ', ResultName_
    palmxyfile = cfg_parameters.palmxyfile_cfg
    print 'label file ', palmxyfile
    N_small_img = 4000
    Size_small_img = 500
    print 'Size_small_img', Size_small_img
    
    palm_box  = "./preprocess_train/oilpalm-train/gt_palm.txt"
    img_dir   = "./preprocess_train/oilpalm-train/IMG_Palm"
    img_name_ = "./preprocess_train/oilpalm-train/IMG_Palm/IMG_"

    if not os.path.exists('./preprocess_train/oilpalm-train'):
        os.mkdir('./preprocess_train/oilpalm-train')

    if os.path.exists(palm_box):
        os.remove(palm_box)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)
    
    cut = lib_so.cut_img(palm_file, palmxyfile, N_small_img, Size_small_img, 
            palm_box, img_name_, 
            cfg_parameters.ImgBeginX_cfg, cfg_parameters.ImgBeginY_cfg)
    
    print('cut finished')
    raw_input('press enter to continue')


#---------------Third step:GET TRAIN.TXT CORRESPOND TO THE CUTTED IMAGE------
path = './preprocess_train/oilpalm-train/'
with open(path + 'train.txt', 'w') as wf:
    imfiles = os.listdir(os.path.join(path + 'IMG_Palm'))
    imfiles = natsort.natsorted(imfiles)
    frame_id = [os.path.basename(fi)[:-4] for fi in imfiles]
    for frame in frame_id:
        # print('frame:', frame)
        wf.write('%s \n' % (frame))

print('ok, get train.txt finished')
quit()

# ------------------------Forth step:data augmentation------------------------------
if DATA_AUG == True:
    image_path = path
    origin_acacia_path = os.listdir(os.path.join(image_path + 'IMG_Palm'))
    origin_acacia_path = natsort.natsorted(origin_acacia_path)
    from datasets.data_augmentation import data_augmentation
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        for i in origin_acacia_path:
            image_i = cv2.imread(os.path.join(image_path + 'IMG_Palm' + '/' + i))
            im = data_augmentation(image_i)
            cv2.imwrite(os.path.join(image_path + 'IMG_Palm' + '/' + i), im)

    print('ok, data augmentation finished')

quit()

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

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.weights = cfg_parameters.pretrain_weights
    args.data_path = '/home/ins/tree-count-train/preprocess_train/oilpalm-train'

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
