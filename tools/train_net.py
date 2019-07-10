#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import os
import cv2
import sys
import pdb
import natsort
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# data augmentation setting
DATA_AUG = False

image_path = './preprocess_train/acacia-train/'

origin_acacia_path = os.listdir(os.path.join(image_path + 'IMG_Palm'))
origin_acacia_path = natsort.natsorted(origin_acacia_path)

if DATA_AUG == True:

    from datasets.data_augmentation import data_augmentation
# ------------------------------data_load-----------------
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        for i in origin_acacia_path:
            print('i: ', i)
            image_i = cv2.imread(os.path.join(image_path + 'IMG_Palm' + '/' + i))

            im = data_augmentation(image_i)
            # print('type(im): ',type(im))
            cv2.imwrite(os.path.join(image_path + 'IMG_Palm' + '/' + i), im)

    print('ok,data augmentation finished')


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
                        default='./pretrain_model/VGG_imagenet.npy', type=str)
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
    parser.add_argument('--data_path',dest='data_path',
                        help='the training data',
                        default=None,type=str)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.weights = "./pretrain_model/VGG_imagenet.npy"
    args.data_path = '/mnt/a409/users/tongpinmo/projects/oilpalm/oilpalm_count_train/preprocess_train/acacia-train'

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
    imdb = get_imdb(args.imdb_name,args.data_path)
    print('args.imdb_name: ',args.imdb_name)            #'palm_train'
    print('imdb.name: ',imdb.name)
    print ('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print ('Output will be saved to `{:s}`'.format(output_dir))

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print (device_name)

    network = get_network(args.network_name)
    print ('Use network `{:s}` in training'.format(args.network_name))

    train_net(network, imdb, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
