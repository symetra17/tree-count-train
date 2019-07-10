# coding:utf-8
import cv2
import os
import natsort
import numpy as np
import tensorflow as tf

def data_augmentation(image):
    if np.random.random(1) < 0.5:
        if np.random.random(1) < 0.1:
            im = tf.image.adjust_gamma(image)
            # print('gamma')
        elif np.random.random(1) < 0.2:
            im = tf.image.random_contrast(image, lower=0.4, upper=0.8)
            # print('contrast')
        elif np.random.random(1) < 0.3:
            im = tf.image.adjust_brightness(image, 0.15)
            # print('brightness')
        elif np.random.random(1) < 0.4:
            im = tf.image.random_hue(image, max_delta=0.17)
            # print('Hue')
        else:
            im = tf.image.random_saturation(image, lower=0.4, upper=0.8)
            # print('saturation')

        image_numpy = im.eval()
    else:
        image_numpy = image
    # print('no change')
    return image_numpy

