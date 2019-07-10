# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
import datasets.palm_Tree as palm_Tree

__sets = {}
image_set = 'train'
# data_path = '/mnt/a409/users/tongpinmo/projects/oilpalm/oilpalm_count_train/preprocess_train/acacia-train'

def get_imdb(name,data_path):
    """Get an imdb (image database) by name."""
    __sets[name] = (lambda image_set=image_set, data_path=data_path:palm_Tree.palm_Tree(image_set, data_path))
    print('name: ',name)
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
