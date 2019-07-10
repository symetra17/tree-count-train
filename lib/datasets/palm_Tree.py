#coding:utf-8
# --------------------------------------------------------

import datasets
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle as cPickle
import subprocess

class palm_Tree(datasets.imdb):
    def __init__(self, image_set, data_path=None):
        #checkpoints saving path
        datasets.imdb.__init__(self, 'tree_weights_' + image_set) #image_set 为train或者val或者trainval或者test。
        self._image_set = image_set # image_set以train为例
        self._data_path = data_path # 数据所在的路径，根据传进来的参数data_path而定。传进来的参数data_path在我这里就是Data/ID_card/
        self._classes = ('__background__','text') #object的类别，只有两类：背景和文本
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes))) #构成字典{'__background__':'0','text':'1'}
        self._image_ext = '.jpg' #图片后缀
        self._image_index = self._load_image_set_index() #读取train.txt，获取图片名称（该图片名称没有后缀.jpg）
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb #获取图片的gt
        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._data_path), \
                'Image Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):#获得_image_index 下标为i的图像的路径
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):#根据_image_index获取图像路径
        """
        Construct an image path from the image's "index" identifier.
        """
#        print(index)
        image_path = os.path.join(self._data_path, 'IMG_Palm', index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):#已做修改
        """
        Load the indexes listed in this dataset's image set file.
        得到图片名称的list。这个list里面是集合self._image_set=train中所有图片的名字（注意，图片名字没有后缀.jpg）
        """
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt') 
        #image_set_file是Data/ID_card/train/train.txt
        #之所以要读这个train.txt文件，是因为train.txt文件里面写的是集合train中所有图片的名字（没有后缀.jpg）
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f: #读取train.txt，获取图片名称（没有后缀.jpg）
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        读取并返回图片gt的db。这个函数就是将图片的gt加载进来。
        其中，图片的gt信息在gt_ID_card.txt文件中
        并且，图片的gt被提前放在了一个.pkl文件里面。（这个.pkl文件需要我们自己生成，代码就在该函数中）

        This function loads/saves from/to a cache file to speed up future calls.
        之所以会将图片的gt提前放在一个.pkl文件里面，是为了不用每次都再重新读图片的gt，直接加载这个文件就可以了，可以提升速度。
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        '''if os.path.exists(cache_file):#若存在cache file则直接从cache file中读取
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb'''

        gt_roidb = self._load_annotation()  #读入整个gt文件的具体实现
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    #def selective_search_roidb(self):#在没有使用RPN的时候，是这样提取候选框，fast-rcnn会用到。我直接删除了这个函数，faster-rcnn用不到     
    #def _load_selective_search_roidb(self, gt_roidb):#用不到，删除
    #def selective_search_IJCV_roidb(self):  #用不到，删除      
    #def _load_selective_search_IJCV_roidb(self, gt_roidb): #用不到，删除      

    def _load_annotation(self):
        """
        Load image and bounding boxes info from txt format.
        读取图片的gt的具体实现。
        我把train集合中所有图片的gt，集中放在了一个gt_ID_card.txt文件里面
        gt_ID_card.txt中每行的格式如下：ID_card/train/back_1.jpg 1 147 65 443 361      
        后面的四个数值分别是文本框左上角和右下角的坐标。我的图片里面只有一个文本，所以只有一组文本框的坐标
        """
        gt_roidb = []      
        txtfile = os.path.join(self._data_path, 'gt_palm.txt')
        f = open(txtfile)
        split_line = f.readline().strip().split()
        # print('split_line: ',split_line)
        print('self.num_classes: ',self.num_classes)
        num = 1
        while(split_line):
            num_objs = int(split_line[1])
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            #oilpalm: i*5  acacia: i*4
            if self._data_path.find('oilpalm') > 0:
                for i in range(num_objs):
                    x1 = float(split_line[2 + i * 5])
                    y1 = float(split_line[3 + i * 5])
                    x2 = float(split_line[4 + i * 5])
                    y2 = float(split_line[5 + i * 5])
                    cls = self._class_to_ind['text']
                    boxes[i,:] = [x1, y1, x2, y2]
                    gt_classes[i] = cls
                    overlaps[i,cls] = 1.0
            else:
                for i in range(num_objs):
                    x1 = float(split_line[2 + i * 4])
                    y1 = float(split_line[3 + i * 4])
                    x2 = float(split_line[4 + i * 4])
                    y2 = float(split_line[5 + i * 4])
                    cls = self._class_to_ind['text']
                    boxes[i,:] = [x1, y1, x2, y2]
                    gt_classes[i] = cls
                    overlaps[i,cls] = 1.0

            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_roidb.append({'boxes' : boxes, 'gt_classes': gt_classes, 'gt_overlaps' : overlaps, 'flipped' : False})
            split_line = f.readline().strip().split()

        f.close()
        return gt_roidb

    #def _write_voc_results_file(self, all_boxes):#没用，删掉        
    #def _do_matlab_eval(self, comp_id, output_dir='output'): #没用，删掉       
    #def evaluate_detections(self, all_boxes, output_dir):# 没用，删掉

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    import datasets.palm_Tree #作了修改
    d = datasets.palm_Tree('train', 'Data/company_palm')#datasets.ID_card()在factory.py中用到了，
    res = d.roidb
    from IPython import embed; embed()
