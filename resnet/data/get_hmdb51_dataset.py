from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import scipy.io
import os
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib, xrange
import tensorflow as tf
import resnet.data.input_data_c3d 
from PIL import Image


class DataSet(object):
    """
    Loads data and Keeps track of dataset, data paths, image height/width, channels, number of data/classes 
    """
    def __init__(self, batchsize, testbatchsize, dataset="hmdb51_img"):
         
        self._dataset = dataset
        self._batchsize = batchsize
        self._test_batchsize = testbatchsize

    
        if self._dataset == "hmdb51_img":
            
            print("HMDB51 RGB dataset is used now")
            
            self._height = 56
            self._width = 56
            self._channels = 3
            self._num_train = 3570
            self._num_test = 1530
            self._num_classes = 51
            self._padding = 0

        else: 
            raise Exception("Dataset: %s has not been implemented yet. Please check spelling." % dataset)


    def get_hmdb51_batch(self, batchsize, seq_len, testing=False):
      """Reads and parses HMDB51 batches from HMDB51 data files"""
      mode = 'test' if testing else 'train'
      images, labels, _, _, _ = resnet.data.input_data_c3d.read_clip_and_label(
                        filename='./data_list/HMDB51/list1/%s.list' % mode,
                        batch_size = batchsize,
                        num_frames_per_clip = seq_len,
                        crop_size = 56,
                        shuffle = True)
      
      return np.squeeze(images), labels


