from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cPickle as pkl

import numpy as np
import tensorflow as tf
import input_data_c3d

from resnet.utils import logger

log = logger.get()

# Global constants describing the CIFAR-10 data set.
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 56
NUM_CLASSES = 51
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 9357
NUM_TEST_IMG = 3780


def unpickle(file):
  fo = open(file, 'rb')
  dict = pkl.load(fo)
  fo.close()
  return dict


def get_hmdb51_batch(self, batchsize, seq_len, testing=False):
  """Reads and parses HMDB51 batches from HMDB51 data files"""
  mode = 'test' if testing else 'train'
  images, labels, _, _, _ = input_data_c3d.read_clip_and_label(
                    filename='./data_list/HMDB51/list1/%s.list' % mode,
                    batch_size = batchsize,
                    num_frames_per_clip = seq_len,
                    crop_size = 56,
                    shuffle = True)
  
  return np.squeeze(images), labels



def read_HMDB51_img(data_folder)
  """Reads and parses examples from HMDB51 data files"""

  train_img = np.load(os.path.join(data_folder, 'train_imgs.npy'))
  train_label = np.load(os.path.join(data_folder, 'train_labels.npy'))

  test_img = np.load(os.path.join(data_folder, 'test_imgs.npy'))
  test_label = np.load(os.path.join(data_folder, 'test_labels.npy'))

  print(train_img.shape)
  print(test_img.shape)

  mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

  hmdb51_img_data = {}
  hmdb51_img_data["train_img"] = train_img - mean_img
  hmdb51_img_data["test_img"] = test_img - mean_img
  hmdb51_img_data["train_label"] = train_label
  hmdb51_img_data["test_label"] = test_label

  return hmdb51_img_data

def hmdb51_tf_preprocess(random_crop=True, random_flip=True, whiten=True):
  image_size = 56
  inp = tf.placeholder(tf.float32, [image_size, image_size, 3])
  image = inp
  # image = tf.cast(inp, tf.float32)
  if random_crop:
    log.info("Apply random cropping")
    image = tf.image.resize_image_with_crop_or_pad(inp, image_size + 4,
                                                   image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
  if random_flip:
    log.info("Apply random flipping")
    image = tf.image.random_flip_left_right(image)
  # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
  # image = tf.image.random_brightness(image, max_delta=63. / 255.)
  # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
  if whiten:
    log.info("Apply whitening")
    image = tf.image.per_image_whitening(image)
  return inp, image


