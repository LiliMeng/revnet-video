from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cPickle as pkl

from numpy import genfromtxt
import numpy as np
import tensorflow as tf

from resnet.utils import logger

log = logger.get()



def unpickle(file):
  fo = open(file, 'rb')
  dict = pkl.load(fo)
  fo.close()
  return dict


def read_hmdb51(data_folder):
  """ Reads and parses examples from CIFAR10 data files """

  train_img = np.load(os.path.join(data_folder, 'train_imgs.npy'))
  #train_label = genfromtxt(os.path.join(data_folder, 'train_labels.txt'), delimiter = ",")
  train_label = np.load(os.path.join(data_folder, 'train_labels.npy'))
  test_img = np.load(os.path.join(data_folder, 'test_imgs.npy'))
  #test_label = genfromtxt(os.path.join(data_folder, 'test_labels.txt'), delimiter = ",")
  test_label = np.load(os.path.join(data_folder, 'test_labels.npy'))

  print(train_img.shape)
  print(train_label.shape)
  print(test_img.shape)
  print(test_label.shape)

  mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

  hmdb51_data = {}
  hmdb51_data["train_img"] = train_img - mean_img
  hmdb51_data["test_img"] = test_img - mean_img
  hmdb51_data["train_label"] = train_label
  hmdb51_data["test_label"] = test_label

  return hmdb51_data


def hmdb51_optflow_preprocess(config, random_crop=True, random_flip=True, whiten=True):
  img_width = config.width
  img_height = config.height
  
  inp = tf.placeholder(tf.float32, [img_height, img_width, config.num_optflow_channel])

  image = inp
  # image = tf.cast(inp, tf.float32)
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


def hmdb51_rgb_preprocess(config, random_crop=True, random_flip=True, whiten=True):
  img_width = config.width
  img_height = config.height
  
  inp = tf.placeholder(tf.float32, [img_height, img_width, config.num_rgb_channel])

  image = inp
  # image = tf.cast(inp, tf.float32)
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
