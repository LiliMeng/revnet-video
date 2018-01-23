from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.data import hmdb51_input
from resnet.utils import logger

log = logger.get()


class HMDB51_img_op_Dataset():

  def __init__(self,
               config,
               img_folder,
               op_folder,
               split,
               num_fold=10,
               fold_id=0,
               data_aug=False,
               whiten=False,
               div255=False):
    self.split = split
    self.img_data = hmdb51_input.read_hmdb51(img_folder)
    self.op_data = hmdb51_input.read_hmdb51(op_folder)
    num_ex = config.num_train_imgs
    self.split_idx = np.arange(num_ex)
    rnd = np.random.RandomState(0)
    rnd.shuffle(self.split_idx)

    num_valid = config.num_test_imgs
    valid_start = fold_id * num_valid
    valid_end = min((fold_id + 1) * num_valid, num_ex)
    self.valid_split_idx = self.split_idx[valid_start:valid_end]
    self.train_split_idx = np.concatenate(
        [self.split_idx[:valid_start], self.split_idx[valid_end:]])
    if data_aug or whiten:
      with tf.device("/cpu:0"):
        self.inp_preproc_img, self.out_preproc_img, self.inp_preproc_op, self.out_preproc_op = hmdb51_input.hmdb51_tf_preprocess(
            random_crop=data_aug, random_flip=data_aug, whiten=whiten, config=config)
      self.session = tf.Session()
    self.data_aug = data_aug
    self.whiten = whiten
    self.div255 = div255
    if div255 and whiten:
      log.fatal("Applying both /255 and whitening is not recommended.")

  def get_size(self, config):
    if self.split == "train":
      return config.num_train_imgs
    elif self.split == "test":
      return config.num_test_imgs
    else:
      raise Exception("not implemented yet in validation")

  def get_batch_idx(self, idx):
    if self.split == "train":
      result = {
          "img_data": self.img_data["train_img"][idx],
          "img_label": self.img_data["train_label"][idx],
          "op_data": self.op_data["train_img"][idx],
          "op_label": self.op_data["train_label"][idx]
      }
    elif self.split == "test":
      result = {
          "img_data": self.img_data["test_img"][idx],
          "img_label": self.img_data["test_label"][idx],
          "op_data": self.op_data["test_img"][idx],
          "op_label": self.op_data["test_label"][idx]
      }
    else:
      raise Exceptioin("not implemented yet in the get_batch_idx function")
    if self.data_aug or self.whiten:
      img = np.zeros(result["img_data"].shape)
      for ii in range(len(idx)):
        img[ii] = self.session.run(
            self.out_preproc_img, feed_dict={self.inp_preproc_img: result["img_data"][ii]})
      result["img_data"] = img
    if self.div255:
      result["img_data"] = result["img_data"] / 255.0
    return result
