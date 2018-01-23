from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.data import hmdb51_input
from resnet.utils import logger

log = logger.get()


class HMDB51_Dataset():

  def __init__(self,
               config,
               folder,
               split,
               num_fold=10,
               fold_id=0,
               data_aug=False,
               whiten=False,
               div255=False):
    self.split = split
    self.data = hmdb51_input.read_hmdb51(folder)
    num_ex = config.num_train_imgs
    self.split_idx = np.arange(num_ex)
    rnd = np.random.RandomState(0)
    rnd.shuffle(self.split_idx)
    #num_valid = int(np.ceil(num_ex / num_fold))
    num_valid = config.num_test_imgs
    valid_start = fold_id * num_valid
    valid_end = min((fold_id + 1) * num_valid, num_ex)
    self.valid_split_idx = self.split_idx[valid_start:valid_end]
    self.train_split_idx = np.concatenate(
        [self.split_idx[:valid_start], self.split_idx[valid_end:]])
    if data_aug or whiten:
      with tf.device("/cpu:0"):
        if config.rgb_only == True or config.optflow_only == True:
          self.inp_preproc, self.out_preproc = hmdb51_input.hmdb51_preprocess(
              random_crop=data_aug, random_flip=data_aug, whiten=whiten, config=config)
        elif config.double_stream == True:
          self.inp_preproc, self.out_preproc, self.inp_preproc_optflow = hmdb51_input.hmdb51_double_stream_preprocess(random_crop=data_aug, random_flip=data_aug,
              whiten=whiten, config=config)
        else:
          raise Exceptioin("Not implemented yet")

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
      raise Exceptioin("not implemented yet in validation")

  def get_batch_idx(self, idx):
    if self.split == "train":
      result = {
          "img": self.data["train_img"][idx],
          "label": self.data["train_label"][idx]
      }
    elif self.split == "test":
      result = {
          "img": self.data["test_img"][idx],
          "label": self.data["test_label"][idx]
      }
    else:
      raise Exceptioin("not implemented yet in the get_batch_idx function")
    # if self.data_aug or self.whiten:
    #   img = np.zeros(result["img"].shape)
    #   for ii in range(len(idx)):
    #     img[ii] = self.session.run(
    #         self.out_preproc, feed_dict={self.inp_preproc: result["img"][ii]})
    #   result["img"] = img
    # if self.div255:
    #   result["img"] = result["img"] / 255.0
    return result
