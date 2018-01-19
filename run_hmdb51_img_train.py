#!/usr/bin/env python
"""
Train a CNN on HMDB51.
Author: Lili Meng (menglili@cs.ubc.ca)  
Revised from Mengye Ren's original run_cifar10_train.py

Usage:
python run_hmdb51_train.py    --model           [MODEL NAME]          \
                             --config          [CONFIG FILE]         \
                             --env             [ENV FILE]            \
                             --dataset         [DATASET]             \
                             --data_folder     [DATASET FOLDER]      \
                             --validation                            \
                             --no_validation                         \
                             --logs            [LOGS FOLDER]         \
                             --results         [SAVE FOLDER]         \
                             --gpu             [GPU ID]

Flags:
  --model: See resnet/configs/cifar_exp_config.py. Default resnet-32.
  --config: Not using the pre-defined configs above, specify the JSON file
  that contains model configurations.
  --dataset: Dataset name. Available options are: 1) cifar-10 2) cifar-100.
  --data_folder: Path to data folder, default is data/{DATASET}.
  --validation: Evaluating experiments on validation set.
  --no_validation: Evaluating experiments on test set.
  --logs: Path to logs folder, default is logs/default.
  --results: Path to save folder, default is results.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm

from resnet.configs import get_config, get_config_from_json
#from resnet.data import get_dataset
from resnet.data.get_hmdb51_dataset import *
from resnet.data import input_data_c3d
from resnet.models import get_model
from resnet.utils import ExperimentLogger, FixedLearnRateScheduler
from resnet.utils import logger, gen_id


log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file.")
flags.DEFINE_string("dataset", "hmdb51_img", "Dataset name.")
flags.DEFINE_string("id", None, "Experiment ID.")
#flags.DEFINE_string("batchsize", None, "Experiment ID.")
flags.DEFINE_string("results", "./results/cifar", "Saving folder.")
flags.DEFINE_string("logs", "./logs/public", "Logging folder.")
flags.DEFINE_string("model", "resnet-32", "Model type.")
flags.DEFINE_bool("validation", False, "Whether run validation set.")
flags.DEFINE_bool("restore", False, "Whether restore model.")
FLAGS = flags.FLAGS


def _get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return get_config_from_json(FLAGS.config)
  else:
    if FLAGS.restore:
      save_folder = os.path.realpath(
          os.path.abspath(os.path.join(FLAGS.results, FLAGS.id)))
      return get_config_from_json(os.path.join(save_folder, "conf.json"))
    else:
      return get_config(FLAGS.model)


def _get_models(config):
  # Builds models.
  log.info("Building models")
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      with log.verbose_level(2):
        m = get_model(
            config.model_class,
            config,
            is_training=True,
            num_pass=1,
            batch_size=config.batch_size)

  with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
      with log.verbose_level(2):
        mvalid = get_model(
            config.model_class,
            config,
            is_training=False,
            batch_size=config.batch_size)
  return m, mvalid




def train_step(sess, model, img_batch, label_batch):
  """Train step"""
  return model.train_step(sess, img_batch, label_batch)


def evaluate(sess, model, data_iter):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  for batch in data_iter:
    y = model.infer_step(sess, batch["img"])
    pred_label = np.argmax(y, axis=1)
    num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
    count += pred_label.size
  acc = (num_correct / count)
  return acc

def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(json.dumps(dict(config.__dict__)))
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def train_model(exp_id,
                config,
                dataset,
                #train_iter,
                #test_iter,
                trainval_iter=None,
                save_folder=None,
                logs_folder=None):
  """Trains a CIFAR model.

  Args:
      exp_id: String. Experiment ID.
      config: Config object
      train_data: Dataset iterator.
      test_data: Dataset iterator.

  Returns:
      acc: Final test accuracy
  """
  # log.info("Config: {}".format(config.__dict__))
  batch_size=30
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)

  # Initializes variables.
  with tf.Graph().as_default():
    np.random.seed(0)
    if not hasattr(config, "seed"):
      tf.set_random_seed(1234)
      log.info("Setting tensorflow random seed={:d}".format(1234))
    else:
      log.info("Setting tensorflow random seed={:d}".format(config.seed))
      tf.set_random_seed(config.seed)
    m, mvalid = _get_models(config)

    with tf.Session() as sess:
      saver = tf.train.Saver()
      if FLAGS.restore:
        log.info("Restore checkpoint \"{}\"".format(save_folder))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
      else:
        sess.run(tf.global_variables_initializer())
      niter_start = int(m.global_step.eval())
      w_list = tf.trainable_variables()
      log.info("Model initialized.")
      num_params = np.array([
          np.prod(np.array([int(ss) for ss in w.get_shape()])) for w in w_list
      ]).sum()
      log.info("Number of parameters {}".format(num_params))

      # Set up learning rate schedule.
      if config.lr_scheduler_type == "fixed":
        lr_scheduler = FixedLearnRateScheduler(
            sess,
            m,
            config.base_learn_rate,
            config.lr_decay_steps,
            lr_list=config.lr_list)
      else:
        raise Exception("Unknown learning rate scheduler {}".format(
            config.lr_scheduler))

      for niter in tqdm(range(niter_start, config.max_train_iter), desc=exp_id):
        lr_scheduler.step(niter)
        if FLAGS.dataset == 'hmdb51_img':
          image_batch, label_batch = dataset.get_hmdb51_batch(batch_size, 1)       
        else:
          raise Exception("dataset not implemented yet for video activity recogntion")
        ce  = train_step(sess, m, image_batch, label_batch)
        #ce = train_step(sess, m, train_iter.next())

        if (niter + 1) % config.disp_iter == 0 or niter == 0:
          exp_logger.log_train_ce(niter, ce)

        if (niter + 1) % config.valid_iter == 0 or niter == 0:
         
          num_test_steps = int(dataset._num_test / batch_size)
          num_correct = 0.0
          count = 0
          for test_step in xrange(num_test_steps):
            if FLAGS.dataset == "hmdb51_img":
              test_image_batch, test_label_batch = dataset.get_hmdb51_batch(batch_size, 1, testing=True)
            else:
              raise Exception("dataset not implemented yet for video action recognition")
            y = mvalid.infer_step(sess, test_image_batch)
            pred_label = np.argmax(y, axis=1)
            num_correct += np.sum(np.equal(pred_label, test_label_batch).astype(float))
            count += pred_label.size
          acc = (num_correct / count)
          
          #acc = evaluate(sess, mvalid, test_iter)
          exp_logger.log_valid_acc(niter, acc)

        if (niter + 1) % config.save_iter == 0 or niter == 0:
          save(sess, saver, m.global_step, config, save_folder)
          exp_logger.log_learn_rate(niter, m.lr.eval())

      #test_iter.reset()
      #acc = evaluate(sess, mvalid, test_iter)
  return acc


def main():
  # Loads parammeters.
  config = _get_config()
  if FLAGS.dataset == "hmdb51_img":
    config.num_classes = 51
  else:
    raise ValueError("Unknown dataset name {}".format(FLAGS.dataset))

   # Fetch data 
  dataset = DataSet(batchsize=30, 
                    testbatchsize=30,  
                    dataset=FLAGS.dataset)

  save_folder = "/media/lci/storage/Video/revnet-video/save_folder/"
  logs_folder = "/media/lci/storage/Video/revnet-video/save_folder/logs/"
  
  # Trains a model.
  acc = train_model(
      FLAGS.id,
      config,
      dataset,
      #train_data,
      #test_data,
      #trainval_data,
      save_folder=save_folder,
      logs_folder=logs_folder)
  log.info("Final test accuracy = {:.3f}".format(acc * 100))


if __name__ == "__main__":
  main()
