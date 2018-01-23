#!/usr/bin/env python
"""

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
from resnet.data import get_dataset
from resnet.models import get_model
from resnet.utils import ExperimentLogger, FixedLearnRateScheduler
from resnet.utils import logger, gen_id

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file.")
flags.DEFINE_string("dataset", "cifar-10", "Dataset name.")
flags.DEFINE_string("id", None, "Experiment ID.")
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


def train_step(sess, config, model, batch_img=None, batch_op=None):
  """Train step."""
  if config.rgb_only == True:
    return model.train_step(config, sess, batch_img["img"], batch_img["label"])
  elif config.optflow_only == True:
    return model.train_step(config, sess, batch_op["img"], batch_op["label"])
  elif config.double_stream == True:
    return model.train_step(config, sess, batch_img["img"], batch_img["label"], batch_op["img"], batch_op["label"])


def evaluate(sess, config, model, data_iter_img=None, data_iter_op=None):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  if config.rgb_only == True:
    for batch in data_iter_img:
      y = model.infer_step(sess, inp_img= batch["img"])
      pred_label = np.argmax(y, axis=1)
      num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
      count += pred_label.size
    acc = (num_correct / count)
  elif config.optflow_only == True:
    for batch in data_iter_op:
      y = model.infer_step(sess, inp_op=batch["img"])
      pred_label = np.argmax(y, axis=1)
      num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
      count += pred_label.size
    acc = (num_correct / count)
  # elif config.double_stream == True:
  #    for batch in data_iter_img:
  #      y = model.
  else:
    raise Exception("Not implemented yet")
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
                train_iter_img=None,
                test_iter_img=None,
                train_iter_op=None,
                test_iter_op=None,
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
        if config.rgb_only == True:
          ce = train_step(sess, config, m, batch_img = train_iter_img.next(), batch_op = None)
        elif config.optflow_only == True:
          ce = train_step(sess, config, m, batch_img = None, batch_op = train_iter_op.next())
        elif config.double_stream == True:
          ce = train_step(sess, config, m, batch_img = train_iter_img.next(), batch_op = train_iter_op.next())
        else:
          raise Exception("Not implemented yet")

        if (niter + 1) % config.disp_iter == 0 or niter == 0:
          exp_logger.log_train_ce(niter, ce)

        if (niter + 1) % config.valid_iter == 0 or niter == 0:
          test_iter_img.reset()
          test_iter_op.reset()
          if config.rgb_only == True:
            acc = evaluate(sess, config, mvalid, data_iter_img=test_iter_img, data_iter_op=None)
          elif config.optflow_only == True:
            acc = evaluate(sess, config, mvalid, data_iter_img = None, data_iter_op=test_iter_op)
          elif config.double_stream == True:
            acc = evaluate(sess, config, mvalid, data_iter_img = test_iter_img, data_iter_op=test_iter_op)
          else:
            raise Exception("Not implemented yet")
          exp_logger.log_valid_acc(niter, acc)

        if (niter + 1) % config.save_iter == 0 or niter == 0:
          save(sess, saver, m.global_step, config, save_folder)
          exp_logger.log_learn_rate(niter, m.lr.eval())

      test_iter.reset()
      if config.rgb_only == True:
        acc = evaluate(sess, config, mvalid, test_iter_img=test_iter_img, test_iter_op=None)
      elif config.optflow_only == True:
        acc = evaluate(sess, config, mvalid, test_iter_img = None, test_iter_op=test_iter_op)
      elif config.double_stream == True:
        acc = evaluate(sess, config, mvalid, test_iter_img = test_iter_img, test_iter_op=test_iter_op)
      else:
        raise Exception("Not implemented yet")
     
  return acc


def main():
  # Loads parammeters.
  config = _get_config()
  config.num_classes = 51

  if FLAGS.validation:
    train_str = "traintrain"
    test_str = "trainval"
    log.warning("Running validation set")
  else:
    train_str = "train"
    test_str = "test"

  if FLAGS.id is None:
    dataset_name = FLAGS.dataset
    exp_id = "exp_" + dataset_name + "_" + FLAGS.model
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id
    dataset_name = exp_id.split("_")[1]

  if FLAGS.results is not None:
    save_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.results, exp_id)))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
  else:
    save_folder = None

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Configures dataset objects.
  log.info("Building dataset")
  if config.rgb_only == True:
    train_data_img = get_dataset(name="hmdb51-img", split=train_str, config=config)

    test_data_img = get_dataset(
        name="hmdb51-img", split=test_str, data_aug=False, cycle=False, prefetch=False,config=config)
        # Trains a model.
    acc = train_model(
        exp_id,
        config,
        train_data_img,
        test_data_img,
        train_iter_op = None,
        test_iter_op = None,
        save_folder=save_folder,
        logs_folder=logs_folder)
    log.info("Final test accuracy = {:.3f}".format(acc * 100))
  elif config.optflow_only == True:


    train_data_op = get_dataset(name="hmdb51-op", split=train_str, config=config)

    test_data_op = get_dataset(
        name="hmdb51-op", split=test_str, data_aug=False, cycle=False, prefetch=False,config=config)

    acc = train_model(
        exp_id,
        config,
        train_iter_img=None,
        test_iter_img=None,
        train_iter_op = train_data_op,
        test_iter_op = test_data_op,
        save_folder=save_folder,
        logs_folder=logs_folder)
    log.info("Final test accuracy = {:.3f}".format(acc * 100))

  elif config.double_stream == True:
    
    train_data_img = get_dataset(name="hmdb51-img", split=train_str, config=config)
    test_data_img = get_dataset(
        name="hmdb51-img", split=test_str, data_aug=False, cycle=False, prefetch=False,config=config)

    train_data_op = get_dataset(name="hmdb51-op", split=train_str, config=config)
    test_data_op = get_dataset(
        name="hmdb51-op", split=test_str, data_aug=False, cycle=False, prefetch=False,config=config)

    acc = train_model(
        exp_id,
        config,
        train_iter_img=train_data_img,
        test_iter_img=test_data_img,
        train_iter_op = train_data_op,
        test_iter_op = test_data_op,
        save_folder=save_folder,
        logs_folder=logs_folder)
    log.info("Final test accuracy = {:.3f}".format(acc * 100))
  else:
    raise Exception("Not implemented yet")



if __name__ == "__main__":
  main()
