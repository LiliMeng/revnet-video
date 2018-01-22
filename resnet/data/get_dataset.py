from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.data.cifar10 import CIFAR10Dataset
from resnet.data.cifar100 import CIFAR100Dataset
from resnet.utils.batch_iter import BatchIterator
from resnet.utils.concurrent_batch_iter import ConcurrentBatchIterator
from resnet.data.hmdb51 import HMDB51_Dataset


path_op = "./dataset/op_txtfiles"
path_img = "./dataset/img_txtfiles"


def get_dataset(name,
                split,
                config,
                data_aug=True,
                cycle=True,
                prefetch=True,
                shuffle=True,
                num_batches=-1):
  """Gets a dataset.

  Args:
      name: "cifar-10" or "cifar-100".
      split: "train", "traintrain", "trainval", or "test".

  Returns:
      dp: Dataset Iterator.
  """
  print(name)
  if name == "cifar-10":
    dp = CIFAR10Dataset(
        "data/cifar-10", split, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=config.batch_size,
        config=config,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  elif name == "cifar-100":
    dp = CIFAR100Dataset(
        "data/cifar-100", split, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=config.batch_size,
        config=config,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  elif name == "hmdb51-op":
    dp = HMDB51_Dataset(
      config=config, split=split, folder=path_op, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=config.batch_size,
        config=config,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  elif name == "hmdb51-img":
    dp = HMDB51_Dataset(
      config=config, split=split, folder=path_img, data_aug=data_aug, whiten=False, div255=False)
    return get_iter(
        dp,
        batch_size=config.batch_size,
        config=config,
        shuffle=shuffle,
        cycle=cycle,
        prefetch=prefetch,
        num_worker=20,
        queue_size=300,
        num_batches=num_batches)
  else:
    raise Exception("Unknown dataset {}".format(dataset))


def get_iter(dataset,
             batch_size,
             config,
             shuffle=False,
             cycle=False,
             log_epoch=-1,
             seed=0,
             prefetch=False,
             num_worker=20,
             queue_size=300,
             num_batches=-1):
  """Gets a data iterator.

  Args:
      dataset: Dataset object.
      batch_size: Mini-batch size.
      shuffle: Whether to shuffle the data.
      cycle: Whether to stop after one full epoch.
      log_epoch: Log progress after how many iterations.

  Returns:
      b: Batch iterator object.
  """
  b = BatchIterator(
      dataset.get_size(config),
      batch_size=batch_size,
      shuffle=shuffle,
      cycle=cycle,
      get_fn=dataset.get_batch_idx,
      log_epoch=log_epoch,
      seed=seed,
      num_batches=num_batches)
  if prefetch:
    b = ConcurrentBatchIterator(
        b, max_queue_size=queue_size, num_threads=num_worker, log_queue=-1)
  return b
