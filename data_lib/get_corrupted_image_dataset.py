"""Corrupted CIFAR10/100
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from data_lib import transforms
import data_lib.cifar100_corrupted.cifar100_corrupted as cifar100_corrupted


CIFAR10_CORRUPTIONS = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'frosted_glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic',
    'pixelate',
    'jpeg_compression',
]

CIFAR100_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "frosted_glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic",
    "pixelate",
    "jpeg_compression",
]

CIFAR_SEVERITY = [1, 2, 3, 4, 5]

DATASET_SHAPES = {
  'cifar10_corrupted': (32, 32, 3),
  'cifar100_corrupted': (32, 32, 3),
}


def get_cifar10_corrupted(data_root: str,
                          corruption: str,
                          severity: int):
  """
  Args:
      corruption: string from CIFAR10_CORRUPTIONS
      severity: indicating severity of corruption (1-5)

  Returns:
      Corrupted CIFAR10
  """

  dataset_name = f'cifar10_corrupted/{corruption}_{severity}'
  data = tfds.load(dataset_name, data_dir=data_root, split='test', shuffle_files=False)

  data = data.map(transforms.get_rescaling(255))
  label_to_one_hot = transforms.get_label_to_one_hot(
      num_classes=10)
  data = data.map(label_to_one_hot).map(
      transforms.produce_tuple)

  # cache dataset
  data = data.cache()

  return data


def get_cifar100_corrupted(data_root: str,
                           corruption: str,
                           severity: int):
  """
  Args:
      corruption: string from CIFAR100_CORRUPTIONS
      severity: indicating severity of corruption (1-5)

  Returns:
      Corrupted CIFAR100
  """
  dataset_name = f'cifar100_corrupted/{corruption}_{severity}'
  data = tfds.load(dataset_name, data_dir=data_root, split='test', shuffle_files=False)

  data = data.map(transforms.get_rescaling(255))
  label_to_one_hot = transforms.get_label_to_one_hot(
      num_classes=100)
  data = data.map(label_to_one_hot).map(
      transforms.produce_tuple)

  # cache dataset
  data = data.cache()

  return data


CORRUPTED_DATASETS = {
    'cifar10': (CIFAR10_CORRUPTIONS, CIFAR_SEVERITY, get_cifar10_corrupted),
    'cifar100': (CIFAR100_CORRUPTIONS, CIFAR_SEVERITY, get_cifar100_corrupted)
}


def get_corruptions_and_severity(dataset: str):

  if dataset not in CORRUPTED_DATASETS:
    raise ValueError(f'There exists no corrupted dataset for {dataset}!')

  return CORRUPTED_DATASETS[dataset]
