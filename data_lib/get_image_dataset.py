# Lint as: python3
"""Gets image classification datasets (MNIST, FashionMNIST, CIFAR10 & SVHN)"""

import random
from typing import Optional
import tensorflow as tf
import tensorflow_datasets as tfds

from data_lib import transforms


DATASET_SHAPES = {
  'mnist': (28, 28, 1),
  'fashion_mnist': (28, 28, 1),
  'cifar10': (32, 32, 3),
  'cifar100': (32, 32, 3),
  'svhn_cropped': (32, 32, 3)
}

DATASET_NUM_CLASSES = {
  'mnist': 10,
  'fashion_mnist': 10,
  'cifar10': 10,
  'cifar100': 100,
  'svhn_cropped': 10,
  'omniglot': 10
}


def apply_perturbation(perturbation_type: str,
                       perturbation: float,
                       data_train,
                       data_val,
                       data_test):
  """
  Applies pertrubation to dataset.

  Args:
    perturbation_type: string. rotation/additive_gaussian/jpeg_quality/brightness/hflip/vflip
    perturbation: float. Magnitude of perturbation
    data_train:
    data_val:
    data_test:

  Returns:

  """

  if perturbation_type == 'rotation':
    data_train = data_train.map(
      transforms.get_rotate_transform(perturbation))
    data_val = data_val.map(
      transforms.get_rotate_transform(perturbation))
    data_test = data_test.map(
      transforms.get_rotate_transform(perturbation))
  elif perturbation_type == 'additive_gaussian':
    data_train = data_train.map(
      transforms.get_additive_gaussian_transform(perturbation))
    data_val = data_val.map(
      transforms.get_additive_gaussian_transform(perturbation))
    data_test = data_test.map(
      transforms.get_additive_gaussian_transform(perturbation))
  elif perturbation_type == 'jpeg_quality':
    data_train = data_train.map(
      transforms.get_jpeg_quality_transform(perturbation))
    data_val = data_val.map(
      transforms.get_jpeg_quality_transform(perturbation))
    data_test = data_test.map(
      transforms.get_jpeg_quality_transform(perturbation))
  elif perturbation_type == 'brightness':
    data_train = data_train.map(
      transforms.get_brightness_transform(perturbation))
    data_val = data_val.map(
      transforms.get_brightness_transform(perturbation))
    data_test = data_test.map(
      transforms.get_brightness_transform(perturbation))
  elif perturbation_type == 'hflip':
    data_train = data_train.map(
      transforms.flip_horizontal)
    data_val = data_val.map(
      transforms.flip_horizontal)
    data_test = data_test.map(
      transforms.flip_horizontal)
  elif perturbation_type == 'vflip':
    data_train = data_train.map(
      transforms.flip_vertical)
    data_val = data_val.map(
      transforms.flip_vertical)
    data_test = data_test.map(
      transforms.flip_vertical)

  return data_train, data_val, data_test


def get_image_dataset(dataset: str,
                      data_root: str,
                      perturbation_type: Optional[str] = None,
                      perturbation: float = 0.0,
                      **kwargs):
  return get_classification_dataset(
      dataset,
      data_root,
      perturbation_type=perturbation_type,
      perturbation=perturbation,
      **kwargs)


def get_classification_dataset(dataset: str,
                               data_root: str,
                               data_augmentation: bool = True,
                               perturbation_type: Optional[str] = None,
                               perturbation: float = 0.0):
  """
  Args:
      dataset: string. mnist, omniglot, fashion_mnist, cifar10 or svhn_cropped
      data_root: string. folder (from) where datasets are stored (loaded)
      preprocess: bool. preprocess trainset or not.
      data_augmentation: bool.
      perturbation_type; str. perturbation of data: rotation or
        additive_gaussian, hflip, vflip,
      perturbation: float. magnitude of perturbation

  Returns:
      Image dataset
  """

  seed = random.randint(1, 1000000)
  tf.random.set_seed(1337)  # fixed train-validation-split

  data_train_val = tfds.load(dataset, data_dir=data_root, split='train', shuffle_files=True)
  data_test = tfds.load(dataset, data_dir=data_root, split='test', shuffle_files=False)

  N = len(list(data_train_val))
  # N = data_train_val.cardinality().numpy()
  data_train_val = data_train_val.shuffle(N)
  data_train = data_train_val.take(int(0.8*N))
  data_val = data_train_val.skip(int(0.8*N))

  tf.random.set_seed(seed)  # non-deterministic

  data_train = data_train.map(transforms.get_rescaling(255))
  data_val = data_val.map(transforms.get_rescaling(255))
  data_test = data_test.map(transforms.get_rescaling(255))

  if perturbation_type is not None:
    data_train, data_val, data_test = apply_perturbation(perturbation_type=perturbation_type,
                                                         perturbation=perturbation,
                                                         data_train=data_train,
                                                         data_val=data_val,
                                                         data_test=data_test)

  label_to_one_hot = transforms.get_label_to_one_hot(
      num_classes=DATASET_NUM_CLASSES[dataset])
  data_train = data_train.map(label_to_one_hot).map(
      transforms.produce_tuple)
  data_val = data_val.map(label_to_one_hot).map(
      transforms.produce_tuple)
  data_test = data_test.map(label_to_one_hot).map(
      transforms.produce_tuple)

  if dataset == 'omniglot':
    data_train = data_train.map(lambda input0, input1: (tf.image.resize(input0, [28, 28]), input1))
    data_val = data_val.map(lambda input0, input1: (tf.image.resize(input0, [28, 28]), input1))
    data_test = data_test.map(lambda input0, input1: (tf.image.resize(input0, [28, 28]), input1))

    map_func = lambda input0, input1: (1 - tf.reduce_mean(input0, axis=-1, keepdims=True), input1)
    data_train = data_train.map(map_func)
    data_val = data_val.map(map_func)
    data_test = data_test.map(map_func)
  elif dataset == 'stl10':
    data_train = data_train.map(lambda input0, input1: (tf.image.resize(input0, [32, 32]), input1))
    data_val = data_val.map(lambda input0, input1: (tf.image.resize(input0, [32, 32]), input1))
    data_test = data_test.map(lambda input0, input1: (tf.image.resize(input0, [32, 32]), input1))

  # cache dataset
  data_train = data_train.cache()
  data_val = data_val.cache()
  data_test = data_test.cache()

  data_train = data_train.shuffle(N)

  # Data augmentation
  if data_augmentation and dataset in [
      'cifar100', 'cifar10', 'svhn_cropped', 'stl10']:
    data_train = data_train.map(transforms.random_flip_horizontal)
    data_train = data_train.map(transforms.random_brightness(max_delta=0.5))
    data_train = data_train.map(transforms.random_contrast(lower=0.5, upper=1.2))

  return data_train, data_val, data_test


def get_ood_datasets(dataset: str,
                     data_root: str):
  """
  Returns OOD datasets for a given dataset.

  Type of OOD datasets:
    HFlip
    VFlip
    Gaussian Noise
    Rotate 90°
    OOD datasets
      MNIST: FashionMNIST, OMNIGLOT
      FashionMNIST: MNIST, OMNIGLOT
      SVHN: CIFAR10, STL10
      CIFAR10: SVHN, STL10
  """

  ood_datasets = dict()

  if dataset == 'mnist':
    ood_datasets.update({
        'fashion_mnist': get_image_dataset(dataset='fashion_mnist',
                                           data_root=data_root)[-1],
        'omniglot': get_image_dataset(dataset='omniglot',
                                      data_root=data_root)[-1]
    })
  elif dataset == 'fashion_mnist':
    ood_datasets.update({
        'mnist': get_image_dataset(dataset='mnist',
                                   data_root=data_root)[-1],
        'omniglot': get_image_dataset(dataset='omniglot',
                                      data_root=data_root)[-1]
    })
  elif dataset == 'cifar10':
    ood_datasets.update({
        'svhn_cropped': get_image_dataset(dataset='svhn_cropped',
                                          data_root=data_root)[-1],
        'cifar100': get_image_dataset(dataset='cifar100',
                                      data_root=data_root)[-1]
    })
  elif dataset == 'svhn_cropped':
    ood_datasets.update({
        'cifar10': get_image_dataset(dataset='cifar10',
                                     data_root=data_root)[-1],
        'cifar100': get_image_dataset(dataset='cifar100',
                                      data_root=data_root)[-1]
    })
  elif dataset == 'cifar100':
    ood_datasets.update({
        'cifar10': get_image_dataset(dataset='cifar10',
                                     data_root=data_root)[-1],
        'svhn_cropped': get_image_dataset(dataset='svhn_cropped',
                                      data_root=data_root)[-1]
    })

  return ood_datasets
