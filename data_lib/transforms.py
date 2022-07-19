# Lint as: python3
"""Data transformations."""

import math
import random
import tensorflow as tf
import tensorflow_addons.image as tfaimg


def random_flip_horizontal(inputs, *args):
  """Flips image randomly horizontally"""
  if isinstance(inputs, dict):
    return {
        'image': tf.image.random_flip_left_right(inputs['image']),
        'label': inputs['label']
    }
  else:
    return (tf.image.random_flip_left_right(inputs), args[0])


def random_flip_horizontal_with_label(inputs, *args):
  """Flips image randomly horizontally"""
  if random.random() <= 0.5:
    if isinstance(inputs, dict):
      return {
          'image': tf.image.flip_left_right(inputs['image']),
          'label': tf.image.flip_left_right(inputs['label']),
      }
    else:
      return (tf.image.flip_left_right(inputs),
              tf.image.flip_left_right(args[0]))
  else:
    return inputs


def get_random_crop(crop_size_img: tuple):

  def random_crop(inputs, *args):
    """Randomly crops image and segmentation gt"""

    # seed
    seed = tf.random.uniform(shape=(2,), minval=1, maxval=10000, dtype=tf.int32)
    if isinstance(inputs, dict):
      return {
          'image':
              tf.image.stateless_random_crop(
                  value=inputs['image'], size=crop_size_img, seed=seed),
          'label': inputs['label']
      }
    else:
      return (tf.image.stateless_random_crop(
          value=inputs, size=crop_size_img, seed=seed), args[0])

  return random_crop


def get_random_crop(crop_size_img: tuple):
  def random_crop(inputs):
    """Randomly crops image and segmentation gt"""

    # seed
    seed = tf.random.uniform(shape=(2,), minval=1, maxval=10000, dtype=tf.int32)
    return {
        'image': tf.image.stateless_random_crop(
            value=inputs['image'], size=crop_size_img, seed=seed),
        'label': inputs['label'],
    }
  return random_crop


def get_random_crop_with_label(crop_size_img: tuple, crop_size_lbl: tuple):

  def random_crop_with_label(inputs, *args):
    """Randomly crops image and segmentation gt"""

    # seed
    seed = tf.random.uniform(shape=(2,), minval=1, maxval=10000, dtype=tf.int32)
    if isinstance(inputs, dict):
      return {
          'image':
              tf.image.stateless_random_crop(
                  value=inputs['image'], size=crop_size_img, seed=seed),
          'label':
              tf.image.stateless_random_crop(
                  value=inputs['label'], size=crop_size_lbl, seed=seed)
      }
    else:
      return (tf.image.stateless_random_crop(
          value=inputs, size=crop_size_img, seed=seed),
              tf.image.stateless_random_crop(
                  value=args[0], size=crop_size_lbl, seed=seed))
  return random_crop_with_label


def flip_vertical(inputs):
  """Flips image vertically"""
  return {
      'image': tf.image.flip_up_down(inputs['image']),
      'label': inputs['label']
  }


def flip_horizontal(inputs):
  """Flips image vertically"""
  return {
      'image': tf.image.flip_left_right(inputs['image']),
      'label': inputs['label']
  }


def random_brightness(max_delta: float = 0.5):

  def _random_brightness(inputs, *args):
    """Random brightness adjustment"""
    if isinstance(inputs, dict):
      return {
          'image':
              tf.image.random_brightness(inputs['image'], max_delta=max_delta),
          'label':
              inputs['label']
      }
    else:
      return (tf.image.random_brightness(inputs, max_delta=max_delta),
              args[0])

  return _random_brightness


def random_contrast(lower, upper):

  def _random_contrast(inputs, *args):
    """Random contrast adjustment"""
    if isinstance(inputs, dict):
      return {
          'image': tf.image.random_contrast(inputs['image'], lower, upper),
          'label': inputs['label']
      }
    else:
      return (tf.image.random_contrast(inputs, lower, upper), args[0])

  return _random_contrast


def get_rotate_transform(angle: float):
  """Rotates image counterclockwise by angle degrees.

  Args:
      angle: float. Angle in degrees.
  """

  def rotate(inputs):
    return {
        'image': tfaimg.rotate(inputs['image'], angle * math.pi / 180),
        'label': inputs['label']
    }

  return rotate


def get_jpeg_quality_transform(level: float):
  """Rotates image counterclockwise by angle degrees.

  Args:
      level: float. [0, 1], 0: full quality, 1:poor quality.
  """

  def adjust_jpeg_quality(inputs):
    return {
        'image':
            tf.image.adjust_jpeg_quality(inputs['image'], 100 - 100 * level),
        'label':
            inputs['label']
    }

  return adjust_jpeg_quality


def get_brightness_transform(level: float):
  """Rotates image counterclockwise by angle degrees.

  Args:
      level: float. [0, 1], 0: no adjust, 1: max adjust
  """

  def adjust_brightness(inputs):
    return {
        'image': tf.image.adjust_brightness(inputs['image'], level),
        'label': inputs['label']
    }

  return adjust_brightness


def get_additive_gaussian_transform(std: float):
  """Adds iid Gaussian noise of with standard deviation std to image.

  Args:
    std (float): standard deviation of additive Gaussian noise.
  """

  def addtive_gaussian(inputs):
    return {
        'image':
            inputs['image'] +
            tf.random.normal(inputs['image'].shape, stddev=std),
        'label':
            inputs['label']
    }

  return addtive_gaussian


def image_resize(image_size=[400, 640], resize_label=True):

  def resize(inputs, *args):
    if isinstance(inputs, dict):
      out= {'image': tf.image.resize(inputs['image'], image_size)}
      if resize_label:
        out['label'] = tf.image.resize(inputs['label'], image_size)
      return out
    else:
      if resize_label:
        out = (tf.image.resize(inputs, image_size),
               tf.image.resize(args[0], image_size))
      else:
        out = (tf.image.resize(inputs, image_size), args[0])

  return resize


def get_label_to_one_hot(num_classes: int = 10):

  def label_to_one_hot(inputs):
    """Converts label to one hot tensor Args:

      inputs: input dict.
      num_classes (int): number of classes.
    """
    return {
        'image': inputs['image'],
        'label': tf.keras.backend.one_hot(inputs['label'], num_classes)
    }

  return label_to_one_hot


def get_segmentation_label_to_one_hot_tuple(num_classes: int = 20):
  def segmentation_label_to_one_hot(x, y):
    label_img = y
    label_img = tf.clip_by_value(label_img[:, :, 0], 0, num_classes)
    label_img = tf.cast(label_img, tf.uint8)
    return x, tf.keras.backend.one_hot(label_img, num_classes)
  return segmentation_label_to_one_hot


def get_segmentation_label_to_one_hot(num_classes: int = 20):
  def segmentation_label_to_one_hot(inputs):
    label_img = inputs['label']
    label_img = tf.clip_by_value(label_img[:, :, 0], 0, num_classes)
    label_img = tf.cast(label_img, tf.uint8)
    return {
        'image': inputs['image'],
        'label': tf.keras.backend.one_hot(label_img, num_classes)
    }
  return segmentation_label_to_one_hot


def rgb_label_to_depth(inputs, rescale=True):
  """Converts label to one hot tensor Args:

    inputs: input dict.
  """
  depth_image = tf.case(inputs['label_depth'], tf.float32)
  depth = 255.0 * 255.0 * depth_image[:, :,
                                      0] + 255.0 * depth_image[:, :,
                                                               1] + depth_image[:, :,
                                                                                2]
  if rescale:
    depth /= 255.0 * 255.0 * 255.0
  return {
      'image': inputs['image'],
      'label_seg': inputs['label_seg'],
      'label_depth': depth
  }


def select_label(inputs, select):
  return {'image': inputs['image'], 'label': inputs[select]}


def produce_selected_tuple(inputs, select):
  return inputs['image'], inputs[select]


def produce_tuple(inputs):
  return inputs['image'], inputs['label']


def get_rescaling(scale: float, label_list=['label']):
  """Rescales image by dividing with float scale.

  Args:
    scale (float): image pixel values are scaled with 1/scale
  """

  def rescale(inputs):
    output_dict = {'image': inputs['image'] / scale}
    for label in label_list:
      output_dict[label] = inputs[label]
    return output_dict

  return rescale


def get_normalize(mean: float, std: float, eps: float = 1e-5):
  """Normalizes image with mean and std.

  Args:
    mean (float): translation
    std (float): scale (-> 1/std)
    eps (float): used for numerical stability (1/(std + eps))
  """

  def normalize(inputs):
    return {
        'image': (inputs['image'] - mean) / (std + eps),
        'label': inputs['label']
    }

  return normalize
