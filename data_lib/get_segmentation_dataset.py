# Lint as: python3
"""Gets image classification datasets (MNIST, FashionMNIST, CIFAR10 & SVHN)"""

import os
import random
from collections import namedtuple
from typing import Optional
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
import numpy as np
import tqdm

from data_lib import transforms
from data_lib.corruptions.apply_corruptions import get_corruption

# a label and all meta information
Label = namedtuple(
    'Label',
    [
        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class
        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.
        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!
        'category',  # The name of the category that this label belongs to
        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.
        'hasInstances',  # Whether this label distinguishes between single instances or not
        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        'color',  # The color of this label
    ])

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (0, 0, 0)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (0, 0, 0)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (0, 0, 0)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (0, 0, 0)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (0, 0, 0)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (0, 0, 0)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 0)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 0)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 0)),
]

ID_TO_TRAINID_CITYSCAPES = {label.id: label.trainId for label in labels}
TRAINID_TO_COLOR_CITYSCAPES = {label.trainId: label.color for label in labels}


def cityscapes_id_to_trainid(inputs):

  lbls = inputs['label']
  for key in ID_TO_TRAINID_CITYSCAPES:
    indices = tf.where(tf.equal(lbls, key))
    if ID_TO_TRAINID_CITYSCAPES[key] == 255:
      lbls = tf.tensor_scatter_nd_update(lbls, indices, 19*tf.cast(tf.ones_like(indices[:, 0]), tf.uint8))
    else:
      lbls = tf.tensor_scatter_nd_update(lbls, indices, ID_TO_TRAINID_CITYSCAPES[key]*tf.cast(tf.ones_like(indices[:, 0]), tf.uint8))

  return {'image': inputs['image'], 'label': lbls}


IOU_MASKS = {
    'cityscapes': [True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, False],
}

DATASET_NUM_CLASSES = {}

RESCALED_SIZE_CITYSCAPES = [200, 320]
DATASET_SHAPES = {
    'cityscapes': RESCALED_SIZE_CITYSCAPES,
}

NUM_CLASSES_CARLA = 20
NUM_CLASSES_CITYSCAPES = 20


def get_segmentation_dataset(dataset: str, data_root: str, **kwargs):
  if dataset == 'cityscapes':
    return get_cityscapes(data_root, **kwargs)
  else:
    raise ValueError(f'Unknown dataset {dataset}!')


def load_cityscapes_testset_np(data_root: str, testset = None, **kwargs):
  """Cityscapes testset as np"""

  print('Preparing numpy testset...')

  if not testset:
    testset_np_imgs = os.path.join(
        data_root, 'cityscapes_test_np', 'imgs.npy')
    testset_np_lbls = os.path.join(
        data_root, 'cityscapes_test_np', 'labels.npy')

    tmp_file1 = f'/tmp/tmp1.npy'
    tmp_file2 = f'/tmp/tmp2.npy'
    tf.io.gfile.copy(testset_np_imgs, tmp_file1, overwrite=True)
    tf.io.gfile.copy(testset_np_lbls, tmp_file2, overwrite=True)
    testset_np_imgs = np.load(tmp_file1)
    testset_np_lbls = np.load(tmp_file2)

  else:
    testset_np_imgs, testset_np_lbls = [], []
    for i, batch in tqdm.tqdm(enumerate(testset)):
      testset_np_imgs += [batch[0].numpy()]
      testset_np_lbls += [batch[1].numpy()]
    testset_np_imgs = np.concatenate(testset_np_imgs, axis=0)
    testset_np_lbls = np.concatenate(testset_np_lbls, axis=0)

  return testset_np_imgs, testset_np_lbls


def get_cityscapes_testset(testset_np_imgs,
                           testset_np_lbls,
                           resize: bool = True,
                           batch_size: int = 8,
                           corruption: Optional[str] = None,
                           severity: int = 0,
                           **kwargs):

  # corrupt data
  corrupted_testset = np.zeros_like(testset_np_imgs)
  corruption_fn = get_corruption(name=corruption, severity=severity)
  for i in range(testset_np_imgs.shape[0]):
    corrupted_testset[i] = corruption_fn(testset_np_imgs[i]*255.)/255.  # corruption lib aussumes value range [0...255]

  testset = tf.data.Dataset.from_tensor_slices(
      (corrupted_testset, testset_np_lbls))

  testset = testset.batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  return testset


def get_cityscapes(data_root: str,
                   data_augmentation: bool = True,
                   resize: bool = True,
                   perturbation_type: Optional[str] = None,
                   perturbation: float = 0.0,
                   batch_size: int = 8,
                   corruption: Optional[str] = None,
                   severity: int = 0,
                   **kwargs):

  data_train = tfds.load(
      'cityscapes/semantic_segmentation',
      data_dir=data_root,
      split='train',
      shuffle_files=True)
  data_val = tfds.load(
      'cityscapes/semantic_segmentation',
      data_dir=data_root,
      split='validation',
      shuffle_files=False)
  data_test = tfds.load(
      'cityscapes/semantic_segmentation',
      data_dir=data_root,
      split='test',
      shuffle_files=False)

  # change dict keys for compatibility with transforms
  dict_key_transform = lambda data: {
      'image': data['image_left'],
      'label': data['segmentation_label']
  }
  data_train = data_train.map(dict_key_transform)
  data_val = data_val.map(dict_key_transform)
  data_test = data_test.map(dict_key_transform)

  # eliminate unnecessary IDs
  data_train = data_train.map(cityscapes_id_to_trainid)
  data_val = data_val.map(cityscapes_id_to_trainid)
  data_test = data_test.map(cityscapes_id_to_trainid)

  # if corruption:
  #   corruption_fn = get_corruption(name=corruption, severity=severity)
  #   def set_tensor_shape(x):
  #     x['image'].set_shape((1024, 2048, 3))
  #     return x
  #   data_test = data_test.map(corruption_fn).map(set_tensor_shape)

  # rescaling
  data_train = data_train.map(transforms.get_rescaling(255))
  data_val = data_val.map(transforms.get_rescaling(255))
  data_test = data_test.map(transforms.get_rescaling(255))

  if resize:
    data_train = data_train.map(
        transforms.image_resize(RESCALED_SIZE_CITYSCAPES))
    data_val = data_val.map(transforms.image_resize(RESCALED_SIZE_CITYSCAPES))
    data_test = data_test.map(transforms.image_resize(RESCALED_SIZE_CITYSCAPES))

  data_train = data_train.map(
      transforms.get_segmentation_label_to_one_hot(
          num_classes=NUM_CLASSES_CITYSCAPES))
  data_val = data_val.map(
      transforms.get_segmentation_label_to_one_hot(
          num_classes=NUM_CLASSES_CITYSCAPES))
  # data_test = data_test.map(
  #     transforms.get_segmentation_label_to_one_hot(
  #         num_classes=NUM_CLASSES_CITYSCAPES))

  if data_augmentation:
    data_train = data_train.map(transforms.random_flip_horizontal_with_label)
    data_train = data_train.map(transforms.random_brightness(max_delta=0.2))
    data_train = data_train.map(
        transforms.random_contrast(lower=0.8, upper=1.2))
    cropping_factor = kwargs.get('cropping_factor')
    if cropping_factor is not None and cropping_factor < 1.0:
      data_train = data_train.map(
          transforms.get_random_crop_with_label(
              crop_size_img=(int(RESCALED_SIZE_CITYSCAPES[0] * cropping_factor),
                             int(RESCALED_SIZE_CITYSCAPES[1] * cropping_factor),
                             3),
              crop_size_lbl=(int(RESCALED_SIZE_CITYSCAPES[0] * cropping_factor),
                             int(RESCALED_SIZE_CITYSCAPES[1] * cropping_factor),
                             20)))
      data_train = data_train.map(
          transforms.image_resize(RESCALED_SIZE_CITYSCAPES))

  data_train = data_train.map(transforms.produce_tuple)
  data_val = data_val.map(transforms.produce_tuple)
  data_test = data_test.map(transforms.produce_tuple)

  data_train = data_train.batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)
  data_val = data_val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  data_test = data_test.batch(batch_size).prefetch(
      tf.data.experimental.AUTOTUNE)

  return data_train, data_val, data_test
