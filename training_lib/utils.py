"""Util functions for optimizing NN."""

import os
from typing import List, Dict, Any, Tuple, Optional, Callable
import tensorflow as tf

from model_lib.baselines import BaseUncertaintyModel
from model_lib.baselines import SoftmaxModel
from model_lib.baselines import MCDropoutModel
from model_lib.duq import DUQ
from model_lib.ddu import DDUModel
from model_lib.mir import MIR
from model_lib.mir import extract_features
from model_lib.mir_segmentation import extract_features_segmentation
from model_lib.sngp import SNGP
from model_lib.baselines_seg import SoftmaxSeg
from model_lib.sngp_segmentation import SNGPSeg
from model_lib.baselines_seg import MCDropoutSeg
from model_lib.mir_segmentation import MIRSeg
from model_lib.ddu_segmentation import DDUSeg


def multi_step_lr_schedule(steps: List[int], gamma: float) -> Callable:
  def _lr_schedule(e: int, lr: tf.Tensor) -> tf.Tensor:
    if e in steps:
      return lr * tf.constant(gamma)
    else:
      return lr
  return _lr_schedule


def load_model(
    method: str,
    hyperparameters: Dict,
    exp_folder: str,
    trainset: tf.data.Dataset,
    valset: tf.data.Dataset,
    load_weights: bool = False,
    evaluation: bool = False,
    normalize_features: bool = True
) -> Tuple[BaseUncertaintyModel, Any, Optional[Callable]]:
  """Helper for initializing image classification models.

  Args:
    method: str. Name of Method
    hyperparameters: dict. Containing hyperparameters for intantiating Model
    exp_folder: str. Root folder of an exeriment. Used to load trained weights
      for evaluation.
    trainset: training dataset. Used for fitting density model on the
      hidden representations for uncertainty estimation.
    valset: training dataset. Used for evaluating density model on the
      hidden representations for uncertainty estimation.
    load_weights: bool. Whether to load pretrained weights or not.
    evaluation: bool. Whether model is used for evaluation or not.
      Determines whether density model needs to be fitted.
    normalize_features: bool. Whether to normalize features prior to
      density estimation or not.
  """

  lr_schedule = None
  if hyperparameters.get('lr_schedule_steps'):
    lr_schedule = multi_step_lr_schedule(
        steps=hyperparameters['lr_schedule_steps'],
        gamma=hyperparameters['lr_schedule_gamma'])

  if method == 'softmax':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = SoftmaxModel(**hyperparameters)

  elif method == 'dropout':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = MCDropoutModel(**hyperparameters)

  elif method == 'ddu':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = DDUModel(**hyperparameters)

  elif method == 'duq':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hyperparameters['lr'], momentum=0.9)
    model = DUQ(**hyperparameters)

  elif method == 'mir':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = MIR(**hyperparameters)

  elif method == 'sngp':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hyperparameters['lr'], momentum=0.9)
    model = SNGP(trainset=trainset, **hyperparameters)

  else:
    raise ValueError('Unknown method!')

  model.compile(optimizer=optimizer)

  if load_weights:
    _, init_input = enumerate(valset.batch(10)).__next__()
    _ = model.initialize(data=init_input)
    if tf.io.gfile.exists(os.path.join(exp_folder, 'best_model.h5')):
      model.custom_load_weights(
          filepath=os.path.join(exp_folder, 'best_model.h5'))
    else:
      model.custom_load_weights(
          filepath=os.path.join(exp_folder, 'best_model.tf'))
    if method in ['mir', 'ddu'] and evaluation:
      model.init_density(normalize_features=normalize_features)
      features_train, pred_train = extract_features(
          model=model,
          dataset=trainset.batch(hyperparameters['batch_size']))
      features_val, pred_val = extract_features(
          model=model,
          dataset=valset.batch(hyperparameters['batch_size']))
      model.density.fit(
          x=features_train,
          y=pred_train,
          x_val=features_val,
          y_val=pred_val)

  return model, optimizer, lr_schedule


def load_model_segmentation(
    method: str,
    hyperparameters: Dict,
    exp_folder: str,
    trainset: tf.data.Dataset,
    valset: tf.data.Dataset,
    load_weights: bool = False,
    evaluation: bool = False,
) -> Tuple[BaseUncertaintyModel, Optional[Callable]]:
  """Helper for initializing semantic segmentation models.

  For MIR/DDU features are normalized by default.

  Args:
    method: str. Name of Method
    hyperparameters: dict. Containing hyperparameters for intantiating Model
    exp_folder: str. Root folder of an exeriment. Used to load trained weights
      for evaluation.
    trainset: training dataset. Used for fitting density model on the
      hidden representations for uncertainty estimation.
    valset: training dataset. Used for evaluating density model on the
      hidden representations for uncertainty estimation.
    load_weights: bool. Whether to load pretrained weights or not.
    evaluation: bool. Whether model is used for evaluation or not.
      Determines whether density model needs to be fitted.
  """

  lr_schedule = None
  if hyperparameters.get('lr_schedule_steps'):
    lr_schedule = multi_step_lr_schedule(
        steps=hyperparameters['lr_schedule_steps'],
        gamma=hyperparameters['lr_schedule_gamma'])

  print('Initializing model...', flush=True)
  if method == 'softmax':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = SoftmaxSeg(**hyperparameters)

  elif method == 'dropout':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = MCDropoutSeg(**hyperparameters)

  elif method == 'sngp':
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=hyperparameters['lr'], momentum=0.9)
    model = SNGPSeg(trainset=trainset, **hyperparameters)

  elif method == 'mir':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = MIRSeg(**hyperparameters)

  elif method == 'ddu':
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters['lr'])
    model = DDUSeg(**hyperparameters)

  else:
    raise ValueError('Unknown method!')

  print('Compiling model...', flush=True)
  model.compile(optimizer=optimizer)

  if load_weights:
    _, init_input = enumerate(valset).__next__()
    if method == 'sngp':
      print('Building backbone...', flush=True)
      model.backbone.build([None] + init_input[0].shape[1:])
    if method in ['mir', 'ddu']:
      print('Building model...', flush=True)
      model.build([None] + init_input[0].shape[1:])
    print('Initial model call...', flush=True)
    _ = model.initialize(data=init_input)
    print('Loading weights...', flush=True)
    if tf.io.gfile.exists(os.path.join(exp_folder, 'best_model.h5')):
      model.custom_load_weights(
          filepath=os.path.join(exp_folder, 'best_model.h5'))
    else:
      model.custom_load_weights(
          filepath=os.path.join(exp_folder, 'best_model.tf'))

    if method in ['mir', 'ddu'] and evaluation:
      model.init_density()
      features_train = extract_features_segmentation(
          model=model, dataset=trainset)
      features_val = extract_features_segmentation(
          model=model,
          dataset=valset,
          max_samples=10000)
      model.density.fit(x=features_train, x_val=features_val)

  return model, lr_schedule
