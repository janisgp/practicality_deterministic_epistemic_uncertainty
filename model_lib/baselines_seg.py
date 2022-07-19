import os
import pickle
from typing import Dict, Tuple, List, Optional
import tensorflow as tf
import numpy as np

from model_lib.baselines import BaseUncertaintyModel
from model_lib.dilated_resnet_segmentation import DRNSeg
from model_lib.utils import entropy_tf
from model_lib.utils import dice_coef_loss
from model_lib.utils import weighted_categorical_crossentropy
from data_lib.get_segmentation_dataset import DATASET_SHAPES

import tensorflow_probability as tfp


class BaseSegModel(BaseUncertaintyModel):
  """Base segmentation model for uncertainty prediction.

  Attributes:
    nr_classes: int
    backbone: tf.keras.models.Model
    exp_folder: path to experiment folder
    uncertainty_agg: Type of uncertainty aggregation across pixels
    batch_size: batch size
    loss_func: loss function used during training
  """

  def __init__(self, batch_size: int = 16, **kwargs):
    super().__init__(**kwargs)
    self.dataset = kwargs.get('dataset')
    self.uncertainty_agg = kwargs.get('uncertainty_agg')
    self.backbone = self.get_backbone(**kwargs)
    self.batch_size = batch_size
    self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

  def aggregate_uncertainty(self, uncertainty: tf.Tensor) -> tf.Tensor:
    if self.uncertainty_agg == 'mean':
      return tf.reduce_mean(tf.reduce_mean(uncertainty, axis=-1), axis=-1)
    else:
      raise NotImplementedError

  def get_backbone(self, **kwargs) -> tf.keras.models.Model:
    # create backbone
    backbone = DRNSeg(
        kwargs.get('backbone'),
        classes=self.nr_classes,
        l2_reg=kwargs.get('l2_reg'),
        dropout=kwargs.get('dropout'),
        batch_size=kwargs.get('batch_size'),
        spectral_normalization=kwargs.get('spectral_normalization'),
        spectral_batchnormalization=kwargs.get('spectral_batchnormalization'),
        soft_spectral_normalization=kwargs.get('soft_spectral_normalization'),
        coeff=kwargs.get('coeff'),
        power_iterations=kwargs.get('power_iterations'),
        smoothing_conv=kwargs.get('smoothing_conv'))

    return backbone


class SoftmaxSeg(BaseSegModel):

  def call(self,
           inputs: tf.Tensor,
           training=True,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    prediction, x = self.backbone(inputs, training=training)
    output_dict = {'prediction': prediction}
    if return_features:
      output_dict['features'] = x
    return output_dict

  def train_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                 step: int=0) -> Dict[str, tf.Tensor]:
    x, y = data

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      loss_value = self.loss_func(y, out['prediction'])

    grads = tape.gradient(loss_value, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    out.update({'loss': loss_value})

    return out

  def test_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                step: int=0) -> Dict[str, tf.Tensor]:
    x, y = data

    out = self.call(inputs=x, training=False)
    loss_value = self.loss_func(y, out['prediction'])

    out.update({'loss': loss_value})

    return out

  def uncertainty(self, data: Tuple[tf.Tensor,
                                    tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Computes predictions and uncertainty estimates given x.

    Uncertainty is the entropy of the softmax.
    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    inputs = data[0]
    output_dict = self.call(inputs=inputs, training=False)
    output_dict['uncertainty_pixel'] = entropy_tf(
        inputs=output_dict['prediction'], axis=-1)
    output_dict['uncertainty'] = self.aggregate_uncertainty(
        uncertainty=output_dict['uncertainty_pixel'])
    return output_dict


class MCDropoutSeg(SoftmaxSeg):

  def __init__(self, nr_samples: int = 10, **kwargs):
    super().__init__(**kwargs)
    self.nr_samples = nr_samples

  def uncertainty(self, data: Tuple[tf.Tensor,
                                    tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Computes predictions and uncertainty estimates given x.

    Uncertainty is the entropy of the softmax.
    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    inputs = data[0]
    output_dict = dict()
    predictions = []
    for _ in range(self.nr_samples):
      predictions.append(
          tf.expand_dims(
              self.call(inputs=inputs, training=True)['prediction'], axis=0))
    predictions = tf.concat(predictions, axis=0)
    mean_prediction = tf.reduce_mean(predictions, axis=0)
    entropy_marginal = entropy_tf(inputs=mean_prediction, axis=-1)
    mean_entropy = tf.reduce_mean(
        entropy_tf(inputs=predictions, axis=-1), axis=0)
    output_dict['prediction'] = mean_prediction
    output_dict['uncertainty_pixel'] = entropy_marginal - mean_entropy
    # output_dict['uncertainty_pixel'] = entropy_marginal
    output_dict['uncertainty'] = self.aggregate_uncertainty(
        uncertainty=output_dict['uncertainty_pixel'])

    return output_dict


class EnsembleSeg(SoftmaxSeg):

  def __init__(self, nr_samples: int = 10, **kwargs):
    super().__init__(**kwargs)
    self.ensemble_size = kwargs.get('ensemble_size')

    model_class = None
    if kwargs.get('method') == 'softmax':
      model_class = SoftmaxSeg
    else:
      ValueError(f'Base model: {kwargs.get("method")} not compatible with ensembles!')
    self.ensemble = [model_class(**kwargs) for _ in range(self.ensemble_size)]

  def call(self, inputs, training=None, **kwargs) -> Dict[str, tf.Tensor]:
    member_predictions = []
    output_dict = dict()
    for member in self.ensemble:
      member_predictions.append(
        member(inputs=inputs, training=training))
    output_dict['member_prediction'] = tf.concat(
        [tf.expand_dims(d['prediction'], axis=1) for d in member_predictions],
        axis=1)
    output_dict['prediction'] = tf.reduce_mean(output_dict['member_prediction'], axis=1)
    return output_dict

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    x = data[0]
    output_dict = self.call(inputs=x, training=False)
    entropy_marginal = entropy_tf(
      inputs=output_dict['prediction'], axis=-1)
    mean_entropy = tf.reduce_mean(
      entropy_tf(inputs=output_dict['member_prediction'], axis=-1), axis=1)
    output_dict['uncertainty_pixel'] = entropy_marginal - mean_entropy
    output_dict['uncertainty'] = self.aggregate_uncertainty(
        uncertainty=output_dict['uncertainty_pixel'])
    return output_dict

  def custom_load_weights(self, filepath: List[str], *kwargs):
    assert len(filepath) == self.ensemble_size, 'Number of paths has to match ensemble size!'
    for i in range(self.ensemble_size):
      self.ensemble[i].custom_load_weights(filepath=filepath[i])
