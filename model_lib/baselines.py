import os
import pickle
from typing import Dict, Tuple, List, Optional, Any
import tensorflow as tf

from model_lib.base_model import BaseModel
from model_lib.basic_backbones import SimpleFC
from model_lib.resnet_backbone import ResNet
from model_lib.utils import entropy_tf
from data_lib.get_image_dataset import DATASET_SHAPES


class BaseUncertaintyModel(BaseModel):
  """
  Base (classification) model for uncertainty prediction.

  Attributes:
    nr_classes: int
    backbone: tf.keras.models.Model
    exp_folder: path to experiment folder
  """

  def __init__(self, **kwargs):
    super().__init__()
    self.nr_classes = kwargs.get('nr_classes')
    self.backbone = self.get_backbone(**kwargs)
    self.exp_folder = kwargs.get('exp_folder')

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    Computes predictions and uncertainty estimates given x.

    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    raise NotImplementedError

  def train_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                 **kwargs) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def get_backbone(self, **kwargs) -> tf.keras.models.Model:
    """Gets model backbone."""

    # create backbone
    input_shape = DATASET_SHAPES[kwargs.get('dataset')]
    if kwargs.get('backbone') == 'simple_fc':
      backbone = SimpleFC(
        input_shape=input_shape,
        hidden_dim=kwargs.get('hidden_dims'),
        hidden_layer=kwargs.get('hidden_layer'),
        dropout=kwargs.get('dropout'),
        l2_reg=kwargs.get('l2_reg'),
        batch_normalization=kwargs.get('bn'),
        output_dim=kwargs.get('feature_dims'),
        spectral_normalization=kwargs.get('spectral_normalization'),
        spectral_batchnormalization=kwargs.get('spectral_batchnormalization'),
        soft_spectral_normalization=kwargs.get('soft_spectral_normalization'),
        coeff=kwargs.get('coeff'),
        power_iterations=kwargs.get('power_iterations'))
    elif kwargs.get('backbone') == 'resnet':
      backbone = ResNet(
        input_shape,
        kwargs.get('resnet_depth'),
        kwargs.get('batch_size'),
        dropout=kwargs.get('dropout'),
        batchnorm=kwargs.get('bn'),
        l2_reg=kwargs.get('l2_reg'),
        num_classes=kwargs.get('feature_dims'),
        spectral_normalization=kwargs.get('spectral_normalization'),
        spectral_batchnormalization=kwargs.get('spectral_batchnormalization'),
        soft_spectral_normalization=kwargs.get('soft_spectral_normalization'),
        coeff=kwargs.get('coeff'),
        power_iterations=kwargs.get('power_iterations'))
    else:
      raise ValueError('Unknown backbone model!')

    return backbone


class SoftmaxModel(BaseUncertaintyModel):
  """Softmax model for classification. Inherits from BaseUncertaintyModel.

  Attributes:
    final_layer: tf.keras.layers.Dense. Final classification layer
    softmax: tf.keras.layers.Softmax. Softmax layer
    loss_func: tf.keras.losses.CategoricalCrossentropy. CE loss
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.final_layer = tf.keras.layers.Dense(self.nr_classes)
    self.softmax = tf.keras.layers.Softmax()
    self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None,
           mask: Optional[Any] = None,
           return_features: bool = False) -> Dict[str, tf.Tensor]:
    """ Call to model.

    Args:
      inputs: model inputs.
      training: Training mode.
      mask: legacy
      return_features: bool. Whether to return features or not.
        Used post-training for MIR and DDU.

    Returns:
      output_dict: Dictioniory containing predictions.
    """
    backbone_output = self.backbone(inputs, training=training)
    if isinstance(backbone_output, list):
      features = backbone_output[0]
    else:
      features = backbone_output
    output_dict = {'prediction': self.softmax(self.final_layer(features))}
    if return_features:
      output_dict['features'] = features
      output_dict['features_large'] = backbone_output[1]
    return output_dict

  def train_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                 **kwargs) -> Dict[str, tf.Tensor]:

    x, y = data

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      loss_value = self.loss_func(y, out['prediction'])

    grads = tape.gradient(loss_value, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    out.update({'loss': loss_value})

    return out

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:
    x, y = data

    out = self.call(inputs=x, training=False)
    loss_value = self.loss_func(y, out['prediction'])

    out.update({'loss': loss_value})
    return out

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:

    x, y = data

    out = self.call(inputs=x, training=False)
    loss_value = self.loss_func(y, out['prediction'])

    out.update({'loss': loss_value})
    return out

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    Computes predictions and uncertainty estimates given x.
    Uncertainty is the entropy of the softmax.

    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    inputs = data[0]
    output_dict = self.call(inputs=inputs, training=False)
    output_dict['uncertainty'] = entropy_tf(
      inputs=output_dict['prediction'], axis=-1)
    return output_dict


class MCDropoutModel(SoftmaxModel):

  def __init__(self, nr_samples: int = 10, **kwargs):
    super().__init__(**kwargs)
    self.nr_samples = nr_samples

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    Computes predictions and uncertainty estimates given x.
    Uncertainty is computed as the conditional mutual information
    between the weights and the predictions:
    MI(theta,y|x) = H(y|x) - H(y|theta,x)

    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction', 'uncertainty' (epistemic)
      'aleatoric' and 'total_uncertainty'
    """
    inputs = data[0]
    output_dict = dict()
    predictions = []
    for _ in range(self.nr_samples):
      predictions.append(
        tf.expand_dims(self.call(inputs=inputs, training=True)['prediction'], axis=0))
    predictions = tf.concat(predictions, axis=0)
    mean_prediction = tf.reduce_mean(predictions, axis=0)
    entropy_marginal = entropy_tf(
      inputs=mean_prediction, axis=-1)
    mean_entropy = tf.reduce_mean(
      entropy_tf(inputs=predictions, axis=-1), axis=0)
    output_dict['prediction'] = mean_prediction
    output_dict['epistemic_uncertainty'] = entropy_marginal - mean_entropy
    output_dict['uncertainty'] = entropy_marginal
    output_dict['aleatoric'] = mean_entropy
    return output_dict


class Ensemble(BaseUncertaintyModel):
  """
  Pure inference model. Initializes ensemble members from a set
  of pretrained models. In order to use ensembles, first train
  several base model (currently only supports SoftmaxModel) and
  then create an ensemble from these independently trained models.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.ensemble_size = kwargs.get('ensemble_size')

    model_class = None
    if kwargs.get('method') == 'softmax':
      model_class = SoftmaxModel
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
    output_dict['uncertainty'] = entropy_marginal
    output_dict['epistemic'] = entropy_marginal - mean_entropy
    output_dict['aleatoric'] = mean_entropy
    return output_dict

  def custom_load_weights(self, filepath: List[str], *kwargs):
    assert len(filepath) == self.ensemble_size, 'Number of paths has to match ensemble size!'
    for i in range(self.ensemble_size):
      tf.io.gfile.copy(filepath[i], '/tmp/temp.h5', overwrite=True)
      self.ensemble[i].load_weights('/tmp/temp.h5')
