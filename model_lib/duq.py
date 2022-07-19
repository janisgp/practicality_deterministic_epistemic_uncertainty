from typing import Dict, Tuple
import tensorflow as tf

from model_lib.baselines import BaseUncertaintyModel
from model_lib.utils import bce_from_torch


class RBFLayer(tf.keras.layers.Layer):

  def __init__(self,
               nr_classes: int = 10,
               initial_n: int = 12,
               gamma: float = 0.999,
               length_scale: float = 0.1):
    super().__init__()
    self.initial_n = initial_n
    self.gamma = gamma
    self.length_scale = length_scale
    self.nr_classes = nr_classes

  def build(self, input_shape):

    w_init = tf.random_normal_initializer(stddev=0.05)
    self.w = self.add_weight(
        name='kernel',
        shape=(input_shape[-1], self.nr_classes, input_shape[-1]),
        initializer=w_init,
        trainable=True)

    n_init = tf.constant_initializer(value=self.initial_n)
    self.n = self.add_weight(
        name='n', shape=(self.nr_classes), initializer=n_init, trainable=True)

    m_init = tf.random_normal_initializer(stddev=self.initial_n)
    self.m = self.add_weight(
        name='m',
        shape=(input_shape[-1], self.nr_classes),
        initializer=m_init,
        trainable=True)

  def update_mean(self, z: tf.Tensor, y: tf.Tensor):
    """Updates means self.m of DUQ.

    Args:
      z: features: (batch x feature_dims x nr_classes)
      y: one-hot target: (batch x nr_classes)
    """
    # normalizing value per class, assumes y is one_hot encoded
    self.n.assign(self.gamma * self.n +
                  (1 - self.gamma) * tf.reduce_sum(y, axis=0))

    # compute sum of embeddings on class by class basis
    features_sum = tf.einsum('ijk,ik->jk', z, y)

    self.m.assign(self.gamma * self.m + (1 - self.gamma) * features_sum)

  def call(self, z: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    z = tf.einsum('ij,mnj->imn', z, self.w)
    mean = self.m / self.n
    diff = z - mean
    distances = tf.exp(-1 * tf.reduce_mean(diff**2, axis=1) /
                       (2 * self.length_scale**2))
    return distances, z


class DUQ(BaseUncertaintyModel):
  """Uncertainty quantification using Deterministic Uncertainty Quantification (DUQ).

  https://arxiv.org/abs/2003.02037
  """

  def __init__(self,
               length_scale: float = 0.1,
               l_gradient_penalty: float = 0.5,
               gamma: float = 0.999,
               **kwargs):
    super().__init__(**kwargs)

    self.length_scale = length_scale
    self.l_gradient_penalty = l_gradient_penalty
    self.gamma = gamma
    self.initial_n = 12

    self.bce = bce_from_torch(reduction='mean')
    self.l2_reg = tf.keras.regularizers.L2(kwargs.get('l2_reg'))

    self.rbf = RBFLayer(
        nr_classes=kwargs.get('nr_classes'),
        initial_n=self.initial_n,
        gamma=self.gamma,
        length_scale=self.length_scale)

  def call(self, inputs, training=None, mask=None) -> Dict[str, tf.Tensor]:
    backbone_output = self.backbone(inputs, training=training)
    if isinstance(backbone_output, list):
      z = backbone_output[0]
    else:
      z = backbone_output
    prediction, z = self.rbf(z=z)
    return {'prediction': prediction, 'z': z}

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Computes predictions and uncertainty estimates given x.

    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    x = data[0]
    output_dict = self.call(inputs=x, training=False)
    output_dict['uncertainty'] = -1 * tf.math.log(
        tf.reduce_max(output_dict['prediction'], axis=-1))
    return output_dict

  def calculate_gradient_penalty(self, grads: tf.Tensor) -> tf.Tensor:
    """Computes two-sided gradient penalty.

    Args:
      grads:

    Returns:

    """
    grads_norm = tf.reduce_sum(tf.square(grads), axis=(1, 2, 3))
    max_zero_gp = tf.nn.relu((grads_norm - 1)**2)
    gradient_penalty = tf.reduce_mean(max_zero_gp)
    return gradient_penalty

  def train_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                 **kwargs) -> Dict[str, tf.Tensor]:

    x, y = data

    use_gradient_penalty = self.l_gradient_penalty > 0.0
    with tf.GradientTape(persistent=use_gradient_penalty) as tape:
      if use_gradient_penalty:
        tape.watch(x)
      out = self.call(inputs=x, training=True)
      loss_value = self.bce(out['prediction'], y) + self.l2_reg(self.rbf.w)
      if use_gradient_penalty:
        inputs_grads = tape.gradient(out['prediction'], [x])
        loss_value += tf.constant(self.l_gradient_penalty) *\
                      self.calculate_gradient_penalty(inputs_grads[0])
    self.rbf.update_mean(z=out['z'], y=y)

    # We make n & m trainable and remove them manually,
    # so that tf.keras.Model.save_weights() saves them.
    variables = [
        v for v in self.trainable_variables
        if not any([n in v.name for n in ['rbf_layer/n:0', 'rbf_layer/m:0']])
    ]
    grads = tape.gradient(loss_value, variables)
    self.optimizer.apply_gradients(zip(grads, variables))

    out.update({'loss': loss_value})
    return out

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:

    x, y = data

    out = self.call(inputs=x, training=False)
    loss_value = self.bce(out['prediction'], y)

    out.update({'loss': loss_value})
    return out
