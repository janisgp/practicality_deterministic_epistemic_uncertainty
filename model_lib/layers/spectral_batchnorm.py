# Soft Spectral Normalization (not enforced, only <= coeff) for Batch Normalization layers
# Based on: "Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression" (van
# Amersfoort et al., 2021) and "Regularisation of Neural Networks by Enforcing Lipschitz Continuity" (Gouk et al., 2018)
# Implementation based on the Tensorflow Addons Spectral Normalization layer
# In this version, we extend the layer to the Soft Spectral Normalization version
# I adapted the pytorch implementation from
# https://github.com/y0ast/DUE/blob/64a1a8935e8ac1059c1f705bc0659db6e7e17165/due/layers/spectral_batchnorm.py
# to make it compatible with tf 2.0

import tensorflow as tf
from typing import Optional


class SpectralBatchNormalization(tf.keras.layers.Wrapper):
  """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the batch normalization
    layer by
    constraining the spectral norm of its scale parameter [See Sec 2.1 of
    Improving Deterministic Uncertainty Estimation
    in Deep Learning for Classification and Regression
    (https://arxiv.org/pdf/2102.11409.pdf)].
    Wrap `tf.keras.layers.BatchNormalization`:
    >>> x = np.random.rand(3, 10, 10, 1)
    >>> sbn = SpectralBatchNormalization(tf.keras.layers.BatchNormalization())
    >>> y = sbn(x)
    >>> y.shape
    TensorShape([3, 10, 10, 1])
    Args:
      layer: A `tf.keras.layers.Layer` instance that has a `gamma` attribute.
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not have `gamma` attribute.
    """

  def __init__(self,
               layer: tf.keras.layers.Layer,
               coeff: Optional[int] = None,
               power_iterations: int = 1,
               **kwargs):
    super().__init__(layer, **kwargs)
    if power_iterations <= 0:
      raise ValueError("`power_iterations` should be greater than zero, got "
                       "`power_iterations={}`".format(power_iterations))
    self.coeff = coeff
    self.power_iterations = power_iterations
    self._initialized = False

  def build(self, input_shape):
    """Build `Layer`"""
    super().build(input_shape)
    input_shape = tf.TensorShape(input_shape)
    self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

    if hasattr(self.layer, "gamma") and hasattr(
        self.layer, "moving_variance") and hasattr(self.layer, "epsilon"):
      self.gamma = self.layer.gamma
      self.moving_variance = self.layer.moving_variance
      self.epsilon = self.layer.epsilon
      self.scale = self.layer.scale
    else:
      raise AttributeError("{} object has no attribute 'gamma' nor "
                           "'moving_variance' nor 'epsilon'".format(
                               type(self.layer).__name__))

  def call(self, inputs, training=None):
    """Call `Layer`"""
    if training is None:
      training = tf.keras.backend.learning_phase()

    if training:
      self.normalize_weights()

    output = self.layer(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())

  @tf.function
  def normalize_weights(self):
    """Generate spectral normalized weights.

        This method will update the value of `self.gamma` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
    # before the forward pass, estimate the lipschitz constant of the layer and
    # divide through by it so that the lipschitz constant of the batch norm operator is approximately
    # 1
    gamma = (
        tf.ones_like(self.moving_variance)
        if self.scale is None else self.gamma)
    # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
    sigma = tf.reduce_max(
        tf.abs(gamma * (self.moving_variance + self.epsilon)**-0.5))

    self.gamma.assign(gamma / sigma)

  def get_config(self):
    config = {"power_iterations": self.power_iterations}
    base_config = super().get_config()
    return {**base_config, **config}


class SoftSpectralBatchNormalization(SpectralBatchNormalization):
  """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the batch normalization
    layer by
    constraining the spectral norm of its scale parameter [See Sec 2.1 of
    Improving Deterministic Uncertainty Estimation
    in Deep Learning for Classification and Regression
    (https://arxiv.org/pdf/2102.11409.pdf)].
    Wrap `tf.keras.layers.BatchNormalization`:
    >>> x = np.random.rand(3, 10, 10, 1)
    >>> ssbn =
    SoftSpectralBatchNormalization(tf.keras.layers.BatchNormalization())
    >>> y = ssbn(x)
    >>> y.shape
    TensorShape([3, 10, 10, 1])
    Args:
      layer: A `tf.keras.layers.Layer` instance that has a `gamma` attribute.
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not have `gamma` attribute.
    """

  def __init__(self,
               layer: tf.keras.layers.Layer,
               coeff: Optional[int] = 3,
               power_iterations: int = 1,
               **kwargs):
    super().__init__(layer, coeff, power_iterations, **kwargs)

  @tf.function
  def normalize_weights(self):
    """Generate spectral normalized weights.

        This method will update the value of `self.gamma` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """
    # before the forward pass, estimate the lipschitz constant of the layer and
    # divide through by it so that the lipschitz constant of the batch norm operator is approximately
    # 1
    gamma = (
        tf.ones_like(self.moving_variance)
        if self.scale is None else self.gamma)
    # see https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct.
    sigma = tf.reduce_max(
        tf.abs(gamma * (self.moving_variance + self.epsilon)**-0.5))

    # if lipschitz of the operation is greater than coeff, then we want to divide the input by a constant to
    # force the overall lipchitz factor of the batch norm to be exactly coeff
    factor = tf.maximum(tf.ones_like(sigma), sigma / self.coeff)

    self.gamma.assign(gamma / factor)
