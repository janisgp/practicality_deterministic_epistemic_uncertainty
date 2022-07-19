# Soft Spectral Normalization (not enforced, only <= coeff) for Dense layers (and Conv2D layers with kernel_size=1)
# Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity (Gouk et al. 2018)
# Implementation based on the Tensorflow Addons Spectral Normalization layer
# In this version, we extend the layer to the Soft Spectral Normalization version
# I adapted the tf implementation from
# https://github.com/azraelzhor/tf-invertible-resnet/blob/a52a4fa5635834db8f59f7f05c4802dda4ce5c49/modules/spectral_norm.py
# to make it compatible with tf 2.0

import tensorflow as tf
from typing import Optional


class SpectralNormalization(tf.keras.layers.Wrapper):
  """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs
    [See "Spectral Normalization for Generative Adversarial Networks"
    (https://arxiv.org/abs/1802.05957)] and enforce smoothness of the feature
    space
    of a generic feature extractor [See "Regularisation of Neural Networks by
    Enforcing Lipschitz Continuity"
    (https://arxiv.org/abs/1804.04368)].
    Wrap `tf.keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])
    Wrap `tf.keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])
    Args:
      layer: A `tf.keras.layers.Layer` instance that has either `kernel` or
        `embeddings` attribute.
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not have `kernel` or `embeddings`
      attribute.
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

    if hasattr(self.layer, "kernel"):
      self.w = self.layer.kernel
    elif hasattr(self.layer, "embeddings"):
      self.w = self.layer.embeddings
    else:
      raise AttributeError("{} object has no attribute 'kernel' nor "
                           "'embeddings'".format(type(self.layer).__name__))

    self.w_shape = self.w.shape.as_list()

    self.u = self.add_weight(
        shape=(1, self.w_shape[-1]),
        initializer=tf.initializers.TruncatedNormal(stddev=0.02),
        trainable=False,
        name="sn_u",
        dtype=self.w.dtype,
    )

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

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

    w = tf.reshape(self.w, [-1, self.w_shape[-1]])
    u = self.u

    with tf.name_scope("spectral_normalize"):
      for _ in range(self.power_iterations):
        v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        u = tf.math.l2_normalize(tf.matmul(v, w))

      sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

      self.w.assign(self.w / sigma)
      self.u.assign(u)

  def get_config(self):
    config = {"power_iterations": self.power_iterations}
    base_config = super().get_config()
    return {**base_config, **config}


class SoftSpectralNormalization(SpectralNormalization):
  """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs
    [See "Spectral Normalization for Generative Adversarial Networks"
    (https://arxiv.org/abs/1802.05957)] and enforce smoothness of the feature
    space
    of a generic feature extractor [See "Regularisation of Neural Networks by
    Enforcing Lipschitz Continuity"
    (https://arxiv.org/abs/1804.04368)].
    Wrap `tf.keras.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SoftSpectralNormalization(tf.keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])
    Wrap `tf.keras.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SoftSpectralNormalization(tf.keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])
    Args:
      layer: A `tf.keras.layers.Layer` instance that has either `kernel` or
        `embeddings` attribute.
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not have `kernel` or `embeddings`
      attribute.
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

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

    w = tf.reshape(self.w, [-1, self.w_shape[-1]])
    u = self.u

    with tf.name_scope("spectral_normalize"):
      for _ in range(self.power_iterations):
        v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
        u = tf.math.l2_normalize(tf.matmul(v, w))

      sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

      # soft normalization: only when sigma larger than coeff
      factor = tf.maximum(tf.ones_like(w), sigma / self.coeff)

      self.w.assign(self.w / factor)
      self.u.assign(u)
