"""Custom keras layers that implement explicit (approximate) kernel feature maps."""

import numpy as np
import tensorflow as tf

import tensorflow.python as tfp

dtypes = tfp.framework.dtypes
ops = tfp.framework.ops
tensor_shape = tfp.framework.tensor_shape
initializers = tfp.keras.initializers
base_layer = tfp.keras.engine.base_layer
input_spec = tfp.keras.engine.input_spec
gen_math_ops = tfp.ops.gen_math_ops
init_ops = tfp.ops.init_ops
math_ops = tfp.ops.math_ops
nn = tfp.ops.nn

_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian', 'laplacian']


class CustomRandomFourierFeatures(
    tf.keras.layers.experimental.RandomFourierFeatures):

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape.rank != 3:
      raise ValueError(
          'The rank of the input tensor should be 3. Got {} instead.'.format(
              input_shape.ndims))
    if input_shape.dims[2].value is None:
      raise ValueError(
          'The last dimension of the inputs to `RandomFourierFeatures` '
          'should be defined. Found `None`.')
    self.input_spec = input_spec.InputSpec(
        ndim=3, axes={2: input_shape.dims[2].value})
    input_dim = input_shape.dims[2].value

    kernel_initializer = _get_random_features_initializer(
        self.kernel_initializer, shape=(input_dim, self.output_dim))

    self.unscaled_kernel = self.add_weight(
        name='unscaled_kernel',
        shape=(input_dim, self.output_dim),
        dtype=dtypes.float32,
        initializer=kernel_initializer,
        trainable=False)

    self.bias = self.add_weight(
        name='bias',
        shape=(self.output_dim,),
        dtype=dtypes.float32,
        initializer=init_ops.random_uniform_initializer(
            minval=0.0, maxval=2 * np.pi, dtype=dtypes.float32),
        trainable=False)

    if self.scale is None:
      self.scale = _get_default_scale(self.kernel_initializer, input_dim)
    self.kernel_scale = self.add_weight(
        name='kernel_scale',
        shape=(1,),
        dtype=dtypes.float32,
        initializer=init_ops.constant_initializer(self.scale),
        trainable=True,
        constraint='NonNeg')
    super(CustomRandomFourierFeatures, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank(3)
    if input_shape.dims[-1].value is None:
      raise ValueError(
          'The innermost dimension of input shape must be defined. Given: %s' %
          input_shape)
    return input_shape[:-1].concatenate(self.output_dim)

  def get_config(self):
    kernel_initializer = self.kernel_initializer
    if not isinstance(kernel_initializer, str):
      kernel_initializer = initializers.serialize(kernel_initializer)
    config = {
        'output_dim': self.output_dim,
        'kernel_initializer': kernel_initializer,
        'scale': self.scale,
    }
    base_config = super(CustomRandomFourierFeatures, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _get_random_features_initializer(initializer, shape):
  """Returns Initializer object for random features."""

  def _get_cauchy_samples(loc, scale, shape):
    probs = np.random.uniform(low=0., high=1., size=shape)
    return loc + scale * np.tan(np.pi * (probs - 0.5))

  random_features_initializer = initializer
  if isinstance(initializer, str):
    if initializer.lower() == 'gaussian':
      random_features_initializer = init_ops.random_normal_initializer(
          stddev=1.0)
    elif initializer.lower() == 'laplacian':
      random_features_initializer = init_ops.constant_initializer(
          _get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

    else:
      raise ValueError(
          'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'.format(
              random_features_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
  return random_features_initializer


def _get_default_scale(initializer, input_dim):
  if (isinstance(initializer, str) and initializer.lower() == 'gaussian'):
    return np.sqrt(input_dim / 2.0)
  return 1.0
