from typing import Optional
import tensorflow as tf

from model_lib.layers.spectral_batchnorm import SpectralBatchNormalization
from model_lib.layers.spectral_batchnorm import SoftSpectralBatchNormalization
from model_lib.layers.spectral_normalization import SpectralNormalization
from model_lib.layers.spectral_normalization import SoftSpectralNormalization
from model_lib.layers.spectral_normalization_conv import SpectralNormalizationConv
from model_lib.layers.spectral_normalization_conv import SoftSpectralNormalizationConv


def spectralnorm_wrapper(
    layer,
    batch_size: int = 128,
    spectral_normalization: bool = False,
    spectral_batchnormalization: bool = False,
    soft_spectral_normalization: bool = False,
    coeff: Optional[int] = None,
    power_iterations: int = 1,
):

  if isinstance(layer, tf.keras.layers.Conv2D):
    layer = wrap_conv2D(layer, batch_size, spectral_normalization,
                        soft_spectral_normalization, coeff, power_iterations)
  elif isinstance(layer, tf.keras.layers.BatchNormalization):
    layer = wrap_batchnorm(layer, spectral_batchnormalization,
                           soft_spectral_normalization, coeff, power_iterations)
  elif isinstance(layer, tf.keras.layers.Dense):
    layer = wrap_dense(layer, spectral_normalization,
                       soft_spectral_normalization, coeff, power_iterations)
  else:
    raise AttributeError(f'{layer.__class__} class is not supported')

  return layer


def wrap_batchnorm(bn_layer,
                   spectral_normalization: bool = False,
                   soft_spectral_normalization: bool = False,
                   coeff: Optional[int] = None,
                   power_iterations: int = 1):
  """ This wrapper controls the usage of spectral normalization, standard or soft, on batch normalization layers.

    Wrap `tf.keras.layers.BatchNormalization`:
    >>> bn_layer = tf.keras.layers.BatchNormalization()
    >>> bn_layer = wrap_batchnorm(bn_layer, True, False, 3, 1)
    >>> bn_layer
    <model_lib.layers.spectral_batchnorm.SpectralBatchNormalization object at
    0x7fe5db930c90>

    Args:
      bn_layer: A `tf.keras.layers.BatchNormalization` instance.
      spectral_normalization: `bool`, if True wrap the layer with spectral
        normalization
      soft_spectral_normalization: `bool`, if True wrap the layer with soft
        spectral normalization
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.
  """
  if spectral_normalization:
    if soft_spectral_normalization:
      wrapped_bn_layer = SoftSpectralBatchNormalization(
          bn_layer, coeff, power_iterations, name='ssn_' + bn_layer.name)
    else:
      wrapped_bn_layer = SpectralBatchNormalization(
          bn_layer, coeff, power_iterations, name='sn_' + bn_layer.name)
  else:
    return bn_layer
  return wrapped_bn_layer


def wrap_conv2D(conv_layer,
                batch_size: int,
                spectral_normalization: bool = False,
                soft_spectral_normalization: bool = False,
                coeff: Optional[int] = None,
                power_iterations: int = 1):
  """ This wrapper controls the usage of spectral normalization, standard or soft, on conv layers.

    Wrap `tf.keras.layers.Conv2D`:
    >>> conv_layer = tf.keras.layers.Conv2D(2, 2)
    >>> conv_layer = wrap_conv2D(conv_layer, 4, True, False, 3, 1)
    >>> conv_layer
    <model_lib.layers.spectral_normalization_conv.SpectralNormalizationConv
    object at 0x7fe5db930dd0>

    Args:
      conv_layer: A `tf.keras.layers.Conv2D` instance.
      spectral_normalization: `bool`, if True wrap the layer with spectral
        normalization
      soft_spectral_normalization: `bool`, if True wrap the layer with soft
        spectral normalization
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.

    Raises:
      AttributeError: If `layer` does not have `kernel_size`.
    """
  if not hasattr(conv_layer, 'kernel_size'):
    raise AttributeError("{} object has no attribute 'kernel_size'".format(
        type(conv_layer).__name__))

  if not spectral_normalization:
    return conv_layer

  if conv_layer.kernel_size == 1:
    # use spectral norm fc, because bound are tight for 1x1 convolutions
    if soft_spectral_normalization:
      wrapped_conv_layer = SoftSpectralNormalization(
          conv_layer, coeff, power_iterations, name='ssn_' + conv_layer.name)
    else:
      wrapped_conv_layer = SpectralNormalization(
          conv_layer, coeff, power_iterations, name='sn_' + conv_layer.name)
  else:
    # Otherwise use spectral norm conv, with loose bound
    if soft_spectral_normalization:
      wrapped_conv_layer = SoftSpectralNormalizationConv(
          conv_layer,
          batch_size,
          coeff,
          power_iterations,
          name='ssn_' + conv_layer.name)
    else:
      wrapped_conv_layer = SpectralNormalizationConv(
          conv_layer,
          batch_size,
          coeff,
          power_iterations,
          name='sn_' + conv_layer.name)

  return wrapped_conv_layer


def wrap_dense(dense_layer,
               spectral_normalization: bool = False,
               soft_spectral_normalization: bool = False,
               coeff: Optional[int] = None,
               power_iterations: int = 1):
  """ This wrapper controls the usage of spectral normalization, standard or soft, on dense layers.

    Wrap `tf.keras.layers.Dense`:
    >>> dense_layer = tf.keras.layers.Dense(10)
    >>> dense_layer = wrap_dense(dense_layer, True, False, 3, 1)
    >>> dense_layer
    <model_lib.layers.spectral_normalization.SpectralNormalization object at
    0x7fe5db930d90>

    Args:
      bn_layer: A `tf.keras.layers.Dense` instance.
      spectral_normalization: `bool`, if True wrap the layer with spectral
        normalization
      soft_spectral_normalization: `bool`, if True wrap the layer with soft
        spectral normalization
      coeff: coefficient to which the Lipschitz constant must be restricted
      power_iterations: `int`, the number of iterations during normalization.
  """

  if not spectral_normalization:
    return dense_layer

  if soft_spectral_normalization:
    wrapped_dense_layer = SoftSpectralNormalization(
        dense_layer, coeff, power_iterations, name='ssn_' + dense_layer.name)
  else:
    wrapped_dense_layer = SpectralNormalization(
        dense_layer, coeff, power_iterations, name='sn_' + dense_layer.name)

  return wrapped_dense_layer
