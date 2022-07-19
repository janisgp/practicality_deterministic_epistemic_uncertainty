# Lint as: python3
"""Simple fully-connected and convolutional neural networks."""

from typing import Optional
import tensorflow as tf

from model_lib.layers.wrap_common_layers_sn import wrap_dense
from model_lib.layers.wrap_common_layers_sn import wrap_batchnorm


def get_simple_fc_net(hidden_dim: int = 100,
                      hidden_layer: int = 5,
                      input_shape: tuple = (28, 28, 1),
                      dropout: float = 0.0,
                      l2_reg: float = 1e-4,
                      batch_normalization: bool = True,
                      output_dim: int = 10,
                      spectral_normalization: bool = False,
                      spectral_batchnormalization: bool = False,
                      soft_spectral_normalization: bool = False,
                      coeff: Optional[int] = None,
                      power_iterations: int = 1):
  """Creates simple fully-connected model Args:

      hidden_dim: int. dim of hidden representations
      hidden_layer: int. # of hidden layers
      input_shape: tuple. Shaphe of input
      dropout: float. dropout probability
      output_dim: dimensionality of the output
      l2_reg (float): l2 regularization weight
      batch_normalization (bool): whether to include batch normalization
      spectral_normalization (bool): whether to include spectral normalization
      spectral_batchnormalization (bool): whether to include spectral
      normalization
        on batchnorm scale
      soft_spectral_normalization (bool): whether to relax spectral
      normalization
      coeff (bool): coefficient to which the Lipschitz constant must be
      restricted in soft
        spectral normalization
      power_iterations (int): the number of iterations during spectral
      normalization

  Returns:
      model: keras.Sequential
  """

  hidden_dims = [hidden_dim for _ in range(hidden_layer)]

  # create layer
  layers = [tf.keras.layers.Flatten(input_shape=input_shape)]
  for d in hidden_dims:
    layers.append(
        wrap_dense(
            tf.keras.layers.Dense(
                d, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            spectral_normalization, soft_spectral_normalization, coeff,
            power_iterations))
    if batch_normalization:
      layers.append(
          wrap_batchnorm(tf.keras.layers.BatchNormalization(),
                         spectral_batchnormalization,
                         soft_spectral_normalization, coeff, power_iterations))
    layers.append(tf.keras.layers.ReLU())
    layers.append(tf.keras.layers.Dropout(dropout))
  layers.append(
      wrap_dense(
          tf.keras.layers.Dense(
              output_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
          spectral_normalization, soft_spectral_normalization, coeff,
          power_iterations))

  return layers


class SimpleFC(tf.keras.models.Sequential):

  def __init__(self,
               hidden_dim: int = 100,
               hidden_layer: int = 5,
               input_shape: tuple = (28, 28, 1),
               dropout: float = 0.0,
               l2_reg: float = 1e-4,
               batch_normalization: bool = True,
               output_dim: int = 10,
               spectral_normalization: bool = False,
               spectral_batchnormalization: bool = False,
               soft_spectral_normalization: bool = False,
               coeff: Optional[int] = None,
               power_iterations: int = 1,
               **kwargs):
    layers = get_simple_fc_net(
        hidden_dim=hidden_dim,
        hidden_layer=hidden_layer,
        input_shape=input_shape,
        dropout=dropout,
        l2_reg=l2_reg,
        batch_normalization=batch_normalization,
        output_dim=output_dim,
        spectral_normalization=spectral_normalization,
        spectral_batchnormalization=spectral_batchnormalization,
        soft_spectral_normalization=soft_spectral_normalization,
        coeff=coeff,
        power_iterations=power_iterations)
    super().__init__(layers)
