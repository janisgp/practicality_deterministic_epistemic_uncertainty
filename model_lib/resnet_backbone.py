# Lint as: python3
"""Resnet."""

from typing import Optional
import tensorflow as tf

from model_lib.layers.wrap_common_layers_sn import wrap_dense
from model_lib.layers.wrap_common_layers_sn import wrap_conv2D
from model_lib.layers.wrap_common_layers_sn import wrap_batchnorm


def resnet_layer(inputs,
                 batch_size: int = 128,
                 num_filters: int = 16,
                 kernel_size: int = 3,
                 strides: int = 1,
                 activation: str = 'relu',
                 l2_reg: float = 1e-4,
                 batch_normalization: bool = True,
                 conv_first: bool = True,
                 spectral_normalization: bool = False,
                 spectral_batchnormalization: bool = False,
                 soft_spectral_normalization: bool = False,
                 coeff: Optional[int] = None,
                 power_iterations: int = 1):
  """2D Convolution-Batch Normalization-Activation stack builder

  Args:
      inputs (tensor): input tensor from input image or previous layer
      num_filters (int): Conv2D number of filters
      kernel_size (int): Conv2D square kernel dimensions
      strides (int): Conv2D square stride dimensions
      activation (string): activation name
      batch_normalization (bool): whether to include batch normalization
      l2_reg (float): l2 regularization weight
      conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
      spectral_normalization (bool): whether to include spectral normalization
      spectral_batchnormalization (bool): whether to include spectral
        normalization on batchnorm scale
      soft_spectral_normalization (bool): whether to relax spectral
        normalization
      coeff (bool): coefficient to which the Lipschitz constant must be
        restricted in soft spectral normalization
      power_iterations (int): the number of iterations during spectral
        normalization

  Returns:
      x (tensor): tensor as input to the next layer
  """
  conv = tf.keras.layers.Conv2D(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

  conv = wrap_conv2D(conv, batch_size, spectral_normalization,
                     soft_spectral_normalization, coeff, power_iterations)

  x = inputs

  if conv_first:
    x = conv(x)
    if batch_normalization:
      bn_1 = tf.keras.layers.BatchNormalization()
      bn_1 = wrap_batchnorm(bn_1, spectral_batchnormalization,
                            soft_spectral_normalization, coeff,
                            power_iterations)
      x = bn_1(x)
    if activation != '':
      x = tf.keras.layers.Activation(activation)(x)
  else:
    if batch_normalization:
      bn_2 = tf.keras.layers.BatchNormalization()
      bn_2 = wrap_batchnorm(bn_2, spectral_batchnormalization,
                            soft_spectral_normalization, coeff,
                            power_iterations)
      x = bn_2(x)
    if activation != '':
      x = tf.keras.layers.Activation(activation)(x)
    x = conv(x)
  return x


class ResNet(tf.keras.Model):
  """ResNet Version 1 Model builder [a]

  Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
  Last ReLU is after the shortcut connection.
  At the beginning of each stage, the feature map size is halved (downsampled)
  by a convolutional layer with strides=2, while the number of filters is
  doubled. Within each stage, the layers have the same number filters and the
  same number of filters.
  Features maps sizes:
  stage 0: 32x32, 16
  stage 1: 16x16, 32
  stage 2:  8x8,  64
  The Number of parameters is approx the same as Table 6 of [a]:
  ResNet20 0.27M
  ResNet32 0.46M
  ResNet44 0.66M
  ResNet56 0.85M
  ResNet110 1.7M

  Args:
      input_shape (tensor): shape of input image tensor
      depth (int): number of core convolutional layers
      num_classes (int): number of classes (CIFAR10 has 10)

  Returns:
      model (Model): Keras model instance
  """

  def __init__(self,
               input_shape: tuple,
               depth: int,
               batch_size: int = 128,
               batchnorm: bool = True,
               num_classes: int = 10,
               l2_reg: float = 1e-4,
               dropout: float = 0.0,
               spectral_normalization: bool = False,
               spectral_batchnormalization: bool = False,
               soft_spectral_normalization: bool = False,
               coeff: Optional[int] = None,
               power_iterations: int = 1,):

    if (depth - 2) % 6 != 0:
      raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = tf.keras.layers.Input(shape=input_shape)
    xi = resnet_layer(inputs=inputs,
                     batch_size=batch_size,
                     batch_normalization=batchnorm,
                     l2_reg=l2_reg,
                     spectral_normalization=spectral_normalization,
                     spectral_batchnormalization=spectral_batchnormalization,
                     soft_spectral_normalization=soft_spectral_normalization,
                     coeff=coeff,
                     power_iterations=power_iterations)
    # Instantiate the stack of residual units
    features = []
    for stack in range(3):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # downsample
        y = resnet_layer(
            inputs=xi,
            batch_size=batch_size,
            num_filters=num_filters,
            strides=strides,
            batch_normalization=batchnorm,
            l2_reg=l2_reg,
            spectral_normalization=spectral_normalization,
            spectral_batchnormalization=spectral_batchnormalization,
            soft_spectral_normalization=soft_spectral_normalization,
            coeff=coeff,
            power_iterations=power_iterations)
        y = resnet_layer(
            inputs=y,
            batch_size=batch_size,
            num_filters=num_filters,
            activation='',
            batch_normalization=batchnorm,
            l2_reg=l2_reg,
            spectral_normalization=spectral_normalization,
            spectral_batchnormalization=spectral_batchnormalization,
            soft_spectral_normalization=soft_spectral_normalization,
            coeff=coeff,
            power_iterations=power_iterations)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match
          # changed dims
          xi = resnet_layer(
              inputs=xi,
              batch_size=batch_size,
              num_filters=num_filters,
              kernel_size=1,
              strides=strides,
              activation='',
              batch_normalization=batchnorm,
              l2_reg=l2_reg,
              spectral_normalization=spectral_normalization,
              spectral_batchnormalization=spectral_batchnormalization,
              soft_spectral_normalization=soft_spectral_normalization,
              coeff=coeff,
              power_iterations=power_iterations)
        x = tf.keras.layers.add([xi, y])
        features.append(x)
        xi = tf.keras.layers.Dropout(rate=dropout)(features[-1])
        xi = tf.keras.layers.Activation('relu')(xi)
      num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x_pooled = tf.keras.layers.AveragePooling2D(pool_size=8)(xi)
    y = tf.keras.layers.Flatten()(x_pooled)

    # Instantiate model.
    super().__init__(inputs=inputs, outputs=[y, x])
