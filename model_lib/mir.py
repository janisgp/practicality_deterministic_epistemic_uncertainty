"""Code for maximially informative representations (mir)."""

import os
from typing import Tuple, Dict, List
import numpy as np
import tensorflow as tf

from model_lib.baselines import BaseModel
from model_lib.baselines import SoftmaxModel
from model_lib.density_models import ClassConditionalGMM
from data_lib.get_image_dataset import DATASET_SHAPES
from model_lib.resnet_backbone import resnet_layer


def extract_features(model: BaseModel,
                     dataset: tf.data.Dataset,
                     norm_features: bool=True):
  features = []
  predictions = []
  for i, batch in enumerate(dataset):
    x, _ = batch

    out = model(inputs=x, return_features=True)

    features.append(out['features'].numpy())
    predictions.append(np.argmax(out['prediction'].numpy(), axis=-1))

  features = np.concatenate(features, axis=0)
  predictions = np.concatenate(predictions, axis=0)
  return features, predictions


class FCDecoder(tf.keras.models.Sequential):
  """Fully-connected decoder model."""

  def __init__(self, dims: Tuple[int], data_shape: Tuple[int, ...]):
    super().__init__()
    self.data_shape = data_shape

    layers = []
    for d in dims:
      layers.append(tf.keras.layers.Dense(d))
      layers.append(tf.keras.layers.ReLU())
    layers.append(tf.keras.layers.Dense(np.prod(data_shape)))
    super().__init__(layers)

  def call(self, inputs, training=None, mask=None) -> tf.Tensor:
    out = super().call(inputs=inputs)
    return tf.reshape(out, (out.shape[0],) + self.data_shape)


class ConvDecoder(tf.keras.models.Sequential):
  """Convolutional decoder model."""

  def __init__(self, feature_dims: int, data_shape: Tuple[int, ...]):
    super().__init__()
    self.data_shape = data_shape
    self.feature_dims = feature_dims

    layers = []
    layers.append(tf.keras.layers.Reshape(
        (1, 1, self.feature_dims)))  # (B, 1, 1, feature_dims)
    layers.append(tf.keras.layers.Conv2DTranspose(64, (3, 3),
                                                  strides=(2,
                                                           2)))  # (B, 3, 3, 64)
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.ReLU())
    layers.append(tf.keras.layers.Conv2DTranspose(32, (3, 3),
                                                  strides=(2,
                                                           2)))  # (B, 7, 7, 32)
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.ReLU())
    layers.append(tf.keras.layers.Conv2DTranspose(
        16, (3, 3), strides=(2, 2)))  # (B, 15, 15, 16)
    layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.ReLU())

    if data_shape[0] == 28:
      layers.append(tf.keras.layers.Conv2DTranspose(
          8, (1, 1), strides=(2, 2)))  # (B, 28, 28, 8)
      layers.append(tf.keras.layers.BatchNormalization())
      layers.append(tf.keras.layers.ReLU())
      layers.append(tf.keras.layers.Conv2D(data_shape[-1], 5,
                                           padding='same'))  # (B, 28, 28, 3)
    elif data_shape[0] == 32:
      layers.append(tf.keras.layers.Conv2DTranspose(
          8, (4, 4), strides=(2, 2)))  # (B, 32, 32, 8)
      layers.append(tf.keras.layers.BatchNormalization())
      layers.append(tf.keras.layers.ReLU())
      layers.append(tf.keras.layers.Conv2D(data_shape[-1], 5,
                                           padding='same'))  # (B, 32, 32, 3)
    else:
      raise ValueError(f'Invalid shape: {data_shape}!')
    super().__init__(layers)

  def call(self, inputs, training=None, mask=None) -> tf.Tensor:
    # inputs = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=1)
    out = super().call(inputs=inputs)
    return out


def resnet_transpose_conv_layer(inputs,
                                num_filters: int = 16,
                                kernel_size: int = 3,
                                strides: int = 1,
                                activation: str = 'relu',
                                batch_normalization: bool = True,
                                conv_first: bool = True):
  """2D Transpose Convolution-Batch Normalization-Activation stack builder

    Args:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or bn-activation-conv
          (False)

    Returns:
        x (tensor): tensor as input to the next layer
    """
  conv = tf.keras.layers.Conv2DTranspose(
      num_filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(1e-4))

  x = inputs
  if conv_first:
    x = conv(x)
    if batch_normalization:
      x = tf.keras.layers.BatchNormalization()(x)
    if activation != '':
      x = tf.keras.layers.Activation(activation)(x)
  else:
    if batch_normalization:
      x = tf.keras.layers.BatchNormalization()(x)
    if activation != '':
      x = tf.keras.layers.Activation(activation)(x)
    x = conv(x)
  return x


class ResNetDecoder(tf.keras.Model):
  """ResNet Decoder Version 1 Model builder [a]

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
               batchnorm: bool = True,
               num_res_blocks: int = 1,
               dropout: float = 0.0):

    # Start model definition.
    inputs = tf.keras.layers.Input(shape=input_shape)
    input_channels = input_shape[-1]

    if len(input_shape) == 1:
      #Reconstruction from flattened latent shape
      x = tf.keras.layers.Dense(8 * 8 * 64, activation='relu')(inputs)
      x = tf.keras.layers.Reshape((8, 8, 64))(x)
      input_channels = 64
    else:
      x = inputs

    if input_channels == 64:
      num_stacks = 3
    elif input_channels == 32:
      num_stacks = 2
    elif input_channels == 16:
      num_stacks = 1
    else:
      assert False, 'Invalid Input Shape! {}'.format(str(input_shape))

    num_filters = input_channels

    x = resnet_layer(
        inputs=x, num_filters=num_filters, batch_normalization=batchnorm)
    # Instantiate the stack of residual units
    for stack in range(num_stacks):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # upsample
          y = resnet_transpose_conv_layer(
              inputs=x,
              num_filters=num_filters,
              strides=strides,
              batch_normalization=batchnorm)
        else:
          y = resnet_layer(
              inputs=x,
              num_filters=num_filters,
              strides=strides,
              batch_normalization=batchnorm)

        y = resnet_layer(
            inputs=y,
            num_filters=num_filters,
            activation='',
            batch_normalization=batchnorm)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match
          # changed dims
          x = resnet_transpose_conv_layer(
              inputs=x,
              num_filters=num_filters,
              kernel_size=1,
              strides=strides,
              activation='',
              batch_normalization=batchnorm)
        x = tf.keras.layers.add([x, y])
        x = tf.keras.layers.Dropout(rate=dropout)(x)
        x = tf.keras.layers.Activation('relu')(x)
      num_filters /= 2

    outputs = resnet_layer(
        inputs=x, num_filters=3, activation='', batch_normalization=False)

    # Instantiate model.
    super().__init__(inputs=inputs, outputs=outputs)

  def train_step(self, data):

    out_dict = super().train_step(data=data)

    out_dict.update({'lr': self.optimizer.lr})

    return out_dict

  def test_step(self, data):

    out_dict = super().test_step(data=data)

    return out_dict


class MIR(SoftmaxModel):

  def __init__(self,
               reconstruction_weight: float = 0.0,
               density_model: str = 'gmm',
               **kwargs):

    super().__init__(**kwargs)
    self.reconstruction_weight = reconstruction_weight
    self.warmup = kwargs.get('warmup')
    self.warmup_counter = self.warmup
    self.feature_dims = kwargs.get('feature_dims')
    self.backbone_type = kwargs.get('backbone')
    self.density_model = density_model

    # decoder & loss function
    self.decoder = MIR._get_decoder(**kwargs)
    self.loss_func_reconstruction = tf.keras.losses.MeanSquaredError()

    self.density = None

  def init_density(self, normalize_features: bool = True):

    # density model
    if self.density_model == 'gmm':
      self.density = ClassConditionalGMM(
          nr_classes=self.nr_classes, red_dim=-1,
          normalize_features=normalize_features)
    else:
      raise ValueError(f'Unknown density model {self.density_model}!')

  def _reconstruction_weight_to_loss_weights(self,
                                             use_warmup: bool = True
                                            ) -> Tuple[float, float]:
    if self.reconstruction_weight >= 1.0:
      w1, w2 = 1 / self.reconstruction_weight, 1.0
    else:
      w1, w2 = 1.0, self.reconstruction_weight
    if not use_warmup:
      return w1, w2
    if not self.warmup is None and self.warmup_counter > 0:
      w1 = w1 * (self.warmup - self.warmup_counter) / self.warmup
    return w1, w2

  @staticmethod
  def _get_decoder(backbone: str, dataset: str,
                   feature_dims: int, **kwargs) -> tf.keras.models.Sequential:
    data_shape = DATASET_SHAPES[dataset]
    if backbone == 'simple_fc':
      return FCDecoder(dims=(200,), data_shape=data_shape)
    elif backbone == 'resnet':
      # return ConvDecoder(feature_dims=feature_dims, data_shape=data_shape)
      return ResNetDecoder(
          input_shape=(8, 8, 64),
          batchnorm=True,
          num_res_blocks=int((8-2)/6),
          dropout=kwargs.get('dropout'))
    else:
      raise ValueError(f'Unknown backbone {backbone}!')

  def on_epoch_end(self, epoch: int, **kwargs):
    if not self.warmup is None and self.warmup_counter > 0:
      self.warmup_counter -= 1
    else:
      pass

  def get_loss_names(self) -> List[str]:
    if self.reconstruction_weight > 0.0:
      return ['loss', 'loss_reconstruction', 'loss_prediction']
    else:
      return ['loss', 'loss_prediction']

  def call(self,
           inputs: tf.Tensor,
           training=None,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    output_dict = super().call(
        inputs=inputs, return_features=True, training=training)
    if self.reconstruction_weight > 0.0:
      if self.backbone_type == 'resnet':
        output_dict['reconstructions'] = self.decoder(
            output_dict['features_large'])
      else:
        output_dict['reconstructions'] = self.decoder(
            output_dict['features'])
    return output_dict

  def compute_loss(self, x: tf.Tensor, y: tf.Tensor,
                   out_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:

    prediction_weight, reconstruction_weight = \
      self._reconstruction_weight_to_loss_weights()

    out_dict['loss_prediction'] = self.loss_func(y, out_dict['prediction'])
    out_dict['loss_weighted'] = prediction_weight * out_dict['loss_prediction']
    if self.reconstruction_weight > 0.0:
      out_dict['loss_reconstruction'] = self.loss_func_reconstruction(
          x, out_dict['reconstructions'])
      out_dict['loss_weighted'] += reconstruction_weight * out_dict[
          'loss_reconstruction']

    prediction_weight, reconstruction_weight = \
      self._reconstruction_weight_to_loss_weights(use_warmup=False)
    if self.reconstruction_weight > 0.0:
      out_dict['loss'] = prediction_weight * out_dict['loss_prediction'] + \
                                reconstruction_weight * out_dict['loss_reconstruction']
    else:
      out_dict['loss'] = out_dict['loss_prediction']

    return out_dict

  def train_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                 **kwargs) -> Dict[str, tf.Tensor]:
    x, y = data

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      out = self.compute_loss(x=x, y=y, out_dict=out)

    grads = tape.gradient(out['loss_weighted'], self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    return out

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:
    x, y = data

    out = self.call(inputs=x, training=False)
    out = self.compute_loss(x=x, y=y, out_dict=out)

    return out

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Computes predictions and uncertainty estimates given x.

    Args:
      data: batch

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    x = data[0]
    output_dict = self.call(inputs=x, training=False, return_features=True)
    output_dict['uncertainty'] = -1 * self.density.marginal_log_probs(
        output_dict['features'])

    # output_dict['uncertainty'] = -1 * tf.reduce_max(self.density.class_conditional_log_probs(output_dict['features_large']), axis=-1)
    return output_dict

  def on_training_end(self, trainset: tf.data.Dataset, valset: tf.data.Dataset,
                      **kwargs):
    """For now: Done in evaluation script in order to tune density model hyperparameters."""
    pass

  def custom_load_weights(self, filepath: str, **kwargs):

    # load neural network weights
    super().custom_load_weights(filepath=filepath)

    # load density model
    if kwargs.get('load_density_model'):
      root = os.path.split(filepath)[0]
      if tf.io.gfile.exists(os.path.join(root, 'density.p')):
        self.density.load(path=root)
