"""
Dilated Resnet implementation for semantic segmentation.
Original PyTorch version can be found at
https://github.com/fyu/drn/blob/master/segment.py
"""
from typing import Optional
import tensorflow as tf

import model_lib.dilated_resnet_backbone as drn
from model_lib.layers.wrap_common_layers_sn import spectralnorm_wrapper


def get_final_block(smoothing_conv, channel: int,
                    spectral_norm_args):
  smooth_layers = [tf.keras.layers.UpSampling2D(8, interpolation='bilinear')]
  if smoothing_conv:
    smooth_layers += [
        spectralnorm_wrapper(
            tf.keras.layers.Conv2D(channel, kernel_size=5, padding='same'),
            **spectral_norm_args),
        tf.keras.layers.ReLU(),
        spectralnorm_wrapper(
            tf.keras.layers.Conv2D(channel, kernel_size=5, padding='same'),
            **spectral_norm_args)
    ]
  return tf.keras.Sequential(smooth_layers)


class DRNSeg(tf.keras.Model):
  def __init__(
      self,
      model_name: str,
      classes: int = 10,
      l2_reg: float = 1e-4,
      dropout: float = 0.0,
      batch_size: int = 128,
      spectral_normalization: bool = False,
      spectral_batchnormalization: bool = False,
      soft_spectral_normalization: bool = False,
      coeff: Optional[int] = None,
      power_iterations: int = 1,
      smoothing_conv: bool = False,
  ):
    super(DRNSeg, self).__init__()
    """
    model_name can be one of:
    (
    'drn_a_50', 'drn_c_26', 'drn_c_42', 'drn_c_58', 'drn_d_22', 'drn_d_24', 
    'drn_d_38', 'drn_d_40', 'drn_d_54', 'drn_d_56', 'drn_d_105', 'drn_d_107'
    )  
    
    Example call: 
    model = DRNSeg('drn_a_50', 10)
    """
    spectral_norm_args = {
        'batch_size': batch_size,
        'spectral_normalization': spectral_normalization,
        'spectral_batchnormalization': spectral_batchnormalization,
        'soft_spectral_normalization': soft_spectral_normalization,
        'coeff': coeff,
        'power_iterations': power_iterations
    }
    model = drn.__dict__.get(model_name)(num_classes=classes,
                                         l2_reg=l2_reg,
                                         dropout=dropout,
                                         **spectral_norm_args)
    self.base = tf.keras.Sequential(model.layers[:-2])
    self.seg = tf.keras.layers.Conv2D(classes, kernel_size=1, use_bias=True,
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    self.seg = spectralnorm_wrapper(self.seg, **spectral_norm_args)
    self.softmax = tf.nn.softmax
    self.final_block = get_final_block(
        smoothing_conv,
        classes,
        spectral_norm_args)

  def call(self, inputs, training=True, mask=None):
    x = self.base(inputs, training)
    x = self.seg(x)
    y = self.final_block(x)
    return self.softmax(y), x
