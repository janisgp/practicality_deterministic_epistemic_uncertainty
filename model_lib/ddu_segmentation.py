import os
from typing import Dict, Tuple, Optional
import tensorflow as tf

from model_lib.baselines_seg import SoftmaxSeg
from model_lib.density_models import ConvFeatureGMM
import model_lib.dilated_resnet_backbone as drn
from model_lib.dilated_resnet_segmentation import DRNSeg
from model_lib.dilated_resnet_segmentation import get_final_block


class DDUSeg(SoftmaxSeg):
  """DDU for semantic segmentation.
  """

  def __init__(self,
               reconstruction_weight: float = 0.0,
               density_model: str = 'gmm',
               **kwargs):
    super().__init__(**kwargs)
    self.feature_dims = kwargs.get('feature_dims')
    self.density_model = density_model

    self.num_classes = kwargs.get('nr_classes')
    self.l2_reg = kwargs.get('l2_reg')
    self.dropout = kwargs.get('dropout')

    # SN args
    spectral_norm_args = {
        'batch_size':
            kwargs.get('batch_size'),
        'spectral_normalization':
            kwargs.get('spectral_normalization'),
        'spectral_batchnormalization':
            kwargs.get('spectral_batchnormalization'),
        'soft_spectral_normalization':
            kwargs.get('soft_spectral_normalization'),
        'coeff':
            kwargs.get('coeff'),
        'power_iterations':
            kwargs.get('power_iterations')
    }

    # Get backbone
    model = drn.__dict__.get(kwargs.get('backbone'))(
        num_classes=self.num_classes,
        l2_reg=self.l2_reg,
        dropout=self.dropout,
        **spectral_norm_args)
    self.backbone = tf.keras.Sequential(model.layers[:-2])

    # head
    self.final_block_seg = get_final_block(
        smoothing_conv=kwargs.get('smoothing_conv'),
        channel=kwargs.get('nr_classes'),
        spectral_norm_args=spectral_norm_args)
    self.fc_seg = tf.keras.layers.Dense(
        self.num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
    self.softmax = tf.nn.softmax

    self.density = None

  def get_backbone(self, **kwargs) -> tf.keras.models.Model:
    # create backbone
    backbone = drn.__dict__.get(kwargs.get('backbone'))(
        num_classes=kwargs.get('nr_classes'),
        l2_reg=kwargs.get('l2_reg'),
        dropout=kwargs.get('dropout'))
    return tf.keras.Sequential(backbone.layers[:-2])

  def init_density(self):

    # density model
    if self.density_model == 'gmm':
      self.density = ConvFeatureGMM(n_components=10, red_dim=32,
                                    normalize_features=False)
    else:
      raise ValueError(f'Unknown density model {self.density_model}!')

  def _reconstruction_weight_to_loss_weights(self,
                                             use_warmup: bool = True
                                            ) -> Tuple[float, float]:
    if self.reconstruction_weight >= 1.0:
      w1, w2 = 1 / self.reconstruction_weight, 1.0
    else:
      w1, w2 = 1.0, self.reconstruction_weight
    return w1, w2

  def on_epoch_end(self, epoch: int, **kwargs):
    pass

  def call(self,
           inputs: tf.Tensor,
           training=None,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    output_dict = dict()
    output_dict['features'] = self.backbone(inputs=inputs, training=training)
    output_dict['prediction'] = self.softmax(
        self.final_block_seg(self.fc_seg(output_dict['features'])))
    return output_dict

  def compute_loss(self, x: tf.Tensor, y: tf.Tensor,
                   out_dict: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    out_dict['loss'] = self.loss_func(y, out_dict['prediction'])
    return out_dict

  def train_step(self,
                 data: Tuple[tf.Tensor, tf.Tensor],
                 step: int = 0) -> Dict[str, tf.Tensor]:
    x, y = data

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      out = self.compute_loss(x=x, y=y, out_dict=out)

    grads = tape.gradient(out['loss'], self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    return out

  def test_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                step: int = 0) -> Dict[str, tf.Tensor]:
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
    output_dict['uncertainty_pixel'] = -1 * self.density.log_probs(
        output_dict['features'])
    output_dict['uncertainty'] = self.aggregate_uncertainty(
        uncertainty=output_dict['uncertainty_pixel'])
    return output_dict

  def custom_load_weights(self, filepath: str, **kwargs):

    # load neural network weights
    super().custom_load_weights(filepath=filepath)

    # load density model
    if kwargs.get('load_density_model'):
      root = os.path.split(filepath)[0]
      if tf.io.gfile.exists(os.path.join(root, 'density.p')):
        self.density.load(path=root)
