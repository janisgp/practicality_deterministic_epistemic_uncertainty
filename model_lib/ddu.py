from typing import Dict, Tuple, Any
import tensorflow as tf

from model_lib.baselines import SoftmaxModel
from model_lib.density_models import ClassConditionalGMM

import tensorflow_probability as tfp
tfd = tfp.distributions


class DDUModel(SoftmaxModel):
  """
  Deep Deterministic Uncertainty (DDU) method.

  DDU disentangles the epistemic and aleatoric uncertainty. The epistemic uncertainty
  comes from the density estimation via Gaussian Mixture Model (GMM), and aleatoric
  uncertainty comes from softmax entropy.

  https://arxiv.org/pdf/2102.11582.pdf
  """

  def __init__(self, density_model: str = 'gmm', **kwargs):
    super().__init__(**kwargs)
    self.data_memory: Tuple[tf.Tensor, tf.Tensor] = None
    self.record_batches = kwargs.get('num_batches_fitting')
    self.n_components = kwargs.get('num_components')
    self.feature_dims = kwargs.get('feature_dims')
    self.density_model = density_model

    self.density = None

  def init_density(self, normalize_features: bool = True):

    # density model
    if self.density_model == 'gmm':
      self.density = ClassConditionalGMM(
          nr_classes=self.nr_classes, red_dim=-1,
          normalize_features=normalize_features)
    else:
      raise ValueError(f'Unknown density model {self.density_model}!')

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
    return output_dict

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """
    Computes predictions and uncertainty estimates given inputs.

    Args:
      data: batch (Tuple of tf.Tensor)

    Returns:
      dictionary with entries 'prediction' and 'uncertainty'
    """
    inputs = data[0]
    output_dict = self.call(inputs=inputs, training=False, return_features=True)
    entropy = entropy_tf(inputs=output_dict['prediction'], axis=-1)
    log_liklihood = self.density.marginal_log_probs(output_dict['features'])
    output_dict['uncertainty'] = -1 * log_liklihood
    output_dict['total_uncertainty'] = entropy - log_liklihood  # Todo: Check it, because this is not clearly stated in their paper.
    output_dict['aleatoric'] = entropy

    return output_dict
