from typing import Dict, Tuple
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import edward2 as ed

from model_lib.baselines import BaseUncertaintyModel
from model_lib.utils import dempster_shafer_metric_tf
from model_lib.layers.random_feature_gaussian_process import CustomRandomFeatureGaussianProcess


def make_random_feature_initializer(random_feature_type):
  # Use stddev=0.05 to replicate the default behavior of
  # tf.keras.initializer.RandomNormal.
  if random_feature_type == 'orf':
    return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
  elif random_feature_type == 'rff':
    return tf.keras.initializers.RandomNormal(stddev=0.05)
  else:
    return random_feature_type


class SNGP(BaseUncertaintyModel):
  """Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression (DUE).

  Describe method.

  https://arxiv.org/pdf/2102.11409.pdf
  """

  def __init__(self, **kwargs):

    super().__init__(**kwargs)

    self.num_classes = kwargs.get('nr_classes')
    self.l2_reg = kwargs.get('l2_reg')
    self.n_inducing_points = kwargs.get('n_inducing_points')
    self.n_inducing_points = kwargs.get(
        'nr_classes'
    ) if self.n_inducing_points is None else self.n_inducing_points
    self.gp_scale = kwargs.get('gp_scale')
    self.gp_bias = kwargs.get('gp_bias')
    self.gp_input_normalization = kwargs.get('gp_input_normalization')
    self.gp_random_feature_type = kwargs.get('gp_random_feature_type')
    self.gp_cov_discount_factor = kwargs.get('gp_cov_discount_factor')
    self.gp_cov_ridge_penalty = kwargs.get('gp_cov_ridge_penalty')
    self.gp_mean_field_factor = kwargs.get('gp_mean_field_factor')

    self.gp = self.get_last_sngp_layer(self.num_classes, self.n_inducing_points,
                                       self.gp_scale, self.gp_bias,
                                       self.gp_input_normalization,
                                       self.gp_random_feature_type,
                                       self.gp_cov_discount_factor,
                                       self.gp_cov_ridge_penalty)

    self.gp.build((None, kwargs.get('feature_dims')))

  def get_last_sngp_layer(self, num_classes, n_inducing_points, gp_scale,
                          gp_bias, gp_input_normalization,
                          gp_random_feature_type, gp_cov_discount_factor,
                          gp_cov_ridge_penalty):
    """Initialize last GP layer to add on top of backbone as in SNGP

    (https://arxiv.org/abs/2006.10108).
    Implementation derived from https://github.com/google/uncertainty-baselines/
    Args:
      num_classes: Number of output classes.
      n_inducing_points: The hidden dimension of the GP layer, which corresponds
        to the number of random features used for the approximation.
      gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
      gp_bias: The bias term for GP layer.
      gp_input_normalization: Whether to normalize the input using LayerNorm for
        GP layer. This is similar to automatic relevance determination (ARD) in
        the classic GP learning.
      gp_random_feature_type: The type of random feature to use for
        `RandomFeatureGaussianProcess`.
      gp_cov_discount_factor: The discount factor to compute the moving average
        of precision matrix.
      gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.

    Returns:
      tf.keras.Model.
    """
    # output_layer = ed.layers.RandomFeatureGaussianProcess(
    output_layer = CustomRandomFeatureGaussianProcess(
        units=num_classes,
        num_inducing=n_inducing_points,
        gp_kernel_scale=float(gp_scale),
        gp_output_bias=gp_bias,
        normalize_input=gp_input_normalization,
        gp_cov_momentum=gp_cov_discount_factor,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        scale_random_features=False,
        use_custom_random_features=True,
        custom_random_features_initializer=make_random_feature_initializer(
            gp_random_feature_type))

    return output_layer

  def call(self,
           inputs: tf.Tensor,
           training=None,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    backbone_output = self.backbone(inputs, training=training)
    if isinstance(backbone_output, list):
      features = backbone_output[0]
    else:
      features = backbone_output
    prediction = self.gp(features, training=training)
    output_dict = {'prediction': prediction, 'features': features}
    return output_dict

  def train_step(self,
                 data: Tuple[tf.Tensor, tf.Tensor],
                 step: int = 0) -> Dict[str, tf.Tensor]:

    x, y = data
    if tf.equal(step, 0) and float(self.gp_cov_discount_factor) < 0:
      # Reset covaraince estimator at the begining of a new epoch.
      self.gp.reset_covariance_matrix()

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      logits = out['prediction']

      if isinstance(logits, (list, tuple)):
        # If model returns a tuple of (logits, covmat), extract logits
        logits, _ = logits
        out['prediction'], _ = out['prediction']

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
      filtered_variables = []
      for var in self.backbone.trainable_variables:
        # Apply l2 on the weights. This excludes BN parameters and biases, but
        # pay caution to their naming scheme.
        if 'kernel' in var.name or 'bias' in var.name:
          filtered_variables.append(tf.reshape(var, (-1,)))

      l2_loss = self.l2_reg * 2 * tf.nn.l2_loss(
          tf.concat(filtered_variables, axis=0))
      loss_value = negative_log_likelihood + l2_loss

    grads = tape.gradient(loss_value, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    out.update({'loss': loss_value})
    return out

  def test_step(self, data: Tuple[tf.Tensor,
                                  tf.Tensor]) -> Dict[str, tf.Tensor]:

    x, y = data

    out = self.call(inputs=x, training=False)

    logits = out['prediction']
    if isinstance(logits, (list, tuple)):
      # If model returns a tuple of (logits, covmat), extract both
      logits, covmat = logits
    else:
      covmat = tf.eye(x.shape[0])

    logits = ed.layers.utils.mean_field_logits(
        logits, covmat, mean_field_factor=self.gp_mean_field_factor)
    out['prediction'] = logits

    stddev = tf.sqrt(tf.linalg.diag_part(covmat))
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))

    out.update({'loss': negative_log_likelihood})
    return out

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """SNGP (https://arxiv.org/pdf/2102.11409.pdf) computes the uncertainty as the Dempster-Shafer metric
    """
    x = data[0]
    output_dict = self.call(inputs=x, training=False)
    logits = output_dict['prediction']

    if isinstance(logits, (list, tuple)):
      # If model returns a tuple of (logits, covmat), extract both
      logits, covmat = logits
    else:
      covmat = tf.eye(x.shape[0])
    logits = ed.layers.utils.mean_field_logits(
        logits, covmat, mean_field_factor=self.gp_mean_field_factor)
    output_dict['prediction'] = logits

    # entropy_marginal = entropy_tf(inputs=output_dict['prediction'], axis=-1)
    dempster_shafer_value = dempster_shafer_metric_tf(
        inputs=logits, num_classes=self.num_classes, axis=-1)
    output_dict['uncertainty'] = dempster_shafer_value
    return output_dict
