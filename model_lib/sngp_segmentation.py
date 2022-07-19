import functools
from typing import Dict, Tuple, Optional
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import edward2 as ed

tfd = tfp.distributions

from model_lib.baselines_seg import BaseSegModel
import model_lib.dilated_resnet_backbone as drn
from model_lib.utils import dempster_shafer_metric_tf
from model_lib.layers.random_feature_gaussian_process_from_patches import RandomFeatureGaussianProcessFromPatches


def make_random_feature_initializer(random_feature_type):
  # Use stddev=0.05 to replicate the default behavior of
  # tf.keras.initializer.RandomNormal.
  if random_feature_type == 'orf':
    return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
  elif random_feature_type == 'rff':
    return tf.keras.initializers.RandomNormal(stddev=0.05)
  else:
    return random_feature_type


def mean_field_logits(logits,
                      covmat=None,
                      mean_field_factor=1.,
                      likelihood='logistic'):
  """Adjust the model logits so its softmax approximates the posterior mean [1].

  Arguments:
    logits: A float tensor of shape (batch_size, num_classes).
    covmat: A float tensor of shape (batch_size, batch_size). If None then it
      assumes the covmat is an identity matrix.
    mean_field_factor: The scale factor for mean-field approximation, used to
      adjust the influence of posterior variance in posterior mean
      approximation. If covmat=None then it is used as the scaling parameter for
      temperature scaling.
    likelihood: Likelihood for integration in Gaussian-approximated latent
      posterior.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  """
  if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
    raise ValueError(
        f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
    )

  if float(mean_field_factor) < 0:
    return logits

  # Compute standard deviation.
  if covmat is None:
    variances = 1.
  else:
    # variances = tf.linalg.diag_part(covmat)
    variances = np.diagonal(covmat, axis1=0, axis2=-1)
    variances = tf.transpose(variances, [2, 0, 1])

  # Compute scaling coefficient for mean-field approximation.
  if likelihood == 'poisson':
    logits_scale = tf.exp(-variances * mean_field_factor / 2.)
  else:
    logits_scale = tf.sqrt(1. + variances * mean_field_factor)

  # Cast logits_scale to compatible dimension.
  if len(logits.shape) > 1:
    logits_scale = tf.expand_dims(logits_scale, axis=-1)

  return logits / logits_scale


def numpy_diagonal_function(x):
  return np.diagonal(x, axis1=0, axis2=-1)


class GP2D(tf.keras.Model):

  def __init__(self,
               num_classes,
               n_inducing_points,
               gp_scale,
               gp_bias,
               gp_input_normalization,
               gp_random_feature_type,
               gp_cov_discount_factor,
               gp_cov_ridge_penalty,
               share_gp=False,
               patch_size=(3, 3)):
    super(GP2D, self).__init__()
    self.num_classes = num_classes
    self.n_inducing_points = n_inducing_points
    self.gp_scale = gp_scale
    self.gp_bias = gp_bias
    self.gp_input_normalization = gp_input_normalization
    self.gp_random_feature_type = gp_random_feature_type
    self.gp_cov_discount_factor = gp_cov_discount_factor
    self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
    self.patch_size = patch_size
    self.share_gp = share_gp

  def build(self, input_shape):
    gp_layer_partial = functools.partial(
        RandomFeatureGaussianProcessFromPatches,
        num_inducing=self.n_inducing_points,
        gp_kernel_scale=self.gp_scale,
        gp_output_bias=self.gp_bias,
        normalize_input=self.gp_input_normalization,
        gp_cov_momentum=self.gp_cov_discount_factor,
        gp_cov_ridge_penalty=self.gp_cov_ridge_penalty,
        scale_random_features=False,
        use_custom_random_features=True,
        custom_random_features_initializer=make_random_feature_initializer(
            self.gp_random_feature_type))

    num_patches = input_shape[1] * input_shape[
        2]  # TODO(mattiasegu): defined num_patches according to conv
    # import pdb
    # pdb.set_trace()
    if self.share_gp:
      gp_input_shape = [input_shape[0], num_patches, input_shape[-1]]
      self.gp = gp_layer_partial(units=self.num_classes)
      self.gp.build(gp_input_shape)
    else:
      gp_input_shape = [input_shape[0], 1, input_shape[-1]]
      self.gps = []
      for h in range(input_shape[1]):
        self.gps.append([])
        for w in range(input_shape[2]):
          gp_layer = gp_layer_partial(units=self.num_classes)
          gp_layer.build(gp_input_shape)
          self.gps[h].append(gp_layer)

  def reset_covariance_matrix(self):
    if self.share_gp:
      self.gp.reset_covariance_matrix()
    else:
      for h in range(len(self.gps)):
        for w in range(len(self.gps[h])):
          self.gps[h][w].reset_covariance_matrix()

  def call(self, inputs, training=None, mask=None):
    if self.share_gp:
      reshaped_inputs = tf.reshape(inputs,
                                   (inputs.shape[0], -1, inputs.shape[-1]))
      # TODO(mattiasegu): use extract_patches_from_image to get patches as in conv so
      # to support also conv that are not 1x1
      means, vars = self.gp(reshaped_inputs)
      means = tf.reshape(means, inputs.shape[:-1] + [means.shape[-1]])
      vars = tf.reshape(vars, inputs.shape[:-1] + [vars.shape[-1]])

    else:
      means = []
      vars = []
      for h in range(inputs.shape[1]):
        means_w = []
        vars_w = []
        for w in range(inputs.shape[2]):
          m, v = self.gps[h][w](inputs[:, h:h+1, w, :])
          means_w.append(tf.squeeze(m))
          vars_w.append(tf.squeeze(v))
        means.append(tf.stack(means_w, axis=1))
        vars.append(tf.stack(vars_w, axis=1))
      means = tf.stack(means, axis=1)
      vars = tf.stack(vars, axis=1)
    return means, vars


class DRNSegGP(tf.keras.Model):

  def __init__(self, model_name: str, **kwargs):
    super(DRNSegGP, self).__init__()
    """model_name can be one of: ( 'drn_a_50', 'drn_c_26', 'drn_c_42', 'drn_c_58', 'drn_d_22', 'drn_d_24', 'drn_d_38', 'drn_d_40', 'drn_d_54', 'drn_d_56', 'drn_d_105', 'drn_d_107' )

    Example call:
    model = DRNSeg('drn_a_50', 10)
    """
    # GP args
    self.num_classes = kwargs.get('nr_classes')
    self.l2_reg = kwargs.get('l2_reg')
    self.dropout = kwargs.get('dropout')
    self.n_inducing_points = kwargs.get('num_inducing_points')
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
    self.share_gp = kwargs.get('share_gp')

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
    # Get model
    model = drn.__dict__.get(model_name)(
        num_classes=self.num_classes,
        l2_reg=self.l2_reg,
        dropout=self.dropout,
        **spectral_norm_args)
    self.base = tf.keras.Sequential(model.layers[:-2])
    self.gp_seg = self.get_last_sngp_layer(
        self.num_classes, self.n_inducing_points, self.gp_scale, self.gp_bias,
        self.gp_input_normalization, self.gp_random_feature_type,
        self.gp_cov_discount_factor, self.gp_cov_ridge_penalty, self.share_gp)

    self.up = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')

  def get_last_sngp_layer(self,
                          num_classes,
                          n_inducing_points,
                          gp_scale,
                          gp_bias,
                          gp_input_normalization,
                          gp_random_feature_type,
                          gp_cov_discount_factor,
                          gp_cov_ridge_penalty,
                          share_gp=False):
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
    # TODO(mattiasegu): will probably need to tune this parameters
    output_layer = GP2D(
        num_classes,
        n_inducing_points,
        gp_scale,
        gp_bias,
        gp_input_normalization,
        gp_random_feature_type,
        gp_cov_discount_factor,
        gp_cov_ridge_penalty,
        share_gp=share_gp,
        patch_size=(1, 1))

    return output_layer

  def build(self, input_shape):
    self.base.build(input_shape)
    gp_seg_input_shape = self.base.compute_output_shape(input_shape)
    self.gp_seg.build(gp_seg_input_shape)

  # @tf.function
  def call(self, inputs, training=True, mask=None):
    x = self.base(inputs, training)
    m, v = self.gp_seg(x)
    m, v = self.up(m), self.up(v)
    return (m, v), x


class SNGPSeg(BaseSegModel):
  """Improving Deterministic Uncertainty Estimation in Deep Learning for Classification and Regression (DUE).

  Describe method.

  https://arxiv.org/pdf/2102.11409.pdf
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.num_classes = kwargs.get('nr_classes')
    self.l2_reg = kwargs.get('l2_reg')
    self.n_inducing_points = kwargs.get('num_inducing_points')
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

    # self.backbone = self.get_backbone(**kwargs)
    self.built = False

  def get_backbone(self, **kwargs) -> tf.keras.models.Model:
    """Create backbone.

    This method overwrites that of parent function BaseSegModel.
    """
    backbone = DRNSegGP(kwargs.get('backbone'), **kwargs)
    return backbone

  def call(self,
           inputs: tf.Tensor,
           training=None,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    # Model call returns prediction=(mean, var) and features
    # import pdb
    # pdb.set_trace()
    prediction, features = self.backbone(inputs, training=training)
    output_dict = {'prediction': prediction}
    if return_features:
      output_dict['features'] = features
    return output_dict

  def train_step(self,
                 data: Tuple[tf.Tensor, tf.Tensor],
                step: int=0) -> Dict[str, tf.Tensor]:

    x, y = data
    if not self.built:
      self.backbone.build([None] + x.shape[1:])
      self.built = True
      # self.backbone(tf.ones((1, 400, 640, 3)))

    if tf.equal(step, 0) and float(self.gp_cov_discount_factor) < 0:
      # Reset covariance estimator at the begining of a new epoch.
      self.backbone.gp_seg.reset_covariance_matrix()

    with tf.GradientTape() as tape:
      out = self.call(inputs=x, training=True)
      logits = out['prediction']

      if isinstance(logits, (list, tuple)):
        # If model returns a tuple of (logits, covmat), extract logits
        logits, _ = logits
        out['prediction'] = tf.nn.softmax(out['prediction'][0])
      else:
        out['prediction'] = tf.nn.softmax(out['prediction'])

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

  def test_step(self, data: Tuple[tf.Tensor, tf.Tensor],
                step: int=0) -> Dict[str, tf.Tensor]:

    x, y = data
    if not self.built:
      self.backbone.build([None] + x.shape[1:])
      self.built = True

    out = self.call(inputs=x, training=False)

    logits = out['prediction']
    if isinstance(logits, (list, tuple)):
      # If model returns a tuple of (logits, covmat), extract both
      logits, covmat = logits
    else:
      covmat = tf.tile(
          tf.expand_dims(
              tf.expand_dims(tf.eye(x.shape[0], dtype=self.dtype), 1), 1),
          [1, x.shape[1], x.shape[2], 1])

    logits = mean_field_logits(
        logits, covmat, mean_field_factor=self.gp_mean_field_factor)
    out['prediction'] = tf.nn.softmax(logits)

    stddev = tf.sqrt(tf.linalg.diag_part(covmat))
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))

    out.update({'loss': negative_log_likelihood})

    return out

  def uncertainty(self, data: Tuple[tf.Tensor]) -> Dict[str, tf.Tensor]:
    """SNGP (https://arxiv.org/pdf/2102.11409.pdf) computes the uncertainty as the Dempster-Shafer metric
    """
    x = data[0]

    if not self.built:
      self.backbone.build([None] + x.shape[1:])
      self.built = True

    output_dict = self.call(inputs=x, training=False)
    logits = output_dict['prediction']

    if isinstance(logits, (list, tuple)):
      # If model returns a tuple of (logits, covmat), extract both
      logits, covmat = logits
    else:
      covmat = tf.tile(
          tf.expand_dims(
              tf.expand_dims(tf.eye(x.shape[0], dtype=self.dtype), 1), 1),
          [1, x.shape[1], x.shape[2], 1])

    logits = mean_field_logits(
        logits, covmat, mean_field_factor=self.gp_mean_field_factor)
    output_dict['prediction'] = logits

    # entropy_marginal = entropy_tf(inputs=output_dict['prediction'], axis=-1)
    output_dict['uncertainty_pixel'] = dempster_shafer_metric_tf(
        inputs=logits, num_classes=self.num_classes, axis=-1)
    output_dict['uncertainty'] = self.aggregate_uncertainty(
        output_dict['uncertainty_pixel'])
    return output_dict
