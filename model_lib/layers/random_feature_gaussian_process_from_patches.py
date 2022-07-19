import edward2
from model_lib.layers.random_fourier_features_from_patches import CustomRandomFourierFeatures

import tensorflow.compat.v2 as tf

initializers = edward2.tensorflow.initializers

_SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')


class RandomFeatureGaussianProcessFromPatches(
    edward2.layers.RandomFeatureGaussianProcess):

  def _build_sublayer_classes(self):
    """Defines sublayer classes."""
    self.bias_layer = tf.Variable
    self.dense_layer = tf.keras.layers.Dense
    self.covariance_layer = CustomLaplaceRandomFeatureCovariance
    self.input_normalization_layer = tf.keras.layers.LayerNormalization

  def _make_random_feature_layer(self, name):
    """Defines random feature layer depending on kernel type."""
    if not self.use_custom_random_features:
      # Use CustomRandomFourierFeatures
      return CustomRandomFourierFeatures(
          output_dim=self.num_inducing,
          kernel_initializer=self.gp_kernel_type,
          scale=self.gp_kernel_scale,
          trainable=self.gp_kernel_scale_trainable,
          dtype=self.dtype,
          name=name)

    if self.gp_kernel_type.lower() == 'linear':
      custom_random_feature_layer = tf.keras.layers.Lambda(
          lambda x: x, name=name)
    else:
      # Use user-supplied configurations.
      custom_random_feature_layer = self.dense_layer(
          units=self.num_inducing,
          use_bias=True,
          activation=self.custom_random_features_activation,
          kernel_initializer=self.custom_random_features_initializer,
          bias_initializer=self.random_features_bias_initializer,
          trainable=False,
          name=name)

    return custom_random_feature_layer

  def call(self, inputs, global_step=None, training=None):
    """ Call method of Random Feature GP from patches.

    Inputs must have shape [batch_size, num_patches, num_input_features].

    Notice that num_patches = H_out * W_out and depends on the conv filter
    shape, stride and padding.

    If instead has shape [batch_size, num_patches, patch_length,
    num_input_features], where patch_length
    is patch.shape[0]*patch.shape[1], reshape to [batch_size, num_patches,
    patch_length*num_input_features]
    before feeding the layer.

    Output_shape will always be [batch_size, num_patches, num_output_features].
    If you want to retrieve
    the corresponding 2D feature map, reshape to [batch_size, H_out, W_out,
    num_output_features].

    Args:
      inputs: inputs of shape [batch_size, num_patches, num_input_features]
    """
    # Computes random features.
    gp_inputs = inputs
    if self.normalize_input:
      # import pdb
      # pdb.set_trace()
      gp_inputs = self._input_norm_layer(gp_inputs)
      # gp_inputs = tf.squeeze(
      #     self._input_norm_layer(tf.expand_dims(gp_inputs, axis=1)))
    elif self.use_custom_random_features:
      # Supports lengthscale for custom random feature layer by directly
      # rescaling the input.
      gp_input_scale = tf.cast(self.gp_input_scale, inputs.dtype)
      gp_inputs = gp_inputs * gp_input_scale

    gp_feature = self._random_feature(gp_inputs)

    if self.scale_random_features:
      # Scale random feature by 2. / sqrt(num_inducing) following [1].
      # When using GP layer as the output layer of a nerual network,
      # it is recommended to turn this scaling off to prevent it from changing
      # the learning rate to the hidden layers.
      gp_feature_scale = tf.cast(self.gp_feature_scale, inputs.dtype)
      gp_feature = gp_feature * gp_feature_scale

    # Computes posterior center (i.e., MAP estimate) and variance.
    gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias

    if self.return_gp_cov:
      gp_covmat = self._gp_cov_layer(gp_feature, gp_output, training)
      gp_covmat = tf.transpose(gp_covmat, [1, 0, 2])

    # Assembles model output.
    model_output = [
        gp_output,
    ]
    if self.return_gp_cov:
      model_output.append(gp_covmat)
    if self.return_random_features:
      model_output.append(gp_feature)

    return model_output


class CustomLaplaceRandomFeatureCovariance(
    edward2.layers.LaplaceRandomFeatureCovariance):

  def build(self, input_shape):
    gp_feature_dim = input_shape[-1]
    num_patches = input_shape[-2]

    # Convert gp_feature_dim to int value for TF1 compatibility.
    if isinstance(gp_feature_dim, tf.compat.v1.Dimension):
      gp_feature_dim = gp_feature_dim.value

    # Posterior precision matrix for the GP's random feature coefficients.
    self.initial_precision_matrix = (
        self.ridge_penalty * tf.tile(
            tf.expand_dims(tf.eye(gp_feature_dim, dtype=self.dtype), 0),
            [num_patches, 1, 1]))

    self.precision_matrix = (
        self.add_weight(
            name='gp_precision_matrix',
            shape=(num_patches, gp_feature_dim, gp_feature_dim),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Constant(
                self.initial_precision_matrix),
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA))
    self.built = True

  def make_precision_matrix_update_op(self, gp_feature, logits,
                                      precision_matrix):
    """Defines update op for the precision matrix of feature weights."""
    if self.likelihood != 'gaussian':
      if logits is None:
        raise ValueError(
            f'"logits" cannot be None when likelihood={self.likelihood}')

      if logits.shape[-1] != 1:
        raise ValueError(
            f'likelihood={self.likelihood} only support univariate logits.'
            f'Got logits dimension: {logits.shape[-1]}')

    batch_size = tf.shape(gp_feature)[0]
    batch_size = tf.cast(batch_size, dtype=gp_feature.dtype)

    # Computes batch-specific normalized precision matrix.
    if self.likelihood == 'binary_logistic':
      prob = tf.sigmoid(logits)
      prob_multiplier = prob * (1. - prob)
    elif self.likelihood == 'poisson':
      prob_multiplier = tf.exp(logits)
    else:
      prob_multiplier = 1.

    gp_feature_adjusted = tf.sqrt(prob_multiplier) * gp_feature

    # Notice that squeezing is fine under the assumption of iid patches
    # squeezed_gp_feature_adjusted = tf.reshape(gp_feature_adjusted, (-1, gp_feature_adjusted.shape[-1]))
    # precision_matrix_minibatch = tf.matmul(
    #   squeezed_gp_feature_adjusted, squeezed_gp_feature_adjusted, transpose_a=True)

    gp_feature_transposed = tf.transpose(gp_feature_adjusted, [1, 0, 2])
    precision_matrix_minibatch = tf.matmul(
        gp_feature_transposed, gp_feature_transposed, transpose_a=True)

    # Updates the population-wise precision matrix.
    if self.momentum > 0:
      # Use moving-average updates to accumulate batch-specific precision
      # matrices.
      precision_matrix_minibatch = precision_matrix_minibatch / batch_size
      precision_matrix_new = (
          self.momentum * precision_matrix +
          (1. - self.momentum) * precision_matrix_minibatch)
    else:
      # Compute exact population-wise covariance without momentum.
      # If use this option, make sure to pass through data only once.
      precision_matrix_new = precision_matrix + precision_matrix_minibatch

    # Returns the update op.
    return precision_matrix.assign(precision_matrix_new)

  def compute_predictive_covariance(self, gp_feature):
    """Computes posterior predictive variance.

    Approximates the Gaussian process posterior using random features.
    Given training random feature Phi_tr (num_train, num_hidden) and testing
    random feature Phi_ts (batch_size, num_hidden). The predictive covariance
    matrix is computed as (assuming Gaussian likelihood):

    s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

    where s is the ridge factor to be used for stablizing the inverse, and I is
    the identity matrix with shape (num_hidden, num_hidden).

    Args:
      gp_feature: (tf.Tensor) The random feature of testing data to be used for
        computing the covariance matrix. Shape (batch_size, gp_hidden_size).

    Returns:
      (tf.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
    """
    # Computes the covariance matrix of the feature coefficient.
    feature_cov_matrix = tf.linalg.inv(self.precision_matrix)
    gp_feature = tf.transpose(gp_feature, [1, 0, 2])

    # Computes the covariance matrix of the gp prediction.
    cov_feature_product = tf.matmul(
        feature_cov_matrix, gp_feature, transpose_b=True) * self.ridge_penalty
    gp_cov_matrix = tf.matmul(gp_feature, cov_feature_product)
    return gp_cov_matrix

  def call(self, inputs, logits=None, training=None):
    """Minibatch updates the GP's posterior precision matrix estimate.

    Args:
      inputs: (tf.Tensor) GP random features, shape (batch_size,
        gp_hidden_size).
      logits: (tf.Tensor) Pre-activation output from the model. Needed for
        Laplace approximation under a non-Gaussian likelihood.
      training: (tf.bool) whether or not the layer is in training mode. If in
        training mode, the gp_weight covariance is updated using gp_feature.

    Returns:
      gp_stddev (tf.Tensor): GP posterior predictive variance,
        shape (batch_size, batch_size).
    """
    batch_size = tf.shape(inputs)[0]
    num_patches = tf.shape(inputs)[1]
    training = self._get_training_value(training)

    if training:
      # Define and register the update op for feature precision matrix.
      precision_matrix_update_op = self.make_precision_matrix_update_op(
          gp_feature=inputs,
          logits=logits,
          precision_matrix=self.precision_matrix)
      self.add_update(precision_matrix_update_op)
      # Return null estimate during training.
      return tf.tile(
          tf.expand_dims(tf.eye(batch_size, dtype=self.dtype), 0),
          [num_patches, 1, 1])
    else:
      # Return covariance estimate during inference.
      return self.compute_predictive_covariance(gp_feature=inputs)
