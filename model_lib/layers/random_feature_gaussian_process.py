import edward2 as ed
import tensorflow as tf


class CustomRandomFeatureGaussianProcess(ed.layers.RandomFeatureGaussianProcess
                                        ):
  """Custom version of ed.layers.RandomFeatureGaussianProcess that does not share same name across different variables.

  Necessary to work fine with keras model saving.
  """

  def __init__(self, **kwargs):
    super(CustomRandomFeatureGaussianProcess, self).__init__(**kwargs)

  def build(self, input_shape):
    self._build_sublayer_classes()
    if self.normalize_input:
      self._input_norm_layer = self.input_normalization_layer(
          name='gp_input_normalization')
      self._input_norm_layer.build(input_shape)
      input_shape = self._input_norm_layer.compute_output_shape(input_shape)

    self._random_feature = self._make_random_feature_layer(
        name='gp_random_feature')
    self._random_feature.build(input_shape)
    input_shape = self._random_feature.compute_output_shape(input_shape)

    if self.return_gp_cov:
      self._gp_cov_layer = self.covariance_layer(
          momentum=self.gp_cov_momentum,
          ridge_penalty=self.gp_cov_ridge_penalty,
          likelihood=self.gp_cov_likelihood,
          dtype=self.dtype,
          name='gp_covariance')
      with tf.name_scope('gp_covariance'):
        self._gp_cov_layer.build(input_shape)

    self._gp_output_layer = self.dense_layer(
        units=self.units,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(self.l2_regularization),
        dtype=self.dtype,
        name='gp_output_weights',
        **self.gp_output_kwargs)
    with tf.name_scope('gp_output_weights'):
      self._gp_output_layer.build(input_shape)

    self._gp_output_bias = self.bias_layer(
        initial_value=[self.gp_output_bias] * self.units,
        dtype=self.dtype,
        trainable=self.gp_output_bias_trainable,
        name='gp_output_bias')

    self.built = True
