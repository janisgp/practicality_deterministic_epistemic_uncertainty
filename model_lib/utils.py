# Lint as: python3
"""Utils."""

from typing import Any
import numpy as np
import tensorflow as tf


def categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-5):
  """ Implements categorical crossentropy. """
  y_pred = y_pred / tf.reduce_sum(y_pred, -1, True)
  return -tf.reduce_sum(y_true * tf.math.log(y_pred + eps), -1)


def dempster_shafer_metric_tf(inputs: tf.Tensor, num_classes: int, axis: int = -1, eps: float = 1e-5):
  exp_inputs = tf.math.exp(inputs + tf.convert_to_tensor(eps, dtype=tf.float32))
  return num_classes/(num_classes+tf.reduce_sum(exp_inputs*inputs, axis=axis))


def entropy_tf(inputs: tf.Tensor, axis: int = -1, eps: float = 1e-5, dtype=tf.float32):
  log_inputs = tf.math.log(inputs + tf.convert_to_tensor(eps, dtype=dtype))
  return -tf.reduce_sum(log_inputs*inputs, axis=axis)


def custom_model_checkpoint(model: tf.keras.models.Model, path: str):
  print(path)
  tmp_file = '/tmp/model.h5'
  model.save_weights(filepath=tmp_file, overwrite=True)
  tf.io.gfile.copy(tmp_file, path, overwrite=True)


def trunc_log(val: tf.Tensor) -> tf.Tensor:
  return tf.cond(
    tf.math.reduce_any(tf.math.is_inf(tf.math.log(val))),
    lambda: -100,
    lambda: tf.math.log(val))


def bce_from_torch(reduction: str = 'mean') -> Any:
  """
  Replicates binary cross entropy loss implemented in pytorch.

  Args:
    reduction: reduce the result accordingly
      none: no reduction, mean: return the average

  Returns:
    BCE loss function
  """

  def _bce_from_torch(y1: tf.Tensor, y2: tf.Tensor) -> tf.Tensor:

    log = trunc_log(y1)
    log_inv = trunc_log(tf.constant(1, dtype=tf.float32) - y1)
    vals = -1 * (y2*log + (tf.constant(1, dtype=tf.float32) - y2)*log_inv)

    if reduction == 'none':
      return vals
    elif reduction == 'mean':
      return tf.reduce_mean(vals)
    else:
      raise ValueError(f'Unknwn reduction type {reduction}!')

  return _bce_from_torch


def get_epistemic_aleatoric_uncertainty(samples,
                                        from_logits: bool = False,
                                        eps: float = 1e-5):
  """Computes epistemic and aleatoric uncertainty from samples (np.array).

  Args:
    samples (np.array): default: batch_size x classes x mc_samples
    from_logits (bool): whether samples are logits or post-softmax
    eps (float): used for numerical stability

  Returns:
    epistemic_uncertainty (np.array): batch_size
    aleatoric_uncertainty (np.array): batch_size
  """

  if from_logits:
    samples_exp = np.exp(samples)
    samples = samples_exp / np.sum(samples_exp,
                                   axis=1,
                                   keepdims=True)
  entropy = -1 * (samples * np.log(samples + eps)).sum(1)
  mean_of_entropy = entropy.mean(-1)
  mean_probs = samples.mean(-1)
  entropy_of_mean = -1 * (mean_probs *
                          np.log(mean_probs + eps)).sum(1)

  epistemic_uncertainty = entropy_of_mean - mean_of_entropy
  aleatoric_uncertainty = mean_of_entropy

  return epistemic_uncertainty, aleatoric_uncertainty


def dice_coef(y_true, y_pred, smooth=1):
  """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
  intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
  denominator = tf.reduce_sum(tf.square(y_true), -1) + tf.reduce_sum(tf.square(y_pred), -1) + smooth
  return (2. * intersection + smooth) / denominator


def dice_coef_loss(y_true, y_pred):
  losses = 1 - dice_coef(y_true, y_pred)
  return tf.reduce_mean(losses)


def weighted_categorical_crossentropy(weights):
  """A weighted version of keras.objectives.categorical_crossentropy

  Variables:
      weights: numpy array of shape (C,) where C is the number of classes

  Usage:
      weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the
      normal weights, class 3 10x.
      loss = weighted_categorical_crossentropy(weights)
      model.compile(loss=loss,optimizer='adam')
  """

  weights = tf.keras.backend.variable(weights)

  def loss(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(),
                                   1 - tf.keras.backend.epsilon())
    # calc
    loss = y_true * tf.keras.backend.log(y_pred) * weights
    loss = -tf.keras.backend.sum(loss, -1)
    return tf.reduce_mean(loss)

  return loss
