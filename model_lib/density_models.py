# Lint as: python3
"""Class-conditional density models."""

import os
from typing import Any
import pickle
import numpy as np
import scipy
import sklearn.mixture as mixture
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import scipy.ndimage
import tensorflow as tf


class GMM(object):
  """Wraps densities."""

  def __init__(self, n_components: int = 20, red_dim: int = 256,
               normalize_features: bool = True):
    super(GMM, self).__init__()
    self.n_components = n_components
    self.red_dim = red_dim
    self.gmm = mixture.GaussianMixture(
        n_components=self.n_components, covariance_type='full')

    if red_dim != -1:
      self.pca = decomposition.PCA(n_components=red_dim)
    else:
      self.pca = None

    self.normalize_features = normalize_features

  def fit(self, x: Any, x_val: Any):
    """Fit output-conditional density.

    The distribution of x associated with each class is estimated using
    separate GMM.

    Args:
      x: Training data
      x_val: Validation data
      n_components: Number components in each GMM

    Returns:
      List of GMMs
    """

    print(f'Fitting GMM on data of shape {x.shape}')

    if self.normalize_features:
      print('Normalizing data...', flush=True)
      x = preprocessing.normalize(x)
      x_val = preprocessing.normalize(x_val)

    if self.pca:
      print(f'Fitting PCA to reduce dim to {self.red_dim}...', flush=True)
      self.pca.fit(x)
      x = self.pca.transform(x)
      x_val = self.pca.transform(x_val)

    print('Fitting GMM...', flush=True)
    self.gmm.fit(X=x)

    log_prob_train = self.gmm.score(x)
    log_prob_val = self.gmm.score(x_val)

    print(
        f'log prob train: {log_prob_train} | log prob val: {log_prob_val}',
        flush=True)

  def log_probs(self, x: Any) -> Any:
    if self.normalize_features:
      x = preprocessing.normalize(x)
    if self.pca:
      x = self.pca.transform(x)
    log_probs = self.gmm.score_samples(x)
    return log_probs


class ConvFeatureGMM(GMM):
  """Wraps densities on conv features."""

  def __init__(self,
               n_components: int = 20,
               red_dim: int = 256,
               upsampling: int = 8,
               normalize_features: bool = True):
    super(ConvFeatureGMM, self).__init__(
        n_components=n_components, red_dim=red_dim,
        normalize_features=normalize_features)
    self.upsampling = 8

  def log_probs(self, x: Any) -> Any:
    orig_shape = x.shape
    x_unraveled = np.reshape(x, (-1, x.shape[-1]))
    log_probs = super().log_probs(x=x_unraveled)
    log_probs = np.reshape(log_probs, orig_shape[:-1])
    log_probs = np.concatenate([
        np.expand_dims(scipy.ndimage.zoom(l, zoom=8, order=1), axis=0)
        for l in log_probs
    ], axis=0)
    return log_probs


class ClassConditionalGMM(object):
  """Wraps conditional densities."""

  def __init__(self,
               n_components: int = 1,
               nr_classes: int = 10,
               red_dim: int = 64,
               normalize_features: bool = True):
    super(ClassConditionalGMM, self).__init__()
    self.n_components = n_components
    self.nr_classes = nr_classes
    self.normalize_features = normalize_features
    self.class_conditional_densities = []
    if red_dim != -1:
      self.pca = decomposition.PCA(n_components=red_dim)
    else:
      self.pca = None
    for i in range(self.nr_classes):
      self.class_conditional_densities.append(
          mixture.GaussianMixture(
              n_components=self.n_components, covariance_type='diag'))

  def fit(self, x: Any, y: Any, x_val: Any, y_val: Any):
    """Fit output-conditional density. The distribution of x associated with each class is estimated using separate GMM.

    Args:
      x: Training data
      y: Predictions on training data
      x_val: Validation data
      y_val: Predictions on validation data
      n_components: Number components in each GMM
      nr_classes:

    Returns:
      List of GMMs
    """

    if self.normalize_features:
      x = preprocessing.normalize(x)
      x_val = preprocessing.normalize(x_val)

    if self.pca:
      print('Fitting PCA...', flush=True)
      self.pca.fit(x)
      x = self.pca.transform(x)
      x_val = self.pca.transform(x_val)

    for i in range(self.nr_classes):

      if np.sum(y == i) > 1:  # sanity check whether this idx exists
        self.class_conditional_densities[i] = \
          self.class_conditional_densities[i].fit(X=x[y == i])

        # log log_prob on train/val
        if np.sum(y_val == i) > 0:
          log_prob_train = np.mean(
              self.class_conditional_densities[i].score_samples(x[y == i]))
          log_prob_val = np.mean(
              self.class_conditional_densities[i].score_samples(
                  x_val[y_val == i]))
          print(
              f'{i}-th component log probs | Train: {log_prob_train} | Val: {log_prob_val}'
          )

  def class_conditional_log_probs(self, x: Any) -> Any:
    log_probs = []
    if self.normalize_features:
      x = preprocessing.normalize(x)
    if self.pca:
      x = self.pca.transform(x)
    for density in self.class_conditional_densities:
      try:
        log_probs.append(np.expand_dims(density.score_samples(x), -1))
      except:
        pass
    return np.concatenate(log_probs, -1)

  def marginal_log_probs(self, x: Any):
    """Computes marginal likelihood (epistemic uncertainty)of x. Assuming class balance.

    Args:
        x (np.array): array of dim batch_size x features

    Returns:
      epistemic uncertainty: dim batch_size
    """
    cc_log_probs = self.class_conditional_log_probs(x)
    return scipy.special.logsumexp(cc_log_probs, axis=-1)

  def predict(self, x: Any) -> Any:
    """Predictions.

    Args:
      x:

    Returns:
    """
    log_probs = self.class_conditional_log_probs(x)
    return np.argmax(log_probs, 1)

  def sample(self, n: int) -> Any:
    """Assuming class balance.

    Args:
        n (int): number of samples

    Returns:
      samples: dim batch_size x features
    """
    class_idxs = np.random.choice(
        range(len(self.class_conditional_densities)), size=n, replace=True)
    samples = []
    for idx in class_idxs:
      samples.append(self.class_conditional_densities[idx].sample()[0])
    return np.concatenate(samples, 0), class_idxs

  def save(self, path: str):
    """Saves means and covariance matrices to pickle file

    Args:
      path: Folder where covariance and means will be saved
    """
    weights, means, covariances, precisions_cholesky = [], [], [], []
    for i in range(self.nr_classes):
      try:
        weights.append(self.class_conditional_densities[i].weights_)
        means.append(self.class_conditional_densities[i].means_)
        covariances.append(self.class_conditional_densities[i].covariances_)
        precisions_cholesky.append(
            self.class_conditional_densities[i].precisions_cholesky_)
      except:
        weights.append(None)
        means.append(None)
        covariances.append(None)
        precisions_cholesky.append(None)

    params = {
        'weights': weights,
        'means': means,
        'covariances': covariances,
        'precisions_cholesky': precisions_cholesky
    }
    tmp_path = os.path.join('/tmp', 'density.p')
    pickle.dump(params, open(tmp_path, 'wb'))
    tf.io.gfile.copy(tmp_path, os.path.join(path, 'density.p'), overwrite=True)

  def load(self, path):
    """Loads parameters.

    Args:
      path: Root folder containing density.p
    """
    tmp_path = os.path.join('/tmp', 'density.p')
    tf.io.gfile.copy(os.path.join(path, 'density.p'), tmp_path, overwrite=True)
    params = pickle.load(open(tmp_path, 'rb'))

    for i in range(self.nr_classes):
      self.class_conditional_densities[i].weights_ = params['weights'][i]
      self.class_conditional_densities[i].means_ = params['means'][i]
      self.class_conditional_densities[i].covariances_ = params['covariances'][
          i]
      self.class_conditional_densities[i].precisions_cholesky_ = params[
          'precisions_cholesky'][i]
