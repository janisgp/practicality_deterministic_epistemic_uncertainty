"""BaseModel: Abstracts functionality used by several projects."""

import os
from typing import Dict, List, Tuple
import tensorflow as tf


class BaseModel(tf.keras.models.Model):

  def get_parameter_count(self):
    total_parameters = 0
    for variable in self.trainable_variables:
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim
      total_parameters += variable_parameters
    return total_parameters

  def custom_load_weights(self, filepath: str, **kwargs):

    print(f'Loading weights from {filepath}')
    fname = os.path.split(filepath)[-1]
    suffix = fname[fname.find('.')+1:]

    # load model weights
    if suffix == 'h5':
      tmp_file = f'/tmp/tmp_model.h5'
      tf.io.gfile.copy(filepath, tmp_file, overwrite=True)
      super().load_weights(tmp_file)
    elif suffix == 'tf':
      super().load_weights(filepath)
    else:
      raise ValueError(f'Unkown weight format {suffix}!')

  def custom_save_weights(self, filepath: str, tmp_name: str = 'tmp'):

    print(f'Saving weights to {filepath}')
    fname = os.path.split(filepath)[-1]
    suffix = fname[fname.find('.')+1:]

    # saving model weights
    if suffix == 'h5':
      tmp_file = f'/tmp/{tmp_name}_model.h5'
      self.save_weights(filepath=tmp_file, overwrite=True)
      tf.io.gfile.copy(tmp_file, filepath, overwrite=True)
    elif suffix == 'tf':
      self.save_weights(filepath=filepath, overwrite=True)
    else:
      raise ValueError(f'Unkown weight format {suffix}!')

  def call(self,
           inputs: tf.Tensor,
           training=None,
           mask=None,
           return_features=False) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def on_epoch_end(self, epoch: int, **kwargs):
    pass

  def on_training_end(self, trainset: tf.data.Dataset, valset: tf.data.Dataset,
                      logging_path: str, **kwargs):
    """Allows to perform post training operation.

    E.g.

    fit the distribution of hidden representations.
    """
    pass

  def get_loss_names(self) -> List[str]:
    """Returns list of keys of losses.

    Used by trainer to for logging purpose.

    Any key in this list should be returned by train_step/test_step and
    will be logged to tensorboard and to console.
    """
    return ['loss']

  def train_step(self, data, step: int):
    raise NotImplementedError

  def initialize(self, data: Tuple[tf.Tensor, tf.Tensor]):
    _ = self(inputs=data[0])
    _ = self.train_step(data=data, step=1)
