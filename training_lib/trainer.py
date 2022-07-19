import os
import time
from typing import Dict, Tuple, Any, Optional
import numpy as np
import tensorflow as tf

from model_lib.base_model import BaseModel
from model_lib.baselines import BaseUncertaintyModel
from model_lib.mir import extract_features
from evaluation_lib.evaluate_utils import write_dict_to_csv
from data_lib.segmentation_labels import to_colored_label


class BaseTrainer:

  def __init__(self,
               model: BaseModel,
               logging_path: str,
               patience: int = 20,
               lr_schedule: Any = None,
               reduce_lr_on_plateau_gamma: float = 1.0,
               cpkt_frequency: int = 1,
               metric_criterion: str = 'val/loss',
               **kwargs):

    self.model = model
    self.logging_path = logging_path
    self.lr_schedule = lr_schedule
    self.patience = patience
    self.cpkt_frequency = cpkt_frequency
    self.patience_count = tf.Variable(initial_value=0, dtype=tf.int32)
    self.epoch = tf.Variable(initial_value=0, dtype=tf.int32)
    self.reduce_lr_on_plateau_gamma = reduce_lr_on_plateau_gamma
    self.metric_criterion = metric_criterion
    self.init_best_criterion()

    _log_dir = os.path.join(self.logging_path, 'log')
    self.tb_writer = tf.summary.create_file_writer(_log_dir)

    self.metrics = self.get_metrics()

    self.checkpoint_manager, self.checkpoint = self.setup_checkpoint()

  def init_best_criterion(self):
    if 'acc' in self.metric_criterion:
      self.best_criterion = tf.Variable(
          initial_value=-1 * np.infty, dtype=tf.float32)
    else:
      self.best_criterion = tf.Variable(
          initial_value=np.infty, dtype=tf.float32)

  def check_criterion(self, value: float):
    if 'acc' in self.metric_criterion:
      return self.best_criterion.numpy() < value
    else:
      return self.best_criterion.numpy() > value

  def setup_checkpoint(self):
    # , optimizer=model.optimizer
    restore_kwargs = dict(
        model=self.model,
        epoch=self.epoch,
        optimizer=self.model.optimizer,
        patience_count=self.patience_count,
        best_criterion=self.best_criterion)
    checkpoint = tf.train.Checkpoint(**restore_kwargs)

    cpkt_path = os.path.join(self.logging_path, 'checkpoints')
    tf.io.gfile.mkdir(cpkt_path)
    return tf.train.CheckpointManager(checkpoint, cpkt_path, max_to_keep=10), checkpoint

  def restore_checkpoint(self, trainset: tf.data.Dataset):

    list_ckp = self.checkpoint_manager.checkpoints

    # if a checkpoint exists, restore the latest checkpoint.
    if list_ckp:

      # init params
      _, data = enumerate(trainset).__next__()
      self.train_step(batch=data, step=0)

      print(f"Found: {len(list_ckp)} checkpoints")
      status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
      print("Latest checkpoint restored!!")
      status.assert_existing_objects_matched()
      print(f"Training starts from epoch {self.epoch.numpy()}")

  def get_metrics(self) -> Dict:
    raise NotImplementedError

  def update_metrics(self,
                     batch: Tuple[tf.Tensor, tf.Tensor],
                     output_dict: dict,
                     prefix: str = 'train'):
    raise NotImplementedError

  def reset_metrics(self):
    for key in self.metrics:
      self.metrics[key].reset_states()

  def write_tb(self, step: int, trainset: tf.data.Dataset, valset: tf.data.Dataset):
    with self.tb_writer.as_default():
      for key in self.metrics.keys():
        tf.summary.scalar(key, self.metrics[key].result(), step=step)

  def log(self, step: int):
    log_string = f' Epoch {step} | '
    for key in self.metrics:
      result = self.metrics[key].result()
      log_string += f'{key}: {result} | '
    print(log_string, flush=True)

  def on_epoch_end(self, e):

    # call model.on_epoch_end
    self.model.on_epoch_end(epoch=e)

    # save trainer state
    self.epoch.assign_add(1)
    e = self.epoch.numpy()
    if e % self.cpkt_frequency == 0:
      self.checkpoint_manager.save()

    # lr schedule
    # if self.reduce_lr_on_plateau_gamma < 1.0, we only reduce lr on plateau after patience
    if self.lr_schedule is not None and self.reduce_lr_on_plateau_gamma == 1.0:
      self.model.optimizer.learning_rate.assign(
        self.lr_schedule(e + 1, self.model.optimizer.learning_rate))

    # model checkpoint & early stopping
    if self.check_criterion(value=self.metrics[self.metric_criterion].result()):
      self.best_criterion.assign(
          tf.cast(self.metrics[self.metric_criterion].result(),
                  tf.float32))  # DUE produces float64
      self.patience_count.assign(0)
      print('Saving best model...')
      self.model.custom_save_weights(
        filepath=os.path.join(self.logging_path, 'best_model.tf'),
          tmp_name='tmp')

      # save metrics to log file for comparison
      metrics_dict = dict()
      metrics_dict['epochs'] = e
      for k, v in self.metrics.items():
        metrics_dict[k] = float(v.result())
      write_dict_to_csv(metrics_dict, os.path.join(self.logging_path, 'metrics.csv'))
    else:
      self.patience_count.assign_add(1)
      if self.patience_count.numpy() > self.patience:
        if self.reduce_lr_on_plateau_gamma == 1.0:
          return False
        else:
          self.model.optimizer.learning_rate.assign(
              self.model.optimizer.learning_rate *
              self.reduce_lr_on_plateau_gamma)
          self.patience_count.assign(0)

    return True

  def train_step(self, batch: Tuple[tf.Tensor, Any],
                 step: int) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def test_step(self, batch: Tuple[tf.Tensor, Any],
                step: int) -> Dict[str, tf.Tensor]:
    raise NotImplementedError

  def measure_training_time_per_sample(
      self, trainset: tf.data.Dataset, valset: tf.data.Dataset, epochs: int):

    times = []
    memory = []
    for e in range(self.epoch.numpy(), epochs):

      # train
      for i, batch in enumerate(trainset):
        start = time.time()
        _ = self.train_step(batch=batch, step=i)
        times += [(time.time() - start)/batch[0].numpy().shape[0]]
        memory += [tf.config.experimental.get_memory_info('GPU:0')['peak']]
        tf.config.experimental.reset_memory_stats

    return {'time': np.mean(times), 'memory': np.mean(memory)}

  def measure_inference_time_per_sample(
      self, trainset: tf.data.Dataset, valset: tf.data.Dataset, epochs: int):

    times = []
    memory = []
    for e in range(self.epoch.numpy(), epochs):

      # train
      for i, batch in enumerate(valset):
        start = time.time()
        _ = self.model.uncertainty(data=batch)
        times += [(time.time() - start)/batch[0].numpy().shape[0]]
        memory += [tf.config.experimental.get_memory_info('GPU:0')['peak']]
        tf.config.experimental.reset_memory_stats

    return {'time_inf': np.mean(times), 'memory_inf': np.mean(memory)}

  def train(self, trainset: tf.data.Dataset, valset: tf.data.Dataset,
            epochs: int, restore_checkpoint: bool = True):

    # restore checkpoint if exists
    if restore_checkpoint:
      self.restore_checkpoint(trainset=trainset)

    for e in range(self.epoch.numpy(), epochs):

      # reset metrics
      self.reset_metrics()

      # train
      for i, batch in enumerate(trainset):
        output = self.train_step(batch=batch, step=i)
        self.update_metrics(batch=batch, output_dict=output, prefix='train')

      # val
      for i, batch in enumerate(valset):
        output = self.test_step(batch=batch, step=i)
        self.update_metrics(batch=batch, output_dict=output, prefix='val')

      # write to tb and console
      self.write_tb(step=e, trainset=trainset, valset=valset)
      self.log(step=e)

      # run standard ops on epoch end
      # e.g. lr schedule, early stopping, checkpoint
      continue_training = self.on_epoch_end(e)

      if not continue_training:
        print(f'Patience reached at epoch {e + 1}!')
        break

    self.model.on_training_end(trainset, valset, logging_path=self.logging_path)


class DUMTrainer(BaseTrainer):

  def __init__(self,
               model: BaseUncertaintyModel,
               logging_path: str,
               lr_schedule: Any = None,
               reduce_lr_on_plateau_gamma: float = 1.0,
               cpkt_frequency: int = 1,
               **kwargs):
    super().__init__(
        model=model,
        logging_path=logging_path,
        lr_schedule=lr_schedule,
        reduce_lr_on_plateau_gamma=reduce_lr_on_plateau_gamma,
        cpkt_frequency=cpkt_frequency,
        **kwargs)

  def get_metrics(self) -> Dict:
    losses = self.model.get_loss_names()
    metrics_dict = {
        'train/acc': tf.keras.metrics.CategoricalAccuracy(),
        'val/acc': tf.keras.metrics.CategoricalAccuracy()
    }
    for loss in losses:
      metrics_dict[f'train/{loss}'] = tf.keras.metrics.Mean()
      metrics_dict[f'val/{loss}'] = tf.keras.metrics.Mean()
    metrics_dict[f'summary/lr'] = tf.keras.metrics.Mean()
    return metrics_dict

  def update_metrics(self,
                     batch: Tuple[tf.Tensor, tf.Tensor],
                     output_dict: dict,
                     prefix: str = 'train'):
    for loss in self.model.get_loss_names():
      self.metrics[f'{prefix}/{loss}'].update_state(output_dict[loss])
    self.metrics[f'{prefix}/acc'].update_state(
        tf.cast(batch[1], dtype=tf.int32), output_dict['prediction'])
    self.metrics[f'summary/lr'].update_state(self.model.optimizer.learning_rate)

  def train_step(self, batch: Tuple[tf.Tensor, Any],
                 step: int) -> Dict[str, tf.Tensor]:
    return self.model.train_step(data=batch, step=step)

  def test_step(self, batch: Tuple[tf.Tensor, Any],
                step: int) -> Dict[str, tf.Tensor]:
    return self.model.test_step(data=batch)


class SegmentationTrainer(DUMTrainer):

  def __init__(self,
               model: BaseUncertaintyModel,
               logging_path: str,
               nr_classes: int,
               patience: int = 20,
               lr_schedule: Any = None,
               orig_trainset=None,
               orig_valset=None,
               **kwargs):
    self.nr_classes = nr_classes
    super().__init__(model=model, logging_path=logging_path,
                     patience=patience, lr_schedule=lr_schedule, **kwargs)
    self.orig_trainset = orig_trainset
    self.orig_valset = orig_valset

  def get_metrics(self):
    losses = self.model.get_loss_names()
    metrics_dict = {
        'train/acc': tf.keras.metrics.CategoricalAccuracy(),
        'val/acc': tf.keras.metrics.CategoricalAccuracy(),
        'train/iou': tf.keras.metrics.MeanIoU(num_classes=self.nr_classes),
        'val/iou': tf.keras.metrics.MeanIoU(num_classes=self.nr_classes),
    }
    for loss in losses:
      metrics_dict[f'train/{loss}'] = tf.keras.metrics.Mean()
      metrics_dict[f'val/{loss}'] = tf.keras.metrics.Mean()
    return metrics_dict

  def update_metrics(self,
                     batch: Tuple[tf.Tensor, tf.Tensor],
                     output_dict: dict,
                     prefix: str = 'train'):
    for loss in self.model.get_loss_names():
      self.metrics[f'{prefix}/{loss}'].update_state(output_dict[loss])
    self.metrics[f'{prefix}/acc'].update_state(
        tf.cast(batch[1], dtype=tf.int32), output_dict['prediction'])
    self.metrics[f'{prefix}/iou'].update_state(
        tf.argmax(tf.cast(batch[1], dtype=tf.int32), axis=-1),
        tf.argmax(output_dict['prediction'], axis=-1))

  def reset_metrics(self):
    for key in self.metrics:
      self.metrics[key].reset_states()

  def get_imgs(self, dataset: tf.data.Dataset, step: int):
    _, val_batch = enumerate(dataset).__next__()
    preds = self.model(val_batch[0], training=False)['prediction']
    if isinstance(preds, (list, tuple)):
      preds, _ = preds
    preds_np = preds.numpy()
    lbls = []
    preds = []
    lbls_np = val_batch[1].numpy()
    for i in range(lbls_np.shape[0]):
      lbls.append(tf.expand_dims(to_colored_label(
        np.argmax(lbls_np[i], axis=-1), num_classes=self.model.nr_classes), axis=0))
      preds.append(tf.expand_dims(to_colored_label(
        np.argmax(preds_np[i], axis=-1), num_classes=self.model.nr_classes), axis=0))
    lbls = np.concatenate(lbls, axis=0)
    preds = np.concatenate(preds, axis=0)
    return val_batch[0], lbls, preds

  def write_tb(self, step: int, trainset: tf.data.Dataset, valset: tf.data.Dataset):
    with self.tb_writer.as_default():
      # for key in self.model.metrics_dict.keys():
      #   tf.summary.scalar(key, self.model.metrics_dict[key].result(), step=step)
      for key in self.metrics.keys():
        tf.summary.scalar(key, self.metrics[key].result(), step=step)

      val_img, val_lbl, val_pred = self.get_imgs(dataset=self.orig_valset, step=step)
      train_img, train_lbl, train_pred = self.get_imgs(dataset=self.orig_trainset, step=step)
      tf.summary.image('val/img', val_img, step=step, max_outputs=25)
      tf.summary.image('val/lbl', val_lbl, step=step, max_outputs=25)
      tf.summary.image('val/pred', val_pred, step=step, max_outputs=25)
      tf.summary.image('train/img', train_img, step=step, max_outputs=25)
      tf.summary.image('train/lbl', train_lbl, step=step, max_outputs=25)
      tf.summary.image('train/pred', train_pred, step=step, max_outputs=25)
