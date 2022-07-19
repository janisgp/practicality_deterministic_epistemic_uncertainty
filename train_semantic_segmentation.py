"""Training script."""

import os
import yaml
from absl import app
from absl import flags
import datetime
import tensorflow as tf

from data_lib.get_segmentation_dataset import get_segmentation_dataset
from training_lib.trainer import SegmentationTrainer
from training_lib.utils import load_model_segmentation
from configs.configs_seg import get_cfg

# Define Flags.
flags.DEFINE_string('data_root',
                    '',
                    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    '',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_folder', '',
                    'When given overrides exp_root.')
flags.DEFINE_enum('dataset', 'cityscapes', ['cityscapes'],
                  'Dataset used during training.')
flags.DEFINE_enum('backbone', 'drn_a_50', [
    'drn_a_50', 'drn_c_26', 'drn_c_42', 'drn_c_58', 'drn_d_22', 'drn_d_24',
    'drn_d_38', 'drn_d_40', 'drn_d_54', 'drn_d_56', 'drn_d_105', 'drn_d_107'
], 'Backbone used during training.')
flags.DEFINE_enum('method', 'softmax',
                  ['softmax', 'dropout', 'sngp', 'mir', 'ddu'],
                  'Uncertainty method used during training.')
flags.DEFINE_enum('uncertainty_agg', 'mean', ['mean'],
                  'Uncertainty aggregation method used for dense prediction.')

# Optimization
flags.DEFINE_integer(
    'batch_size', 1, 'Batch size used during training.', lower_bound=1)
flags.DEFINE_integer('epochs', 200, 'Maximum number of epochs.', lower_bound=1)
flags.DEFINE_integer(
    'patience', 200, 'Patience for early stopping.', lower_bound=1)

# Backbone
flags.DEFINE_integer('nr_classes', 20, 'Number of classes.', lower_bound=1)
flags.DEFINE_boolean('bn', True, 'Whether to use batchnormalization layers.')
flags.DEFINE_boolean(
    'smoothing_conv', False,
    'Whether to use final smoothing convolution after upsampling.')

# Optimization
flags.DEFINE_boolean(
    'use_optimal_params', True,
    'Use optimization parameters provided in configs/<method>.yml.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_enum('loss', 'ce', ['ce'],
                  'Loss used during training.')
flags.DEFINE_float(
    'lr', 0.03, 'Learning rate used during training.', lower_bound=0.0)
flags.DEFINE_float('l2_reg', 1e-4, 'L2 weight regularization.', lower_bound=0.0)
flags.DEFINE_float(
    'dropout', 0.1, 'Dropout probability.', lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float(
    'cropping_factor',
    1.0,
    'Factor used for cropping',
    lower_bound=0.0,
    upper_bound=1.0)
flags.DEFINE_enum(
    'metric_criterion', 'val/acc', ['val/acc', 'val/loss'],
    'Main metric tracked during Training',)

# Spectral normalization
flags.DEFINE_boolean(
    'spectral_normalization', False,
    'Whether to use spectral normalization on model weights.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_boolean(
    'spectral_batchnormalization', False,
    'Whether to use spectral normalization on batchnorm scale.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_boolean(
    'soft_spectral_normalization', False,
    'Whether to relax the constraint imposed by spectral normalization on model weights.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_float(
    'coeff', 3,
    'coefficient to which the Lipschitz constant must be restricted in soft spectral normalization.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_integer(
    'power_iterations', 1,
    'the number of iterations during spectral normalization.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')

### GP PARAMS ###
flags.DEFINE_integer(
    'num_inducing_points', None,
    'The hidden dimension of the GP layer, which corresponds '
    'to the number of random features used for the approximation.')
flags.DEFINE_bool('ard', False,
                  'Whether to use automatic relevance detection for GP.')
flags.DEFINE_enum('kernel_name', 'RBF',
                  ['RBF'],
                  'Kernel choice for GP.')
flags.DEFINE_boolean(
    'share_gp', False,
    'Whether to share same gp kernel for all elements in output'
    ' feature map.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')

### SNGP PARAMS ###
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_input_dim', 128,
    'The dimension to reduce the neural network input for the GP layer '
    '(via random Gaussian projection which preserves distance by the '
    ' Johnson-Lindenstrauss lemma). If -1, no dimension reduction.')
flags.DEFINE_bool(
    'gp_input_normalization', True,
    'Whether to normalize the input using LayerNorm for GP layer.'
    'This is similar to automatic relevance determination (ARD) in the classic '
    'GP learning.')
flags.DEFINE_string(
    'gp_random_feature_type', 'orf',
    'The type of random feature to use. One of "rff" (random fourier feature), '
    '"orf" (orthogonal random feature).')
flags.DEFINE_float('gp_cov_ridge_penalty', 1.,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', -1.,
    'The discount factor to compute the moving average of precision matrix'
    'across epochs. If -1 then compute the exact precision matrix within the '
    'latest epoch.')
flags.DEFINE_float(
    'gp_mean_field_factor', 25.,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')

### MIR PARAMS ###
flags.DEFINE_float(
    'reconstruction_weight',
    10.0,
    'Weight of reconstruction regularization.',
    lower_bound=0)

flags.DEFINE_bool('debug', False, 'Debug mode. Eager execution.')

FLAGS = flags.FLAGS


def main(_):

  tf.config.run_functions_eagerly(FLAGS.debug)

  # get experiment folder
  if FLAGS.exp_folder == '':
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_folder = os.path.join(FLAGS.exp_root, current_time)
  else:
    exp_folder = os.path.join(FLAGS.exp_root, FLAGS.exp_folder)

  # get dataset
  trainset, valset, t = get_segmentation_dataset(
      FLAGS.dataset,
      FLAGS.data_root,
      cropping_factor=FLAGS.cropping_factor,
      batch_size=FLAGS.batch_size)

  # hyperparameters
  hyperparameters = FLAGS.flag_values_dict()
  hyperparameters_opt = dict()
  if FLAGS.use_optimal_params:
    hyperparameters_opt = get_cfg(method=FLAGS.method, backbone=FLAGS.backbone)
  hyperparameters.update(hyperparameters_opt)
  hyperparameters.update({'exp_folder': exp_folder})

  # save Hyperparameters to yml
  tf.io.gfile.makedirs(exp_folder)
  yaml.dump(
      hyperparameters,
      tf.io.gfile.GFile(os.path.join(exp_folder, 'FLAGS.yml'), 'w'),
  )

  # init model, optimizer and trainer
  load_weights = tf.io.gfile.exists(os.path.join(exp_folder, 'model.h5'))
  print(f'Initializing model with load_weights={load_weights}...')
  model, lr_schedule = load_model_segmentation(
      method=FLAGS.method,
      hyperparameters=hyperparameters,
      exp_folder=exp_folder,
      trainset=trainset,
      valset=valset,
      load_weights=tf.io.gfile.exists(os.path.join(exp_folder, 'model.h5')))
  trainer = SegmentationTrainer(
      model=model,
      logging_path=exp_folder,
      lr_schedule=lr_schedule,
      load_state=True,
      orig_trainset=trainset,
      orig_valset=valset,
      **hyperparameters)

  # train
  trainer.train(
      trainset=trainset,
      valset=valset,
      epochs=hyperparameters['epochs'],
      restore_checkpoint=True)


if __name__ == '__main__':
  app.run(main)
