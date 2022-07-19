# Lint as: python3
"""Training script."""

import os
import yaml
import datetime
from absl import app
from absl import flags
import tensorflow as tf

from data_lib.get_image_dataset import get_image_dataset
from data_lib.get_image_dataset import DATASET_NUM_CLASSES
from training_lib.trainer import DUMTrainer
from training_lib.utils import load_model
import evaluation_lib.evaluate_utils as evaluate_utils
from configs.configs import get_cfg

# Define Flags.
flags.DEFINE_string(
    'data_root',
    '',
    'Root directory of data.')
flags.DEFINE_string('exp_root',
                    '',
                    'Root directory of experiments.')
flags.DEFINE_string('exp_folder', '',
                    'When given overrides exp_root. Used with XManager.')
flags.DEFINE_enum(
    'dataset', 'cifar10',
    ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'svhn_cropped'],
    'Dataset used during training.')
flags.DEFINE_enum('backbone', 'simple_fc', ['simple_fc', 'resnet'],
                  'Backbone used during training.')
flags.DEFINE_enum('method', 'softmax',
                  ['softmax', 'dropout', 'duq', 'ddu', 'mir', 'due', 'sngp'],
                  'Uncertainty method used during training.')
flags.DEFINE_boolean(
    'measure_training_time', False,
    'Whether to measure training time. Model will not be trained.')

# Optimization
flags.DEFINE_integer(
    'batch_size', 128, 'Batch size used during training.', lower_bound=1)
flags.DEFINE_integer(
    'feature_dims',
    100,
    'Dimensionality of the output of the backbone network.',
    lower_bound=1)
flags.DEFINE_integer('epochs', 200, 'Maximum number of epochs.', lower_bound=1)
flags.DEFINE_integer(
    'patience', 10, 'Patience for early stopping.', lower_bound=1)
flags.DEFINE_list('override_args', [],
                  'Args in config.py that should be overridden')
flags.DEFINE_boolean(
    'use_optimal_params', True,
    'Use optimization parameters provided in configs/<method>.yml.'
    'Note: params in configs/<method>.yml should not included '
    'hyperparameters specific to <method>.')
flags.DEFINE_float(
    'lr', 0.01, 'Learning rate used during training.', lower_bound=0.0)
flags.DEFINE_float('l2_reg', 1e-4, 'L2 weight regularization.', lower_bound=0.0)
flags.DEFINE_float(
    'dropout', 0.05, 'Dropout probability.', lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_integer(
    'cpkt_frequency', 1, 'Save checkpoint every X epochs.', lower_bound=1)

# Backbone
flags.DEFINE_integer(
    'hidden_dims',
    100,
    'Hidden dimension of fully-connected neural network.',
    lower_bound=1)
flags.DEFINE_integer(
    'hidden_layer',
    3,
    'Number of hidden layer of ' + 'fully-connected neural network.',
    lower_bound=1)
flags.DEFINE_integer('resnet_depth', 50, 'Depth of ResNet (8 => ResNet18).')
flags.DEFINE_integer('nr_classes', 10, 'Number of classes.', lower_bound=1)
flags.DEFINE_boolean('bn', True, 'Whether to use batchnormalization layers.')

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
                  ['RBF', 'matern12', 'matern32', 'matern52'],
                  'Kernel choice for GP.')

### DUQ PARAMS ###
flags.DEFINE_float(
    'length_scale', 0.1, 'Length scale for DUQ.', lower_bound=0.0)
flags.DEFINE_float(
    'l_gradient_penalty',
    0.0,
    'Weight of input gradient penalty.',
    lower_bound=0.0)

### DDU PARAMS ###
flags.DEFINE_enum(
    'fitting_method', 'sample', ['sample', 'sgd'], 'Method for fitting the GMM.'
    'Options: "sample" - Using the sample mean and variance as the GMM param.'
    '"sgd" - Optmizing the GMM param using SGD.')
flags.DEFINE_boolean(
    'use_covariance', True,
    'Whether to use full covariance matrix for each multivariate Gaussian distribution.'
    'If `False`, it will use the diagonal covariance matrix.')
flags.DEFINE_integer(
    'gmm_epochs',
    100,
    'Epochs for fitting the GMM. (Only avaiable when `fitting_method`="sgd")',
    lower_bound=0)
flags.DEFINE_integer(
    'num_components',
    10,
    'Number of components in the GMM. (Only avaiable when `fitting_method`="sgd")',
    lower_bound=1)
flags.DEFINE_integer(
    'num_batches_fitting',
    64,
    'Number of batches used for fitting.',
    lower_bound=1)

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
    1.0,
    'Weight of reconstruction regularization.',
    lower_bound=0)

flags.DEFINE_bool('debug', False, 'Debug mode. Eager execution.')

FLAGS = flags.FLAGS


def main(_):

  # Run eagerly in debug mode
  tf.config.run_functions_eagerly(FLAGS.debug)

  # Resnet for mnist and fashion_mnist not implemented
  if FLAGS.backbone == 'resnet':
    assert FLAGS.dataset in ['cifar10', 'cifar100', 'svhn_cropped']

  # get experiment folder
  if FLAGS.exp_folder == '':
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_folder = os.path.join(FLAGS.exp_root, current_time)
  else:
    exp_folder = os.path.join(FLAGS.exp_root, FLAGS.exp_folder)

  # get dataset
  trainset, valset, _ = get_image_dataset(FLAGS.dataset, FLAGS.data_root)

  # hyperparameters
  hyperparameters = FLAGS.flag_values_dict()
  if FLAGS.backbone == 'resnet':
      hyperparameters['feature_dims'] = 64
  hyperparameters_opt = dict()
  if FLAGS.use_optimal_params:
    hyperparameters_opt = get_cfg(method=FLAGS.method, backbone=FLAGS.backbone)
  for key in hyperparameters_opt:
    if key not in FLAGS.override_args:
      hyperparameters[key] = hyperparameters_opt[key]
  hyperparameters['nr_classes'] = DATASET_NUM_CLASSES[FLAGS.dataset]
  hyperparameters.update({'exp_folder': exp_folder})

  # save Hyperparameters to yml
  tf.io.gfile.makedirs(exp_folder)
  yaml.dump(
      hyperparameters,
      tf.io.gfile.GFile(os.path.join(exp_folder, 'FLAGS.yml'), 'w'),
  )

  # init model, optimizer and trainer
  load_weights = False
  model, optimizer, lr_schedule = load_model(
      method=FLAGS.method,
      hyperparameters=hyperparameters,
      exp_folder=exp_folder,
      trainset=trainset,
      valset=valset,
      load_weights=load_weights)
  trainer = DUMTrainer(
      model=model,
      logging_path=exp_folder,
      optimizer=optimizer,
      lr_schedule=lr_schedule,
      load_state=True,
      **hyperparameters)

  # train
  trainset = trainset.batch(hyperparameters['batch_size'])
  valset = valset.batch(hyperparameters['batch_size'])
  if not FLAGS.measure_training_time:
    trainer.train(
        trainset=trainset,
        valset=valset,
        epochs=hyperparameters['epochs'],
        restore_checkpoint=True)
  else:  # measure training time
    result = trainer.measure_training_time_per_sample(
        trainset=trainset,
        valset=valset,
        epochs=2)
    result.update(trainer.measure_inference_time_per_sample(
        trainset=trainset,
        valset=valset,
        epochs=2))
    evaluate_utils.write_dict_to_csv(
        results_dict=result,
        file_name=os.path.join(exp_folder, 'time_memory.csv'))


if __name__ == '__main__':
  app.run(main)
