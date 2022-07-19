# Lint as: python3
"""Evaluation script."""

import os
import yaml
from absl import app
from absl import flags
import time
import tensorflow as tf

from data_lib.get_image_dataset import get_image_dataset
from data_lib.get_corrupted_image_dataset import CORRUPTED_DATASETS
from data_lib.get_image_dataset import get_ood_datasets
import evaluation_lib.evaluate_utils as evaluate_utils
from model_lib.baselines import Ensemble
from training_lib.utils import load_model

# Define Flags.
flags.DEFINE_list(  # exp_name is list for compatibility of ensembles
  'exp_name',
  '',
  'Name of the experiment to be evaluated.')
flags.DEFINE_string(
  'exp_root',
  '',
  'Root directory of experiments.')

# configs
flags.DEFINE_integer('nr_samples', 15, 'Number of samples for MC dropout.')
flags.DEFINE_boolean('evaluate_ood_detection', True,
                     'Whether to evaluate performance on OOD detection.')
flags.DEFINE_boolean('evaluate_calibration', True,
                     'Whether to evaluate calibration of epistemic '
                     'uncertainty on continuous distributional shifts.')
flags.DEFINE_boolean('use_corrupted', True,
                     'Whether to evaluate calibration of epistemic '
                     'uncertainty on continuous distributional shifts using '
                     'of the shelf corrupted datasets.')
flags.DEFINE_boolean('normalize_features', True,
                     'Whether to normalize features before fitting a density '
                     'model to the hidden representations of NN.')

flags.DEFINE_bool('debug',
                  False,
                  'Debug mode. Eager execution.')

FLAGS = flags.FLAGS


def main(_):
  tf.config.run_functions_eagerly(FLAGS.debug)

  # get experiment folder and create dir for plots
  exp_folders = [os.path.join(FLAGS.exp_root, name) for name in FLAGS.exp_name]
  print(exp_folders, flush=True)
  results_folder = os.path.join(exp_folders[0], 'results')
  tf.io.gfile.makedirs(results_folder)

  # get experiment FLAGS
  TRAINING_FLAGS = yaml.load(
    tf.io.gfile.GFile(os.path.join(exp_folders[0], 'FLAGS.yml'), 'r'),
    Loader=yaml.Loader)
  if 'ensemble' not in TRAINING_FLAGS.keys():
    TRAINING_FLAGS['ensemble'] = False

  # get in-distribution datasets
  trainset, valset, testset = get_image_dataset(TRAINING_FLAGS['dataset'],
                                                TRAINING_FLAGS['data_root'])

  # create model and load weights
  if len(exp_folders) == 1:
    model, _, _ = load_model(
      method=TRAINING_FLAGS['method'],
      hyperparameters=TRAINING_FLAGS,
      exp_folder=exp_folders[0],
      trainset=trainset,
      valset=valset,
      load_weights=True,
      evaluation=True,
      normalize_features=FLAGS.normalize_features)
  elif len(exp_folders) > 1:  # more than 1 exp_name --> ensemble method
    TRAINING_FLAGS['ensemble_size'] = len(exp_folders)
    model = Ensemble(**TRAINING_FLAGS)

    # call model once for init
    _, init_input = enumerate(testset.batch(10)).__next__()
    _ = model(init_input[0])

    model.custom_load_weights(filepath=[os.path.join(f, 'best_model.h5') for f in exp_folders])
    model.compile()
  else:
    raise ValueError('Unknown method!')

  #########################################
  # compute metrics for distinguishing
  # between OOD data and in-distribution data
  #########################################
  if FLAGS.evaluate_ood_detection:
    print('Evaluate OOD')
    results_OOD_folder = os.path.join(results_folder, 'ood_detection')
    tf.io.gfile.mkdir(results_OOD_folder)

    # get OOD data
    ood_datasets_dict = get_ood_datasets(
      dataset=TRAINING_FLAGS['dataset'],
      data_root=TRAINING_FLAGS['data_root'])

    metrics = dict()
    meta_info = dict()
    meta_info['TEST_ACC'] = float(evaluate_utils.get_accuracy(
      model=model, data_loader=testset.batch(TRAINING_FLAGS['batch_size'])))
    for ood_dataset in ood_datasets_dict.keys():
      metrics[ood_dataset] = evaluate_utils.compute_ood_detection_metrics(
        model=model,
        id_loader=testset.batch(TRAINING_FLAGS['batch_size']),
        ood_loader=ood_datasets_dict[ood_dataset].batch(
          TRAINING_FLAGS['batch_size']))

    yaml.dump(
      meta_info,
      tf.io.gfile.GFile(os.path.join(
        results_OOD_folder, 'meta.yml'), 'w'),
    )
    evaluate_utils.write_csv_ood_detection(
      results_dict=metrics,
      file_name=os.path.join(results_OOD_folder, 'metrics.csv'))
  #########################################

  #########################################
  # compute metrics for calibration of
  # epistemic uncertainty under distributional shift
  #########################################
  if FLAGS.evaluate_calibration:

    if FLAGS.use_corrupted:
      perturbations, perturbation_range, dataset_getter = CORRUPTED_DATASETS[
        TRAINING_FLAGS['dataset']]
    else:
      perturbations = evaluate_utils.PERTURBATIONS[TRAINING_FLAGS['dataset']]

    print('Evaluate Calibration')
    for perturbation in perturbations:
      results_perturbation_folder = os.path.join(
        results_folder, f'perturbation_{perturbation}')
      tf.io.gfile.mkdir(results_perturbation_folder)

      if not FLAGS.use_corrupted:
        perturbation_range = evaluate_utils.PERTURBATION_RANGES[perturbation]
      predictions = dict()
      if FLAGS.use_corrupted:
        predictions[0] = evaluate_utils.predict_uncertainty(
          model=model, data_loader=testset.batch(TRAINING_FLAGS['batch_size']))
      for p in perturbation_range:

        # get OOD data
        if FLAGS.use_corrupted:
          oodset = dataset_getter(TRAINING_FLAGS['data_root'], perturbation, p)
        else:
          _, _, oodset = get_image_dataset(
            TRAINING_FLAGS['dataset'],
            TRAINING_FLAGS['data_root'],
            perturbation_type=perturbation,
            perturbation=p)

        # make predictions
        predictions[p] = evaluate_utils.predict_uncertainty(
          model=model, data_loader=oodset.batch(TRAINING_FLAGS['batch_size']))

      metrics = evaluate_utils.compute_calibration_metrics(
        predictions=predictions)
      evaluate_utils.postprocess_calibration_metrics(
        calibration_metrics=metrics, results_path=results_perturbation_folder)
  #########################################

  print("Evaluation done.")


if __name__ == '__main__':
  app.run(main)
