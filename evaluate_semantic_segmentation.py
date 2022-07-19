# Lint as: python3
"""Evaluation script."""

import os
import yaml
from absl import app
from absl import flags
import tensorflow as tf
import gc

from data_lib.get_segmentation_dataset import get_segmentation_dataset
from evaluation_lib import evaluate_utils
from model_lib.baselines_seg import EnsembleSeg
from training_lib.utils import load_model_segmentation
from data_lib.get_segmentation_dataset import get_cityscapes_testset
from data_lib.get_segmentation_dataset import load_cityscapes_testset_np


# Define Flags.
flags.DEFINE_list(  # exp_name is list for compatibility of ensembles
    'exp_name', '', 'Name of the experiment to be evaluated.')
flags.DEFINE_string(
    'exp_root', '', 'Root directory of experiments.')
flags.DEFINE_bool('mcd', False, 'Use MC dropout for uncertainty estimation.')
flags.DEFINE_bool('mir', False, 'Use MIR for uncertainty estimation.')
flags.DEFINE_integer('nr_samples', 10, 'Number of samples for MC dropout.')
flags.DEFINE_bool('debug', False, 'Debug mode. Eager execution.')

FLAGS = flags.FLAGS


def main(_):

  print('Evaluating model...', flush=True)

  tf.config.run_functions_eagerly(FLAGS.debug)

  # get experiment folder and create dir for plots
  exp_folders = [os.path.join(FLAGS.exp_root, name) for name in FLAGS.exp_name]
  results_folder = os.path.join(exp_folders[0], 'results')
  tf.io.gfile.mkdir(results_folder)

  # get experiment FLAGS
  TRAINING_FLAGS = yaml.load(
      tf.io.gfile.GFile(os.path.join(exp_folders[0], 'FLAGS.yml'), 'r'),
      Loader=yaml.Loader)
  if 'ensemble' not in TRAINING_FLAGS.keys():
    TRAINING_FLAGS['ensemble'] = False

    # get in-distribution datasets
    print('Loading dataset...', flush=True)
  TRAINING_FLAGS['batch_size'] = 1  # evaluate with batch_size 1

  # get dataset
  data_root = TRAINING_FLAGS['data_root']
  trainset, valset, testset = get_segmentation_dataset(
      TRAINING_FLAGS['dataset'],
      data_root=data_root,
      batch_size=TRAINING_FLAGS['batch_size'])
  if TRAINING_FLAGS['dataset'] == 'cityscapes':
    print('Loading numpy testset ...', flush=True)
    testset_np = load_cityscapes_testset_np(
        data_root=data_root, testset=testset)
    print('Finished loading numpy testset ...', flush=True)

  if FLAGS.mcd:
    TRAINING_FLAGS['method'] = 'dropout'
  if FLAGS.mir:
    TRAINING_FLAGS['method'] = 'mir'
    TRAINING_FLAGS['reconstruction_weight'] = 0  # does not matter during inference

  # create model and load weights
  if len(exp_folders) == 1:
    model, _ = load_model_segmentation(
        method=TRAINING_FLAGS['method'],
        hyperparameters=TRAINING_FLAGS,
        exp_folder=exp_folders[0],
        trainset=trainset,
        valset=valset,
        load_weights=True,
        evaluation=True)
  elif len(exp_folders) > 1:  # more than 1 exp_name --> ensemble method
    TRAINING_FLAGS['ensemble_size'] = len(exp_folders)
    model = EnsembleSeg(**TRAINING_FLAGS)

    # call model once for init
    _, init_input = enumerate(testset).__next__()
    _ = model(init_input[0])

    try:
      model.custom_load_weights(filepath=[os.path.join(f, 'best_model.h5') for f in exp_folders])
    except:
      model.custom_load_weights(filepath=[os.path.join(f, 'best_model.tf') for f in exp_folders])
    model.compile()
  else:
    raise ValueError('Unknown method!')

  #########################################
  # compute metrics for calibration of
  # epistemic uncertainty under distributional shift
  #########################################
  print('Evaluate Calibration')
  # if TRAINING_FLAGS['dataset'] == 'cityscapes':
  #   test_predictions = evaluate_utils.predict_uncertainty(
  #     model=model, data_loader=valset)
  #   predictions[0] = evaluate_utils.predict_uncertainty(
  #     model=model, data_loader=testset)
  for perturbation in evaluate_utils.PERTURBATIONS[TRAINING_FLAGS['dataset']]:
    results_perturbation_folder = os.path.join(results_folder,
                                               f'perturbation_{perturbation}')
    tf.io.gfile.mkdir(results_perturbation_folder)

    perturbation_range = evaluate_utils.PERTURBATION_RANGES[perturbation]
    predictions = dict()
    if TRAINING_FLAGS['dataset'] == 'cityscapes':
      # predictions[0] = test_predictions
      predictions[0] = evaluate_utils.predict_uncertainty(
        model=model, data_loader=testset)
    for p in perturbation_range:

      print(f'evaluating {perturbation} = {p}', flush=True)

      # get OOD data
      if TRAINING_FLAGS['dataset'] == 'cityscapes':
        oodset = get_cityscapes_testset(
            testset_np[0], testset_np[1],
            batch_size=TRAINING_FLAGS['batch_size'],
            corruption=perturbation, severity=p) # perturbation
      else:
        raise ValueError(f'Unknown dataset {TRAINING_FLAGS["dataset"]}!')

      # make predictions
      predictions[p] = evaluate_utils.predict_uncertainty(
          model=model, data_loader=oodset)

      del oodset
      gc.collect()

    metrics = evaluate_utils.compute_calibration_metrics(
        predictions=predictions, segmentation=True,
        dataset=TRAINING_FLAGS["dataset"])
    evaluate_utils.postprocess_calibration_metrics(
        calibration_metrics=metrics, results_path=results_perturbation_folder)
  #########################################

  print('Evaluation done.')


if __name__ == '__main__':
  app.run(main)
