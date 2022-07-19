# Lint as: python3
"""Functions used for evaluation."""

import os
import csv
import pickle
from typing import List, Text, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
import scipy

pearsonr = scipy.stats.pearsonr
spearmanr = scipy.stats.spearmanr

from model_lib.baselines import BaseModel
from data_lib.segmentation_labels import to_colored_label
from data_lib.corruptions.apply_corruptions import BENCHMARK_CORRUPTIONS
from data_lib.get_segmentation_dataset import IOU_MASKS

# determines which perturbation is run on which dataset
PERTURBATIONS = {
  'mnist': ['rotation'],
  'fashion_mnist': ['rotation'],
  'cifar10': ['additive_gaussian', 'jpeg_quality'],
  'svhn_cropped': ['additive_gaussian', 'jpeg_quality'],
  'cityscapes': BENCHMARK_CORRUPTIONS
}

# fixed perturbation ranges
PERTURBATION_RANGES = {
  'rotation':
    np.arange(0, 200, 20),
  'additive_gaussian':
    np.arange(0.0, 0.3, 0.05),
  'jpeg_quality':
    np.arange(0.0, 1.0, 0.1),
  'brightness':
    np.arange(0.0, 1.0, 0.1),
  'time_of_day':
    np.concatenate([np.arange(90, 10, -5),
                    np.arange(10, -6,
                              -1)]),  # 'time_of_day': np.array([90])
  'rain_strength':
    np.arange(1, 5, 1)
}
PERTURBATION_RANGES.update({c: [1, 2, 3, 4, 5] for c in BENCHMARK_CORRUPTIONS})


def get_accuracy(model: BaseModel, data_loader: tf.data.Dataset) -> float:
  accuracies = []
  for i, batch in enumerate(data_loader):
    x = batch[0]
    out = model(x)
    targets = batch[-1].numpy().argmax(-1)
    if isinstance(out['prediction'], (list, tuple)):
      predictions = out['prediction'][0].numpy().argmax(-1)
    else:
      predictions = out['prediction'].numpy().argmax(-1)
    accuracies += [(targets == predictions).mean()]
  return np.mean(accuracies)


def predict_uncertainty(model: BaseModel,
                        data_loader: tf.data.Dataset,
                        max_i: int = -1,
                        return_input: bool = False) -> Dict:
  """Accumulates all predictions and uncertainties from a data loader

  Args:
    model: BaseModel. implements method uncertainty()
    data_loader: tf.data.Dataset. Iterable were element -1 corresponds to the
      label

  Returns:
    output_dict: dict.
     entries np.arrays 'prediction', 'target', 'uncertainty'
  """

  output_dict = dict()
  for i, batch in enumerate(data_loader):

    if max_i != -1 and i > max_i:
      break

    if len(batch[0].shape) == 3:
      imgs = tf.expand_dims(batch[0], 0)
      lbls = tf.expand_dims(batch[1], 0)
      inputs = (imgs, lbls)
    else:
      inputs = batch

    out = model.uncertainty(inputs)
    if i == 2:
      break

    if i == 0:
      for key in out:
        if isinstance(out[key], tf.Tensor):
          output_dict[key] = [out[key].numpy()]
        else:
          output_dict[key] = [out[key]]
      output_dict['target'] = [inputs[-1].numpy()]
      if return_input:
        output_dict['x'] = [inputs[0].numpy()]
    else:
      for key in out:
        if isinstance(out[key], tf.Tensor):
          output_dict[key] += [out[key].numpy()]
        else:
          output_dict[key] += [out[key]]
      output_dict['target'] += [inputs[-1].numpy()]
      if return_input:
        output_dict['x'] += [inputs[0].numpy()]

  for key in output_dict:
    output_dict[key] = np.concatenate(output_dict[key], axis=0)
  return output_dict


def write_csv_ood_detection(results_dict: Dict[str, Dict[str, float]],
                            file_name: str):
  """Writes the results for OOD detection to csv file.

  Args:
    results_dict: dict. one entry for each ood dataset with corresponding
      metrics
    file_name: str. csv file path
  """
  oodsets = list(results_dict.keys())
  csv_columns = ['Name'] + list(results_dict[oodsets[0]].keys())

  with tf.io.gfile.GFile(file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for ood in oodsets:
      entry = {'Name': ood}
      for key in results_dict[ood]:
        entry[key] = results_dict[ood][key]
      writer.writerow(entry)


def write_dict_to_csv(results_dict: Dict, file_name: str):
  """Writes the results for dictionary to csv file.

  Args:
    results_dict: dict.
    file_name: str. csv file path
  """
  csv_columns = list(results_dict.keys())
  with tf.io.gfile.GFile(file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    if isinstance(results_dict[csv_columns[0]], list):
      for i in range(len(results_dict[csv_columns[0]])):
        tmp_dict = {k: results_dict[k][i] for k in csv_columns}
        writer.writerow(tmp_dict)
    else:
      writer.writerow(results_dict)


def load_csv_to_dict(file_path: str) -> Dict:
  with tf.io.gfile.GFile(file_path, 'r') as infile:
    reader = csv.reader(infile)
    keys = reader.__next__()
    mydict = {k: [] for k in keys}
    for _, vals in enumerate(reader):
      for i in range(len(keys)):
        mydict[keys[i]] += [vals[i]]
  return mydict


def compute_auroc_ap_aupr(labels: Any,
                          scores: Any) -> Tuple[float, float, float]:
  """Computes the metrics AUROC, AUPR and AP.

  Args:
    labels: np.array. 1: in-distribution, 0: ood
    scores: np.array. Uncertainty values
  """
  AUROC = sklearn.metrics.roc_auc_score(labels, -1 * scores)
  AP = sklearn.metrics.average_precision_score(labels, -1 * scores)
  p, r, _ = sklearn.metrics.precision_recall_curve(labels, -1 * scores)
  AUPR = sklearn.metrics.auc(r, p)
  return AUROC, AP, AUPR


def compute_auroc(labels: Any, scores: Any) -> float:
  """Computes the metrics AUROC, AUPR and AP.

  Args:
    labels: np.array. 1: in-distribution, 0: ood
    scores: np.array. Uncertainty values
  """
  AUROC = sklearn.metrics.roc_auc_score(labels, -1 * scores)
  return AUROC


def compute_ood_detection_metrics(
        model: BaseModel, id_loader: tf.data.Dataset,
        ood_loader: tf.data.Dataset) -> Dict[str, float]:
  """Computes metrics for OOD detection.

  Args:
    model: BaseModel implementing uncertainty(
    id_loader: in-distribution dataset
    ood_loader: OOD dataset

  Returns:
    metrics_dict: dictionary containing entries AUROC, AUPR, AP
  """

  # make predictions
  id_dict = predict_uncertainty(model=model, data_loader=id_loader)
  ood_dict = predict_uncertainty(model=model, data_loader=ood_loader)

  metrics_dict = dict()
  # Compute AUROC, AP & AUPRC
  merged = np.concatenate([id_dict['uncertainty'], ood_dict['uncertainty']], 0)
  merged_labels = np.zeros(merged.shape[0])
  merged_labels[:id_dict['uncertainty'].shape[0]] = 1
  metrics_dict['AUROC'], metrics_dict['AP'], metrics_dict[
    'AUPR'] = compute_auroc_ap_aupr(merged_labels, merged)

  return metrics_dict


def get_calibration(correct: Any,
                    uncertainty: Any,
                    n_neighbour: int = 100,
                    n_sample: int = 1000) -> Tuple[List[Any], List[Any]]:
  """Computes a calibration plot (Accuracy vs. Uncertainty). First sorts arrays according to uncertainty magnitude.

  Then samples n_sample-times and uncertainty values and n_neighbour
  uncertainty values close to it and computes the corresponding accuracy.
  Args:
    correct: np.array indictating whether prediction on a point is correct or
      not
    uncertainty: np.array containing uncertainty estimates
    n_neighbour: int. number of nearby uncertainty values for binning
    n_sample: int. number of bins to generate

  Returns:
    unc: list of average uncertainty values in each bin
    acc: list of average accuracies of each bin
  """
  sorted_idxs = np.argsort(uncertainty)
  c = correct[sorted_idxs]
  u = uncertainty[sorted_idxs]

  unc, acc = [], []
  for _ in range(n_sample):

    # random sample
    idx = np.random.choice(range(c.shape[0]))

    if idx - int(n_neighbour / 2) < 0:
      idx1 = 0
      idx2 = 2 * int(n_neighbour / 2)
    elif idx + int(n_neighbour / 2) > c.shape[0] - 1:
      idx1 = c.shape[0] - 1 - 2 * int(n_neighbour / 2)
      idx2 = c.shape[0] - 1
    else:
      idx1 = idx - int(n_neighbour / 2)
      idx2 = idx + int(n_neighbour / 2)
    unc.append(np.mean(u[idx1:idx2]))
    acc.append(np.mean(c[idx1:idx2]))

  return unc, acc


def aulc(labels, scores, reverse_sort: bool = False):
  n = labels.shape[0]

  if reverse_sort:
    sorted_idx = np.flip(np.argsort(scores), axis=0)
  else:
    sorted_idx = np.argsort(scores)
  sorted_labels = labels[sorted_idx]

  lift = np.zeros(n)
  lift[0] = sorted_labels[0] / np.mean(sorted_labels)
  _m = np.mean(sorted_labels)
  for i in range(1, n):
    lift[i] = (i * lift[i - 1] + sorted_labels[i] / _m) / (i + 1)

  step = 1 / n
  return lift, np.arange(step, 1 + step, step), np.sum(lift) * step - 1


def compute_raulc(labels, scores):
  y, x, area = aulc(labels, scores)
  y_opt, x_opt, area_opt = aulc(labels, labels, reverse_sort=True)

  return area / area_opt


def visualize_segmentation(predictions: Dict[float, Dict[str, Any]],
                           path: str,
                           num_classes: int = 20):
  for k in predictions:
    current_path = os.path.join(path, str(k))
    tf.io.gfile.mkdir(current_path)

    for i in range(predictions[k]['prediction'].shape[0]):
      pred = to_colored_label(
        np.argmax(predictions[k]['prediction'][i], axis=-1),
        num_classes=num_classes)
      targ = to_colored_label(
        np.argmax(predictions[k]['target'][i], axis=-1),
        num_classes=num_classes)
      unc = predictions[k]['uncertainty_pixel'][i]
      img = predictions[k]['x'][i]

      fig = plt.figure()
      plt.imshow(pred)
      plt.savefig(os.path.join(current_path, f'{i}_pred.png'))
      plt.close(fig)

      fig = plt.figure()
      plt.imshow(targ)
      plt.savefig(os.path.join(current_path, f'{i}_targ.png'))
      plt.close(fig)

      fig = plt.figure()
      plt.imshow(unc)
      plt.savefig(os.path.join(current_path, f'{i}_unc.png'))
      plt.close(fig)

      fig = plt.figure()
      plt.imshow(img)
      plt.savefig(os.path.join(current_path, f'{i}_img.png'))
      plt.close(fig)


def approximation(x1, x2, func, n_samples=1000000, iters=20):
  n_samples = min(n_samples, x1.shape[0])

  results = []
  for _ in range(iters):
    idxs = np.random.choice(
      np.arange(x1.shape[0]), size=n_samples, replace=True)
    results += [func(x1[idxs], x2[idxs])]
  return np.mean(results)


def compute_calibration_metrics(predictions: Dict[float, Dict[str, Any]],
                                num_classes: int = 20,
                                segmentation: bool = False,
                                dataset: str = 'cityscapes') -> Dict[str, Any]:
  """Computes all calibration metrics for each perturbation step and across all perturbations.

  Metrics:
  - ACC
  - AUROC between correct and incorrect predictions
  - spearman rank correlation coefficient
  - RAULC
  - mean uncertainty
  (- IoU/mIoU for semantic segmentation)

  Args:
    predictions: dictionary with dictionary for each perturbation. each
      dictionary contains prediction, uncertainty and target.

  Returns:
    metrics_dict: dictionary with entries perturbation and overall.
      perturbation contains development of etrics over perturbations
      overall contains metrics computed across perturbations
  """

  metrics_dict = {
    'perturbation': {
      'range': list(predictions.keys()),
      'ACC': [],
      'AUROC': [],
      'AULC': []
    },
    'overall': dict()
  }
  if segmentation:
    metrics_dict['perturbation']['mIoU'] = []  # mean IoU
  else:
    metrics_dict['perturbation']['spearman'] = []  # mean IoU

  #################################################
  # Iterate over perturbations and compute metrics
  # for each perturbation step
  all_correct = []
  all_uncertainty = []
  for key in predictions:

    print(f'Postprocessing key={key}', flush=True)

    pred = np.argmax(predictions[key]['prediction'], -1)
    targ = np.argmax(predictions[key]['target'], -1)
    correct = pred == targ

    # Compute ACC
    print('Computing Accuracy', flush=True)
    metrics_dict['perturbation']['ACC'].append(np.mean(correct))

    if segmentation:
      print('Computing IoU', flush=True)
      iou = sklearn.metrics.jaccard_score(
        y_true=np.reshape(targ, -1),
        y_pred=np.reshape(pred, -1),
        labels=np.arange(num_classes),
        average=None)
      print(f'Per class IoU: {iou}')
      print('Computing mIoU', flush=True)
      metrics_dict['perturbation']['mIoU'].append(
        np.mean(iou[IOU_MASKS[dataset]]))

    # Compute AUROC between correct and incorrect predictions
    print('Computing AUROC', flush=True)
    if segmentation:
      scores = predictions[key]['uncertainty_pixel']
      metrics_dict['perturbation']['AUROC'].append(approximation(
        np.reshape(correct, -1), np.reshape(scores, -1), compute_auroc))
    else:
      scores = predictions[key]['uncertainty']
      metrics_dict['perturbation']['AUROC'].append(
        compute_auroc(correct, scores))

    # id_dict = predictions[metrics_dict['perturbation']['range'][0]]
    # ood_dict = predictions[key]
    # merged = np.concatenate(
    #     [id_dict['uncertainty'], ood_dict['uncertainty']], 0)
    # merged_labels = np.zeros(merged.shape[0])
    # merged_labels[:id_dict['uncertainty'].shape[0]] = 1
    # res = compute_auroc_ap_aupr(merged_labels, merged)
    # metrics_dict['perturbation']['AUROC'].append(res[0])
    # metrics_dict['perturbation']['AP'].append(res[1])
    # metrics_dict['perturbation']['AUPR'].append(res[2])
    # metrics_dict['perturbation']['mean_uncertainty'].append(
    #     np.mean(ood_dict['uncertainty']))

    # Compute pearson & spearman correlation coefficients
    if not segmentation:
      random_unc_bins, random_acc_bins = get_calibration(
        correct, -1 * predictions[key]['uncertainty'],
        n_neighbour=100, n_sample=1000)
      metrics_dict['perturbation']['spearman'].append(
        spearmanr(random_unc_bins, random_acc_bins)[0])

    # Compute relative area under the lift curve
    print('Computing AULC', flush=True)
    if segmentation:
      # area_lc, curves = compute_raulc(
      #     np.reshape(correct, (-1)),
      #     np.reshape(predictions[key]['uncertainty_pixel'], (-1)))
      # area_lc = compute_raulc(
      #     np.mean(correct, axis=(1, 2)), predictions[key]['uncertainty'])
      area_lc = approximation(np.reshape(correct, -1), np.reshape(scores, -1),
                              compute_raulc, n_samples=100000, iters=10)
    else:
      area_lc = compute_raulc(correct, predictions[key]['uncertainty'])
    metrics_dict['perturbation']['AULC'].append(area_lc)

    all_correct.append(correct)
    if segmentation:
      all_uncertainty.append(predictions[key]['uncertainty_pixel'])
    else:
      all_uncertainty.append(predictions[key]['uncertainty'])
  all_correct = np.reshape(np.concatenate(all_correct, axis=0), -1)
  all_uncertainty = np.reshape(np.concatenate(all_uncertainty, axis=0), -1)
  #################################################

  #################################################
  # Compute overall metrics.

  # Compute AUROC between correct and incorrect predictions
  print('Computing AUROC', flush=True)
  id_dict = predictions[metrics_dict['perturbation']['range'][0]]
  scores = all_uncertainty
  labels = all_correct
  if segmentation:
    metrics_dict['overall']['AUROC'] = approximation(
      labels, scores, compute_auroc)
  else:
    metrics_dict['overall']['AUROC'] = compute_auroc(labels, scores)
  # print('Computing overall AUROC, AP & AUPr', flush=True)
  # id_dict = predictions[metrics_dict['perturbation']['range'][0]]
  # merged = np.concatenate([id_dict['uncertainty'], all_uncertainty], 0)
  # merged_labels = np.zeros(merged.shape[0])
  # merged_labels[:id_dict['uncertainty'].shape[0]] = 1
  # res = compute_auroc_ap_aupr(merged_labels, merged)
  # metrics_dict['overall']['AUROC'] = res[0]
  # metrics_dict['overall']['AP'] = res[1]
  # metrics_dict['overall']['AUPR'] = res[2]

  # Compute pearson & spearman correlation coefficients
  if not segmentation:
    random_unc_bins, random_acc_bins = get_calibration(
      all_correct, -1 * all_uncertainty, n_neighbour=100, n_sample=1000)
    metrics_dict['overall']['spearman'] = spearmanr(random_unc_bins,
                                                    random_acc_bins)[0]

  # Compute relative area under the lift curve
  print('Computing overall AULC', flush=True)
  if segmentation:
    # area_lc, curves = compute_raulc(
    #     np.mean(all_correct, axis=(1, 2)), all_uncertainty)
    area_lc = approximation(all_correct, all_uncertainty, compute_raulc,
                            n_samples=100000, iters=10)
  else:
    area_lc = compute_raulc(all_correct, all_uncertainty)
  metrics_dict['overall']['AULC'] = area_lc
  #################################################

  return metrics_dict


def postprocess_calibration_metrics(calibration_metrics: Dict[str, Any],
                                    results_path: str):
  """Postprocessing of calibration metrics -> Save visualizations (png) and metrics (csv) Also dumps all results (calibration_metrics)

  Args:
    calibration_metrics: dictionary with entries perturbation and overall.
      perturbation contains development of etrics over perturbations overall
      contains metrics computed across perturbations
    results_path: string containing Path to result folder
  """

  # results regarding metrics vs. perturbation magnitude
  fname = os.path.join(results_path, 'perturbation.csv')
  write_dict_to_csv(calibration_metrics['perturbation'], fname)
  # perturbation_path = os.path.join(results_path, 'perturbation')
  # tf.io.gfile.mkdir(perturbation_path)
  # for key in calibration_metrics['perturbation']:
  #   if key != 'range' and key != 'IoU':
  #     print(f'Plotting key... {key}')
  #     plot(
  #         calibration_metrics['perturbation']['range'],
  #         calibration_metrics['perturbation'][key],
  #         labels=[key],
  #         xlabel='Perturbation',
  #         ylabel=key,
  #         path=os.path.join(perturbation_path, f'{key}.png'),
  #         scatter=False)

  # log overall results
  write_dict_to_csv(
    results_dict=calibration_metrics['overall'],
    file_name=os.path.join(results_path, 'overall.csv'))

  # save data
  tmp_path = os.path.join('/tmp', 'metrics.p')
  pickle.dump(calibration_metrics, open(tmp_path, 'wb'))
  tf.io.gfile.copy(
    tmp_path, os.path.join(results_path, 'metrics.p'), overwrite=True)


def plot(X,
         Y,
         labels: List[Text],
         xlabel: str = '',
         ylabel: str = '',
         path: Optional[str] = None,
         display: bool = False,
         scatter: bool = False,
         **kwargs):
  fig = plt.figure()
  if isinstance(X, list) or len(X.shape) == 1:  # one graph
    if scatter:
      plt.scatter(X, Y, label=labels[0], **kwargs)
    else:
      plt.plot(X, Y, label=labels[0], **kwargs)
  else:  # several graphs
    for i in range(X.shape[0]):
      if scatter:
        plt.scatter(X[i], Y[i], label=labels[i], **kwargs)
      else:
        plt.plot(X[i], Y[i], label=labels[i], **kwargs)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  if path is not None:
    with tf.io.gfile.GFile(path, 'wb') as f:
      plt.savefig(f)
  if display:
    plt.show()
  else:
    plt.close(fig)
