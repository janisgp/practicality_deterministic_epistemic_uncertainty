"""Optimization configs for image classification."""

from typing import Dict, Any


BASELINES = {
    'simple_fc': {
        'lr': 0.003,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 200,
        'patience': 200
    },
    'resnet': {
        'lr': 0.003,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 400,
        'patience': 200,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [100, 200, 300]
    }
}

MIR = {
    'simple_fc': {
        'lr': 0.001,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 200,
        'patience': 200
    },
    'resnet': {
        'lr': 0.003,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 400,
        'patience': 200,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [100, 200, 300]
    }
}

DDU = {
    'simple_fc': {
        'lr': 0.001,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 200,
        'patience': 200,
        'power_iterations': 1,
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'dropout': 0.0,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [80, 120, 180],
        'fitting_method': 'sample',
        'use_covariance': True,
        'num_batches_fitting': True
    },
    'resnet': {
        'lr': 0.001,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 400,
        'patience': 200,
        'power_iterations': 1,
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [50, 80, 110],
        'fitting_method': 'sample',
        'use_covariance': True,
        'num_batches_fitting': True
    }
}

SNGP = {
    'simple_fc': {
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'power_iterations': 1,
        'dropout': 0.0,
        'lr': 0.05,
        'l2_reg': 0.0004,
        'batch_size': 128,
        'epochs': 250,
        'patience': 200,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [60, 120, 160],
        'num_inducing_points': 1024
    },
    'resnet': {
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'power_iterations': 1,
        'dropout': 0.0,
        'lr': 0.05,
        'l2_reg': 0.0003,
        'batch_size': 128,
        'epochs': 400,
        'patience': 200,
        'lr_schedule_gamma': 0.2,
        'lr_schedule_steps': [100, 200, 300],
        'num_inducing_points': 128
    }
}

DUQ = {
    'simple_fc': {
        'lr': 0.01,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 200,
        'patience': 200,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [60, 120, 160],
    },
    'resnet': {
        'lr': 0.03,
        'l2_reg': 0.0001,
        'batch_size': 128,
        'epochs': 400,
        'patience': 200,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [200, 250, 300]
    }
}


def get_cfg(method: str, backbone: str) -> Dict[str, Any]:
  """Helper for getting configs.

  Args:
    method: str. Name of Method
    backbone: str. Type of backbone architecture

  Returns:
    Dictionary containing the method/backbone-specific config
  """
  if method in ['softmax', 'dropout']:
    return BASELINES[backbone]
  elif method == 'mir':
    return MIR[backbone]
  elif method == 'ddu':
    return DDU[backbone]
  elif method == 'duq':
    return DUQ[backbone]
  elif method == 'sngp':
    return SNGP[backbone]
  else:
    raise ValueError('Unknown method!')
