"""
Optimization configs for semantic segmentation
"""

from typing import Dict, Any


BASELINES = {
    'drn': {
        'lr': 2e-4,
        'l2_reg': 0.0001,
        'epochs': 200,
        'patience': 200,
        'reduce_lr_on_plateau_gamma': 0.3,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [30, 60, 90, 120]
    }
}

MIR = {
    'drn': {
        'lr': 2e-4,
        'l2_reg': 0.0001,
        'epochs': 200,
        'patience': 200,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [30, 60, 90, 120]
    }
}

DDU = {
    'drn': {
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'coeff': 6,
        'lr': 0.0001,
        'l2_reg': 1e-4,
        'epochs': 200,
        'patience': 200,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [30, 60, 90, 120, 150]
    }
}

SNGP = {
    'drn': {
        'spectral_normalization': True,
        'soft_spectral_normalization': True,
        'coeff': 6,
        'power_iterations': 1,
        'dropout': 0.0,
        'lr': 0.05,
        'l2_reg': 0.0003,
        'epochs': 200,
        'patience': 200,
        'lr_schedule_gamma': 0.3,
        'lr_schedule_steps': [20, 40, 60, 80, 100],
        'num_inducing_points': 128
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
  if 'drn' in backbone:
    backbone_key = 'drn'
  else:
    raise ValueError(f'Unknown backbone {backbone}!')
  if method in ['softmax', 'dropout']:
    return BASELINES[backbone_key]
  elif method in ['sngp', 'snconvgp']:
    return SNGP[backbone_key]
  elif method in ['mir']:
    return MIR[backbone_key]
  elif method in ['ddu']:
    return DDU[backbone_key]
  else:
    raise ValueError('Unknown method!')
