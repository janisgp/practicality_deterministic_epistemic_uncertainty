"""Helper for applying corruptions."""

from typing import Dict
import tensorflow as tf

import data_lib.corruptions.corruptions_transforms as transforms


CORRUPTION_NAME_TO_FUNCTION = {
    "gaussian_noise": transforms.gaussian_noise,
    "shot_noise": transforms.shot_noise,
    "impulse_noise": transforms.impulse_noise,
    "defocus_blur": transforms.defocus_blur,
    "frosted_glass_blur": transforms.glass_blur,
    "motion_blur": transforms.motion_blur,
    "zoom_blur": transforms.zoom_blur,
    "snow": transforms.snow,
    "fog": transforms.fog,
    "brightness": transforms.brightness,
    "contrast": transforms.contrast,
    "elastic": transforms.elastic_transform,
    "pixelate": transforms.pixelate,
    "jpeg_compression": transforms.jpeg_compression,
}


BENCHMARK_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "frosted_glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic",
    "pixelate",
    "jpeg_compression",
]


def get_corruption(name: str, severity: int):
  def _corrupt(x):
    img_corrupted_numpy = CORRUPTION_NAME_TO_FUNCTION[name](
        x, severity=severity)
    return img_corrupted_numpy
  return _corrupt
