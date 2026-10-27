"""Observation-image conversion for OpenVLA.

Upstream ``get_vla_action()`` accepts one RGB frame, optionally applies the
training-time center crop (``scale=0.9`` means each side keeps ``sqrt(0.9)`` of
its length, then the crop is resampled to 224x224), and lets the HF processor
normalise the result.

This module reproduces that with Pillow alone, so it stays importable (and
testable) without torch.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from PIL import Image

CENTER_CROP_SCALE = 0.9
#: Side length the processor reduces to; upstream crop_and_resize targets it too.
_PROCESSOR_SIZE = 224


def to_pil_image(image: npt.NDArray[np.uint8] | Image.Image) -> Image.Image:
    """Return ``image`` as an RGB ``PIL.Image``, rejecting anything ambiguous."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got array shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected uint8 image data, got dtype {array.dtype}")
    return Image.fromarray(array).convert("RGB")


def prepare_image(
    image: npt.NDArray[np.uint8] | Image.Image,
    *,
    center_crop: bool = False,
    crop_scale: float = CENTER_CROP_SCALE,
) -> Image.Image:
    """Convert an observation frame into the image OpenVLA's processor expects.

    ``center_crop=True`` mirrors the RLDS training augmentation: crop to
    ``sqrt(crop_scale)`` of each side (centered), then resample straight to the
    224 px the processor consumes, like upstream's ``crop_and_resize``. A no-op
    when ``center_crop`` is false.
    """
    pil = to_pil_image(image)
    if not center_crop:
        return pil
    if not 0.0 < crop_scale <= 1.0:
        raise ValueError(f"crop_scale must be in (0, 1]; got {crop_scale!r}")

    width, height = pil.size
    crop_width = max(1, round(width * math.sqrt(crop_scale)))
    crop_height = max(1, round(height * math.sqrt(crop_scale)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((_PROCESSOR_SIZE, _PROCESSOR_SIZE), resample=Image.Resampling.BILINEAR)
