"""Image loading / preprocessing shared by dataset preparation, training and inference.

Single source of truth for how a picture becomes model input, so that training and
inference cannot drift apart:

    open -> fix EXIF rotation -> flatten transparency -> RGB -> resize (bilinear, squash)
         -> float32 array with values in [0, 255]

The model itself contains the final scaling layer (see ``model.py``), so nothing
else needs to be applied at inference time.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import IMAGE_SIZE, MAX_IMAGE_PIXELS, MAX_UPLOAD_MB

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

MIN_SIDE_PIXELS = 32
ImageSource = Union[str, Path, bytes, BinaryIO]


class ImageValidationError(ValueError):
    """The file is not a usable image. The message is safe to show to end users."""


def open_rgb(source: ImageSource) -> Image.Image:
    """Decode an image fully into RGB, raising ImageValidationError on any problem."""
    try:
        if isinstance(source, (bytes, bytearray)):
            fh: BinaryIO = io.BytesIO(source)
            img = Image.open(fh)
        else:
            img = Image.open(source)
        img.load()  # forces a full decode so truncated files fail here, not later
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageValidationError("The file could not be read as an image "
                                   "(it may be corrupt or not a JPG/PNG).") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        img = img.convert("RGB")
    if min(img.size) < MIN_SIDE_PIXELS:
        raise ImageValidationError(f"The image is too small ({img.size[0]}x{img.size[1]} px); "
                                   f"each side must be at least {MIN_SIDE_PIXELS} px.")
    return img


def validate_upload(data: bytes, max_mb: float = MAX_UPLOAD_MB) -> Image.Image:
    """Validate an uploaded file (size + real decodability) and return an RGB image."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ImageValidationError(f"The file is larger than {max_mb:g} MB.")
    return open_rgb(data)


def resize_for_model(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize an RGB image to size x size (bilinear; aspect ratio is not preserved,
    exactly as during dataset preparation)."""
    return img.resize((size, size), Image.BILINEAR)


def to_model_array(img: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB PIL image -> float32 array (size, size, 3) with values in [0, 255]."""
    arr = np.asarray(resize_for_model(img, size), dtype=np.float32)
    assert arr.shape == (size, size, 3)
    return arr


def prepare_batch(img: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Array with a leading batch axis: (1, size, size, 3)."""
    return to_model_array(img, size)[None, ...]
