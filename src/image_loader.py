
import os
import logging
from typing import Dict, Any

import cv2
import numpy as np

from src.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    pass


def validate_path(image_path: str) -> None:
    if not image_path:
        raise ImageLoadError("No image path was provided.")

    if not os.path.exists(image_path):
        raise ImageLoadError(f"Image file not found: '{image_path}'")

    if not os.path.isfile(image_path):
        raise ImageLoadError(f"Path exists but is not a file: '{image_path}'")


def validate_extension(image_path: str) -> None:
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ImageLoadError(
            f"Unsupported file extension '{ext}'. Supported formats: {supported}"
        )


def load_image(image_path: str) -> np.ndarray:
    validate_path(image_path)
    validate_extension(image_path)


    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError(
            f"Failed to read image (it may be corrupted or in an "
            f"unsupported internal format): '{image_path}'"
        )

    logger.info("Loaded image '%s' with shape %s", image_path, image.shape)
    return image


def get_image_metadata(image: np.ndarray, image_path: str) -> Dict[str, Any]:
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1

    try:
        file_size_bytes = os.path.getsize(image_path)
    except OSError:
        file_size_bytes = None

    return {
        "path": image_path,
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "file_size_bytes": file_size_bytes,
    }
