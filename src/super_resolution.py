
import logging
import os
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ocr_sr_models")

_MODEL_URLS = {
    2: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb",
    3: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x3.pb",
    4: "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x4.pb",
}


_readers = {}
_unavailable = False  


def _get_model_path(scale: int) -> Optional[str]:
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    path = os.path.join(_MODEL_CACHE_DIR, f"FSRCNN_x{scale}.pb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = _MODEL_URLS.get(scale)
    if url is None:
        return None

    try:
        logger.info("Downloading FSRCNN x%d super-resolution model (one-time, ~40 KB)...", scale)
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as exc:
        logger.info("Could not download super-resolution model (%s) - "
                    "falling back to interpolation-based upscaling.", exc)
        return None


def _get_reader(scale: int):
    global _unavailable
    if _unavailable:
        return None
    if scale in _readers:
        return _readers[scale]

    try:
        import cv2
        if not hasattr(cv2, "dnn_superres"):
            logger.info(
                "cv2.dnn_superres not available (needs opencv-contrib-python "
                "instead of opencv-python) - falling back to interpolation-based upscaling."
            )
            _unavailable = True
            return None
    except ImportError:
        _unavailable = True
        return None

    model_path = _get_model_path(scale)
    if model_path is None:
        _unavailable = True
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", scale)
        _readers[scale] = sr
        return sr
    except Exception as exc:
        logger.info("Failed to initialize super-resolution model: %s", exc)
        _unavailable = True
        return None


def super_resolve(gray_image: np.ndarray, scale: int = 2) -> Optional[np.ndarray]:
    reader = _get_reader(scale)
    if reader is None:
        return None

    try:
        import cv2

        if gray_image.ndim == 2:
            bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = gray_image

        upsampled = reader.upsample(bgr)

        if gray_image.ndim == 2:
            upsampled = cv2.cvtColor(upsampled, cv2.COLOR_BGR2GRAY)
        return upsampled
    except Exception as exc:
        logger.warning("Super-resolution inference failed: %s", exc)
        return None
