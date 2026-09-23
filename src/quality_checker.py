
import logging
from typing import Dict, Any

import cv2
import numpy as np

from src.config import (
    BLUR_THRESHOLD,
    CONTRAST_THRESHOLD,
    LOW_RESOLUTION_PIXEL_AREA,
    MIN_TEXT_HEIGHT_PX,
    VERY_SMALL_CROP_HEIGHT_PX,
)

logger = logging.getLogger(__name__)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_blur_score(gray_image: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def compute_contrast_score(gray_image: np.ndarray) -> float:
    return float(gray_image.std())


def estimate_noise(gray_image: np.ndarray) -> float:
    denoised = cv2.medianBlur(gray_image, 3)
    diff = cv2.absdiff(gray_image, denoised)
    return float(diff.mean())


def analyze_quality(
    image: np.ndarray,
    blur_threshold: float = BLUR_THRESHOLD,
    contrast_threshold: float = CONTRAST_THRESHOLD,
    low_resolution_pixel_area: int = LOW_RESOLUTION_PIXEL_AREA,
) -> Dict[str, Any]:
    height, width = image.shape[:2]
    total_pixels = width * height

    gray = _to_grayscale(image)

    blur_score = compute_blur_score(gray)
    contrast_score = compute_contrast_score(gray)
    noise_score = estimate_noise(gray)

    is_low_resolution = total_pixels < low_resolution_pixel_area or height < MIN_TEXT_HEIGHT_PX
    is_blurry = blur_score < blur_threshold
    is_low_contrast = contrast_score < contrast_threshold


    is_noisy = noise_score > 6.0


    is_very_small_crop = height < VERY_SMALL_CROP_HEIGHT_PX

    report = {
        "width": int(width),
        "height": int(height),
        "total_pixels": int(total_pixels),
        "blur_score": round(blur_score, 2),
        "contrast_score": round(contrast_score, 2),
        "noise_score": round(noise_score, 2),
        "is_low_resolution": bool(is_low_resolution),
        "is_blurry": bool(is_blurry),
        "is_low_contrast": bool(is_low_contrast),
        "is_noisy": bool(is_noisy),
        "is_very_small_crop": bool(is_very_small_crop),
    }

    logger.info("Quality report: %s", report)
    return report


def needs_preprocessing(quality_report: Dict[str, Any]) -> bool:
    return (
        quality_report["is_blurry"]
        or quality_report["is_low_contrast"]
        or quality_report["is_low_resolution"]
        or quality_report["is_noisy"]
    )
