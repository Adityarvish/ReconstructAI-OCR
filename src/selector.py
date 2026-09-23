
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

from src.config import MIN_CROP_WIDTH, MIN_CROP_HEIGHT, ROI_PADDING_PIXELS

logger = logging.getLogger(__name__)

WINDOW_NAME = "Select Text Region - ENTER to confirm, ESC to cancel"


class ROISelectionError(Exception):
    pass


def select_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    print("\nSelect the text region and press ENTER. Press ESC to cancel.\n")


    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    roi = cv2.selectROI(WINDOW_NAME, image, showCrosshair=True, fromCenter=False)

    cv2.destroyWindow(WINDOW_NAME)

    x, y, w, h = roi

    if w <= 0 or h <= 0:
        raise ROISelectionError(
            "ROI selection was cancelled or empty. No region was selected."
        )

    if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
        raise ROISelectionError(
            f"Selected region is too small ({w}x{h}). "
            f"Minimum required size is {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT}. "
            f"Please select a larger region."
        )

    logger.info("ROI selected: x=%d, y=%d, width=%d, height=%d", x, y, w, h)
    return int(x), int(y), int(w), int(h)


def crop_image(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    padding: int = ROI_PADDING_PIXELS,
) -> np.ndarray:
    x, y, w, h = roi
    img_h, img_w = image.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    cropped = image[y1:y2, x1:x2]

    if cropped is None or cropped.size == 0:
        raise ROISelectionError("Cropping failed - resulting region is empty.")

    return cropped


def save_crop(cropped_image: np.ndarray, output_path: str) -> None:
    success = cv2.imwrite(output_path, cropped_image)
    if not success:
        raise ROISelectionError(f"Failed to save cropped image to '{output_path}'")

    logger.info("Saved selected crop to '%s'", output_path)


def select_and_crop(
    image: np.ndarray,
    output_path: Optional[str] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    if roi is None:
        roi = select_roi(image)
    else:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ROISelectionError(f"Invalid ROI: width/height must be positive, got {roi}.")
        if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
            raise ROISelectionError(
                f"Given region is too small ({w}x{h}). "
                f"Minimum required size is {MIN_CROP_WIDTH}x{MIN_CROP_HEIGHT}."
            )
        roi = (int(x), int(y), int(w), int(h))

    cropped = crop_image(image, roi)

    if output_path:
        save_crop(cropped, output_path)

    return cropped, roi
