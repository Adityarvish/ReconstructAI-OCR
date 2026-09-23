
import logging
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from src.config import (
    UPSCALE_FACTOR,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    SHARPEN_AMOUNT,
    MEDIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_KERNEL,
    MAX_DESKEW_ANGLE,
    ILLUMINATION_BLUR_KSIZE,
    TARGET_UPSCALE_HEIGHT_PX,
    MAX_ADAPTIVE_UPSCALE_FACTOR,
    ENABLE_DECONVOLUTION_CANDIDATES,
    ENABLE_SUPER_RESOLUTION,
    SR_SCALE_SMALL,
    SR_SCALE_TINY,
)
from src import deblur as _deblur
from src import super_resolution as _super_resolution

logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale_image(
    image: np.ndarray, scale_factor: float = UPSCALE_FACTOR, strong: bool = False
) -> np.ndarray:
    interpolation = cv2.INTER_LANCZOS4 if strong else cv2.INTER_CUBIC
    new_size = (
        max(1, int(image.shape[1] * scale_factor)),
        max(1, int(image.shape[0] * scale_factor)),
    )
    return cv2.resize(image, new_size, interpolation=interpolation)


def adaptive_upscale(
    image: np.ndarray,
    target_height: int = TARGET_UPSCALE_HEIGHT_PX,
    max_factor: float = MAX_ADAPTIVE_UPSCALE_FACTOR,
) -> np.ndarray:
    height = image.shape[0]
    if height <= 0:
        return image

    factor = target_height / float(height)
    factor = max(1.0, min(factor, max_factor))

    return upscale_image(image, scale_factor=factor, strong=True)


def denoise_gaussian(image: np.ndarray, kernel: Tuple[int, int] = GAUSSIAN_BLUR_KERNEL) -> np.ndarray:
    return cv2.GaussianBlur(image, kernel, 0)


def denoise_median(image: np.ndarray, kernel_size: int = MEDIAN_BLUR_KERNEL) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1  
    return cv2.medianBlur(image, kernel_size)


def denoise_non_local_means(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=10)
    return cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def normalize_illumination(
    gray_image: np.ndarray, blur_ksize: int = ILLUMINATION_BLUR_KSIZE
) -> np.ndarray:
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    background = cv2.GaussianBlur(gray_image, (blur_ksize, blur_ksize), 0)

    background = np.where(background == 0, 1, background)

    normalized = cv2.divide(gray_image.astype(np.float32), background.astype(np.float32), scale=255)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def sharpen_image(image: np.ndarray, amount: float = SHARPEN_AMOUNT) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def threshold_otsu(gray_image: np.ndarray) -> np.ndarray:
    _, thresholded = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def threshold_adaptive(gray_image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=15,
    )


def _binary_looks_degenerate(binary_image: np.ndarray) -> bool:
    h, w = binary_image.shape[:2]
    if h == 0 or w == 0:
        return True

    dark = binary_image < 127
    dark_fraction = float(dark.mean())
    if dark_fraction > 0.6 or dark_fraction < 0.02:
        return True


    row_dark_fraction = dark.mean(axis=1)
    solid_rows = int((row_dark_fraction > 0.85).sum())
    if solid_rows / h > 0.3:
        return True


    dark_u8 = dark.astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
    if num_labels > 1:  
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        total_dark_area = int(areas.sum())
        largest_idx = int(np.argmax(areas))
        largest_fraction = areas[largest_idx] / total_dark_area if total_dark_area > 0 else 0.0
        largest_spans_wide = widths[largest_idx] > 0.5 * w
        largest_spans_tall = heights[largest_idx] > 0.4 * h
        if largest_fraction > 0.6 and largest_spans_wide and largest_spans_tall:
            return True

    return False


def threshold_then_upscale(gray_image: np.ndarray, method: str = "otsu") -> np.ndarray:
    thresh_fn = threshold_otsu if method == "otsu" else threshold_adaptive
    illum = normalize_illumination(gray_image)
    native_binary = thresh_fn(illum)

    if _binary_looks_degenerate(native_binary):
        logger.info(
            "threshold_then_upscale(%s): native threshold looked degenerate, "
            "falling back to non-thresholded upscale instead of amplifying garbage",
            method,
        )
        return adaptive_upscale(illum)

    upscaled = adaptive_upscale(native_binary)
    _, clean = cv2.threshold(upscaled, 127, 255, cv2.THRESH_BINARY)

    if _binary_looks_degenerate(clean):
        logger.info(
            "threshold_then_upscale(%s): result looked degenerate after "
            "upscale/re-threshold, falling back to non-thresholded upscale",
            method,
        )
        return adaptive_upscale(illum)

    return clean


def estimate_skew_angle(gray_image: np.ndarray) -> float:


    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 10:

        return 0.0

    angle = cv2.minAreaRect(coords)[-1]


    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    return float(angle)


def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def try_deskew(gray_image: np.ndarray, max_angle: float = MAX_DESKEW_ANGLE) -> Tuple[np.ndarray, bool, float]:
    angle = estimate_skew_angle(gray_image)

    if abs(angle) < 0.5:


        return gray_image, False, 0.0

    if abs(angle) > max_angle:


        logger.info("Skipping deskew: estimated angle %.2f exceeds max %.2f", angle, max_angle)
        return gray_image, False, angle

    return deskew_image(gray_image, angle), True, angle


def preprocess_for_ocr(
    image: np.ndarray, quality_report: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    steps: List[str] = []

    processed = to_grayscale(image)
    steps.append("grayscale")


    processed, deskewed, angle = try_deskew(processed)
    if deskewed:
        steps.append(f"deskew({angle:.1f} deg)")


    if quality_report.get("is_noisy"):
        processed = denoise_median(processed)
        steps.append("median_filter")
    elif quality_report.get("is_blurry"):


        processed = denoise_gaussian(processed)
        steps.append("gaussian_denoise")


    if quality_report.get("is_low_resolution"):

        processed = upscale_image(processed, scale_factor=UPSCALE_FACTOR, strong=True)
        steps.append(f"upscale_{UPSCALE_FACTOR:g}x_lanczos")
    elif quality_report.get("is_blurry"):


        processed = upscale_image(processed, scale_factor=1.5, strong=False)
        steps.append("upscale_1.5x_cubic")


    if quality_report.get("is_low_contrast"):
        processed = apply_clahe(processed)
        steps.append("CLAHE")


    if quality_report.get("is_blurry"):
        processed = sharpen_image(processed)
        steps.append("sharpen")


    if quality_report.get("is_low_contrast") and quality_report.get("is_blurry"):
        thresholded = threshold_adaptive(processed)
        if not _binary_looks_degenerate(thresholded):
            processed = thresholded
            steps.append("adaptive_threshold")
        else:
            logger.info("Skipping adaptive_threshold: result looked degenerate")
    elif quality_report.get("is_low_contrast"):
        thresholded = threshold_otsu(processed)
        if not _binary_looks_degenerate(thresholded):
            processed = thresholded
            steps.append("otsu_threshold")
        else:
            logger.info("Skipping otsu_threshold: result looked degenerate")

    logger.info("Preprocessing steps applied: %s", steps)
    return processed, steps


def generate_candidates(
    image: np.ndarray, quality_report: Dict[str, Any]
) -> List[Tuple[str, np.ndarray]]:
    gray = to_grayscale(image)
    candidates: List[Tuple[str, np.ndarray]] = [("grayscale", gray)]


    deskewed, was_deskewed, angle = try_deskew(gray)
    if was_deskewed:
        candidates.append((f"deskew_{angle:.1f}deg", deskewed))
        base = deskewed
    else:
        base = gray


        candidates.append(("deskew_try_+3deg", deskew_image(gray, 3.0)))
        candidates.append(("deskew_try_-3deg", deskew_image(gray, -3.0)))


    illum = normalize_illumination(base)
    candidates.append(("illumination_normalized", illum))
    illum_otsu_candidate = threshold_otsu(illum)
    if not _binary_looks_degenerate(illum_otsu_candidate):
        candidates.append(("illum_norm_otsu", illum_otsu_candidate))


    candidates.append(("median_filter", denoise_median(base)))
    candidates.append(("gaussian_denoise", denoise_gaussian(base)))


    clahe_img = apply_clahe(base)
    candidates.append(("CLAHE", clahe_img))


    candidates.append(("sharpen", sharpen_image(base)))
    candidates.append(("CLAHE_sharpen", sharpen_image(clahe_img)))
    candidates.append(("illum_norm_sharpen", sharpen_image(illum)))


    candidates.append(("upscale_2x_cubic", upscale_image(base, 2.0, strong=False)))
    candidates.append(("upscale_3x_lanczos", upscale_image(base, 3.0, strong=True)))
    candidates.append(
        ("upscale_2x_CLAHE_sharpen", sharpen_image(apply_clahe(upscale_image(base, 2.0, strong=True))))
    )


    adaptive = adaptive_upscale(base)
    candidates.append(("adaptive_upscale", adaptive))
    candidates.append(("adaptive_upscale_sharpen", sharpen_image(adaptive)))
    candidates.append(("adaptive_upscale_CLAHE_sharpen", sharpen_image(apply_clahe(adaptive))))
    candidates.append(("adaptive_upscale_illum_norm", normalize_illumination(adaptive)))


    otsu_candidate = threshold_otsu(base)
    if not _binary_looks_degenerate(otsu_candidate):
        candidates.append(("otsu_threshold", otsu_candidate))

    adaptive_candidate = threshold_adaptive(base)
    if not _binary_looks_degenerate(adaptive_candidate):
        candidates.append(("adaptive_threshold", adaptive_candidate))


    candidates.append(("otsu_native_then_upscale", threshold_then_upscale(base, "otsu")))
    candidates.append(("adaptive_native_then_upscale", threshold_then_upscale(base, "adaptive")))


    if quality_report.get("is_blurry") and ENABLE_DECONVOLUTION_CANDIDATES:
        for label, deconv_img in _deblur.generate_deblur_candidates(base):
            candidates.append((label, deconv_img))


        rl_psf = _deblur.gaussian_psf(size=9, sigma=2.0)
        rl_result = _deblur.richardson_lucy(base, rl_psf, iterations=10)
        candidates.append(("richardson_lucy_sharpen", sharpen_image(rl_result)))


    if ENABLE_SUPER_RESOLUTION and (
        quality_report.get("is_low_resolution") or quality_report.get("is_very_small_crop")
    ):
        scale = SR_SCALE_TINY if quality_report.get("is_very_small_crop") else SR_SCALE_SMALL
        sr_image = _super_resolution.super_resolve(base, scale=scale)
        if sr_image is not None:
            candidates.append((f"fsrcnn_sr_x{scale}", sr_image))
            candidates.append((f"fsrcnn_sr_x{scale}_sharpen", sharpen_image(sr_image)))
            candidates.append((f"fsrcnn_sr_x{scale}_CLAHE", apply_clahe(sr_image)))


            if quality_report.get("is_blurry") and ENABLE_DECONVOLUTION_CANDIDATES:
                rl_psf = _deblur.gaussian_psf(size=9, sigma=2.0)
                deblurred_first = _deblur.richardson_lucy(base, rl_psf, iterations=8)
                sr_after_deblur = _super_resolution.super_resolve(deblurred_first, scale=scale)
                if sr_after_deblur is not None:
                    candidates.append((f"deblur_then_sr_x{scale}", sr_after_deblur))


    combined, _ = preprocess_for_ocr(image, quality_report)
    candidates.append(("combined_pipeline", combined))

    return _prioritize_candidates(candidates, quality_report)


def _prioritize_candidates(
    candidates: List[Tuple[str, np.ndarray]], quality_report: Dict[str, Any]
) -> List[Tuple[str, np.ndarray]]:

    boosts = []
    if quality_report.get("is_low_contrast"):
        boosts += ["illumination_normalized", "illum_norm_otsu", "illum_norm_sharpen", "CLAHE", "CLAHE_sharpen"]
    if quality_report.get("is_blurry"):
        boosts += [
            "sharpen", "CLAHE_sharpen", "illum_norm_sharpen", "upscale_2x_CLAHE_sharpen",
            "richardson_lucy_defocus", "richardson_lucy_sharpen",
        ]
    if quality_report.get("is_low_resolution"):
        boosts += [
            "adaptive_upscale",
            "adaptive_upscale_sharpen",
            "adaptive_upscale_CLAHE_sharpen",
            "adaptive_upscale_illum_norm",
            "upscale_3x_lanczos",
            "upscale_2x_cubic",
            "upscale_2x_CLAHE_sharpen",
            "otsu_native_then_upscale",
            "adaptive_native_then_upscale",
        ]
    if quality_report.get("is_noisy"):
        boosts += ["median_filter", "gaussian_denoise"]

    if quality_report.get("is_very_small_crop"):
        boosts += ["otsu_native_then_upscale", "adaptive_native_then_upscale"]


    prefix_boosts = []
    if quality_report.get("is_blurry"):
        prefix_boosts += ["wiener_gaussian_", "wiener_motion_"]
    if quality_report.get("is_low_resolution") or quality_report.get("is_very_small_crop"):
        prefix_boosts += ["fsrcnn_sr_", "deblur_then_sr_"]

    def priority(item: Tuple[str, np.ndarray]) -> int:
        label = item[0]


        if label in boosts or any(label.startswith(p) for p in prefix_boosts):
            return 0
        return 1


    return sorted(candidates, key=priority)
