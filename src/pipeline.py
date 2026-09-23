
import json
import logging
import os
from typing import Dict, Any, Tuple

import cv2
import numpy as np

from src import config
from src.image_loader import load_image, get_image_metadata, ImageLoadError
from src.selector import select_and_crop, ROISelectionError
from src.quality_checker import analyze_quality, needs_preprocessing
from src.preprocessing import preprocess_for_ocr, generate_candidates
from src.ocr_engine import OCREngine, OCREngineError, score_result
from src import vlm_ocr

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


def classify_confidence(confidence: float) -> str:
    if confidence >= config.HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    elif confidence >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def _is_strong_result(result: Dict[str, Any]) -> bool:
    text = result["text"].strip()
    return (
        bool(text)
        and len(text) >= config.MIN_STRONG_RESULT_CHARS
        and result["confidence"] >= config.HIGH_CONFIDENCE_THRESHOLD
        and result.get("coverage", 1.0) >= 0.5
    )


def _ensure_output_dir() -> None:
    try:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    except OSError as exc:
        raise PipelineError(f"Could not create output directory: {exc}") from exc


def run_pipeline(
    input_image_path: str,
    show_windows: bool = True,
    roi: "Tuple[int, int, int, int] | None" = None,
) -> Dict[str, Any]:
    _ensure_output_dir()


    try:
        image = load_image(input_image_path)
        metadata = get_image_metadata(image, input_image_path)
    except ImageLoadError as exc:
        raise PipelineError(str(exc)) from exc

    print("=" * 50)
    print("IMAGE OCR EXTRACTION SYSTEM")
    print("=" * 50)
    print(f"\nInput image:\n{input_image_path}")
    print(f"Resolution: {metadata['width']} x {metadata['height']}\n")


    try:
        cropped, roi = select_and_crop(image, output_path=config.SELECTED_CROP_PATH, roi=roi)
    except ROISelectionError as exc:
        raise PipelineError(str(exc)) from exc

    x, y, w, h = roi
    print(f"ROI:\nx={x}, y={y}, width={w}, height={h}\n")


    quality_report = analyze_quality(cropped)

    print("IMAGE QUALITY")
    print("-" * 14)
    print(f"Resolution: {quality_report['width']} x {quality_report['height']}")
    print(f"Blur Score: {quality_report['blur_score']}")
    print(f"Contrast Score: {quality_report['contrast_score']}")
    print(f"Noise Score: {quality_report['noise_score']}")
    print(f"Blur Detected: {'YES' if quality_report['is_blurry'] else 'NO'}")
    print(f"Low Contrast: {'YES' if quality_report['is_low_contrast'] else 'NO'}")
    print(f"Low Resolution: {'YES' if quality_report['is_low_resolution'] else 'NO'}")
    print(f"Noisy: {'YES' if quality_report['is_noisy'] else 'NO'}\n")


    if not quality_report["is_blurry"] and (
        quality_report.get("is_low_resolution") or quality_report.get("is_very_small_crop")
    ):
        print("NOTE:")
        print(
            "This crop is sharp at the pixel level (that's what 'Blur Detected: NO' "
            "means - no smearing/defocus to undo) but still hard to read because it's "
            "LOW RESOLUTION: too few pixels capture each character to begin with. "
            "These are different problems with different fixes - deblurring/sharpening "
            "cannot help here (there's nothing smeared to undo); upscaling and "
            "native-resolution binarization are what actually help, and that's what "
            "this run will prioritize below.\n"
        )

    if quality_report.get("is_very_small_crop"):
        print("NOTE:")
        print(
            f"The selected region is only {quality_report['height']}px tall. "
            f"OCR reliability drops sharply below roughly {config.MIN_TEXT_HEIGHT_PX}px "
            f"of text height, since there aren't enough captured pixels to tell "
            f"character shapes apart - no amount of enhancement can fully make up "
            f"for that. For best results, reselect a region with a bit more margin "
            f"around the text.\n"
        )


    ocr_engine = OCREngine()

    try:
        original_ocr = ocr_engine.recognize(cropped)
    except OCREngineError as exc:
        raise PipelineError(str(exc)) from exc

    best_result = original_ocr
    best_source = "original_crop"
    best_candidate_label = "original"
    preprocessing_steps = []
    attempted_candidates = []
    best_processed_image = None

    quality_bad = needs_preprocessing(quality_report)
    ocr_confidence_low = original_ocr["confidence"] < config.HIGH_CONFIDENCE_THRESHOLD
    ocr_text_empty = not original_ocr["text"].strip()


    ocr_incomplete = not _is_strong_result(original_ocr)


    if quality_report.get("is_very_small_crop") and not _is_strong_result(best_result) and vlm_ocr.is_available():
        print("VISION-LLM FAST PATH")
        print("-" * 20)
        print(f"Crop height is {quality_report['height']}px (below the "
              f"{config.VERY_SMALL_CROP_HEIGHT_PX}px threshold where classical "
              "upscaling/deblurring can reliably help) - trying the vision-LLM "
              "reader before the full classical sweep...\n")
        fast_vlm_result = vlm_ocr.recognize_with_vlm(cropped)
        if fast_vlm_result is not None:
            attempted_candidates.append(
                {
                    "label": "vision_llm_fast_path",
                    "text": fast_vlm_result["text"],
                    "confidence": fast_vlm_result["confidence"],
                    "coverage": fast_vlm_result.get("coverage", 1.0),
                    "ocr_method": fast_vlm_result.get("method"),
                }
            )
            if score_result(fast_vlm_result) > score_result(best_result):
                best_result = fast_vlm_result
                best_source = f"vision_llm_fast_path ({fast_vlm_result.get('model')})"
                print(f"Vision-LLM reading: {fast_vlm_result['text'] or '(model reported UNREADABLE)'}\n")
        else:
            print("Vision-LLM fast path unavailable or failed - continuing to classical preprocessing.\n")

    ocr_incomplete = not _is_strong_result(best_result)

    if quality_bad or ocr_confidence_low or ocr_text_empty or ocr_incomplete:
        candidates = generate_candidates(cropped, quality_report)

        print("PREPROCESSING")
        print("-" * 13)
        print(f"Trying {len(candidates)} correction strategies to find the best result...\n")

        for label, candidate_image in candidates:
            try:
                candidate_ocr = ocr_engine.recognize(candidate_image)
            except OCREngineError as exc:
                logger.warning("OCR failed on candidate '%s': %s", label, exc)
                continue

            attempted_candidates.append(
                {
                    "label": label,
                    "text": candidate_ocr["text"],
                    "confidence": candidate_ocr["confidence"],
                    "coverage": candidate_ocr.get("coverage", 1.0),
                    "ocr_method": candidate_ocr.get("method"),
                }
            )


            candidate_is_better = (
                score_result(candidate_ocr) > score_result(best_result)
                or (
                    score_result(candidate_ocr) == score_result(best_result)
                    and candidate_ocr["text"].strip()
                    and not best_result["text"].strip()
                )
            )

            if candidate_is_better:
                best_result = candidate_ocr
                best_source = f"processed_crop ({label})"
                best_candidate_label = label
                best_processed_image = candidate_image


            if _is_strong_result(best_result):
                break

        if best_processed_image is not None:
            preprocessing_steps = [best_candidate_label]
            try:
                cv2.imwrite(config.PROCESSED_CROP_PATH, best_processed_image)
            except Exception as exc:
                logger.warning("Could not save processed crop: %s", exc)

        print(f"Strategies tried: {len(attempted_candidates)}")
        print(f"Best strategy: {best_candidate_label} "
              f"(confidence {best_result['confidence'] * 100:.1f}%)\n")
    else:
        print("PREPROCESSING")
        print("-" * 13)
        print("- (skipped - initial crop quality and OCR confidence were sufficient)\n")


    vlm_attempted = any(c["label"] == "vision_llm_fast_path" for c in attempted_candidates)
    if not _is_strong_result(best_result) and vlm_ocr.is_available():
        print("VISION-LLM FALLBACK")
        print("-" * 19)
        print("Classical OCR still low-confidence after preprocessing - "
              "asking a vision-language model to read the crop...\n")


        image_for_vlm = best_processed_image if best_processed_image is not None else cropped
        vlm_result = vlm_ocr.recognize_with_vlm(image_for_vlm)
        vlm_attempted = True

        if vlm_result is not None:
            attempted_candidates.append(
                {
                    "label": "vision_llm",
                    "text": vlm_result["text"],
                    "confidence": vlm_result["confidence"],
                    "coverage": vlm_result.get("coverage", 1.0),
                    "ocr_method": vlm_result.get("method"),
                }
            )
            if score_result(vlm_result) > score_result(best_result) or (
                score_result(vlm_result) == score_result(best_result)
                and vlm_result["text"].strip()
                and not best_result["text"].strip()
            ):
                best_result = vlm_result
                best_source = f"vision_llm ({vlm_result.get('model')})"
                print(f"Vision-LLM reading: {vlm_result['text'] or '(model reported UNREADABLE)'}")
                print(f"Vision-LLM self-reported legibility: "
                      f"{vlm_result.get('qualitative_confidence', 'MEDIUM')}\n")
            else:
                print("Vision-LLM result did not improve on the existing best result.\n")
        else:
            print("Vision-LLM fallback unavailable or failed - keeping classical result.\n")
    elif not _is_strong_result(best_result) and not vlm_ocr.is_available():
        print("VISION-LLM FALLBACK")
        print("-" * 19)
        print("Skipped: not configured. To enable this last-resort fallback for "
              "hard-to-read crops, install the 'openai' package (OpenRouter is "
              "OpenAI-API-compatible) and set the "
              f"{config.VLM_API_KEY_ENV} environment variable (free key at "
              "https://openrouter.ai/keys). See README.md.\n")


    effective_confidence = score_result(best_result)
    confidence_status = classify_confidence(effective_confidence)
    result_is_from_vlm = best_result.get("method") == "vision_llm"
    coverage = best_result.get("coverage", 1.0)
    is_partial_read = (
        bool(best_result["text"].strip())
        and not _is_strong_result(best_result)
        and best_result["confidence"] >= config.HIGH_CONFIDENCE_THRESHOLD
    )
    warning = None
    if confidence_status == "LOW":
        warning = (
            "Image quality is insufficient for reliable OCR. "
            "The extracted text may be inaccurate. No information has "
            "been guessed or fabricated."
        )
    if is_partial_read:
        warning = (
            f"The OCR engine confidently read {best_result['text'].strip()!r}, "
            f"but that only spans about {coverage * 100:.0f}% of the crop's "
            "width - it likely captured a fragment of the field (e.g. one "
            "character out of several), not the whole thing. Treat this as "
            "an incomplete reading, not a confirmed full value. No "
            "information has been guessed or fabricated."
        )
    if not best_result["text"].strip() and result_is_from_vlm:
        warning = (
            "Both classical OCR and the vision-LLM fallback found this crop "
            "unreadable. No information has been guessed or fabricated."
        )

    print("OCR RESULT")
    print("-" * 10)
    print(f"Text: {best_result['text'] if best_result['text'] else '(no text detected)'}")
    if result_is_from_vlm:
        print(f"Confidence: {best_result['confidence'] * 100:.1f}% "
              f"(model self-estimate, not a calibrated OCR probability)")
    else:
        print(f"Confidence: {best_result['confidence'] * 100:.1f}% "
              f"(character-recognition confidence only)")
    print(f"Text coverage: {best_result.get('coverage', 1.0) * 100:.0f}% of crop width")
    print(f"Effective confidence (coverage-adjusted): {effective_confidence * 100:.1f}%")
    print(f"Status: {confidence_status} CONFIDENCE")
    print(f"Source: {best_source}\n")

    if warning:
        print("WARNING:")
        print(warning)
        print()


    result = {
        "input_image": input_image_path,
        "selected_region": {"x": x, "y": y, "width": w, "height": h},
        "quality": quality_report,
        "quality_note": (
            "Crop is sharp (not blurry) but low-resolution: too few pixels per "
            "character, not smearing. Deblurring cannot fix this; upscaling / "
            "native-resolution binarization and the vision-LLM fallback are the "
            "relevant strategies, and were prioritized for this run."
            if not quality_report["is_blurry"]
            and (quality_report.get("is_low_resolution") or quality_report.get("is_very_small_crop"))
            else None
        ),
        "preprocessing_applied": preprocessing_steps,
        "strategies_tried": attempted_candidates,
        "ocr_text": best_result["text"],
        "ocr_confidence": best_result["confidence"],
        "ocr_text_coverage": round(best_result.get("coverage", 1.0), 3),
        "effective_confidence": round(effective_confidence, 4),
        "is_partial_read": is_partial_read,
        "confidence_status": confidence_status,
        "result_source": best_source,
        "confidence_is_estimated": best_result.get("confidence_is_estimated", False),
        "vision_llm_fallback_attempted": vlm_attempted,
        "warning": warning,
    }

    save_result(result, config.RESULT_JSON_PATH)

    print("Saved:")
    print(f"{config.SELECTED_CROP_PATH}")
    if preprocessing_steps:
        print(f"{config.PROCESSED_CROP_PATH}")
    print(f"{config.RESULT_JSON_PATH}")
    print("\n" + "=" * 50)


    if show_windows:
        _show_preview_windows(cropped, preprocessing_steps)

    return result


def save_result(result: Dict[str, Any], output_path: str) -> None:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
    except OSError as exc:
        raise PipelineError(f"Failed to save result.json: {exc}") from exc


def _show_preview_windows(original_crop: np.ndarray, preprocessing_steps) -> None:
    try:
        cv2.imshow("Original Crop", original_crop)

        if preprocessing_steps and os.path.exists(config.PROCESSED_CROP_PATH):
            processed = cv2.imread(config.PROCESSED_CROP_PATH)
            if processed is not None:
                cv2.imshow("Processed Crop", processed)

        print("Press any key on an image window to close the preview...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as exc:


        logger.warning("Could not display preview windows: %s", exc)
