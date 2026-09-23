
import logging
from typing import Dict, Any, List, Optional

import numpy as np

from src.config import OCR_LANGUAGES, OCR_USE_GPU, ENABLE_TESSERACT_FALLBACK

logger = logging.getLogger(__name__)


def _text_coverage(detections: List[Dict[str, Any]], image_width: float) -> float:
    if not detections or image_width <= 0:
        return 0.0
    min_x = min(min(p[0] for p in d["box"]) for d in detections)
    max_x = max(max(p[0] for p in d["box"]) for d in detections)
    return float(max(0.0, min(1.0, (max_x - min_x) / image_width)))


def _length_based_coverage(text: str, image_width: float, image_height: float) -> float:
    text = text.strip()
    if not text or image_width <= 0 or image_height <= 0:
        return 0.0


    assumed_char_width = max(image_height * 0.5, 1.0)
    expected_chars = max(image_width / assumed_char_width, 1.0)
    return float(max(0.0, min(1.0, len(text) / expected_chars)))


def score_result(result: Dict[str, Any]) -> float:
    confidence = result.get("confidence", 0.0)
    coverage = result.get("coverage", 1.0)
    return confidence * (0.3 + 0.7 * coverage)


class OCREngineError(Exception):
    pass


class OCREngine:

    def __init__(self, languages: Optional[List[str]] = None, use_gpu=OCR_USE_GPU):
        self.languages = languages or OCR_LANGUAGES
        self.use_gpu = self._resolve_gpu_setting(use_gpu)
        self._reader = None  


        self._easyocr_unavailable = False

    @staticmethod
    def _resolve_gpu_setting(use_gpu) -> bool:
        if use_gpu != "auto":
            return bool(use_gpu)

        try:
            import torch
            available = torch.cuda.is_available()
            logger.info("GPU auto-detect: CUDA available = %s", available)
            return available
        except ImportError:
            return False

    def _get_reader(self):
        if self._easyocr_unavailable:
            raise OCREngineError(
                "EasyOCR is not installed. Run: pip install easyocr"
            )

        if self._reader is None:
            try:
                import easyocr  


            except ImportError as exc:
                self._easyocr_unavailable = True
                raise OCREngineError(
                    "EasyOCR is not installed. Run: pip install easyocr"
                ) from exc

            try:
                logger.info(
                    "Initializing EasyOCR reader (languages=%s, gpu=%s)...",
                    self.languages,
                    self.use_gpu,
                )
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as exc:  
                raise OCREngineError(f"Failed to initialize EasyOCR: {exc}") from exc

        return self._reader

    def recognize(self, image: np.ndarray, sensitive: bool = False) -> Dict[str, Any]:
        attempts = []

        for method_name, fn, kwargs in [
            ("region_recognition", self._recognize_region, {}),
            ("region_recognition_sensitive", self._recognize_region, {"sensitive": True}),
            ("detection_readtext", self._recognize_detection, {}),
            ("detection_readtext_sensitive", self._recognize_detection, {"sensitive": True}),
        ]:
            try:
                result = fn(image, **kwargs)
            except OCREngineError as exc:
                logger.warning("OCR method '%s' raised an error: %s", method_name, exc)
                continue

            result["method"] = method_name
            attempts.append(result)

        if ENABLE_TESSERACT_FALLBACK:
            tesseract_result = self._recognize_tesseract(image)
            if tesseract_result is not None:
                tesseract_result["method"] = "tesseract"
                attempts.append(tesseract_result)

        if not attempts:
            raise OCREngineError("All OCR recognition strategies failed to run.")

        non_empty = [a for a in attempts if a["text"].strip()]
        if non_empty:
            return max(non_empty, key=score_result)


        return max(attempts, key=score_result)

    def _format_results(self, raw_results, image_width: float) -> Dict[str, Any]:
        if not raw_results:
            return {"text": "", "confidence": 0.0, "detections": [], "raw_results": [], "coverage": 0.0}

        detections = []
        for box, text, conf in raw_results:
            detections.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "box": [[float(p[0]), float(p[1])] for p in box],
                }
            )

        combined_text = combine_text_reading_order(detections)
        overall_confidence = sum(d["confidence"] for d in detections) / len(detections)

        return {
            "text": combined_text,
            "confidence": round(float(overall_confidence), 4),
            "detections": detections,
            "raw_results": raw_results,
            "coverage": _text_coverage(detections, image_width),
        }

    def _recognize_region(self, image: np.ndarray, sensitive: bool = False) -> Dict[str, Any]:
        reader = self._get_reader()

        kwargs: Dict[str, Any] = {"detail": 1, "paragraph": False}
        if sensitive:
            kwargs.update(contrast_ths=0.05, adjust_contrast=0.7)

        try:
            raw_results = reader.recognize(image, **kwargs)
        except Exception as exc:
            raise OCREngineError(f"Region-based OCR recognition failed: {exc}") from exc

        result = self._format_results(raw_results, image.shape[1])
        result["coverage"] = _length_based_coverage(result["text"], image.shape[1], image.shape[0])
        return result

    def _recognize_tesseract(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            return None

        try:
            data = pytesseract.image_to_data(
                image, config="--psm 7", output_type=Output.DICT
            )
        except Exception as exc:
            logger.info("Tesseract OCR unavailable or failed: %s", exc)
            return None

        detections = []
        for i in range(len(data.get("text", []))):
            text = data["text"][i].strip()
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < 0:
                continue

            left, top = data["left"][i], data["top"][i]
            width, height = data["width"][i], data["height"][i]
            box = [
                [float(left), float(top)],
                [float(left + width), float(top)],
                [float(left + width), float(top + height)],
                [float(left), float(top + height)],
            ]
            detections.append({"text": text, "confidence": conf / 100.0, "box": box})

        if not detections:
            return {"text": "", "confidence": 0.0, "detections": [], "raw_results": data, "coverage": 0.0}

        combined_text = combine_text_reading_order(detections)
        overall_confidence = sum(d["confidence"] for d in detections) / len(detections)

        return {
            "text": combined_text,
            "confidence": round(float(overall_confidence), 4),
            "detections": detections,
            "raw_results": data,
            "coverage": _text_coverage(detections, image.shape[1]),
        }

    def _recognize_detection(self, image: np.ndarray, sensitive: bool = False) -> Dict[str, Any]:
        reader = self._get_reader()

        kwargs: Dict[str, Any] = {}
        if sensitive:
            kwargs = {
                "text_threshold": 0.5,
                "low_text": 0.3,
                "contrast_ths": 0.05,
                "adjust_contrast": 0.7,
                "mag_ratio": 2.0,
            }

        try:
            raw_results = reader.readtext(image, **kwargs)
        except Exception as exc:
            raise OCREngineError(f"Detection-based OCR recognition failed: {exc}") from exc

        return self._format_results(raw_results, image.shape[1])


def combine_text_reading_order(detections: List[Dict[str, Any]]) -> str:
    def sort_key(det: Dict[str, Any]):
        box = det["box"]
        top_left_y = min(point[1] for point in box)
        top_left_x = min(point[0] for point in box)


        return (round(top_left_y / 10.0), top_left_x)

    ordered = sorted(detections, key=sort_key)
    return " ".join(det["text"] for det in ordered).strip()
