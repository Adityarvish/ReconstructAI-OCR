
import base64
import logging
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src import config

logger = logging.getLogger(__name__)


_QUALITATIVE_CONFIDENCE_MAP = {
    "HIGH": 0.95,
    "MEDIUM": 0.75,
    "LOW": 0.40,
}

_SYSTEM_PROMPT = """You are a strict OCR transcription assistant. You will be shown a small, \
cropped image of a single text field (e.g. an account number, name, date, or amount) taken \
from a document or screenshot. Some crops are blurry, compressed, or very low resolution.

Your ONLY job is to transcribe exactly what is visibly printed in the image. Rules:
- Transcribe only characters you can actually see. Never guess, autocomplete, or infer a \
character from context, "typical formats", or what would "make sense".
- If a specific character is genuinely too degraded to identify, output "?" in its place - \
do not silently drop it and do not guess a plausible replacement.
- If NONE of the text in the crop is legible enough to transcribe, respond with exactly: \
UNREADABLE
- Preserve the exact character case, spacing, punctuation, and any leading zeros you can see.
- Do not correct spelling, do not reformat numbers, do not add words that are not visible.
- Do not add commentary, explanations, quotes, or markdown. Respond with EXACTLY two lines: \
line 1 is the transcription (or the single word UNREADABLE), line 2 is your own confidence \
rating - one of HIGH, MEDIUM, or LOW - reflecting how legible the ORIGINAL characters were to \
you, not how confident you are in a guess."""


class VLMOCRError(Exception):
    pass


def _encode_image(image: np.ndarray) -> Optional[str]:
    try:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        success, buffer = cv2.imencode(".png", image)
        if not success:
            return None
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        logger.warning("Could not encode image for vision-LLM OCR: %s", exc)
        return None


def is_available() -> bool:
    if not config.ENABLE_VLM_FALLBACK:
        return False
    if not os.environ.get(config.VLM_API_KEY_ENV):
        return False
    try:
        import openai  
    except ImportError:
        return False
    return True


def recognize_with_vlm(image: np.ndarray) -> Optional[Dict[str, Any]]:
    if not config.ENABLE_VLM_FALLBACK:
        return None

    api_key = os.environ.get(config.VLM_API_KEY_ENV)
    if not api_key:
        logger.info(
            "Vision-LLM OCR fallback skipped: no %s environment variable set. "
            "Get a free key at https://openrouter.ai/keys",
            config.VLM_API_KEY_ENV,
        )
        return None

    try:
        import openai
    except ImportError:
        logger.info(
            "Vision-LLM OCR fallback skipped: 'openai' package not "
            "installed (pip install openai). OpenRouter is OpenAI-API-"
            "compatible, so no separate SDK is needed."
        )
        return None

    encoded = _encode_image(image)
    if encoded is None:
        return None

    actual_model = config.VLM_MODEL  
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.VLM_API_BASE_URL)
        response = client.chat.completions.create(
            model=config.VLM_MODEL,
            temperature=0,
            max_tokens=config.VLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe this crop following the system rules exactly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                },
            ],
        )


        actual_model = getattr(response, "model", None) or config.VLM_MODEL
    except Exception as exc:


        logger.warning("Vision-LLM OCR call failed: %s", exc)
        return None

    try:
        raw_text = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Could not parse vision-LLM OCR response: %s", exc)
        return None

    if not raw_text:


        message = response.choices[0].message
        reasoning_fallback = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        finish_reason = getattr(response.choices[0], "finish_reason", "unknown")
        if reasoning_fallback and finish_reason != "length":


            raw_text = reasoning_fallback.strip()
            logger.info(
                "Vision-LLM OCR: model '%s' returned empty `content`; recovered "
                "an answer from its reasoning field instead.",
                actual_model,
            )
        else:
            logger.warning(
                "Vision-LLM OCR: model '%s' returned an empty response "
                "(finish_reason=%s%s). This usually means the router picked "
                "a reasoning model that used up its token budget "
                "(VLM_MAX_TOKENS=%d) on internal reasoning before writing an "
                "answer, or the model declined to answer. Keeping the "
                "classical OCR result.",
                actual_model,
                finish_reason,
                " - truncated mid-reasoning, discarding it as unreliable"
                if reasoning_fallback else "",
                config.VLM_MAX_TOKENS,
            )
            return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    transcription = lines[0] if lines else ""
    qualitative_confidence = lines[1].upper() if len(lines) > 1 else "MEDIUM"
    qualitative_confidence = (
        qualitative_confidence if qualitative_confidence in _QUALITATIVE_CONFIDENCE_MAP else "MEDIUM"
    )

    if transcription.upper() == "UNREADABLE":
        return {
            "text": "",
            "confidence": 0.0,
            "qualitative_confidence": "LOW",
            "detections": [],
            "raw_results": raw_text,
            "coverage": 0.0,
            "method": "vision_llm",
            "confidence_is_estimated": True,
            "model": actual_model,
        }

    return {
        "text": transcription,
        "confidence": _QUALITATIVE_CONFIDENCE_MAP[qualitative_confidence],
        "qualitative_confidence": qualitative_confidence,
        "detections": [],
        "raw_results": raw_text,


        "coverage": 1.0,
        "method": "vision_llm",
        "confidence_is_estimated": True,
        "model": actual_model,
    }
