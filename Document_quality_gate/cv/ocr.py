"""
Tesseract OCR Sample Output & Downstream Impact Module
Extracts sample text, measures word-level confidence metrics, renders
confidence-colored word bounding box overlays, and assesses downstream OCR readiness.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


def is_tesseract_available() -> bool:
    """Checks whether pytesseract and the underlying tesseract binary are available."""
    if not HAS_PYTESSERACT:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_sample_ocr(
    image: np.ndarray,
    lang: str = "eng",
    psm: int = 3
) -> Dict[str, Any]:
    """
    Extracts sample OCR text and computes word-level recognition confidence metrics.

    Args:
        image: Input document image (BGR or Grayscale).
        lang: Tesseract language code (default 'eng').
        psm: Page segmentation mode (default 3 = Fully automatic page segmentation).

    Returns:
        Dictionary containing extracted text, word count, character count,
        average word confidence, confidence distribution, and readiness verdict.
    """
    if not is_tesseract_available():
        return {
            "available": False,
            "success": False,
            "error": "Tesseract OCR engine is not installed or not in system PATH.",
            "extracted_text": "",
            "word_count": 0,
            "character_count": 0,
            "average_confidence": 0.0,
            "ocr_readiness": "UNAVAILABLE",
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "words": []
        }

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "available": True,
            "success": False,
            "error": "Invalid image array provided for OCR.",
            "extracted_text": "",
            "word_count": 0,
            "character_count": 0,
            "average_confidence": 0.0,
            "ocr_readiness": "FAILED",
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "words": []
        }

    # Convert to RGB for pytesseract
    if image.ndim == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    config = f"--psm {psm}"

    try:
        # 1. Full text extraction
        full_text = pytesseract.image_to_string(rgb_image, lang=lang, config=config).strip()

        # 2. Word-level bounding boxes and confidence data
        data = pytesseract.image_to_data(rgb_image, lang=lang, config=config, output_type=pytesseract.Output.DICT)

        words_data: List[Dict[str, Any]] = []
        confs: List[float] = []
        n_items = len(data.get("text", []))

        high_conf = 0
        med_conf = 0
        low_conf = 0

        for i in range(n_items):
            word_str = data["text"][i].strip()
            conf_val = float(data["conf"][i])

            # Tesseract reports conf=-1 for non-word blocks
            if conf_val >= 0 and word_str:
                confs.append(conf_val)
                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                if conf_val >= 80.0:
                    high_conf += 1
                elif conf_val >= 50.0:
                    med_conf += 1
                else:
                    low_conf += 1

                words_data.append({
                    "word": word_str,
                    "confidence": round(conf_val, 1),
                    "bbox": [x, y, w, h]
                })

        total_words = len(words_data)
        avg_conf = round(float(np.mean(confs)), 1) if confs else 0.0

        # Classify Downstream OCR Readiness
        if total_words == 0:
            ocr_readiness = "FAILED"
            readiness_desc = "No readable text detected. Optical blur, low contrast, or blank image prevents recognition."
        elif avg_conf >= 85.0:
            ocr_readiness = "OPTIMAL"
            readiness_desc = f"Document is optimal for automated OCR intake (Average confidence: {avg_conf:.1f}%)."
        elif avg_conf >= 65.0:
            ocr_readiness = "ACCEPTABLE"
            readiness_desc = f"Document has acceptable OCR readability ({avg_conf:.1f}%), minor word errors possible."
        else:
            ocr_readiness = "DEGRADED"
            readiness_desc = f"OCR recognition is degraded ({avg_conf:.1f}%). Preprocessing or rescan strongly recommended."

        snippet = full_text[:250] + ("..." if len(full_text) > 250 else "")

        return {
            "available": True,
            "success": True,
            "extracted_text": full_text,
            "preview_snippet": snippet,
            "word_count": total_words,
            "character_count": len(full_text),
            "average_confidence": avg_conf,
            "ocr_readiness": ocr_readiness,
            "readiness_description": readiness_desc,
            "confidence_distribution": {
                "high": high_conf,
                "medium": med_conf,
                "low": low_conf
            },
            "words": words_data
        }

    except Exception as e:
        return {
            "available": True,
            "success": False,
            "error": str(e),
            "extracted_text": "",
            "word_count": 0,
            "character_count": 0,
            "average_confidence": 0.0,
            "ocr_readiness": "ERROR",
            "readiness_description": f"OCR execution failed: {str(e)}",
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "words": []
        }


def generate_ocr_word_overlay(
    image: np.ndarray,
    ocr_result: Dict[str, Any]
) -> np.ndarray:
    """
    Renders an overlay on the document image displaying word bounding boxes
    color-coded by Tesseract recognition confidence:
      - Green: High confidence (>= 80%)
      - Orange: Medium confidence (50-79%)
      - Red: Low confidence (< 50%)
    """
    if image is None or not ocr_result.get("words"):
        return image.copy() if image is not None else np.zeros((300, 300, 3), dtype=np.uint8)

    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for item in ocr_result["words"]:
        x, y, w, h = item["bbox"]
        conf = item["confidence"]

        if conf >= 80.0:
            color = (0, 200, 0)       # Green (high confidence)
        elif conf >= 50.0:
            color = (0, 165, 255)     # Orange (medium confidence)
        else:
            color = (0, 0, 240)       # Red (low confidence)

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)

    return overlay


def compare_ocr_impact(
    raw_image: np.ndarray,
    restored_image: np.ndarray
) -> Dict[str, Any]:
    """
    Compares downstream OCR readability before and after image quality restoration.
    Measures word recovery delta and average confidence gains.
    """
    ocr_raw = extract_sample_ocr(raw_image)
    ocr_restored = extract_sample_ocr(restored_image)

    raw_words = ocr_raw.get("word_count", 0)
    restored_words = ocr_restored.get("word_count", 0)
    raw_conf = ocr_raw.get("average_confidence", 0.0)
    restored_conf = ocr_restored.get("average_confidence", 0.0)

    word_delta = restored_words - raw_words
    conf_delta = round(restored_conf - raw_conf, 1)

    return {
        "raw_word_count": raw_words,
        "restored_word_count": restored_words,
        "word_delta": word_delta,
        "raw_confidence": raw_conf,
        "restored_confidence": restored_conf,
        "confidence_delta": conf_delta,
        "ocr_raw": ocr_raw,
        "ocr_restored": ocr_restored,
        "impact_summary": (
            f"Restoration recovered {word_delta:+d} readable words with a "
            f"{conf_delta:+0.1f}% change in recognition confidence."
        )
    }
