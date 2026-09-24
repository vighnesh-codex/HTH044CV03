"""
Document Contrast & Dynamic Range Module
Measures tonal distribution, standard deviation, and dynamic range
with clipping-aware confidence estimation.
"""

from typing import Dict, Any
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_contrast(image: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates tonal contrast across the document.

    Methodology:
    - Root-Mean-Square (RMS) Contrast: Calculated as standard deviation of gray levels.
    - Percentile Dynamic Spread: 95th percentile minus 5th percentile to resist outlier noise.
    - Confidence Formulation: High clipping at intensity extremes (0 or 255) skews the
      standard deviation. Confidence decreases when clipping ratio is excessive:
      clipped_fraction = (count(I < 5) + count(I > 250)) / N
      confidence = clip(1.0 - 0.75 * clipped_fraction, 0.35, 1.0)

    Returns:
        dict containing score (0-100), status, severity (0-100), confidence (0.0-1.0),
        and evidence metrics.
    """
    _, gray_image = preprocess_image(image)

    contrast = float(gray_image.std())

    # Original scoring logic preservation
    score = min(100, max(0, int(contrast * 2)))

    if score >= 50:
        status = "good"
    elif score >= 35:
        status = "moderate"
    else:
        status = "poor"

    severity = max(0, min(100, 100 - score))

    # Evidence: Percentile dynamic spread & Michelson contrast
    p5 = float(np.percentile(gray_image, 5))
    p95 = float(np.percentile(gray_image, 95))
    dynamic_spread = p95 - p5
    michelson = (p95 - p5) / (p95 + p5 + 1e-6)

    # Clipping detection for confidence
    clipped_pixels = int(np.count_nonzero(gray_image < 5) + np.count_nonzero(gray_image > 250))
    clipped_fraction = float(clipped_pixels / gray_image.size)
    confidence = min(1.0, max(0.35, 1.0 - 0.75 * clipped_fraction))

    return {
        "score": score,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 2),
        "evidence": {
            "contrast_std": round(contrast, 2),
            "dynamic_spread": round(dynamic_spread, 2),
            "michelson_contrast": round(michelson, 3),
            "clipped_ratio": round(clipped_fraction, 3)
        }
    }
