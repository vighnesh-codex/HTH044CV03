"""
Document Blur & Sharpness Detection Module
Computes global and frequency-domain focus metrics including Laplacian variance
and Tenengrad gradient energy with mathematically grounded confidence estimation.
"""

from typing import Dict, Any
import cv2
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_blur(image: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates document image sharpness and blur defect severity.

    Methodology:
    - Laplacian Operator: Computes the 2nd order spatial derivative variance.
      Sharp edges yield high local variance; optical/motion blur diffuses edges,
      reducing variance.
    - Tenengrad Measure: Computes gradient magnitude variance via Sobel operators.
    - Confidence Formulation: Laplacian variance is mathematically dependent on image
      contrast. Under very low contrast (std < 15), variance is suppressed regardless
      of focus. Confidence evaluates signal sufficiency:
      confidence = clip(gray_std / 45.0, 0.25, 1.0).

    Returns:
        dict containing score (0-100), status, severity (0-100), confidence (0.0-1.0),
        and evidence metrics.
    """
    _, gray_image = preprocess_image(image)

    # 1. Laplacian Variance (Sharpness)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Original scoring preservation
    score = min(100, max(0, int(sharpness / 10)))

    if score >= 50:
        status = "good"
    elif score >= 30:
        status = "moderate"
    else:
        status = "poor"

    # Severity: Inverse of sharpness quality (100 = completely blurry, 0 = crisp)
    severity = max(0, min(100, 100 - score))

    # 2. Tenengrad Gradient Energy (Sobel)
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx ** 2 + gy ** 2
    tenengrad = float(np.mean(grad_mag_sq))

    # 3. Confidence: Measurement stability based on dynamic contrast sufficiency
    gray_std = float(gray_image.std())
    # If dynamic range is healthy (>45), confidence in sharpness measurement is high (>0.95)
    confidence = min(1.0, max(0.25, gray_std / 45.0))

    return {
        "score": score,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 2),
        "evidence": {
            "laplacian_variance": round(sharpness, 2),
            "tenengrad_energy": round(tenengrad, 2),
            "contrast_std": round(gray_std, 2)
        }
    }
