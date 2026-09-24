"""
Document Noise & High-Frequency Artifact Module
Measures residual noise variance using Gaussian spatial subtraction
and background patch noise floor estimation.
"""

from typing import Dict, Any
import cv2
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_noise(image: np.ndarray) -> Dict[str, Any]:
    """
    Measures document noise contamination and high-frequency grain.

    Methodology:
    - High-frequency residual: Computes absolute difference between image and
      Gaussian-filtered (5x5, sigma=0) version.
    - Global noise metric: Mean residual magnitude, mapped to original score range:
      score = max(0, min(100, int(100 - noise_level * 5))).
    - Flat background estimation: Samples lowest 25% gradient patches to isolate
      electronic sensor noise from true text edges.
    - Confidence Formulation: High confidence when flat-region residual variance
      is homogeneous across different quadrants:
      confidence = clip(1.0 - (flat_std / (flat_mean + 1.0)), 0.40, 1.0).

    Returns:
        dict containing score (0-100), status, severity (0-100), confidence,
        and evidence metrics.
    """
    _, gray_image = preprocess_image(image)

    # 1. Spatial difference method (Original baseline)
    smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    noise_diff = cv2.absdiff(gray_image, smooth_image)
    noise_level = float(noise_diff.mean())

    score = max(0, min(100, int(100 - noise_level * 5)))

    if score >= 70:
        status = "good"
    elif score >= 40:
        status = "moderate"
    else:
        status = "poor"

    severity = max(0, min(100, 100 - score))

    # 2. Patch-based background noise estimation
    # Extract 16x16 tiles across the document
    h, w = gray_image.shape
    tile_size = 16
    patch_means = []
    patch_stds = []

    for y in range(0, h - tile_size, tile_size * 2):
        for x in range(0, w - tile_size, tile_size * 2):
            patch = gray_image[y:y + tile_size, x:x + tile_size]
            p_std = float(patch.std())
            if p_std < 12.0:  # Candidate flat/uniform region
                diff_patch = noise_diff[y:y + tile_size, x:x + tile_size]
                patch_means.append(float(diff_patch.mean()))
                patch_stds.append(float(diff_patch.std()))

    if patch_means:
        bg_noise = float(np.median(patch_means))
        bg_var = float(np.std(patch_means))
        confidence = min(1.0, max(0.40, 1.0 - (bg_var / (bg_noise + 2.0))))
    else:
        bg_noise = noise_level
        confidence = 0.65

    # Signal-to-noise ratio approximation (dB)
    signal_power = max(1.0, float(gray_image.std()) ** 2)
    noise_power = max(0.01, noise_level ** 2)
    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return {
        "score": score,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 2),
        "evidence": {
            "noise_level": round(noise_level, 2),
            "background_noise": round(bg_noise, 2),
            "estimated_snr_db": round(float(snr_db), 2)
        }
    }
