"""
Forensic Document Integrity & Compression Discrepancy Module
Performs Error Level Analysis (ELA) and local high-frequency compression gradient
inspection to detect digital splicing, pasted signatures, and localized tampering.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def analyze_document_forensics(
    image: np.ndarray,
    jpeg_quality: int = 90
) -> Dict[str, Any]:
    """
    Evaluates compression homogeneity across the document substrate using Error Level Analysis (ELA).

    Methodology:
    - In-memory recompression: Re-encodes the image at a controlled 90% JPEG quality.
    - Error differential: Computes absolute pixel differences between original and recompressed frames.
    - Scale expansion: Multiplies residuals by an adaptive factor to expose compression artifacts.
    - Block dispersion: Measures standard deviation of error levels across 4x4 spatial blocks.
      High spatial variance signifies disparate compression histories (e.g. spliced signatures or inserted stamps).
    """
    bgr, gray, _ = validate_and_preprocess(image)
    h, w = bgr.shape[:2]

    # 1. In-memory JPEG recompression
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, buffer = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        return {
            "tamper_risk": 0.0,
            "uniformity_score": 100,
            "status": "ANALYSIS_UNAVAILABLE",
            "diagnosis": "Unable to execute in-memory JPEG compression.",
            "ela_overlay": bgr.copy(),
            "evidence": {}
        }

    recompressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # 2. Compute absolute difference (ELA residual)
    diff = cv2.absdiff(bgr, recompressed).astype(np.float32)

    # Adaptive scale multiplier based on peak error
    max_diff = float(np.max(diff))
    scale_factor = 255.0 / max(1.0, max_diff) if max_diff > 0 else 15.0
    scale_factor = min(25.0, max(8.0, scale_factor))

    ela_scaled = np.clip(diff * scale_factor, 0, 255).astype(np.uint8)
    ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_BGR2GRAY)

    # 3. Dense ELA Heatmap (Inferno Colormap)
    ela_heatmap = cv2.applyColorMap(ela_gray, cv2.COLORMAP_INFERNO)
    ela_overlay = cv2.addWeighted(bgr, 0.55, ela_heatmap, 0.45, 0)

    # 4. Spatial Block Variance Analysis (4x4 Grid)
    rows, cols = 4, 4
    h_step, w_step = h // rows, w // cols
    block_means = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * h_step
            y2 = h if r == rows - 1 else (r + 1) * h_step
            x1 = c * w_step
            x2 = w if c == cols - 1 else (c + 1) * w_step

            block = ela_gray[y1:y2, x1:x2]
            if block.size > 0:
                block_means.append(float(block.mean()))

    mean_ela = float(np.mean(block_means)) if block_means else 0.0
    ela_spatial_std = float(np.std(block_means)) if block_means else 0.0

    # Tamper Risk Formulation (0 - 100)
    # High spatial standard deviation in ELA residuals signifies local compression anomalies
    tamper_risk = min(100.0, ela_spatial_std * 8.5)
    uniformity_score = max(0.0, min(100.0, 100.0 - tamper_risk))

    if tamper_risk < 25.0:
        status = "AUTHENTIC_UNIFORM"
        diagnosis = "Compression surface is homogeneous across the document canvas."
    elif tamper_risk < 55.0:
        status = "MODERATE_DISPERSION"
        diagnosis = "Minor localized compression variance observed; typical for hybrid print/scan documents."
    else:
        status = "SUSPICIOUS_INCONSISTENCY"
        diagnosis = "Significant localized compression gradient detected - potential spliced signature or digital stamp."

    return {
        "tamper_risk": round(tamper_risk, 1),
        "uniformity_score": round(uniformity_score, 1),
        "status": status,
        "diagnosis": diagnosis,
        "ela_overlay": ela_overlay,
        "evidence": {
            "mean_error_level": round(mean_ela, 2),
            "spatial_dispersion_std": round(ela_spatial_std, 2),
            "scale_factor_applied": round(scale_factor, 1),
            "block_grid": f"{rows}x{cols}"
        }
    }
