"""
Document Completeness & Missing Section Detection Module
Analyzes functional document zones (header, upper, middle, bottom)
to detect blank or missing regions with coordinate-aware visual evidence.
"""

from typing import Dict, Any, List
import cv2
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_completeness(image: np.ndarray) -> Dict[str, Any]:
    """
    Checks whether required functional zones contain readable content.

    Methodology:
    - Partition document into four vertical structural zones:
      header (0-20%), upper (20-40%), middle (40-75%), bottom (75-100%).
    - Inverted binary thresholding with margin exclusion to avoid border shadows.
    - Content ink density: Ratio of non-zero ink pixels to area.
      Regions with < 3% ink density are classified as missing / omitted.
    - Confidence Formulation: High contrast between background and text increases
      confidence in blank section verification:
      confidence = clip(0.70 + 0.30 * (contrast_std / 50.0), 0.50, 1.0).

    Returns:
        dict containing score (0-100), missing (list of names), severity,
        status, confidence, and bounding box coordinates of missing sections.
    """
    _, gray_image = preprocess_image(image)
    height, width = gray_image.shape

    zone_bounds = {
        "header": (0, int(height * 0.20)),
        "upper": (int(height * 0.20), int(height * 0.40)),
        "middle": (int(height * 0.40), int(height * 0.75)),
        "bottom": (int(height * 0.75), height)
    }

    missing: List[str] = []
    region_ratios: Dict[str, float] = {}
    missing_boxes: List[List[int]] = []

    for name, (y1, y2) in zone_bounds.items():
        region = gray_image[y1:y2, :]
        if region.size == 0:
            missing.append(name)
            missing_boxes.append([0, y1, width, y2 - y1])
            continue

        _, binary = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY_INV)

        border = 10
        if region.shape[0] > border * 2 and region.shape[1] > border * 2:
            binary = binary[border:-border, border:-border]

        content_ratio = float(cv2.countNonZero(binary) / binary.size)
        region_ratios[name] = round(content_ratio, 4)

        if content_ratio < 0.03:
            missing.append(name)
            missing_boxes.append([0, y1, width, y2 - y1])

    total_regions = len(zone_bounds)
    completed_regions = total_regions - len(missing)
    score = int((completed_regions / total_regions) * 100)

    status = "good" if score >= 80 else "poor"
    severity = max(0, min(100, 100 - score))

    contrast_std = float(gray_image.std())
    confidence = min(1.0, max(0.50, 0.70 + 0.30 * min(1.0, contrast_std / 50.0)))

    return {
        "score": score,
        "missing": missing,
        "severity": severity,
        "status": status,
        "confidence": round(confidence, 2),
        "evidence": {
            "region_ratios": region_ratios,
            "missing_boxes": missing_boxes
        }
    }


def analyze_missing_section(image: np.ndarray) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for modules.missing_section.
    """
    return analyze_completeness(image)
