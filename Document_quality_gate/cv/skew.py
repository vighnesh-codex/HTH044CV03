"""
Document Skew & Orientation Alignment Module
Estimates geometric tilt angle using Probabilistic Hough Line Transform
with statistical dispersion and support-line confidence estimation.
"""

from typing import Dict, Any, List
import cv2
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_skew(image: np.ndarray) -> Dict[str, Any]:
    """
    Measures document skew angle and alignment quality.

    Methodology:
    - Canny Edge Detection with gradient hysteresis (50, 150).
    - Probabilistic Hough Transform (HoughLinesP) detecting linear text baselines.
    - Angle filtering within [-45, 45] degrees to capture dominant horizontal orientation.
    - Median angle calculation for outlier rejection.
    - Confidence Formulation: High sample size (more detected lines) combined with
      low angular variance indicates high confidence in skew estimation:
      confidence = clip((num_lines / 15.0) * (1.0 / (1.0 + 0.15 * std_dev)), 0.35, 1.0).

    Returns:
        dict containing angle, score (0-100), status, severity (0-100), confidence,
        and evidence lines for visualization.
    """
    _, gray_image = preprocess_image(image)
    edges = cv2.Canny(gray_image, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    angles: List[float] = []
    line_segments: List[List[int]] = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = [int(v) for v in line.reshape(-1)[:4]]
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Normalize angle to near-horizontal domain
            if -45 < angle < 45:
                angles.append(angle)
                line_segments.append([x1, y1, x2, y2])

    if angles:
        angle = float(np.median(angles))
        angle_std = float(np.std(angles))
    else:
        angle = 0.0
        angle_std = 0.0

    abs_angle = abs(angle)

    # Score calculation matching backend_modules/scoring.py
    if abs_angle <= 1.0:
        score = 100
        status = "good"
    elif abs_angle <= 3.0:
        score = 85
        status = "warning"
    elif abs_angle <= 5.0:
        score = 70
        status = "warning"
    elif abs_angle <= 8.0:
        score = 50
        status = "poor"
    else:
        score = 20
        status = "poor"

    # Severity: 0 at 0 deg, 100 at >= 8 deg
    severity = min(100, int(abs_angle * 12.5))

    # Confidence calculation
    if len(angles) >= 3:
        dispersion_factor = 1.0 / (1.0 + 0.15 * angle_std)
        line_count_factor = min(1.0, len(angles) / 15.0)
        confidence = min(1.0, max(0.40, line_count_factor * dispersion_factor))
    else:
        confidence = 0.35

    return {
        "angle": round(angle, 2),
        "score": score,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 2),
        "evidence": {
            "lines_detected": len(angles),
            "angle_variance": round(angle_std ** 2, 3),
            "line_segments": line_segments[:30]  # Store representative segments for overlay
        }
    }
