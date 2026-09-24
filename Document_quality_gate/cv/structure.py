"""
Document Structure & Layout Integrity Module
Performs non-semantic geometric inspection of document boundaries, margins,
illumination uniformity across quadrants, and border truncation.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from cv.preprocessing import preprocess_image


def analyze_structure(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes physical document structure, framing, margins, and illumination symmetry.

    Methodology:
    - Quadrant Illumination Non-Uniformity: Evaluates gradient across 4 document quadrants.
    - Border Truncation & Crop Anomaly: Checks if high-gradient text ink touches outer perimeter
      within 3 pixels, indicating accidental scanner or camera cutoff.
    - Margin estimation: Detects blank borders on left, right, top, and bottom.
    - Page Orientation & Aspect Ratio: Classifies portrait vs landscape vs non-standard aspect ratio.
    - Document Boundary: Identifies page substrate polygon contour.

    Returns:
        dict containing score (0-100), status, severity, alerts, and structural evidence.
    """
    _, gray_image = preprocess_image(image)
    h, w = gray_image.shape

    alerts: List[str] = []
    structural_penalties = 0

    # 1. Page Orientation & Aspect Ratio
    aspect_ratio = float(h / max(1, w))
    orientation = "Portrait" if aspect_ratio >= 1.05 else ("Landscape" if aspect_ratio <= 0.95 else "Square")

    # 2. Quadrant Illumination Analysis
    mid_y, mid_x = h // 2, w // 2
    quadrants = {
        "top_left": float(gray_image[:mid_y, :mid_x].mean()),
        "top_right": float(gray_image[:mid_y, mid_x:].mean()),
        "bottom_left": float(gray_image[mid_y:, :mid_x].mean()),
        "bottom_right": float(gray_image[mid_y:, mid_x:].mean())
    }
    q_vals = list(quadrants.values())
    illumination_spread = max(q_vals) - min(q_vals)
    illumination_mean = float(np.mean(q_vals))
    illumination_coefficient = illumination_spread / max(1.0, illumination_mean)

    if illumination_coefficient > 0.25:
        alerts.append("Uneven illumination detected across page quadrants.")
        structural_penalties += min(30, int(illumination_coefficient * 50))

    # 3. Border Truncation & Unusual Cropping Check
    # High-pass filter or threshold to see if ink touches the outer 3 pixels
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    border_thickness = max(2, min(5, int(min(h, w) * 0.01)))
    top_edge = binary[:border_thickness, :]
    bot_edge = binary[-border_thickness:, :]
    left_edge = binary[:, :border_thickness]
    right_edge = binary[:, -border_thickness:]

    edge_truncation_detected = False
    truncated_edges = []
    if (cv2.countNonZero(top_edge) / top_edge.size) > 0.12:
        edge_truncation_detected = True
        truncated_edges.append("top")
    if (cv2.countNonZero(bot_edge) / bot_edge.size) > 0.12:
        edge_truncation_detected = True
        truncated_edges.append("bottom")
    if (cv2.countNonZero(left_edge) / left_edge.size) > 0.12:
        edge_truncation_detected = True
        truncated_edges.append("left")
    if (cv2.countNonZero(right_edge) / right_edge.size) > 0.12:
        edge_truncation_detected = True
        truncated_edges.append("right")

    if edge_truncation_detected:
        alerts.append(f"Potential incomplete/cropped region detected near document boundary ({', '.join(truncated_edges)}).")
        structural_penalties += 20 * len(truncated_edges)

    # 4. Margin Estimation
    # Project horizontal and vertical ink profiles
    horiz_proj = np.sum(binary, axis=1) / 255.0
    vert_proj = np.sum(binary, axis=0) / 255.0

    ink_rows = np.where(horiz_proj > (0.01 * w))[0]
    ink_cols = np.where(vert_proj > (0.01 * h))[0]

    if len(ink_rows) > 0 and len(ink_cols) > 0:
        top_margin = int(ink_rows[0])
        bot_margin = int(h - ink_rows[-1])
        left_margin = int(ink_cols[0])
        right_margin = int(w - ink_cols[-1])
    else:
        top_margin, bot_margin, left_margin, right_margin = 0, 0, 0, 0

    # 5. Document Boundary Contour (Outer frame detection)
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_box = None
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_c)
        if area > 0.20 * (h * w):
            bx, by, bw, bh = cv2.boundingRect(largest_c)
            boundary_box = [int(bx), int(by), int(bw), int(bh)]

    score = max(10, min(100, 100 - structural_penalties))
    severity = max(0, min(100, 100 - score))

    if score >= 80:
        status = "good"
    elif score >= 55:
        status = "moderate"
    else:
        status = "poor"

    confidence = 0.85

    return {
        "score": score,
        "status": status,
        "severity": severity,
        "confidence": confidence,
        "alerts": alerts,
        "evidence": {
            "orientation": orientation,
            "aspect_ratio": round(aspect_ratio, 2),
            "quadrant_means": quadrants,
            "illumination_non_uniformity": round(illumination_coefficient, 3),
            "margins": {
                "top": top_margin,
                "bottom": bot_margin,
                "left": left_margin,
                "right": right_margin
            },
            "truncated_edges": truncated_edges,
            "boundary_box": boundary_box
        }
    }
