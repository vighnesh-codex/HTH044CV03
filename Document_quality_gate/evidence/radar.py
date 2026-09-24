"""
Quality Radar & Spider Profile Visualization Module
Renders high-resolution multi-axial polar quality diagrams illustrating
the multi-dimensional defect footprint against optimal quality boundaries.
"""

from typing import Dict, Any, List
import math
import cv2
import numpy as np


def generate_quality_radar_chart(
    defects: Dict[str, Any],
    quality_index: float,
    size: int = 560
) -> np.ndarray:
    """
    Renders an authentic Computer Vision polar polygon radar chart.

    Axes:
    1. Sharpness / Focus (Blur Score)
    2. Tonal Contrast (Contrast Score)
    3. Alignment (Skew Score)
    4. Cleanliness (Noise Score)
    5. Non-Interference (100 - Handwriting Severity)
    6. Completeness (Functional Zone Score)
    """
    # Create dark slate canvas
    canvas = np.full((size, size, 3), 16, dtype=np.uint8)  # Deep obsidian #101010
    # Gradient background tint
    center = (size // 2, size // 2)
    max_radius = int(size * 0.36)

    # 6 Axis metrics (0-100)
    blur_score = float(defects["blur"]["score"])
    contrast_score = float(defects["contrast"]["score"])

    # Skew score
    skew_val = defects["skew"].get("score")
    if skew_val is None:
        sk_angle = abs(defects["skew"].get("angle", 0))
        skew_val = 100 if sk_angle <= 1 else (85 if sk_angle <= 3 else (70 if sk_angle <= 5 else 30))
    skew_score = float(skew_val)

    noise_score = float(defects["noise"]["score"])

    hw_detected = defects["handwriting"].get("detected", False)
    hw_conf = float(defects["handwriting"].get("confidence", 0.0))
    hw_clearance = max(0.0, 100.0 - (hw_conf * 100.0 if hw_detected else hw_conf * 40.0))

    comp_score = float(defects.get("missing_section", {}).get("score", 100.0))

    axes = [
        ("Sharpness", blur_score),
        ("Contrast", contrast_score),
        ("Alignment", skew_score),
        ("Cleanliness", noise_score),
        ("Clean Ink", hw_clearance),
        ("Completeness", comp_score)
    ]
    num_axes = len(axes)
    angle_step = 2.0 * math.pi / num_axes

    # 1. Draw Concentric Polar Grid Rings (20, 40, 60, 80, 100)
    for level in [20, 40, 60, 80, 100]:
        r = int(max_radius * (level / 100.0))
        pts = []
        for i in range(num_axes):
            theta = i * angle_step - math.pi / 2.0
            x = int(center[0] + r * math.cos(theta))
            y = int(center[1] + r * math.sin(theta))
            pts.append([x, y])
        pts_arr = np.array([pts], dtype=np.int32)
        cv2.polylines(canvas, pts_arr, True, (45, 50, 60), 1, cv2.LINE_AA)
        # Ring percentage label
        cv2.putText(
            canvas,
            f"{level}",
            (center[0] + 5, center[1] - r + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (90, 100, 115),
            1,
            cv2.LINE_AA
        )

    # 2. Draw Spokes and Axis Labels
    data_pts = []
    for i, (label, val) in enumerate(axes):
        theta = i * angle_step - math.pi / 2.0
        # Spoke line
        sx = int(center[0] + max_radius * math.cos(theta))
        sy = int(center[1] + max_radius * math.sin(theta))
        cv2.line(canvas, center, (sx, sy), (55, 65, 80), 1, cv2.LINE_AA)

        # Data point
        clamped_val = max(5.0, min(100.0, val))
        r_val = int(max_radius * (clamped_val / 100.0))
        dx = int(center[0] + r_val * math.cos(theta))
        dy = int(center[1] + r_val * math.sin(theta))
        data_pts.append([dx, dy])

        # Axis text label
        tx = int(center[0] + (max_radius + 38) * math.cos(theta))
        ty = int(center[1] + (max_radius + 38) * math.sin(theta))

        # Color code label based on metric health
        lbl_color = (60, 220, 120) if val >= 75 else ((40, 180, 240) if val >= 50 else (60, 70, 240))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        (vw, vh), _ = cv2.getTextSize(f"{val:.0f}", cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

        cv2.putText(canvas, label, (tx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:.0f}", (tx - vw // 2, ty + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, lbl_color, 1, cv2.LINE_AA)

    # 3. Draw Filled Data Polygon
    overlay = canvas.copy()
    data_arr = np.array([data_pts], dtype=np.int32)

    # Color scheme depends on Quality Index
    if quality_index >= 85:
        poly_color = (46, 204, 113)   # Emerald
        fill_color = (30, 140, 80)
    elif quality_index >= 60:
        poly_color = (241, 196, 15)   # Amber
        fill_color = (160, 120, 10)
    else:
        poly_color = (231, 76, 60)    # Crimson
        fill_color = (150, 40, 30)

    cv2.fillPoly(overlay, data_arr, fill_color)
    canvas = cv2.addWeighted(canvas, 0.65, overlay, 0.35, 0)
    cv2.polylines(canvas, data_arr, True, poly_color, 2, cv2.LINE_AA)

    # Vertex dots
    for pt in data_pts:
        cv2.circle(canvas, tuple(pt), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pt), 4, poly_color, 1, cv2.LINE_AA)

    # 4. Center Badge: Quality Index
    badge_r = 30
    cv2.circle(canvas, center, badge_r, (25, 30, 38), -1, cv2.LINE_AA)
    cv2.circle(canvas, center, badge_r, poly_color, 2, cv2.LINE_AA)
    qi_str = f"{quality_index:.0f}"
    (qw, qh), _ = cv2.getTextSize(qi_str, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(canvas, qi_str, (center[0] - qw // 2, center[1] + qh // 2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas
