"""
Defect Overlay Generation Module
Draws geometrically accurate computer-vision overlays for detected skew baselines,
handwriting bounding boxes, missing document zones, and boundary margins.
"""

from typing import Dict, Any
import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def generate_defect_overlay(
    image: np.ndarray,
    defects: Dict[str, Any],
    structure_info: Dict[str, Any] = None
) -> np.ndarray:
    """
    Renders an integrated defect overlay visualizing actual detected CV defects.

    Evidence plotted:
    - Cyan lines: Hough transform skew lines with angle annotation.
    - Magenta boxes: Handwriting connected component stroke regions.
    - Orange shaded bands: Detected missing functional sections.
    - Red border markers: Edge truncation cutoff anomalies.
    - Lime rectangle: Document boundary contour.
    """
    bgr, _, _ = validate_and_preprocess(image)
    overlay = bgr.copy()
    h, w = overlay.shape[:2]

    # 1. Plot Missing Sections
    missing_boxes = defects.get("missing_section", {}).get("evidence", {}).get("missing_boxes", [])
    if missing_boxes:
        mask = overlay.copy()
        for box in missing_boxes:
            bx, by, bw, bh = box
            # Orange translucent fill
            cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), (0, 140, 255), -1)
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 100, 240), 2)
            cv2.putText(
                overlay,
                "MISSING SECTION",
                (bx + 15, by + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 80, 230),
                2,
                cv2.LINE_AA
            )
        # Blend translucent fill
        overlay = cv2.addWeighted(overlay, 0.75, mask, 0.25, 0)

    # 2. Plot Handwriting Regions
    hw_boxes = defects.get("handwriting", {}).get("evidence", {}).get("bounding_boxes", [])
    if hw_boxes:
        for box in hw_boxes[:25]:
            hx, hy, hw, hh = box
            # Magenta border
            cv2.rectangle(overlay, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)
            cv2.putText(
                overlay,
                "HW",
                (hx, max(12, hy - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 0, 255),
                1,
                cv2.LINE_AA
            )

    # 3. Plot Skew Lines
    skew_segments = defects.get("skew", {}).get("evidence", {}).get("line_segments", [])
    skew_angle = defects.get("skew", {}).get("angle", 0.0)
    if skew_segments and abs(skew_angle) > 1.0:
        for seg in skew_segments[:12]:
            x1, y1, x2, y2 = seg
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)

        # Skew annotation banner at bottom
        cv2.putText(
            overlay,
            f"Skew Angle: {skew_angle:+.2f} deg",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    # 4. Plot Structural Boundary & Truncations
    if structure_info:
        boundary_box = structure_info.get("evidence", {}).get("boundary_box")
        if boundary_box:
            bx, by, bw, bh = boundary_box
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (50, 220, 50), 2)

        trunc_edges = structure_info.get("evidence", {}).get("truncated_edges", [])
        if trunc_edges:
            for edge in trunc_edges:
                if edge == "top":
                    cv2.line(overlay, (0, 3), (w, 3), (0, 0, 255), 4)
                elif edge == "bottom":
                    cv2.line(overlay, (0, h - 3), (w, h - 3), (0, 0, 255), 4)
                elif edge == "left":
                    cv2.line(overlay, (3, 0), (3, h), (0, 0, 255), 4)
                elif edge == "right":
                    cv2.line(overlay, (w - 3, 0), (w - 3, h), (0, 0, 255), 4)

    return overlay
