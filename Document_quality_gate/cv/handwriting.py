"""
Handwriting & Pen Annotation Readability & Legibility Module
Performs stroke morphology, pen pressure uniformity, stroke contrast,
and crowding analysis to grade handwriting legibility (0-100) and route intake.
Completely OCR-free, mathematically explainable computer vision.
"""

from typing import Dict, Any, List
import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def analyze_handwriting(image: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates handwriting presence and assesses stroke-level legibility & readability quality.

    Methodology:
    - Otsu adaptive binarization to isolate candidate foreground ink strokes.
    - Outer border margin clearance (to eliminate scanner shadows, camera frame borders, desk edges).
    - Directional morphological opening (40x1 and 1x40) to subtract printed document borders,
      form grids, and horizontal/vertical rules.
    - Geometric contour discrimination:
      * Machine-printed text exhibits high solidity (>= 0.50), rigid character heights, and alignment
        into horizontal baselines.
      * Cursive handwriting and signatures exhibit elongated loopy paths with low solidity (<= 0.45)
        and high tortuosity (P^2 / (4*pi*A) >= 4.5).
      * Physical pen stroke constraint: distance transform ridge radius <= 7.0 px (rejecting large
        solid blocks, stamps, logos, or photo elements).
    - Local substrate contrast verification:
      * Ink strokes must have genuine contrast against their local paper background (>= 0.18).
      * Faint blur halos, paper textures, or camera noise with negligible contrast (< 0.18) are rejected.
    - Ballpoint / gel pen color ink verification:
      * Evaluated strictly within verified thin stroke contours (dark saturated blue/red pen ink
        on light paper), completely eliminating false positives from colored desks, headers, or UI elements.
    - Detection Criteria:
      * Requires at least 6 verified cursive handwriting strokes OR at least 3 confirmed color pen strokes.

    Grading & Routing:
    - Legibility >= 75: 'legible' -> PASS (Clear, legible handwriting conforms to intake standards)
    - 50 <= Legibility < 75: 'marginal' -> NEEDS REVIEW (Requires operator review)
    - Legibility < 50: 'illegible' -> REJECT (Smeared, faded, or illegible scribble)

    Returns:
        dict containing detected (bool), confidence (0.0-1.0), score (0-100),
        legibility_score, legibility_status, legibility_verdict, severity, and evidence.
    """
    bgr_image, gray_image, _ = validate_and_preprocess(image)
    h_img, w_img = gray_image.shape[:2]

    # 1. Otsu inverted binary threshold
    _, binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 2. Suppress straight printed lines (horizontal & vertical table/form rules)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    cleaned = cv2.subtract(binary, cv2.bitwise_or(horizontal_lines, vertical_lines))

    # Margin border suppression (ignore outer 1.5% border to prevent scanner/camera boundary artifacts)
    bm_y = max(6, int(h_img * 0.015))
    bm_x = max(6, int(w_img * 0.015))
    cleaned[:bm_y, :] = 0
    cleaned[-bm_y:, :] = 0
    cleaned[:, :bm_x] = 0
    cleaned[:, -bm_x:] = 0

    # 3. Find connected stroke components
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # First pass: collect candidate components and identify horizontal printed text lines
    raw_components = []
    line_buckets: Dict[int, List[Tuple[int, int, int, int]]] = {}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80 or area > 10000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > 6 * h or h > 6 * w:
            continue
        raw_components.append((contour, x, y, w, h, area))

        # Bucket into horizontal lines (within 5 px)
        matched_line = None
        for ly in line_buckets:
            if abs(y - ly) <= 5:
                matched_line = ly
                break
        if matched_line is not None:
            line_buckets[matched_line].append((x, y, w, h))
        else:
            line_buckets[y] = [(x, y, w, h)]

    # Identify lines with >= 5 components and uniform heights (printed text lines)
    printed_line_ys = set()
    for ly, box_list in line_buckets.items():
        if len(box_list) >= 5:
            hs = [b[3] for b in box_list]
            if float(np.std(hs)) <= 8.0:
                printed_line_ys.add(ly)

    valid_contours = []
    bounding_boxes: List[List[int]] = []
    stroke_contrasts: List[float] = []
    pen_strokes = 0

    for contour, x, y, w, h, area in raw_components:
        # Check if component lies within an identified machine-printed text line
        is_printed_line = any(abs(y - ly) <= 5 for ly in printed_line_ys)

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        # Stroke tortuosity (isoperimetric ratio) and convex hull solidity
        tortuosity = (perimeter ** 2) / (4.0 * np.pi * area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(1.0, hull_area)

        # Stroke width constraint via distance transform (physical pen tip radius)
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        dist = cv2.distanceTransform(mask[y:y+h, x:x+w], cv2.DIST_L2, 3)
        max_r = float(dist.max())
        if max_r > 7.0:
            continue

        # Local background and contrast verification against surrounding paper
        y1, y2 = max(0, y - 5), min(h_img, y + h + 5)
        x1, x2 = max(0, x - 5), min(w_img, x + w + 5)
        dil = cv2.dilate(mask[y1:y2, x1:x2], np.ones((5, 5), dtype=np.uint8))
        roi_g = gray_image[y1:y2, x1:x2]
        c_roi = mask[y1:y2, x1:x2]

        bg_pix = roi_g[(dil > 0) & (c_roi == 0)]
        st_pix = roi_g[c_roi > 0]
        bg_m = float(bg_pix.mean()) if bg_pix.size > 0 else 220.0
        st_m = float(st_pix.mean()) if st_pix.size > 0 else 0.0
        loc_contrast = (bg_m - st_m) / max(1.0, bg_m)

        # Reject faint shadows, blur halos, and paper texture artifacts
        if loc_contrast < 0.18:
            continue

        # Color pen ink check strictly within the stroke contour
        is_color_pen = False
        if bgr_image is not None and bgr_image.ndim == 3 and bgr_image.shape[2] == 3:
            roi_bgr = bgr_image[y:y+h, x:x+w]
            stroke_bgr_mask = mask[y:y+h, x:x+w]
            if stroke_bgr_mask.shape == roi_bgr.shape[:2] and cv2.countNonZero(stroke_bgr_mask) > 0:
                hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                # Ballpoint pen ink: dark saturated blue or red (V <= 140, S >= 60)
                blue_m = cv2.inRange(hsv, np.array([95, 60, 30]), np.array([135, 255, 140]))
                red_m1 = cv2.inRange(hsv, np.array([0, 80, 30]), np.array([10, 255, 140]))
                red_m2 = cv2.inRange(hsv, np.array([170, 80, 30]), np.array([180, 255, 140]))
                pen_m = cv2.bitwise_or(blue_m, cv2.bitwise_or(red_m1, red_m2))
                pen_m = cv2.bitwise_and(pen_m, pen_m, mask=stroke_bgr_mask)
                color_ratio = cv2.countNonZero(pen_m) / max(1, cv2.countNonZero(stroke_bgr_mask))
                # Verified pen ink on light paper
                if color_ratio > 0.35 and bg_m > 160:
                    is_color_pen = True

        # Machine-printed text line exclusion:
        # If the component aligns horizontally on a standard printed line, exclude it unless verified colored pen
        if is_printed_line and not is_color_pen:
            continue

        # True handwriting stroke criteria:
        # High contrast ink AND (cursive loopy morphology: tortuosity >= 4.5 and solidity <= 0.45, OR verified color pen)
        if is_color_pen or (tortuosity >= 4.5 and solidity <= 0.45):
            valid_contours.append(contour)
            bounding_boxes.append([int(x), int(y), int(w), int(h)])
            stroke_contrasts.append(loc_contrast)
            if is_color_pen:
                pen_strokes += 1

    handwriting_regions = len(valid_contours)
    color_ink_detected = pen_strokes >= 3

    # Handwriting detected if at least 6 cursive strokes OR verified color pen
    detected = (handwriting_regions >= 6) or color_ink_detected
    confidence = min(1.0, handwriting_regions / 10.0) if detected else 0.0

    # 4. Legibility & Readability Analysis
    if not detected or len(valid_contours) == 0:
        # No handwriting detected -> clean machine printed intake document
        legibility_score = 100.0
        legibility_status = "not_present"
        legibility_verdict = "PASS"
        severity = 0
        status = "good"
        stroke_contrast = 1.0
        stroke_uniformity = 1.0
        stroke_continuity = 1.0
        stroke_spacing = 1.0
    else:
        # Create stroke mask
        stroke_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(stroke_mask, valid_contours, -1, 255, -1)

        # A. Stroke Contrast (C_stroke)
        bg_pixels = gray_image[stroke_mask == 0]
        ink_pixels = gray_image[stroke_mask > 0]
        bg_mean = float(bg_pixels.mean()) if bg_pixels.size > 0 else 240.0
        ink_mean = float(ink_pixels.mean()) if ink_pixels.size > 0 else 50.0
        contrast_ratio = max(0.0, (bg_mean - ink_mean) / max(1.0, bg_mean))
        stroke_contrast = round(contrast_ratio, 3)
        c_subscore = min(100.0, max(15.0, (contrast_ratio / 0.45) * 100.0))

        # B. Stroke Width Regularity (Distance Transform)
        dist = cv2.distanceTransform(stroke_mask, cv2.DIST_L2, 5)
        ridge_widths = dist[dist > 0.5] * 2.0
        if ridge_widths.size > 10:
            w_mean = float(np.mean(ridge_widths))
            w_std = float(np.std(ridge_widths))
            cv_w = w_std / max(0.1, w_mean)
            # Legible pen strokes have CV around 0.35-0.55; messy splotches exceed 0.85
            u_subscore = max(15.0, min(100.0, 100.0 - (cv_w - 0.35) * 110.0))
        else:
            cv_w = 0.50
            u_subscore = 75.0
        stroke_uniformity = round(float(cv_w), 3)

        # C. Stroke Separation / Non-Crowding
        # Measure bounding box overlaps
        overlap_count = 0
        n_boxes = len(bounding_boxes)
        for i in range(min(30, n_boxes)):
            x1, y1, w1, h1 = bounding_boxes[i]
            for j in range(i + 1, min(30, n_boxes)):
                x2, y2, w2, h2 = bounding_boxes[j]
                if not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1):
                    overlap_count += 1

        overlap_ratio = overlap_count / max(1, n_boxes)
        s_subscore = max(20.0, min(100.0, 100.0 - (overlap_ratio * 70.0)))
        stroke_spacing = round(1.0 - min(1.0, overlap_ratio), 3)

        # D. Stroke Continuity
        # Ratio of medium-sized strokes to broken fragments
        fragment_count = sum(1 for c in valid_contours if cv2.contourArea(c) < 300)
        frag_ratio = fragment_count / max(1, len(valid_contours))
        f_subscore = max(20.0, min(100.0, 100.0 - (frag_ratio * 60.0)))
        stroke_continuity = round(1.0 - min(1.0, frag_ratio), 3)

        # Composite Legibility Score (0-100)
        legibility_score = round(
            0.35 * c_subscore +
            0.30 * u_subscore +
            0.20 * f_subscore +
            0.15 * s_subscore,
            1
        )
        legibility_score = max(10.0, min(100.0, legibility_score))

        # Legibility Classification & Action Routing
        if legibility_score >= 75.0:
            legibility_status = "legible"
            legibility_verdict = "PASS"
            status = "good"
            severity = max(0, int(100.0 - legibility_score))
        elif legibility_score >= 50.0:
            legibility_status = "marginal"
            legibility_verdict = "NEEDS REVIEW"
            status = "moderate"
            severity = int(100.0 - legibility_score)
        else:
            legibility_status = "illegible"
            legibility_verdict = "REJECT"
            status = "poor"
            severity = int(100.0 - legibility_score)

    return {
        "detected": detected,
        "confidence": round(confidence, 2),
        "score": legibility_score,
        "legibility_score": legibility_score,
        "legibility_status": legibility_status,
        "legibility_verdict": legibility_verdict,
        "severity": severity,
        "status": status,
        "evidence": {
            "candidate_regions": handwriting_regions,
            "color_ink_detected": color_ink_detected,
            "stroke_contrast": stroke_contrast,
            "stroke_uniformity": stroke_uniformity,
            "stroke_continuity": stroke_continuity,
            "stroke_spacing": stroke_spacing,
            "bounding_boxes": bounding_boxes[:50]
        }
    }
