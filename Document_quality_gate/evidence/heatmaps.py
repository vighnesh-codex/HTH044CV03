"""
Defect Heatmap Generation Module
Renders mathematically authentic computer-vision heatmaps for optical blur,
high-frequency noise grain, local contrast, and spatial quality distribution.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def generate_blur_heatmap(image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Computes dense spatial blur defect heatmap using local Laplacian energy.
    Cold colors (Blue/Cyan) indicate sharp text; Hot colors (Orange/Red) indicate optical blur.
    """
    bgr, gray, _ = validate_and_preprocess(image)
    h, w = gray.shape

    # 1. Local Laplacian variance map
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_energy = cv2.boxFilter(lap ** 2, -1, (25, 25))

    # Log transform to compress dynamic range
    log_energy = np.log1p(lap_energy)

    # Invert so high blur defect = high value (red)
    max_val = np.percentile(log_energy, 98)
    min_val = np.percentile(log_energy, 2)
    spread = max(1e-5, max_val - min_val)

    # Normalized sharpness [0, 1]
    norm_sharpness = np.clip((log_energy - min_val) / spread, 0.0, 1.0)
    # Defect map: 1.0 = most blurry, 0.0 = sharpest
    blur_defect = (1.0 - norm_sharpness)

    heat_gray = np.uint8(blur_defect * 255)
    heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_JET)

    # Blend with original
    overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


def generate_noise_heatmap(image: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    """
    Computes dense spatial noise heatmap using local high-frequency residual energy.
    Bright yellow/red regions pinpoint severe sensor or compression noise.
    """
    bgr, gray, _ = validate_and_preprocess(image)

    # High frequency residual via Gaussian subtraction
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.absdiff(gray, smooth).astype(np.float32)

    # Local noise energy via moving spatial window
    local_noise = cv2.boxFilter(residual, -1, (15, 15))

    p99 = np.percentile(local_noise, 99)
    p5 = np.percentile(local_noise, 5)
    spread = max(1e-5, p99 - p5)

    norm_noise = np.clip((local_noise - p5) / spread, 0.0, 1.0)
    heat_gray = np.uint8(norm_noise * 255)
    heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_HOT)

    overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


def generate_contrast_heatmap(image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Computes local standard deviation (RMS contrast) across the document.
    Dark/cool regions represent washed-out or low-contrast zones.
    """
    bgr, gray, _ = validate_and_preprocess(image)
    gray_f = gray.astype(np.float32)

    # Fast local variance: Var = E[X^2] - (E[X])^2
    mean_i = cv2.boxFilter(gray_f, -1, (29, 29))
    mean_i2 = cv2.boxFilter(gray_f ** 2, -1, (29, 29))
    local_var = np.maximum(0.0, mean_i2 - (mean_i ** 2))
    local_std = np.sqrt(local_var)

    p95 = np.percentile(local_std, 95)
    spread = max(1e-5, p95)

    norm_contrast = np.clip(local_std / spread, 0.0, 1.0)
    heat_gray = np.uint8(norm_contrast * 255)
    heat_color = cv2.applyColorMap(heat_gray, cv2.COLORMAP_VIRIDIS)

    overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


def generate_quality_heatmap(
    image: np.ndarray,
    multiscale_info: Dict[str, Any],
    alpha: float = 0.45
) -> np.ndarray:
    """
    Interpolates regional grid quality scores across the entire document canvas.
    Colors: Green = High Quality, Yellow = Moderate, Red = Degraded Defect.
    """
    bgr, gray, _ = validate_and_preprocess(image)
    h, w = gray.shape

    score_grid = np.array(multiscale_info.get("score_grid", [[75, 75], [75, 75]]), dtype=np.float32)

    # Smoothly upscale grid to image dimensions
    smooth_map = cv2.resize(score_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth_map = np.clip(smooth_map, 0.0, 100.0)

    # Map: 0 = Red, 50 = Yellow, 100 = Green
    # In HSV: Hue goes from 0 (Red) to 60 (Yellow) to 120 (Green)
    hue = np.uint8((smooth_map / 100.0) * 120.0 * (180.0 / 360.0) * 2.0)  # [0, 120 in degrees, OpenCV hue 0-60]
    hue = np.uint8((smooth_map / 100.0) * 60.0)  # 0 (Red) to 60 (Green in OpenCV)
    sat = np.full((h, w), 220, dtype=np.uint8)
    val = np.full((h, w), 235, dtype=np.uint8)

    hsv_heat = cv2.merge([hue, sat, val])
    heat_bgr = cv2.cvtColor(hsv_heat, cv2.COLOR_HSV2BGR)

    overlay = cv2.addWeighted(bgr, 1.0 - alpha, heat_bgr, alpha, 0)
    return overlay


def render_regional_tile_overlay(
    image: np.ndarray,
    multiscale_info: Dict[str, Any]
) -> np.ndarray:
    """
    Renders spatial grid boundaries with score badge labels on each tile.
    """
    bgr, _, _ = validate_and_preprocess(image)
    vis = bgr.copy()

    tile_features = multiscale_info.get("tile_features", [])
    for tile in tile_features:
        x, y, tw, th = tile["bounds"]
        score = tile["score"]
        name = tile["region"]

        # Color based on score
        if score >= 75:
            color = (40, 180, 40)   # Green
        elif score >= 50:
            color = (30, 190, 240)  # Amber
        else:
            color = (40, 40, 230)   # Red

        # Draw tile border
        cv2.rectangle(vis, (x, y), (x + tw, y + th), color, 2)

        # Semi-transparent tinted background for tag
        tag_text = f"{name}: {score:.0f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(0.7, tw / 400.0))
        (tw_txt, th_txt), _ = cv2.getTextSize(tag_text, font, font_scale, 1)

        cv2.rectangle(
            vis,
            (x + 5, y + 5),
            (x + 10 + tw_txt, y + 10 + th_txt + 4),
            (20, 20, 20),
            -1
        )
        cv2.putText(
            vis,
            tag_text,
            (x + 8, y + 8 + th_txt),
            font,
            font_scale,
            color,
            1,
            cv2.LINE_AA
        )

    return vis
