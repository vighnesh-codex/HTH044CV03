"""
Multi-Scale Document Quality Analysis Module
Performs hierarchical quality decomposition across global, regional, and local tile scales.
Extracts localized Laplacian variance, contrast, noise, and edge density maps.
"""

from typing import Dict, Any, Tuple, List
import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


REGION_NAMES_3X3 = [
    ["top_left", "top_center", "top_right"],
    ["middle_left", "center", "middle_right"],
    ["bottom_left", "bottom_center", "bottom_right"]
]


def analyze_multiscale_quality(
    image: np.ndarray,
    grid_size: Tuple[int, int] = (3, 3)
) -> Dict[str, Any]:
    """
    Evaluates quality metrics on a spatial grid across the document.

    Args:
        image: Input document image (BGR or Grayscale).
        grid_size: Tuple (rows, cols) for the regional grid (default 3x3).

    Returns:
        Dict with global_quality, regional_quality, tile_features,
        spatial_variance, localized_defect diagnosis, and score grid.
    """
    bgr_img, gray_img, _ = validate_and_preprocess(image)
    h, w = gray_img.shape
    rows, cols = grid_size

    h_step = h // rows
    w_step = w // cols

    score_grid = np.zeros((rows, cols), dtype=np.float32)
    regional_quality: Dict[str, float] = {}
    tile_features: List[Dict[str, Any]] = []

    # HSV for color saturation analysis
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV) if bgr_img is not None else None

    for r in range(rows):
        for c in range(cols):
            y1 = r * h_step
            y2 = h if r == rows - 1 else (r + 1) * h_step
            x1 = c * w_step
            x2 = w if c == cols - 1 else (c + 1) * w_step

            tile_gray = gray_img[y1:y2, x1:x2]
            if tile_gray.size == 0:
                continue

            # 1. Local Laplacian variance
            lap_var = float(cv2.Laplacian(tile_gray, cv2.CV_64F).var())

            # 2. Local contrast (std dev)
            loc_contrast = float(tile_gray.std())

            # 3. Local brightness
            loc_brightness = float(tile_gray.mean())

            # 4. Local noise estimate
            tile_blur = cv2.GaussianBlur(tile_gray, (5, 5), 0)
            loc_noise = float(cv2.absdiff(tile_gray, tile_blur).mean())

            # 5. Local edge density
            tile_edges = cv2.Canny(tile_gray, 50, 150)
            edge_density = float(cv2.countNonZero(tile_edges) / max(1, tile_gray.size))

            # 6. Gradient strength (Sobel)
            sobelx = cv2.Sobel(tile_gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(tile_gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = float(np.mean(np.sqrt(sobelx ** 2 + sobely ** 2)))

            # 7. Text-like ink density
            _, bin_tile = cv2.threshold(tile_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink_density = float(cv2.countNonZero(bin_tile) / max(1, tile_gray.size))

            # 8. Saturation
            if hsv_img is not None:
                tile_sat = float(hsv_img[y1:y2, x1:x2, 1].mean())
            else:
                tile_sat = 0.0

            # Compute tile composite quality (0-100)
            sharp_sub = min(100.0, lap_var / 10.0)
            cont_sub = min(100.0, loc_contrast * 2.0)
            noise_sub = max(0.0, min(100.0, 100.0 - loc_noise * 5.0))

            # Illumination balance penalty
            illum_sub = 100.0
            if loc_brightness < 40:
                illum_sub = max(20.0, loc_brightness * 2.5)
            elif loc_brightness > 230:
                illum_sub = max(20.0, 100.0 - (loc_brightness - 230) * 3.5)

            tile_score = round(
                0.35 * sharp_sub +
                0.30 * cont_sub +
                0.20 * noise_sub +
                0.15 * illum_sub,
                1
            )
            score_grid[r, c] = tile_score

            # Region naming
            if rows == 3 and cols == 3:
                name = REGION_NAMES_3X3[r][c]
            else:
                name = f"region_r{r}_c{c}"

            regional_quality[name] = float(tile_score)

            tile_features.append({
                "region": name,
                "row": r,
                "col": c,
                "bounds": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "score": float(tile_score),
                "laplacian_var": round(lap_var, 2),
                "contrast": round(loc_contrast, 2),
                "brightness": round(loc_brightness, 2),
                "noise": round(loc_noise, 2),
                "edge_density": round(edge_density, 4),
                "gradient_strength": round(grad_mag, 2),
                "ink_density": round(ink_density, 4),
                "saturation": round(tile_sat, 2)
            })

    # Global multi-scale score
    global_quality = float(round(np.mean(score_grid), 1))
    regional_variance = float(round(np.std(score_grid), 2))

    # Identify localized vs global defects
    worst_region = min(regional_quality.items(), key=lambda kv: kv[1])
    best_region = max(regional_quality.items(), key=lambda kv: kv[1])

    is_localized = False
    diagnosis = "Quality is uniform across the document."

    # If the worst region is severely lower than global average
    if (global_quality - worst_region[1]) >= 20.0 and worst_region[1] < 55.0:
        is_localized = True
        diagnosis = (
            f"Overall image is acceptable ({global_quality:.1f}), but the "
            f"{worst_region[0]} region has severe quality degradation ({worst_region[1]:.1f})."
        )
    elif worst_region[1] < 40.0 and global_quality < 55.0:
        diagnosis = f"Global quality degradation across entire document (Worst: {worst_region[0]} at {worst_region[1]:.1f})."

    return {
        "global_quality": global_quality,
        "regional_quality": regional_quality,
        "score_grid": score_grid.tolist(),
        "regional_variance": regional_variance,
        "is_localized_defect": is_localized,
        "diagnosis": diagnosis,
        "worst_region": {"name": worst_region[0], "score": worst_region[1]},
        "best_region": {"name": best_region[0], "score": best_region[1]},
        "tile_features": tile_features
    }
