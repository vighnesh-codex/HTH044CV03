"""
Contrast Enhancement Module
Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
in LAB perceptual luminance space to enhance faded ink without amplifying noise.
"""

import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def apply_contrast_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid: tuple = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE adaptive contrast optimization.
    """
    bgr, _, _ = validate_and_preprocess(image)

    # Convert to LAB space and equalize only the Luminance (L) channel
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
