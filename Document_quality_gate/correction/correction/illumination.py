"""
Illumination Normalization Module
Compensates for uneven lighting gradients, flash hot-spots, and shadows
via morphological background division.
"""

import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def apply_illumination_normalization(
    image: np.ndarray,
    kernel_size: int = 51
) -> np.ndarray:
    """
    Estimates non-uniform background illumination profile using large morphological closing
    and normalizes pixel luminance across the sheet.
    """
    bgr, _, _ = validate_and_preprocess(image)

    # Convert to LAB space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Estimate background shading surface via large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, kernel)

    # Division normalization: (L / background) * target_mean
    target_mean = 230.0
    normalized_l = np.clip(
        (l_channel.astype(np.float32) / np.maximum(1.0, background.astype(np.float32))) * target_mean,
        0,
        255
    ).astype(np.uint8)

    normalized_lab = cv2.merge([normalized_l, a_channel, b_channel])
    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    return normalized_bgr
