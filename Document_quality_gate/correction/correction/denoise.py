"""
Denoising Module
Suppresses high-frequency sensor grain and compression artifacts
using edge-preserving bilateral filtering.
"""

import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def apply_denoising(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 45.0,
    sigma_space: float = 45.0
) -> np.ndarray:
    """
    Applies edge-preserving bilateral filtering to suppress noise while maintaining sharp text edges.
    """
    bgr, _, _ = validate_and_preprocess(image)
    denoised = cv2.bilateralFilter(bgr, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return denoised
