"""
Geometric Deskewing Module
Rotates document image by affine transformation to correct angular tilt.
"""

import cv2
import numpy as np
from cv.preprocessing import validate_and_preprocess


def apply_deskew(
    image: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Rotates the image to compensate for the measured skew angle.
    Uses white background fill to preserve document margin aesthetics.
    """
    bgr, _, _ = validate_and_preprocess(image)
    if abs(angle) < 0.2:
        return bgr.copy()

    h, w = bgr.shape[:2]
    center = (w // 2, h // 2)

    # Negative angle because OpenCV rotates counterclockwise for positive angles
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Determine bounding rectangle dimensions to prevent corner clipping
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        bgr,
        rot_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated
