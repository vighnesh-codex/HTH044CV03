"""
Document Preprocessing Module
Provides robust image validation, color-space normalization, aspect-ratio-preserving
resizing, and grayscale conversion for document quality analysis.
"""

from typing import Tuple, Dict, Any, Union
import cv2
import numpy as np
from cv.pdf_loader import is_pdf, load_single_pdf_page


def validate_and_preprocess(
    image: Union[np.ndarray, str, bytes, Any],
    max_size: int = 2000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Validates input image or PDF document, standardizes channel format (BGR and Grayscale),
    and resizes large documents while preserving aspect ratio.

    Args:
        image: Input numpy array representing the image, or a PDF file path/bytes.
        max_size: Maximum pixel dimension for width or height.

    Returns:
        Tuple of (bgr_image, gray_image, metadata_dict)
    """
    if image is None:
        raise ValueError("Invalid image: Input is None. Please provide a valid document image.")

    # Automatically handle PDF input by rasterizing page 1
    pdf_meta = None
    if is_pdf(image):
        image, pdf_meta = load_single_pdf_page(image, page_number=1, scale=2.0)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Invalid image: Input must be a numpy ndarray (or valid PDF), got {type(image)}.")

    if image.size == 0 or image.ndim < 2:
        raise ValueError("Invalid image: Image array is empty or has invalid dimensions.")

    orig_shape = image.shape
    h, w = orig_shape[:2]

    if h < 16 or w < 16:
        raise ValueError(f"Invalid image: Document dimensions ({w}x{h}) are too small for CV analysis (min 16x16).")

    # Normalize channel formats to standard 3-channel BGR
    if image.ndim == 2:
        # Grayscale image
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray_image = image.copy()
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 4:
            # BGRA image
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        elif channels == 3:
            # Standard BGR
            bgr_image = image.copy()
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        elif channels == 1:
            # Single channel with shape (H, W, 1)
            bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            gray_image = image.squeeze(axis=2).copy()
        else:
            raise ValueError(f"Unsupported number of image channels: {channels}")
    else:
        raise ValueError(f"Unsupported image array dimensions: {image.ndim}")

    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        bgr_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray_image = cv2.resize(gray_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    metadata = {
        "original_shape": orig_shape,
        "processed_shape": bgr_image.shape,
        "scale_factor": scale,
        "channels": 3,
        "width": bgr_image.shape[1],
        "height": bgr_image.shape[0]
    }
    if pdf_meta is not None:
        metadata["pdf"] = pdf_meta

    return bgr_image, gray_image, metadata


def preprocess_image(image: np.ndarray, max_size: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper matching original signature.
    Returns (image, gray_image).
    """
    bgr_image, gray_image, _ = validate_and_preprocess(image, max_size=max_size)
    return bgr_image, gray_image
