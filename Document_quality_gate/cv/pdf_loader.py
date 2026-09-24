"""
PDF Document Loader and Rasterizer Module
Provides high-performance, OCR-free rendering of PDF pages into OpenCV BGR numpy arrays.
Supports file paths, raw bytes, and file-like objects (e.g. Streamlit UploadedFile).
"""

import io
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
import cv2
import numpy as np

# Primary high-performance C-engine renderer
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

# Secondary fallback renderer
try:
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False


def is_pdf(source: Union[str, Path, bytes, io.BytesIO, Any]) -> bool:
    """
    Detects whether the input represents a PDF document via filename or magic byte header (%PDF-).
    """
    if source is None:
        return False

    if isinstance(source, (str, Path)):
        path_str = str(source)
        if path_str.lower().endswith(".pdf"):
            return True
        if os.path.exists(path_str):
            try:
                with open(path_str, "rb") as f:
                    header = f.read(5)
                    return header.startswith(b"%PDF-")
            except Exception:
                return False
        return False

    if isinstance(source, bytes):
        return source.startswith(b"%PDF-")

    if hasattr(source, "read") and hasattr(source, "seek"):
        try:
            pos = source.tell()
            header = source.read(5)
            source.seek(pos)
            return header.startswith(b"%PDF-")
        except Exception:
            return False

    return False


def get_pdf_page_count(source: Union[str, Path, bytes, io.BytesIO, Any]) -> int:
    """
    Returns the total number of pages in the PDF document without full rasterization.
    """
    if not HAS_PDFIUM and not HAS_PDF2IMAGE:
        raise ImportError("No PDF engine available. Please install pypdfium2 or pdf2image.")

    raw_bytes = _read_source_bytes(source)

    if HAS_PDFIUM:
        pdf = pdfium.PdfDocument(raw_bytes)
        count = len(pdf)
        pdf.close()
        return count

    if HAS_PDF2IMAGE:
        info = pdf2image.pdfinfo_from_bytes(raw_bytes)
        return int(info.get("Pages", 1))

    return 0


def load_pdf_pages(
    source: Union[str, Path, bytes, io.BytesIO, Any],
    scale: float = 2.0,
    max_pages: Optional[int] = None
) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
    """
    Renders pages of a PDF document into standard OpenCV BGR numpy arrays.

    Args:
        source: Filepath, bytes, or file-like stream of the PDF.
        scale: Resolution multiplier (default 2.0 = ~144 DPI for crisp CV feature extraction).
        max_pages: Optional ceiling on the number of pages to render.

    Returns:
        List of tuples: (page_index [1-based], bgr_image_array [uint8], page_metadata_dict)
    """
    if not HAS_PDFIUM and not HAS_PDF2IMAGE:
        raise ImportError(
            "Neither 'pypdfium2' nor 'pdf2image' is installed. "
            "Install pypdfium2 via: pip install pypdfium2"
        )

    raw_bytes = _read_source_bytes(source)
    if not raw_bytes.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF: File header does not start with standard '%PDF-' magic bytes.")

    pages_result: List[Tuple[int, np.ndarray, Dict[str, Any]]] = []

    # 1. Primary Engine: pypdfium2 (Direct C-bindings, fastest, zero subprocess overhead)
    if HAS_PDFIUM:
        pdf = pdfium.PdfDocument(raw_bytes)
        total_pages = len(pdf)
        pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)

        for idx in range(pages_to_read):
            page = pdf[idx]
            # Render page to bitmap
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            rgb_arr = np.array(pil_image)

            # Convert RGB PIL to OpenCV BGR
            if rgb_arr.ndim == 2:
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
            elif rgb_arr.shape[2] == 4:
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
            else:
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            meta = {
                "page_number": idx + 1,
                "total_pages": total_pages,
                "width": bgr_arr.shape[1],
                "height": bgr_arr.shape[0],
                "scale": scale,
                "renderer": "pypdfium2"
            }
            pages_result.append((idx + 1, bgr_arr, meta))

        pdf.close()
        return pages_result

    # 2. Fallback Engine: pdf2image
    if HAS_PDF2IMAGE:
        # 144 DPI equivalent to scale=2.0
        dpi = int(72 * scale)
        pil_pages = pdf2image.convert_from_bytes(
            raw_bytes,
            dpi=dpi,
            first_page=1,
            last_page=max_pages
        )
        total_pages = len(pil_pages)

        for idx, pil_img in enumerate(pil_pages):
            rgb_arr = np.array(pil_img)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
            meta = {
                "page_number": idx + 1,
                "total_pages": total_pages,
                "width": bgr_arr.shape[1],
                "height": bgr_arr.shape[0],
                "scale": scale,
                "renderer": "pdf2image"
            }
            pages_result.append((idx + 1, bgr_arr, meta))

        return pages_result

    return pages_result


def load_single_pdf_page(
    source: Union[str, Path, bytes, io.BytesIO, Any],
    page_number: int = 1,
    scale: float = 2.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Renders a single specified page (1-based index) from a PDF.
    """
    pages = load_pdf_pages(source, scale=scale, max_pages=page_number)
    for p_idx, bgr_img, meta in pages:
        if p_idx == page_number:
            return bgr_img, meta
    if pages:
        return pages[0][1], pages[0][2]
    raise ValueError(f"Could not render page {page_number} from the provided PDF.")


def _read_source_bytes(source: Union[str, Path, bytes, io.BytesIO, Any]) -> bytes:
    """Helper to convert varied source types into raw bytes."""
    if isinstance(source, (str, Path)):
        with open(str(source), "rb") as f:
            return f.read()

    if isinstance(source, bytes):
        return source

    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return data

    raise TypeError(f"Unsupported source type for PDF loader: {type(source)}")
