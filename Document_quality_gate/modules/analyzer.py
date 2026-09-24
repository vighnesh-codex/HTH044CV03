from cv.blur import analyze_blur
from cv.contrast import analyze_contrast
from cv.skew import analyze_skew
from cv.noise import analyze_noise
from cv.handwriting import analyze_handwriting
from cv.completeness import analyze_completeness

def analyze_document(image):
    """
    Backward-compatible analyzer returning original defect dictionary structure.
    """
    blur = analyze_blur(image)
    contrast = analyze_contrast(image)
    skew = analyze_skew(image)
    noise = analyze_noise(image)
    handwriting = analyze_handwriting(image)
    missing_section = analyze_completeness(image)

    return {
        "blur": blur,
        "contrast": contrast,
        "skew": skew,
        "noise": noise,
        "handwriting": handwriting,
        "missing_section": missing_section
    }
