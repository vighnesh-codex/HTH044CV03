from modules.blur import analyze_blur
from modules.contrast import analyze_contrast
from modules.skew import analyze_skew
from modules.noise import analyze_noise
from modules.handwriting import analyze_handwriting
from modules.missing_section import analyze_missing_section


def analyze_document(image):

    blur = analyze_blur(image)
    contrast = analyze_contrast(image)
    skew = analyze_skew(image)
    noise = analyze_noise(image)
    handwriting = analyze_handwriting(image)
    missing_section = analyze_missing_section(image)

    return {
        "blur": blur,
        "contrast": contrast,
        "skew": skew,
        "noise": noise,
        "handwriting": handwriting,
        "missing_section": missing_section
    }