# Quality scoring calculation with original weight schema
WEIGHTS = {
    "blur": 0.25,
    "contrast": 0.20,
    "skew": 0.15,
    "noise": 0.15,
    "missing_section": 0.15,
    "handwriting": 0.10
}

from intelligence.quality_model import calculate_legacy_quality_score, calculate_ensemble_quality

def calculate_quality_score(defects):
    """
    Maintains exact numerical fidelity with the original scoring weights.
    """
    return calculate_legacy_quality_score(defects)
