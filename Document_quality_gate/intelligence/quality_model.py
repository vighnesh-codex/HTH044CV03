"""
Ensemble Quality Scoring Engine
Replaces simplistic single-scalar scoring with a 5-dimension quality model:
1. Visual Quality Score
2. Structural Quality Score
3. Readability Risk Score
4. Completeness Score
5. Interference Risk Score
Also preserves legacy scoring as a documented, configurable fallback.
"""

from typing import Dict, Any


LEGACY_WEIGHTS = {
    "blur": 0.25,
    "contrast": 0.20,
    "skew": 0.15,
    "noise": 0.15,
    "missing_section": 0.15,
    "handwriting": 0.10
}


def calculate_legacy_quality_score(defects: Dict[str, Any]) -> float:
    """
    Original scoring algorithm preserved for baseline reproducibility.
    Supports continuous handwriting legibility score when available.
    """
    blur = defects["blur"]["score"]
    contrast = defects["contrast"]["score"]
    noise = defects["noise"]["score"]
    missing_section = defects.get("missing_section", {}).get("score", 100)

    skew_angle = abs(defects["skew"].get("angle", 0))
    if skew_angle <= 1:
        skew = 100
    elif skew_angle <= 3:
        skew = 85
    elif skew_angle <= 5:
        skew = 70
    elif skew_angle <= 8:
        skew = 50
    else:
        skew = 20

    # If continuous handwriting score is present, utilize it; else binary
    hw_entry = defects.get("handwriting", {})
    if "score" in hw_entry and hw_entry["score"] is not None:
        handwriting = float(hw_entry["score"])
    else:
        handwriting = 0 if hw_entry.get("detected", False) else 100

    score = (
        blur * LEGACY_WEIGHTS["blur"] +
        contrast * LEGACY_WEIGHTS["contrast"] +
        skew * LEGACY_WEIGHTS["skew"] +
        noise * LEGACY_WEIGHTS["noise"] +
        missing_section * LEGACY_WEIGHTS["missing_section"] +
        handwriting * LEGACY_WEIGHTS["handwriting"]
    )

    return round(max(0.0, min(100.0, float(score))), 2)


def calculate_ensemble_quality(
    defects: Dict[str, Any],
    structure_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates explainable 5-dimension subscores and composite Quality Index.
    """
    blur_score = float(defects["blur"]["score"])
    contrast_score = float(defects["contrast"]["score"])
    noise_score = float(defects["noise"]["score"])

    # Skew score
    skew_val = defects["skew"].get("score")
    if skew_val is None:
        angle = abs(defects["skew"].get("angle", 0))
        if angle <= 1:
            skew_val = 100
        elif angle <= 3:
            skew_val = 85
        elif angle <= 5:
            skew_val = 70
        elif angle <= 8:
            skew_val = 50
        else:
            skew_val = 20
    skew_score = float(skew_val)

    # Completeness
    completeness_score = float(defects.get("missing_section", {}).get("score", 100))

    # Handwriting legibility
    hw_detected = defects["handwriting"].get("detected", False)
    hw_legibility = float(defects["handwriting"].get("score", 100.0))
    hw_conf = float(defects["handwriting"].get("confidence", 0.0))

    # Structure
    if structure_result is not None:
        structure_score = float(structure_result.get("score", 95.0))
    else:
        structure_score = 95.0

    # 1. Visual Quality Score
    visual_quality = round(
        0.45 * blur_score + 0.35 * contrast_score + 0.20 * noise_score,
        1
    )

    # 2. Structural Quality Score
    structural_quality = round(
        0.60 * skew_score + 0.40 * structure_score,
        1
    )

    # 3. Readability Risk Score (lower is better)
    readability_risk = round(
        0.50 * (100.0 - blur_score) +
        0.30 * (100.0 - contrast_score) +
        0.20 * (100.0 - noise_score),
        1
    )

    # 4. Completeness Score
    completeness = round(completeness_score, 1)

    # 5. Interference Risk Score (Continuous Legibility-Based)
    if hw_detected:
        hw_risk = max(0.0, 100.0 - hw_legibility)
    else:
        hw_risk = 0.0

    noise_penalty = max(0.0, 100.0 - noise_score)
    interference_risk = round(
        max(hw_risk, 0.70 * hw_risk + 0.30 * noise_penalty),
        1
    )

    # Final Composite Quality Index
    quality_index = round(
        0.30 * visual_quality +
        0.20 * structural_quality +
        0.20 * max(0.0, 100.0 - readability_risk) +
        0.15 * completeness +
        0.15 * max(0.0, 100.0 - interference_risk),
        1
    )

    quality_index = max(0.0, min(100.0, quality_index))

    # Legacy fallback calculation
    legacy_score = calculate_legacy_quality_score(defects)

    return {
        "quality_index": quality_index,
        "legacy_score": legacy_score,
        "subscores": {
            "visual_quality": visual_quality,
            "structural_quality": structural_quality,
            "readability_risk": readability_risk,
            "completeness": completeness,
            "interference_risk": interference_risk
        },
        "weights_documented": {
            "visual_quality": 0.30,
            "structural_quality": 0.20,
            "readability_clearance": 0.20,
            "completeness": 0.15,
            "interference_clearance": 0.15
        }
    }
