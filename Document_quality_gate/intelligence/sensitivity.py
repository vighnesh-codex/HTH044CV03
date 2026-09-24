"""
Counterfactual Quality Sensitivity Analysis Module
Quantifies the exact marginal score degradation caused by each defect factor
by evaluating counterfactual 'what-if' perturbations against the scoring model.
"""

import copy
from typing import Dict, Any, List
from intelligence.quality_model import calculate_ensemble_quality


FACTOR_NAMES = {
    "handwriting": "Handwriting Interference",
    "contrast": "Low Contrast",
    "blur": "Blur / Sharpness",
    "noise": "Image Noise",
    "skew": "Skew Angle",
    "completeness": "Missing Sections"
}


def analyze_quality_sensitivity(
    defects: Dict[str, Any],
    structure_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Performs counterfactual sensitivity analysis on the scoring model.

    For each defect:
    1. Temporarily eliminates the defect by setting it to optimal/ideal quality.
    2. Recomputes the ensemble Quality Index.
    3. Calculates delta: delta = ideal_score - current_score.
    4. Expresses this as the negative score penalty attributable to that defect.

    Returns:
        Dict with baseline_score, sensitivities list sorted by impact,
        dominant_driver, and explainable summary.
    """
    baseline_result = calculate_ensemble_quality(defects, structure_result)
    baseline_score = float(baseline_result["quality_index"])

    sensitivities: List[Dict[str, Any]] = []

    # Factors to perturb
    test_factors = ["handwriting", "contrast", "blur", "noise", "skew", "completeness"]

    for factor in test_factors:
        perturbed = copy.deepcopy(defects)

        if factor == "blur":
            perturbed["blur"]["score"] = 100
            perturbed["blur"]["status"] = "good"
        elif factor == "contrast":
            perturbed["contrast"]["score"] = 100
            perturbed["contrast"]["status"] = "good"
        elif factor == "noise":
            perturbed["noise"]["score"] = 100
            perturbed["noise"]["status"] = "good"
        elif factor == "skew":
            perturbed["skew"]["angle"] = 0.0
            perturbed["skew"]["score"] = 100
            perturbed["skew"]["status"] = "good"
        elif factor == "handwriting":
            perturbed["handwriting"]["detected"] = False
            perturbed["handwriting"]["confidence"] = 0.0
        elif factor == "completeness":
            if "missing_section" in perturbed:
                perturbed["missing_section"]["score"] = 100
                perturbed["missing_section"]["missing"] = []
            perturbed["missing_sections"] = []

        counterfactual_res = calculate_ensemble_quality(perturbed, structure_result)
        ideal_score = float(counterfactual_res["quality_index"])
        score_gain = round(ideal_score - baseline_score, 1)

        # Negative impact represents current drag on score
        impact = round(-score_gain, 1)

        sensitivities.append({
            "key": factor,
            "label": FACTOR_NAMES.get(factor, factor.title()),
            "impact": impact,
            "score_gain_if_fixed": score_gain,
            "potential_score": ideal_score
        })

    # Sort by absolute impact descending
    sensitivities.sort(key=lambda item: abs(item["impact"]), reverse=True)

    dominant = sensitivities[0]
    if dominant["impact"] < 0:
        dominant_driver = dominant["label"]
        summary = f"Dominant quality driver is {dominant_driver} ({dominant['impact']} pts drag on Quality Index)."
    else:
        dominant_driver = "None (Optimal Quality)"
        summary = "No significant defect drag detected. Document is near optimal quality."

    return {
        "baseline_score": baseline_score,
        "sensitivities": sensitivities,
        "dominant_driver": dominant_driver,
        "dominant_key": dominant["key"],
        "summary": summary
    }
