"""
Document Quality Intelligence Engine
Master orchestrator integrating multi-scale computer vision defect detection,
ensemble quality scoring, normalized risk vectors, explainable decision traces,
counterfactual sensitivity analysis, and cohort anomaly evaluation.
"""

from typing import Dict, Any, Optional
import numpy as np

from cv.preprocessing import validate_and_preprocess
from cv.blur import analyze_blur
from cv.contrast import analyze_contrast
from cv.skew import analyze_skew
from cv.noise import analyze_noise
from cv.handwriting import analyze_handwriting
from cv.completeness import analyze_completeness
from cv.structure import analyze_structure

from intelligence.multiscale import analyze_multiscale_quality
from intelligence.quality_model import calculate_ensemble_quality, calculate_legacy_quality_score
from intelligence.risk_engine import compute_risk_vector, evaluate_decision_trace
from intelligence.sensitivity import analyze_quality_sensitivity
from intelligence.fingerprint import generate_quality_fingerprint
from intelligence.anomaly import extract_anomaly_features, evaluate_anomaly
from storage.database import get_all_feature_vectors


def generate_engine_suggestions(defects: Dict[str, Any], structure: Dict[str, Any] = None) -> list:
    """
    Generates actionable corrective suggestions free of any OCR references.
    """
    suggestions = []

    if defects["blur"]["status"] == "poor":
        suggestions.append("Rescan or recapture the document with stabilized camera focus.")
    elif defects["blur"]["status"] == "moderate":
        suggestions.append("Noticeable softness detected; consider capturing under sharper focus.")

    if defects["contrast"]["status"] == "poor":
        suggestions.append("Improve ambient lighting or apply adaptive contrast enhancement.")

    if abs(defects["skew"]["angle"]) > 3.0:
        suggestions.append(f"Deskew document alignment (current tilt: {defects['skew']['angle']:+.1f}°).")

    if defects["noise"]["status"] == "poor":
        suggestions.append("Apply digital denoising or reduce camera ISO grain.")

    if defects["handwriting"].get("detected", False):
        suggestions.append("Route document to manual operator verification due to detected handwriting interference.")

    missing_sections = defects.get("missing_sections", [])
    if not missing_sections:
        missing_sections = defects.get("missing_section", {}).get("missing", [])

    if missing_sections:
        sections_str = ", ".join(missing_sections)
        suggestions.append(f"Resubmit complete document containing missing sections: {sections_str}.")

    if structure and structure.get("alerts"):
        for alert in structure["alerts"]:
            if "illumination" in alert.lower():
                suggestions.append("Ensure uniform document lighting across all page quadrants.")
            elif "cropped" in alert.lower():
                suggestions.append("Reframe capture to ensure all margins and content boundaries are fully visible.")

    if not suggestions:
        suggestions.append("Document satisfies all technical quality standards. Approved for intake.")

    return suggestions


def analyze_document_quality(
    image: np.ndarray,
    historical_feature_matrix: Optional[list] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive Document Quality Intelligence analysis on an input image.

    Returns:
        Master dictionary with all CV defects, multi-scale grid, ensemble scores,
        risk vector, explainable decision trace, sensitivity analysis, fingerprint,
        and anomaly status.
    """
    bgr, gray, meta = validate_and_preprocess(image)

    # 1. Core Computer Vision Defect Analyzers
    blur_res = analyze_blur(bgr)
    contrast_res = analyze_contrast(bgr)
    skew_res = analyze_skew(bgr)
    noise_res = analyze_noise(bgr)
    handwriting_res = analyze_handwriting(bgr)
    completeness_res = analyze_completeness(bgr)
    structure_res = analyze_structure(bgr)

    # Standardized defects map (preserving original keys and types)
    defects = {
        "blur": blur_res,
        "contrast": contrast_res,
        "skew": skew_res,
        "noise": noise_res,
        "handwriting": handwriting_res,
        "missing_section": completeness_res,
        "missing_sections": completeness_res["missing"],
        "structure": structure_res
    }

    # 2. Multi-Scale Quality Decomposition (3x3 Grid)
    multiscale_res = analyze_multiscale_quality(bgr, grid_size=(3, 3))

    # 3. Ensemble Quality Modeling
    ensemble_res = calculate_ensemble_quality(defects, structure_res)
    quality_index = ensemble_res["quality_index"]
    legacy_score = ensemble_res["legacy_score"]

    # 4. Normalized Quality Risk Vector
    risk_info = compute_risk_vector(defects, structure_res)

    # 5. Explainable Decision Engine
    decision_trace = evaluate_decision_trace(
        quality_index=quality_index,
        defects=defects,
        ensemble_results=ensemble_res,
        risk_info=risk_info,
        multiscale_info=multiscale_res
    )
    decision = decision_trace["decision"]

    # 6. Counterfactual Sensitivity Analysis
    sensitivity_res = analyze_quality_sensitivity(defects, structure_res)

    # 7. Document Quality Fingerprint
    fingerprint_res = generate_quality_fingerprint(
        quality_index=quality_index,
        defects=defects,
        decision=decision,
        dominant_risk=risk_info["primary_risk"]["label"],
        multiscale_info=multiscale_res
    )

    # 8. Anomaly Detection (Isolation Forest)
    if historical_feature_matrix is None:
        try:
            historical_feature_matrix = get_all_feature_vectors()
        except Exception:
            historical_feature_matrix = []

    anomaly_features = extract_anomaly_features(defects, multiscale_res, structure_res)
    anomaly_res = evaluate_anomaly(anomaly_features, historical_feature_matrix)

    # 9. Corrective Suggestions (Free of OCR references)
    suggestions = generate_engine_suggestions(defects, structure_res)

    # Assemble comprehensive intelligence record
    return {
        # Core identification
        "quality_index": quality_index,
        "legacy_score": legacy_score,
        "decision": decision,
        "intake_accepted": (decision == "PASS"),

        # Decision trace
        "reasons": [decision_trace["primary_reason"]],
        "decision_trigger": decision_trace["decision_trigger"],
        "supporting_evidence": decision_trace["supporting_evidence"],
        "checklist": decision_trace["checklist"],
        "suggestions": suggestions,

        # Defects & Subscores
        "defects": defects,
        "subscores": ensemble_res["subscores"],
        "weights_documented": ensemble_res["weights_documented"],

        # Advanced Intelligence
        "risk_vector": risk_info,
        "multiscale": multiscale_res,
        "sensitivity": sensitivity_res,
        "fingerprint": fingerprint_res,
        "anomaly": anomaly_res,

        # Metadata
        "image_metadata": meta
    }
