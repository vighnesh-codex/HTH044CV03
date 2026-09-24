"""
Quality Risk Vector & Explainable Decision Engine
Translates multidimensional defect measurements into a normalized risk vector,
ranks primary/secondary risk drivers, and builds dynamic, rule-explainable decision traces.
Includes intelligent handwriting legibility routing (PASS/REVIEW/REJECT).
"""

from typing import Dict, Any, List, Tuple


RISK_LABELS = {
    "blur_risk": "Blur / Optical Defocus",
    "contrast_risk": "Low Tonal Contrast",
    "noise_risk": "Image Noise / Grain",
    "skew_risk": "Geometric Skew Angle",
    "handwriting_risk": "Handwriting Legibility Risk",
    "completeness_risk": "Missing Document Section",
    "structure_risk": "Layout / Illumination Defect"
}


def compute_risk_vector(
    defects: Dict[str, Any],
    structure_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Computes normalized risk probabilities [0.0 - 1.0] across all quality failure modes.
    """
    blur_score = float(defects["blur"]["score"])
    contrast_score = float(defects["contrast"]["score"])
    noise_score = float(defects["noise"]["score"])

    # Skew angle risk
    skew_angle = abs(float(defects["skew"].get("angle", 0.0)))
    skew_risk = min(1.0, round(skew_angle / 8.0, 2))

    # Continuous Handwriting Legibility Risk
    hw_detected = defects["handwriting"].get("detected", False)
    hw_legibility = float(defects["handwriting"].get("score", 100.0))
    hw_risk = round(max(0.0, (100.0 - hw_legibility) / 100.0), 2) if hw_detected else 0.0

    # Completeness risk
    comp_score = float(defects.get("missing_section", {}).get("score", 100.0))
    completeness_risk = round((100.0 - comp_score) / 100.0, 2)

    # Blur, Contrast, Noise risk
    blur_risk = round(max(0.0, (100.0 - blur_score) / 100.0), 2)
    contrast_risk = round(max(0.0, (100.0 - contrast_score) / 100.0), 2)
    noise_risk = round(max(0.0, (100.0 - noise_score) / 100.0), 2)

    # Structure risk
    if structure_result:
        struct_score = float(structure_result.get("score", 95.0))
    else:
        struct_score = 95.0
    structure_risk = round(max(0.0, (100.0 - struct_score) / 100.0), 2)

    risk_vector = {
        "blur_risk": blur_risk,
        "contrast_risk": contrast_risk,
        "noise_risk": noise_risk,
        "skew_risk": skew_risk,
        "handwriting_risk": hw_risk,
        "completeness_risk": completeness_risk,
        "structure_risk": structure_risk
    }

    # Rank risks
    sorted_risks = sorted(risk_vector.items(), key=lambda kv: kv[1], reverse=True)

    primary = {
        "key": sorted_risks[0][0],
        "label": RISK_LABELS.get(sorted_risks[0][0], sorted_risks[0][0]),
        "value": sorted_risks[0][1]
    }

    secondary = {
        "key": sorted_risks[1][0],
        "label": RISK_LABELS.get(sorted_risks[1][0], sorted_risks[1][0]),
        "value": sorted_risks[1][1]
    }

    low_risk_factors = [
        {"key": k, "label": RISK_LABELS.get(k, k), "value": v}
        for k, v in sorted_risks if v <= 0.25
    ]

    return {
        "vector": risk_vector,
        "primary_risk": primary,
        "secondary_risk": secondary,
        "low_risk_factors": low_risk_factors,
        "ranking": sorted_risks
    }


def evaluate_decision_trace(
    quality_index: float,
    defects: Dict[str, Any],
    ensemble_results: Dict[str, Any],
    risk_info: Dict[str, Any],
    multiscale_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Synthesizes an explainable, step-by-step decision trace with trigger provenance.
    Evaluates handwriting legibility dynamically (PASS / REVIEW / REJECT).
    """
    subscores = ensemble_results.get("subscores", {})
    checklist: List[Dict[str, Any]] = []

    # 1. Blur verification
    blur_score = defects["blur"]["score"]
    blur_status = defects["blur"]["status"]
    if blur_status == "good":
        blur_item = {"name": "Blur", "state": "pass", "icon": "✓", "msg": f"Blur acceptable (Score: {blur_score}/100)"}
    elif blur_status == "moderate":
        blur_item = {"name": "Blur", "state": "warn", "icon": "⚠", "msg": f"Moderate blur observed (Score: {blur_score}/100)"}
    else:
        blur_item = {"name": "Blur", "state": "fail", "icon": "✗", "msg": f"Severe optical defocus detected (Score: {blur_score}/100)"}
    checklist.append(blur_item)

    # 2. Contrast verification
    cont_score = defects["contrast"]["score"]
    cont_status = defects["contrast"]["status"]
    if cont_status == "good":
        cont_item = {"name": "Contrast", "state": "pass", "icon": "✓", "msg": f"Tonal contrast acceptable (Score: {cont_score}/100)"}
    elif cont_status == "moderate":
        cont_item = {"name": "Contrast", "state": "warn", "icon": "⚠", "msg": f"Suboptimal contrast observed (Score: {cont_score}/100)"}
    else:
        cont_item = {"name": "Contrast", "state": "fail", "icon": "✗", "msg": f"Severe low contrast detected (Score: {cont_score}/100)"}
    checklist.append(cont_item)

    # 3. Skew verification
    skew_angle = abs(defects["skew"]["angle"])
    if skew_angle <= 1.0:
        skew_item = {"name": "Skew", "state": "pass", "icon": "✓", "msg": f"Skew acceptable ({skew_angle:.1f}°)"}
    elif skew_angle <= 5.0:
        skew_item = {"name": "Skew", "state": "warn", "icon": "⚠", "msg": f"Minor skew deviation ({skew_angle:.1f}°)"}
    else:
        skew_item = {"name": "Skew", "state": "fail", "icon": "✗", "msg": f"Significant document skew detected ({skew_angle:.1f}° > 5°)"}
    checklist.append(skew_item)

    # 4. Noise verification
    noise_score = defects["noise"]["score"]
    noise_status = defects["noise"]["status"]
    if noise_status == "good":
        noise_item = {"name": "Noise", "state": "pass", "icon": "✓", "msg": f"Noise level acceptable (Score: {noise_score}/100)"}
    elif noise_status == "moderate":
        noise_item = {"name": "Noise", "state": "warn", "icon": "⚠", "msg": f"Moderate noise grain detected (Score: {noise_score}/100)"}
    else:
        noise_item = {"name": "Noise", "state": "fail", "icon": "✗", "msg": f"High image noise detected (Score: {noise_score}/100)"}
    checklist.append(noise_item)

    # 5. Completeness verification
    missing_sections = defects.get("missing_sections", [])
    if not missing_sections:
        missing_sections = defects.get("missing_section", {}).get("missing", [])
    comp_score = defects.get("missing_section", {}).get("score", 100)
    if not missing_sections and comp_score >= 80:
        comp_item = {"name": "Completeness", "state": "pass", "icon": "✓", "msg": f"Document complete ({comp_score}%)"}
    elif comp_score >= 50:
        comp_item = {"name": "Completeness", "state": "warn", "icon": "⚠", "msg": f"Missing zones: {', '.join(missing_sections)}"}
    else:
        comp_item = {"name": "Completeness", "state": "fail", "icon": "✗", "msg": f"Critical missing sections: {', '.join(missing_sections)}"}
    checklist.append(comp_item)

    # 6. Handwriting Legibility Verification (Intelligent 3-Tier Routing)
    hw_detected = defects["handwriting"].get("detected", False)
    hw_score = float(defects["handwriting"].get("score", 100.0))
    hw_verdict = defects["handwriting"].get("legibility_verdict", "PASS")

    if not hw_detected:
        hw_item = {"name": "Handwriting", "state": "pass", "icon": "✓", "msg": "No handwriting interference detected (Clean printed text)"}
    elif hw_verdict == "PASS":
        hw_item = {"name": "Handwriting", "state": "pass", "icon": "✓", "msg": f"Handwriting verified legible (Score: {hw_score:.0f}/100 - PASS)"}
    elif hw_verdict == "NEEDS REVIEW":
        hw_item = {"name": "Handwriting", "state": "warn", "icon": "⚠", "msg": f"Marginal handwriting legibility (Score: {hw_score:.0f}/100 - REVIEW)"}
    else:
        hw_item = {"name": "Handwriting", "state": "fail", "icon": "✗", "msg": f"Illegible or scribbled handwriting (Score: {hw_score:.0f}/100 - REJECT)"}
    checklist.append(hw_item)

    # Base Score Decision Thresholds
    if quality_index >= 85.0:
        decision = "PASS"
        trigger = "Quality index exceeds acceptance threshold (>= 85.0)"
        reason = "All quality parameters within operational tolerances."
    elif quality_index >= 60.0:
        decision = "NEEDS REVIEW"
        trigger = "Quality index in review band (60.0 - 84.9)"
        reason = f"Moderate quality degradation identified (Score: {quality_index:.1f})."
    else:
        decision = "REJECT"
        trigger = "Quality index below minimum threshold (< 60.0)"
        reason = f"Severe overall quality failure (Score: {quality_index:.1f})."

    # Handwriting Intelligent Routing Overrides
    if hw_detected:
        if hw_verdict == "REJECT" and decision != "REJECT":
            decision = "REJECT"
            trigger = f"Illegible handwriting threshold (< 50, Score: {hw_score:.0f})"
            reason = f"Handwritten text is illegible, smudged, or scribbled (Legibility Score: {hw_score:.0f}/100)."
        elif hw_verdict == "NEEDS REVIEW" and decision == "PASS":
            decision = "NEEDS REVIEW"
            trigger = f"Marginal handwriting legibility (Score: {hw_score:.0f})"
            reason = f"Handwriting detected with marginal legibility ({hw_score:.0f}/100) requiring operator verification."

    # Completeness Override
    if missing_sections and decision == "PASS":
        decision = "NEEDS REVIEW"
        trigger = "Completeness override rule"
        reason = f"Missing functional document zones detected: {', '.join(missing_sections)}."

    # Fatal Rejection Overrides
    if comp_score < 40:
        decision = "REJECT"
        trigger = "Fatal completeness threshold (< 40%)"
        reason = f"Document is largely blank or truncated ({', '.join(missing_sections)})."

    if blur_status == "poor" and blur_score < 20:
        decision = "REJECT"
        trigger = "Severe blur threshold failure (< 20)"
        reason = "Optical blur is too severe for standard processing."

    # Compound defect rejection: 2 or more poor defects
    poor_defects = [
        name for name, d in [
            ("blur", defects["blur"]),
            ("contrast", defects["contrast"]),
            ("noise", defects["noise"])
        ] if d.get("status") == "poor"
    ]
    if skew_angle > 5.0:
        poor_defects.append("skew")
    if hw_detected and hw_verdict == "REJECT":
        poor_defects.append("handwriting")

    if len(poor_defects) >= 2 and decision != "REJECT":
        decision = "REJECT"
        trigger = f"Compound defect threshold ({len(poor_defects)} defects rated poor: {', '.join(poor_defects)})"
        reason = f"Multiple simultaneous critical defects ({', '.join(poor_defects)}) preclude reliable intake."

    # Supporting evidence list
    evidence_list = [
        f"Overall Quality Index: {quality_index:.1f}/100",
        f"Visual Quality: {subscores.get('visual_quality', 'N/A')}/100",
        f"Structural Quality: {subscores.get('structural_quality', 'N/A')}/100",
        f"Readability Risk: {subscores.get('readability_risk', 'N/A')}/100",
        f"Completeness: {subscores.get('completeness', 'N/A')}%",
        f"Handwriting Legibility: {hw_score:.1f}/100 (Routing: {hw_verdict})",
        f"Primary Risk Driver: {risk_info.get('primary_risk', {}).get('label', 'None')}"
    ]

    return {
        "decision": decision,
        "primary_reason": reason,
        "decision_trigger": trigger,
        "supporting_evidence": evidence_list,
        "checklist": checklist
    }
