# Document intake routing logic with handwriting legibility assessment (OCR-free)

def route_document(score, defects):
    reasons = []

    if defects["blur"]["status"] == "poor":
        reasons.append("Severe blur detected")

    if defects["contrast"]["status"] == "poor":
        reasons.append("Low contrast detected")

    skew_angle = abs(defects["skew"]["angle"])
    if skew_angle > 5:
        reasons.append("Significant document skew detected")

    if defects["noise"]["status"] == "poor":
        reasons.append("High image noise detected")

    hw_detected = defects["handwriting"].get("detected", False)
    hw_score = defects["handwriting"].get("score", 100.0)
    hw_verdict = defects["handwriting"].get("legibility_verdict", "NEEDS REVIEW")

    if hw_detected:
        if hw_verdict == "REJECT":
            reasons.append(f"Illegible or scribbled handwriting detected (Legibility Score: {hw_score:.0f}/100)")
        elif hw_verdict == "NEEDS REVIEW":
            reasons.append(f"Marginal handwriting legibility detected (Score: {hw_score:.0f}/100)")

    missing_sections = defects.get("missing_sections", [])
    if not missing_sections:
        missing_sections = defects.get("missing_section", {}).get("missing", [])

    if missing_sections:
        reasons.append(
            "Missing sections: " + ", ".join(missing_sections)
        )

    # Base thresholds
    if score >= 85:
        decision = "PASS"
    elif score >= 60:
        decision = "NEEDS REVIEW"
    else:
        decision = "REJECT"

    # Handwriting legibility routing
    if hw_detected:
        if hw_verdict == "REJECT":
            decision = "REJECT"
        elif hw_verdict == "NEEDS REVIEW" and decision == "PASS":
            decision = "NEEDS REVIEW"

    if missing_sections and decision == "PASS":
        decision = "NEEDS REVIEW"

    intake_accepted = (decision == "PASS")

    return {
        "decision": decision,
        "reasons": reasons,
        "intake_accepted": intake_accepted
    }
