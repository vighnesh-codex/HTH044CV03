#this program uses the score to decide to wheather forward or reject or review the document
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

    if defects["handwriting"]["detected"]:
        reasons.append("Handwriting interference detected")

    missing_sections = defects.get("missing_sections", [])

    if missing_sections:
        reasons.append(
            "Missing sections: " + ", ".join(missing_sections)
        )

    if score >= 85:
        decision = "PASS"
    elif score >= 60:
        decision = "NEEDS REVIEW"
    else:
        decision = "REJECT"

    if defects["handwriting"]["detected"] and decision == "PASS":
        decision = "NEEDS REVIEW"

    if missing_sections and decision == "PASS":
        decision = "NEEDS REVIEW"

    ocr_allowed = decision == "PASS"

    return {
        "decision": decision,
        "reasons": reasons,
        "ocr_allowed": ocr_allowed
    }