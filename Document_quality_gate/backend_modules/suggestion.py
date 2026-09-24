# Corrective suggestion generator with handwriting legibility guidance (OCR-free)

def generate_suggestions(defects):
    suggestions = []

    if defects["blur"]["status"] == "poor":
        suggestions.append(
            "Rescan the document with better focus."
        )

    if defects["contrast"]["status"] == "poor":
        suggestions.append(
            "Improve lighting or increase image contrast."
        )

    skew_stat = defects["skew"].get("status", "good")
    skew_angle = abs(defects["skew"].get("angle", 0.0))
    if skew_stat == "poor" or skew_angle > 3.0:
        suggestions.append(
            "Straighten or deskew the document."
        )

    if defects["noise"]["status"] == "poor":
        suggestions.append(
            "Apply image denoising before document intake."
        )

    hw = defects.get("handwriting", {})
    if hw.get("detected", False):
        hw_verdict = hw.get("legibility_verdict", "NEEDS REVIEW")
        hw_score = hw.get("score", 50.0)
        if hw_verdict == "REJECT":
            suggestions.append(
                f"Resubmit document: handwritten text is scribbled or illegible (Legibility Score: {hw_score:.0f}/100)."
            )
        elif hw_verdict == "NEEDS REVIEW":
            suggestions.append(
                f"Route document to manual operator verification due to marginal handwriting legibility ({hw_score:.0f}/100)."
            )

    missing_sections = defects.get("missing_sections", [])
    if not missing_sections:
        missing_sections = defects.get("missing_section", {}).get("missing", [])

    if missing_sections:
        sections = ", ".join(missing_sections)
        suggestions.append(
            f"Request the missing sections: {sections}."
        )

    if not suggestions:
        suggestions.append(
            "No corrective action required. Document is suitable for processing."
        )

    return suggestions
