#this program is for the person whos submitting the documents which is rejected
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

    if defects["skew"]["status"] == "poor":
        suggestions.append(
            "Straighten or deskew the document."
        )

    if defects["noise"]["status"] == "poor":
        suggestions.append(
            "Apply image denoising before OCR."
        )

    if defects["handwriting"]["detected"]:
        suggestions.append(
            "Send the document for human review because handwriting interference was detected."
        )

    missing_sections = defects.get("missing_sections", [])

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