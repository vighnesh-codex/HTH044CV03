import pytesseract

def run_ocr(image_path):

    text = pytesseract.image_to_string(image_path)

    data = pytesseract.image_to_data(
        image_path,
        output_type=pytesseract.Output.DICT
    )

    confidence_values = []

    for confidence in data["conf"]:
        if confidence != "-1":
            confidence_values.append(float(confidence))

    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)
    else:
        average_confidence = 0

    return {
        "text": text.strip(),
        "confidence": round(average_confidence, 2)
    }