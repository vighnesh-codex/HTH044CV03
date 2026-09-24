import cv2
from Document_quality_engine.Modules.scoring import calculate_quality_score
from Document_quality_engine.Modules.routing import route_document
from Document_quality_engine.Modules.suggestion import generate_suggestions
from Document_quality_engine.Modules.ocr import run_ocr
from Document_quality_gate.modules.analyzer import analyze_document

def main():
    image = cv2.imread(
    "Document_quality_gate/Datas/miss_full.png"
    )
    if image is None:
        raise FileNotFoundError("Could not load document image")
    analysis = analyze_document(image)

    defects = {
    "blur": analysis["blur"],

    "contrast": analysis["contrast"],

    "skew": analysis["skew"],

    "noise": analysis["noise"],

    "handwriting": {
        "detected": analysis["handwriting"]["detected"],
        "confidence": analysis["handwriting"]["confidence"]
    },

    "completeness": {
        "score": analysis["completeness"]["score"],
        "status": (
            "good"
            if analysis["completeness"]["score"] >= 80
            else "poor"
        )
    },

    "missing_sections": analysis["completeness"]["missing"]
}

    score = calculate_quality_score(defects)

    decision = route_document(score, defects)

    suggestions = generate_suggestions(defects)

    ocr_result = None

    if decision["ocr_allowed"]:
        ocr_result = run_ocr(image_path)

    print("\n================================")
    print("     DOCUMENT QUALITY GATE")
    print("================================")

    print(f"\nQuality Score : {score}/100")
    print(f"Decision      : {decision['decision']}")

    print("\nDefects:")
    print(f"Blur          : {defects['blur']['score']}")
    print(f"Contrast      : {defects['contrast']['score']}")
    print(f"Skew          : {defects['skew']['angle']} degrees")
    print(f"Noise         : {defects['noise']['score']}")
    print(f"Handwriting   : {defects['handwriting']['detected']}")

    print("\nReasons:")

    if decision["reasons"]:
        for reason in decision["reasons"]:
            print(f"- {reason}")
    else:
        print("- No major issues detected")

    print("\nSuggestions:")

    for suggestion in suggestions:
        print(f"- {suggestion}")

    print("\nOCR:")

    if ocr_result:
        print(f"Confidence : {ocr_result['confidence']}%")
        print(f"Text       : {ocr_result['text']}")
    else:
        print("OCR skipped")


if __name__ == "__main__":
    main()