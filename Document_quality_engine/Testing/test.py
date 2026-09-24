from Document_quality_engine.Main.scoring import calculate_quality_score
from Document_quality_engine.Main.routing import route_document
from Document_quality_engine.Main.suggestion import generate_suggestions


defects = {
    "blur": {
        "score": 95,
        "status": "good"
    },

    "contrast": {
        "score": 90,
        "status": "good"
    },

    "skew": {
        "angle": 0.5,
        "status": "good"
    },

    "noise": {
        "score": 90,
        "status": "good"
    },

    "handwriting": {
        "detected": False
    },

    "completeness": {
        "score": 100,
        "status": "good"
    },

    "missing_sections": []
}

score = calculate_quality_score(defects)
result = route_document(score, defects)

suggestions = generate_suggestions(defects)

print("Quality Score:", score)
print("Decision:", result["decision"])
print("Reasons:", result["reasons"])
print("OCR Allowed:", result["ocr_allowed"])
print("Suggestions:", suggestions)
