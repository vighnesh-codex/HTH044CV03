from Document_quality_engine.Main.scoring import calculate_quality_score

defects = {
    "blur": {
        "score": 90,
        "status": "good"
    },
    "contrast": {
        "score": 80,
        "status": "good"
    },
    "skew": {
        "angle": 1.5,
        "status": "warning"
    },
    "noise": {
        "score": 75,
        "status": "moderate"
    },
    "handwriting": {
        "detected": False
    },
    "completeness": {
        "score": 95,
        "status": "good"
    }
}


score = calculate_quality_score(defects)

print("Quality Score:", score)