#this program is to find the score for the documents 
WEIGHTS = {
    "blur": 0.25,
    "contrast": 0.20,
    "skew": 0.15,
    "noise": 0.15,
    "completeness": 0.15,
    "handwriting": 0.10
}


def calculate_quality_score(defects):
    blur = defects["blur"]["score"]
    contrast = defects["contrast"]["score"]
    noise = defects["noise"]["score"]
    completeness = defects.get("completeness", {}).get("score", 100)

    skew_angle = abs(defects["skew"].get("angle", 0))

    if skew_angle <= 1:
        skew = 100
    elif skew_angle <= 3:
        skew = 85
    elif skew_angle <= 5:
        skew = 70
    elif skew_angle <= 8:
        skew = 50
    else:
        skew = 20

    handwriting = 0 if defects["handwriting"]["detected"] else 100

    score = (
        blur * WEIGHTS["blur"] +
        contrast * WEIGHTS["contrast"] +
        skew * WEIGHTS["skew"] +
        noise * WEIGHTS["noise"] +
        completeness * WEIGHTS["completeness"] +
        handwriting * WEIGHTS["handwriting"]
    )

    return round(max(0, min(100, score)), 2)