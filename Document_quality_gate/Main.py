import cv2

from backend_modules.scoring import calculate_quality_score
from backend_modules.routing import route_document
from backend_modules.suggestion import generate_suggestions

from modules.analyzer import analyze_document

def main():

    images = {
    "Missing full Document": "Document_quality_gate/Datas/miss_full.png",
    "Blur Document": "Document_quality_gate/Datas/blur.jpeg",
    "Low Contrast": "Document_quality_gate/Datas/contrast.jpg",
    "Noisy Document": "Document_quality_gate/Datas/noise.png",
    "Skewed Document": "Document_quality_gate/Datas/skew.jpeg",
    "Handwriting": "Document_quality_gate/Datas/handwrit.png",
    "Missing Bottom": "Document_quality_gate/Datas/miss_bottom.png",
    "Missing Middle + Bottom": "Document_quality_gate/Datas/miss_midbot.png",
    "Missing Top": "Document_quality_gate/Datas/miss_top.png"
}

    for name, path in images.items():
        image = cv2.imread(path)

        if image is None:
            print(f"\nCould not load: {name}")
            continue
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

        "missing_section": {
            "score": analysis["missing_section"]["score"],
            "status": (
                "good"
                if analysis["missing_section"]["score"] >= 80
                else "poor"
            )
        },

        "missing_sections": analysis["missing_section"]["missing"]
    }

        score = calculate_quality_score(defects)

        decision = route_document(score, defects)

        suggestions = generate_suggestions(defects)

        print("\n================================")
        print(name)
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



if __name__ == "__main__":
    main()