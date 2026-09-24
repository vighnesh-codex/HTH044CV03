import os
import cv2

from backend_modules.scoring import calculate_quality_score
from backend_modules.routing import route_document
from backend_modules.suggestion import generate_suggestions
from modules.analyzer import analyze_document
from cv.pdf_loader import is_pdf, load_single_pdf_page
from intelligence.engine import analyze_document_quality, analyze_pdf_quality

DATA_DIR = "Datas"

def main():
    documents = {
        "Missing full Document": os.path.join(DATA_DIR, "miss_full.png"),
        "Blur Document": os.path.join(DATA_DIR, "blur.jpeg"),
        "Low Contrast": os.path.join(DATA_DIR, "contrast.jpg"),
        "Noisy Document": os.path.join(DATA_DIR, "noise.png"),
        "Skewed Document": os.path.join(DATA_DIR, "skew.jpeg"),
        "Handwriting": os.path.join(DATA_DIR, "handwrit.png"),
        "Missing Bottom": os.path.join(DATA_DIR, "miss_bottom.png"),
        "Missing Middle + Bottom": os.path.join(DATA_DIR, "miss_midbot.png"),
        "Missing Top": os.path.join(DATA_DIR, "miss_top.png"),
        "Multi-Page PDF Document": os.path.join(DATA_DIR, "sample_document.pdf")
    }

    for name, path in documents.items():
        if not os.path.exists(path):
            print(f"\nCould not find: {name} (path: {path})")
            continue

        # If document is a PDF, run multi-page PDF quality intelligence
        if is_pdf(path):
            print("\n================================")
            print(f"{name} [PDF Document]")
            print("================================")
            pdf_res = analyze_pdf_quality(path)
            print(f"\n[Multi-Page PDF Intake Summary]")
            print(f"Total Pages            : {pdf_res['total_pages']}")
            print(f"Mean Quality Index     : {pdf_res['mean_quality_index']}/100")
            print(f"Bottleneck Page Score  : {pdf_res['bottleneck_quality_index']}/100 (Page {pdf_res['bottleneck_page']})")
            print(f"Overall Intake Decision: {pdf_res['decision']}")
            print(f"Decision Trigger       : {pdf_res['decision_trigger']}")
            print("\nPage-by-Page Quality Breakdown:")
            for p_info in pdf_res["summary_table"]:
                print(f"  - Page {p_info['page']}: Quality Index={p_info['quality_index']:.1f}/100 | Decision={p_info['decision']:12s} | Dominant Risk={p_info['dominant_risk']}")
            continue

        image = cv2.imread(path)
        if image is None:
            print(f"\nCould not load: {name} (path: {path})")
            continue

        # 1. Baseline analysis
        analysis = analyze_document(image)

        defects = {
            "blur": analysis["blur"],
            "contrast": analysis["contrast"],
            "skew": analysis["skew"],
            "noise": analysis["noise"],
            "handwriting": analysis["handwriting"],
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

        print(f"\n[Baseline] Quality Score : {score}/100")
        print(f"[Baseline] Decision      : {decision['decision']}")

        print("\nDefects:")
        print(f"Blur          : {defects['blur']['score']}")
        print(f"Contrast      : {defects['contrast']['score']}")
        print(f"Skew          : {defects['skew']['angle']} degrees")
        print(f"Noise         : {defects['noise']['score']}")
        print(f"Handwriting   : Detected={defects['handwriting']['detected']}, Legibility Score={defects['handwriting'].get('score', 100):.1f}/100 ({defects['handwriting'].get('legibility_verdict', 'PASS')})")

        print("\nReasons:")
        if decision["reasons"]:
            for reason in decision["reasons"]:
                print(f"- {reason}")
        else:
            print("- No major issues detected")

        print("\nSuggestions:")
        for suggestion in suggestions:
            print(f"- {suggestion}")

        # 2. Advanced Document Quality Intelligence
        intel = analyze_document_quality(image)
        print("\n--- Advanced Quality Intelligence ---")
        print(f"Ensemble Quality Index : {intel['quality_index']}/100")
        print(f"Primary Risk Driver    : {intel['risk_vector']['primary_risk']['label']} (Risk: {intel['risk_vector']['primary_risk']['value']:.2f})")
        print(f"Decision Trigger       : {intel['decision_trigger']}")
        print(f"Quality Fingerprint    : {intel['fingerprint']['quality_fingerprint']}")
        print(f"Dominant Quality Driver: {intel['sensitivity']['dominant_driver']}")
        print(f"Regional Uniformity    : {intel['multiscale']['diagnosis']}")


if __name__ == "__main__":
    main()
