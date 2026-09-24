import cv2

from modules.analyzer import analyze_document


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

    result = analyze_document(image)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Blur:", result["blur"])
    print("Contrast:", result["contrast"])
    print("Skew:", result["skew"])
    print("Noise:", result["noise"])
    print("Handwriting:", result["handwriting"])
    print("missing_section:", result["missing_section"])