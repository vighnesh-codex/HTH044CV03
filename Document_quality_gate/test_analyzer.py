import os
import cv2

from modules.analyzer import analyze_document


DATA_DIR = "Datas"

images = {
    "Missing full Document": os.path.join(DATA_DIR, "miss_full.png"),
    "Blur Document": os.path.join(DATA_DIR, "blur.jpeg"),
    "Low Contrast": os.path.join(DATA_DIR, "contrast.jpg"),
    "Noisy Document": os.path.join(DATA_DIR, "noise.png"),
    "Skewed Document": os.path.join(DATA_DIR, "skew.jpeg"),
    "Handwriting": os.path.join(DATA_DIR, "handwrit.png"),
    "Missing Bottom": os.path.join(DATA_DIR, "miss_bottom.png"),
    "Missing Middle + Bottom": os.path.join(DATA_DIR, "miss_midbot.png"),
    "Missing Top": os.path.join(DATA_DIR, "miss_top.png")
}


for name, path in images.items():
    image = cv2.imread(path)

    if image is None:
        print(f"\nCould not load: {name} (path: {path})")
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
