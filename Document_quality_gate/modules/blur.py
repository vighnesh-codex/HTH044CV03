import cv2
from modules.preprocessing import preprocess_image

#Check how clear or blurry the document image is.
def analyze_blur(image):

    _, gray_image = preprocess_image(image)
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    score = min(100, int(sharpness / 10))

    if score >= 50:
        status = "good"
    else:
        status = "poor"

    return {
        "score": score,
        "status": status
    }
