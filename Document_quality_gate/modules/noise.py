import cv2
from modules.preprocessing import preprocess_image

#Check the amount of noise in the document image.
def analyze_noise(image):
   
    _, gray_image = preprocess_image(image)
    smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Difference between original and smooth image
    noise = cv2.absdiff(gray_image, smooth_image)
    

   
    noise_level = noise.mean()
    score = max(0, min(100, int(100 - noise_level * 5)))

    if score >= 70:
        status = "good"
    elif score >= 40:
        status = "moderate"
    else:
        status = "poor"

    return {
        "score": score,
        "status": status
    }