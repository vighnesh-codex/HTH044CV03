from modules.preprocessing import preprocess_image

#Check the contrast of the document image.
def analyze_contrast(image):

    _, gray_image = preprocess_image(image)
    contrast = gray_image.std()

    
    score = min(100, int(contrast * 2))

    
    if score >= 50:
        status = "good"
    else:
        status = "poor"

    return {
        "score": score,
        "status": status
    }

