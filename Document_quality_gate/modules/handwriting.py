import cv2

from modules.preprocessing import preprocess_image


def analyze_handwriting(image):

    _, gray_image = preprocess_image(image)

    # Convert image to black and white
    binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

  
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )


    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 40)
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    # Remove document lines
    cleaned = binary - horizontal_lines - vertical_lines

    # Find connected regions
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    handwriting_regions = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        x, y, width, height = cv2.boundingRect(contour)

        # Ignore small printed characters and noise
        if area < 150:
            continue

        # Ignore extremely large regions
        if area > 15000:
            continue

        # Ignore very long straight regions
        if width > 5 * height or height > 5 * width:
            continue

        handwriting_regions += 1

    # Convert detected regions into confidence
    confidence = min(handwriting_regions / 10, 1.0)

    # Require several handwriting-like regions
    detected = confidence >= 0.5

    return {
        "detected": detected,
        "confidence": round(confidence, 2)
    }