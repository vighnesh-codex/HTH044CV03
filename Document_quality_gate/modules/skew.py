import cv2
import numpy as np

from modules.preprocessing import preprocess_image


def analyze_skew(image):
 

    _, gray_image = preprocess_image(image)
    edges = cv2.Canny(gray_image, 50, 150)


    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    angles = []

    if lines is not None:
        for line in lines:
      
            x1, y1, x2, y2 = line.reshape(-1)[:4]

            angle = np.degrees(
                np.arctan2(y2 - y1, x2 - x1)
            )

           
            if -45 < angle < 45:
                angles.append(angle)

    if angles:
        angle = float(np.median(angles))
    else:
        angle = 0.0

    if abs(angle) < 1:
        status = "good"
    else:
        status = "warning"

    return {
        "angle": round(angle, 2),
        "status": status
    }