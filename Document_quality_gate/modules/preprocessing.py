import cv2

def preprocess_image(image, max_size=2000):

    if image is None:
        raise ValueError("Invalid image. Please provide a valid document.")

    
    height, width = image.shape[:2]

    # Resize only if the image is too large
    if max(height, width) > max_size:

        scale = max_size / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray_image