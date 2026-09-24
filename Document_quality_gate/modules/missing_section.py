import cv2

from modules.preprocessing import preprocess_image


def analyze_missing_section(image):
    """
    Check whether important regions of a document contain content.

    Returns:
        score: Completeness score from 0 to 100.
        missing: List of regions that appear empty.
    """

    # Preprocess image
    _, gray_image = preprocess_image(image)

    height, width = gray_image.shape

    # Divide the document into four regions
    regions = {
        "header": gray_image[:int(height * 0.20), :],
        "upper": gray_image[int(height * 0.20):int(height * 0.40), :],
        "middle": gray_image[int(height * 0.40):int(height * 0.75), :],
        "bottom": gray_image[int(height * 0.75):, :]
    }

    missing = []

    for name, region in regions.items():

        # Convert to black and white
        _, binary = cv2.threshold(
            region,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Ignore the document border
        border = 10

        if region.shape[0] > border * 2 and region.shape[1] > border * 2:
            binary = binary[
                border:-border,
                border:-border
            ]

        # Calculate actual content
        content_ratio = cv2.countNonZero(binary) / binary.size

        # Less than 3% = probably empty
        if content_ratio < 0.03:
            missing.append(name)

    # Calculate score
    total_regions = len(regions)
    completed_regions = total_regions - len(missing)

    score = int(
        (completed_regions / total_regions) * 100
    )

    return {
        "score": score,
        "missing": missing
    }