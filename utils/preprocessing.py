import cv2
import numpy as np


def preprocess(image_path):
    """
    Keep only black pixels in an image, remove everything else.

    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the result.
        threshold (int): Pixel intensity threshold to consider "black". 0-255.

    Returns:
        None
    """

    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150)

    return img


def remove_blue(image, output_path="outputs/preprocessing/no_blue.png"):
    """
    Remove blue parts of an image and keep everything else.

    Args:
        image_path (str): Path to the image.
        output_path (str): Path to save the result.

    Returns:
        None
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue range in HSV
    lower_blue = np.array([100, 60, 50])  # H, S, V
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue pixels
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # replace blue pixel with white
    image[blue_mask > 0] = [255, 255, 255]

    cv2.imwrite(output_path, image)
    print(f"Image with blue removed saved to {output_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    finished_preprocessing = cv2.Canny(gray, 50, 150)
    # cv2.imwrite("outputs/finished_preprocessing.png", finished_preprocessing)
    return finished_preprocessing
