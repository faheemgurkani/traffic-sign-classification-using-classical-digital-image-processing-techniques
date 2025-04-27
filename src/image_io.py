import cv2
import numpy as np



def read_image(path):
    """
    Read an image from disk using OpenCV and return as NumPy array (RGB).
    Returns None if read fails.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)

    if bgr is None:
        return None

    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb