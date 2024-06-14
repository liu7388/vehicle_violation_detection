import cv2
import numpy as np
import pathlib
import os

def detect_and_draw_hough_lines(image_path):
    """
    Detects and draws Hough lines on an input image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Hough line detection
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=160, minLineLength=100, maxLineGap=1)

    # Draw detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the image with Hough lines
    cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough Lines', 1280, 720)
    cv2.imshow('Hough Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
path = os.path.abspath(__file__)
image_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1]) + '/inference/images/lanes_mask/lanes_mask.png'
print(image_path)
detect_and_draw_hough_lines(image_path)
