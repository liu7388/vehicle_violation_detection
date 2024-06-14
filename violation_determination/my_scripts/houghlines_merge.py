import cv2
import numpy as np
from pathlib import Path
import os
from math import acos, sqrt, degrees
from violation_determination.my_scripts.merge_similar_lane_detections import merge_similar_lane_detections

theda_history = {}
grouped_lines = {}
previous_grouped_lines = {}

def houghlines_merge(gray):
    """
    Detects and merges Hough lines with similar angles from a grayscale image.

    Args:
        gray (numpy.ndarray): Grayscale image containing lane markings.

    Returns:
        list: List of merged lane lines represented as endpoints [[x1, y1, x2, y2]].
    """
    def lines_filter(lines):
        """
        Filters lines based on their angle and groups them by similar angles.

        Args:
            lines (numpy.ndarray): Array of detected lines from Hough transform.

        Returns:
            dict: Dictionary containing filtered lines grouped by similar angles.
        """
        global theda_history
        global grouped_lines
        global previous_grouped_lines

        best_lines = {}
        grouped_lines = {}

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                # Group lines with similar angles
                theda_interval = 3
                if y2 - y1 >= 0:
                    theda = 90 - degrees(acos((x2 - x1) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
                else:
                    theda = (-1) * (90 - degrees(acos((x2 - x1) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))))
                q, r = divmod(theda, theda_interval)
                adjusted_theda = int(q * theda_interval + round(r * (1 / theda_interval)) * theda_interval)
                if -80 <= theda <= 80:
                    grouped_lines.setdefault(adjusted_theda, []).append(line)

        theda_max_weight = 250
        theda_accept_weight = 30
        theda_abandon_weight = 20
        theda_weight_increase_step = 2
        theda_weight_decrease_step = 3
        for theda in list(theda_history.keys()):
            if grouped_lines.get(theda) is None:
                if theda_history[theda] <= round(theda_abandon_weight):
                    del theda_history[theda]
                else:
                    grouped_lines[theda] = previous_grouped_lines[theda]
                    theda_history[theda] -= theda_weight_decrease_step

        previous_grouped_lines = grouped_lines.copy()

        for theda in list(grouped_lines.keys()):
            if theda_history.get(theda) is None:
                del grouped_lines[theda]
                theda_history[theda] = 1
            elif theda_history[theda] == theda_max_weight:
                pass
            elif theda_history[theda] >= theda_accept_weight:
                theda_history[theda] += theda_weight_increase_step
            elif 0 < theda_history[theda] < theda_accept_weight:
                del grouped_lines[theda]
                theda_history[theda] += theda_weight_increase_step

        print(f"\n\nHistory lines' weight: {theda_history}")

        # Find the longest line for each angle
        for theda, group in grouped_lines.items():
            max_length = 0
            best_line = None

            for line in group:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if length > max_length:
                    max_length = length
                    best_line = line

            if best_line is not None:
                best_lines[theda] = best_line

        return best_lines

    # Hough line detection
    threshold = 140
    lines = cv2.HoughLinesP(gray, 1, np.pi/360, threshold=threshold, minLineLength=80, maxLineGap=50)

    # Filter detected lines and return as a dictionary
    best_lines_dictionary = lines_filter(lines)

    # Merge lines with similar angles and return as a list [[x1, y1, x2, y2]]
    best_lines_merged_list = merge_similar_lane_detections(best_lines_dictionary)

    return best_lines_merged_list


if __name__ == "__main__":
    # Read the image
    image_path = str(Path(os.path.abspath(__file__)).parents[1]) + '/lanes_mask.png'
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)
    # Dilate the image
    gray = cv2.dilate(gray, kernel, iterations=1)
    # Erode the image
    gray = cv2.erode(gray, kernel, iterations=2)

    # Obtain merged Hough lines
    merged_lines = houghlines_merge(gray)

    # Draw the lines on the image
    if merged_lines is not None:
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough Lines', 1280, 720)
    cv2.imshow('Hough Lines', image)
    cv2.imwrite('houghlines_merge.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
