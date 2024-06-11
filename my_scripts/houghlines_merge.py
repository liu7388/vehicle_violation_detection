import cv2
import numpy as np
from pathlib import Path
import os
from math import acos, sqrt, degrees
from my_scripts.merge_similar_lane_detections import merge_similar_lane_detections

theda_history = {}
grouped_lines = {}
previous_grouped_lines = {}

def houghlines_merge(gray):
    def lines_filter(lines):
        global theda_history
        global grouped_lines
        global previous_grouped_lines

        best_lines = {}
        grouped_lines = {}

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                # 將角度相似的線分組
                theda_interval = 3
                if y2 - y1 >= 0: theda = 90 - degrees(acos((x2 - x1) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
                else: theda = (-1) * (90 - degrees(acos((x2 - x1) / sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))))
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

        # print(f"\n\nAlive lines theda: {grouped_lines.keys()}")
        print(f"\n\nHistory lines' weight: {theda_history}")

        # 找每個角度中最長的線
        for theda, group in grouped_lines.items():
            max_length = 0
            best_line = None

            for line in group:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 计算线段长度

                if length > max_length:
                    max_length = length
                    best_line = line

            if best_line is not None:
                best_lines[theda] = best_line

        return best_lines

    # 霍夫直線偵測
    threshold = 140
    lines = cv2.HoughLinesP(gray, 1, np.pi/360, threshold=threshold, minLineLength=80, maxLineGap=50)

    # 過濾偵測出來的線（），回傳一個 dictionary
    best_lines_dictionary = lines_filter(lines)

    # 合併相似角度的線，回傳得到一個 list [ [[x1, y1, x2, y2]], [[x1, y1, x2, y2]], ... ]
    best_lines_merged_list = merge_similar_lane_detections(best_lines_dictionary)

    return best_lines_merged_list


if __name__ == "__main__":
    # 讀取圖片
    image_path = str(Path(os.path.abspath(__file__)).parents[1]) + '/lanes_mask.png'
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 設定kernel
    kernel = np.ones((5, 5), np.uint8)
    # 膨脹
    gray = cv2.dilate(gray, kernel, iterations=1)
    # 侵蝕
    gray = cv2.erode(gray, kernel, iterations=2)

    merged_lines = houghlines_merge(gray)

    # 畫出直線
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

