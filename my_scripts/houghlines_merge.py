import cv2
import numpy as np
import pathlib
import os

def houghlines_merge(gray):
    def merge_similar_lines(lines):
        merged_lines = []
        grouped_lines = {}

        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算斜率
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                # 将斜率相似的线分组
                grouped_lines.setdefault(round(slope, 1), []).append(line)

        # 对每个斜率组中的线找到位置最置中的线段
        for slope, group in grouped_lines.items():
            print(f"Slope: {slope}")
            print(f"Group: {group}")

            # 计算每条线段的中点
            midpoints = []
            for line in group:
                x1, y1, x2, y2 = line[0]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                midpoints.append((mid_x, mid_y))

            # 计算组中所有中点的平均值
            avg_mid_x = np.mean([pt[0] for pt in midpoints])
            avg_mid_y = np.mean([pt[1] for pt in midpoints])

            # 找到距离平均中点最近的线段
            min_distance = float('inf')
            best_line = None
            for i, (mid_x, mid_y) in enumerate(midpoints):
                distance = np.sqrt((mid_x - avg_mid_x) ** 2 + (mid_y - avg_mid_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_line = group[i]

            # 添加最置中的线段
            merged_lines.append(best_line)

            # 绘制所有的线段
            # for line in group:
            #     x1, y1, x2, y2 = line[0]
            #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return merged_lines

    # 进行霍夫直线变换
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=10)

    # 合并相似斜率的线
    merged_lines = merge_similar_lines(lines)

    return merged_lines


if __name__ == "__main__":
    # 构建图像路径
    image_path = str(pathlib.Path(os.path.abspath(__file__)).parents[1]) + '/lanes_mask.png'

    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 定义内核
    kernel = np.ones((5, 5), np.uint8)
    # 进行膨胀操作
    gray = cv2.dilate(gray, kernel, iterations=1)
    # 进行侵蚀操作
    gray = cv2.erode(gray, kernel, iterations=1)

    merged_lines = houghlines_merge(gray)

    # 绘制检测到的直线
    if merged_lines is not None:
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 显示结果
    cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough Lines', 1280, 720)
    cv2.imshow('Hough Lines', image)
    cv2.imwrite('houghlines_merge.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

