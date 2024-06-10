import cv2
import numpy as np
import pathlib
import os

# 讀取圖片
path = os.path.abspath(__file__)
image = cv2.imread(str(pathlib.Path(os.path.abspath(__file__)).parents[1]) + '/inference/images/lanes_mask/lanes_mask.png')
print(str(pathlib.Path(path).parents[1]) + '/inference/images/lanes_mask/lanes_mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 霍夫直線偵測
# lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=160, minLineLength=100, maxLineGap=10)
lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=160, minLineLength=100, maxLineGap=1)

# 畫出直線
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hough Lines', 1280, 720)
cv2.imshow('Hough Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
