import matplotlib.pyplot as plt
import numpy as np

# 圖片尺寸
width = 1920
height = 1080

# 圖片中心點
center_x = width // 2
center_y = height // 2

# 斜率
slopes = [0.5, 2, -2, -0.5]

left_lines_slope = []
right_lines_slope = []

for i in slopes:
    if i < 0:
        left_lines_slope.append(i)
    if i > 0:
        right_lines_slope.append(i)
left_lines_slope.sort(reverse=True)
right_lines_slope.sort(reverse=True)

print(left_lines_slope, right_lines_slope)

# 計算每條線在圖片下邊界的交點
intersections = []
for slope in slopes:
    b = center_y - slope * center_x
    x_intersection = (height - b) / slope
    intersections.append((x_intersection, height))

# 圖片中點A的座標
# 假設A點在 (x_A, y_A)
x_A = 900  # 這是示例，你需要提供具體的A點座標
y_A = 800   # 這是示例，你需要提供具體的A點座標

# 畫圖以視覺化分割
fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)

# 繪製四條線
x = np.linspace(0, width, 400)
for slope in slopes:
    ax.plot(x, slope * (x - center_x) + center_y, label=f'slope={slope}')

# 繪製中心點
ax.plot(center_x, center_y, 'ro')

# 繪製A點
ax.plot(x_A, y_A, 'bo', label='A Point')

# 標記五個區域
ax.fill_between(x, slopes[0] * (x - center_x) + center_y, slopes[1] * (x - center_x) + center_y, where=(x > center_x), interpolate=True, color='orange', alpha=0.3)
ax.fill_between(x, slopes[2] * (x - center_x) + center_y, slopes[3] * (x - center_x) + center_y, where=(x < center_x), interpolate=True, color='yellow', alpha=0.3)

plt.legend()
plt.gca().invert_yaxis()
plt.show()

# 判斷點A所在的區域
def determine_region(x, y, center_x, center_y):
    if len(right_lines_slope) >= 2 and len(left_lines_slope) >= 2:
        if ((y > right_lines_slope[0] * x + (center_y - right_lines_slope[0] * center_x)) and
                (y > left_lines_slope[-1] * x + (center_y - left_lines_slope[-1] * center_x))):
            return "Middle lane"
        elif x >= center_x:
            if y > right_lines_slope[1] * x + (center_y - right_lines_slope[1] * center_x):
                return "Right lane"
            else:
                return "Other lane on the right side"
        else:
            if y > left_lines_slope[-2] * x + (center_y - left_lines_slope[-2] * center_x):
                return "Left lane"
            else:
                return "Other lane on the left side"
    else:
        return "Unavailable"

# 判斷A點的區域
region = determine_region(x_A, y_A, center_x, center_y)
print(f"A點位於{region}")
