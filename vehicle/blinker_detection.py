import json
import os
import shutil

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def load_images_from_folder(folder, interval=5):
    filenames = []
    file_list = [(os.path.join(folder, f), os.path.getmtime(os.path.join(folder, f))) for f in os.listdir(folder) if
                 f.endswith((".jpg", ".jpeg", ".png"))]
    sorted_files = sorted(file_list, key=lambda x: x[1])
    filenames = [f[0] for f in sorted_files]
    return filenames[::interval]


def load_images_from_folder(folder, interval=5):
    filenames = []
    # 檢索指定資料夾中的圖像檔案
    file_list = [(os.path.join(folder, f), os.path.getmtime(os.path.join(folder, f))) for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png"))]
    # 根據檔案修改時間排序檔案列表
    sorted_files = sorted(file_list, key=lambda x: x[1])
    # 提取已排序的檔案名稱
    filenames = [f[0] for f in sorted_files]
    # 返回間隔的檔案名稱列表
    return filenames[::interval]

def compare_images(img1, img2):
    # 將圖像轉換為灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 將圖像調整為相同大小
    gray1 = cv2.resize(gray1, (256, 256))
    gray2 = cv2.resize(gray2, (256, 256))

    # 計算結構相似性指數（SSIM）
    score, diff = ssim(gray1, gray2, full=True)
    print(f"SSIM: {score}")

    # 如果得分低於閾值，則認為它們差異顯著
    threshold = 0.5
    return score > threshold

def compare_images_bright(img1, img2):
    # 將圖像轉換為灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 將圖像調整為相同大小
    gray1 = cv2.resize(gray1, (256, 256))
    gray2 = cv2.resize(gray2, (256, 256))

    # 計算亮度的絕對差異
    diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(diff)
    print(f"Mean Brightness Difference: {mean_diff}")

    lower_threshold = 10
    higher_threshold = 30
    return lower_threshold <= mean_diff <= higher_threshold

def draw_dividing_lines(image):
    # 獲取圖像的寬度和高度
    height, width, _ = image.shape

    # 定義左半部分的矩形框坐標
    left_rect = (0, 2 * height // 7, 2 * width // 7, 3 * height // 8)

    # 定義右半部分的矩形框坐標
    right_rect = (5 * width // 7, 2 * height // 7, width, 3 * height // 8)

    # 在圖像上繪製矩形框
    image_with_rect = image.copy()
    cv2.rectangle(image_with_rect, (left_rect[0], left_rect[1]),
                  (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), (0, 255, 0), 2)
    cv2.rectangle(image_with_rect, (right_rect[0], right_rect[1]),
                  (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), (0, 255, 0), 2)

    return image_with_rect

def detect_highlight(image):
    # 獲取圖像的寬度和高度
    height, width, _ = image.shape

    # 將圖像分割為左右兩部分
    left_half = image[2*height//7:3*height//8, :2*width//7]
    right_half = image[2*height//7:3*height//8, 5*width//7:width]

    # 計算左右兩部分的亮度值
    left_brightness = np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
    right_brightness = np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))

    print(left_brightness, right_brightness)

    # 設置亮度閾值
    brightness_threshold = 100  # 可根據實際情況調整

    # 檢測左右兩部分是否有高光
    # left_highlight = left_brightness > brightness_threshold
    # right_highlight = right_brightness > brightness_threshold

    if left_brightness > right_brightness:
        return "left"

    elif right_brightness > left_brightness:
        return "right"

    else:
        return "same"


def main():
    # 資料夾路徑
    folder_path = './data/output_deepsort/20230309_073056-1/5'  # 修改為您的資料夾路徑
    image_files = load_images_from_folder(folder_path)

    print(image_files)

    i = 0
    while i < len(image_files) - 1:
        img1_path = image_files[i]
        img2_path = image_files[i + 1]
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if compare_images(img1, img2):
            start_index = i
            # 繼續向後尋找，直到找到下一個返回True的索引值或到達列表末尾
            while i < len(image_files) - 1 and compare_images(cv2.imread(image_files[i]),
                                                              cv2.imread(image_files[i + 1])):
                i += 1
            # 移除從start_index到i之間的所有值
            del image_files[start_index:i + 1]
        else:
            i += 1

    print("Remaining images after removal:")
    print(image_files)

    # 將處理過的影像寫入檔案
    # with open(folder_path +'blinker_detection_temp.json', 'w') as f:
    #         json.dump(image_files, f)

    indices_to_remove = []

    for i in range(len(image_files) - 1):
        img1 = cv2.imread(image_files[i])
        img2 = cv2.imread(image_files[i + 1])

        compare_images(img1, img2)

        # print(os.path.basename(image_files[i]), os.path.basename(image_files[i + 1]))

        if not compare_images_bright(img1, img2):
            # indices_to_remove.append(i+1)
            indices_to_remove.append(i)
            print(os.path.basename(image_files[i]), os.path.basename(image_files[i + 1]))
            print("significantly different.")

    for index in reversed(indices_to_remove):
        del image_files[index]

    # 目標資料夾
    destination_folder = './data/output_blinker/' + os.path.basename(folder_path) + '-1'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for img_file in image_files:
        shutil.copy(img_file, destination_folder)

    image_files = load_images_from_folder(destination_folder, interval=1)
    leftRight = []

    for i in range(len(image_files)):
        image = cv2.imread(image_files[i])
        cv2.imshow("Image with Dividing Lines", draw_dividing_lines(image))
        leftOrRight =  detect_highlight(image)
        print(os.path.basename(image_files[i]), leftOrRight)
        leftRight.append({"file_name":os.path.basename(image_files[i]), "left_or_right": leftOrRight})
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    with open(destination_folder + 'left_or_right_temp.json', 'w') as f:
        json.dump(leftRight, f)

if __name__ == "__main__":
    main()