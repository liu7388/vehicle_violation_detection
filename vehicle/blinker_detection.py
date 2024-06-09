import json
import os
import shutil

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


class ImageProcessor:
    def __init__(self, folder_path, interval=5):
        self.folder_path = folder_path
        self.interval = interval
        self.image_files = self.load_images_from_folder(interval=interval)
        self.original_image_files = self.load_images_from_folder(interval=1)

    def load_images_from_folder(self, interval):
        filenames = []
        file_list = [(os.path.join(self.folder_path, f), os.path.getmtime(os.path.join(self.folder_path, f))) for f in
                     os.listdir(self.folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        sorted_files = sorted(file_list, key=lambda x: x[1])
        filenames = [f[0] for f in sorted_files]
        return filenames[::interval]

    @staticmethod
    def compare_images(img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.resize(gray1, (256, 256))
        gray2 = cv2.resize(gray2, (256, 256))
        score, _ = ssim(gray1, gray2, full=True)
        threshold = 0.3
        return score > threshold

    @staticmethod
    def compare_images_bright(img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.resize(gray1, (256, 256))
        gray2 = cv2.resize(gray2, (256, 256))
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)
        lower_threshold = 14
        higher_threshold = 50
        return lower_threshold <= mean_diff <= higher_threshold

    @staticmethod
    def draw_dividing_lines(image):
        height, width, _ = image.shape
        left_rect = (0, 2 * height // 7, 2 * width // 8, 3 * height // 8)
        right_rect = (5 * width // 8, 2 * height // 7, width, 3 * height // 8)
        image_with_rect = image.copy()
        cv2.rectangle(image_with_rect, (left_rect[0], left_rect[1]),
                      (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), (0, 255, 0), 2)
        cv2.rectangle(image_with_rect, (right_rect[0], right_rect[1]),
                      (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), (0, 255, 0), 2)
        return image_with_rect

    @staticmethod
    def detect_highlight(image):
        height, width, _ = image.shape
        left_half = image[2 * height // 7:3 * height // 8, :2 * width // 8]
        right_half = image[2 * height // 7:3 * height // 8, 5 * width // 8:width]
        left_brightness = np.mean(cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY))
        right_brightness = np.mean(cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY))

        if (left_brightness - right_brightness) < 1:
            return None

        elif left_brightness > right_brightness:
            return "left"

        elif right_brightness > left_brightness:
            return "right"

        else:
            return "same"

    def process_images(self):
        i = 0
        # while i < len(self.image_files) - 1:
        #     img1_path = self.image_files[i]
        #     img2_path = self.image_files[i + 1]
        #     img1 = cv2.imread(img1_path)
        #     img2 = cv2.imread(img2_path)

        # if self.compare_images(img1, img2):
        #     start_index = i
        #     while i < len(self.image_files) - 1 and self.compare_images(cv2.imread(self.image_files[i]),
        #                                                                 cv2.imread(self.image_files[i + 1])):
        #         i += 1
        #         del self.image_files[start_index:i + 1]
        # else:
        #     i += 1

        indices_to_remove = []
        for i in range(len(self.image_files) - 1):
            img1 = cv2.imread(self.image_files[i])
            img2 = cv2.imread(self.image_files[i + 1])
            if not self.compare_images_bright(img1, img2):
                indices_to_remove.append(i)

        for index in reversed(indices_to_remove):
            del self.image_files[index]

        return self.image_files

    def save_processed_images(self, image_files):
        destination_folder = './data/output_blinker/' + os.path.basename(self.folder_path) + '-1/'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for img_file in self.image_files:
            shutil.copy(img_file, destination_folder)

        self.image_files = image_files

        # self.image_files = self.load_images_from_folder(self.interval)
        leftRight = []

        for img_file in self.image_files:
            image = cv2.imread(img_file)
            leftOrRight = self.detect_highlight(image)
            leftRight.append({"file_name": os.path.basename(img_file), "left_or_right": leftOrRight})
            # cv2.imshow('image', self.draw_dividing_lines(image))
            # cv2.waitKey(0)

        remaining_files = set(os.path.basename(f) for f in self.image_files)
        for original_file in self.original_image_files:
            original_file_name = os.path.basename(original_file)
            if original_file_name not in remaining_files:
                leftRight.append({
                    "file_name": original_file_name,
                    "left_or_right": None
                })

        leftRight.sort(key=lambda x: int(x["file_name"].split("_")[1].split(".")[0]))

        # 修改中間四個index的值
        for i in range(0, len(leftRight) - 5, 5):
            current_value = leftRight[i]["left_or_right"]
            next_value = leftRight[i + 5]["left_or_right"]
            if current_value == next_value:
                for j in range(1, 5):
                    if i + j < len(leftRight) - 1:  # 確保不超出範圍
                        leftRight[i + j]["left_or_right"] = current_value

        with open(destination_folder + 'left_or_right_temp.json', 'w') as f:
            json.dump(leftRight, f, indent=4)

    def run(self):
        self.image_files = self.process_images()
        self.save_processed_images(image_files=self.image_files)


if __name__ == "__main__":
    # folder_path = './data/output_deepsort/20230309_073056-1/5'
    folder_path = './data/output_deepsort/night_driving-2/5'
    processor = ImageProcessor(folder_path)
    processor.run()
