import json
import os
import shutil
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class ImageProcessor:
    def __init__(self, folder_path, interval=5, resize_dim=(256, 256)):
        """
        Initialize an ImageProcessor object with parameters for image processing.

        Args:
            folder_path (str): Path to the folder containing images.
            interval (int): Interval between images to process.
            resize_dim (tuple): Dimensions to resize images to (width, height).

        """
        self.folder_path = folder_path
        self.interval = interval
        self.resize_dim = resize_dim
        self.image_files = self.load_images_from_folder(interval=interval)
        self.original_image_files = self.load_images_from_folder(interval=1)

    def load_images_from_folder(self, interval):
        """
        Load image files from the specified folder with a given interval.

        Args:
            interval (int): Interval between images to load.

        Returns:
            list: List of file paths to the loaded images.

        """
        filenames = []
        file_list = [(os.path.join(self.folder_path, f), os.path.getmtime(os.path.join(self.folder_path, f))) for f in
                     os.listdir(self.folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        sorted_files = sorted(file_list, key=lambda x: x[1])
        filenames = [f[0] for f in sorted_files]
        return filenames[::interval]

    def read_and_resize_image(self, filepath):
        """
        Read an image from file and resize it.

        Args:
            filepath (str): Path to the image file.

        Returns:
            numpy.ndarray: Resized image array.

        """
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.resize(image, self.resize_dim)
        return image

    @staticmethod
    def compare_images(img1, img2, threshold=0.3):
        """
        Compare two images using Structural Similarity Index (SSIM).

        Args:
            img1 (numpy.ndarray): First image array.
            img2 (numpy.ndarray): Second image array.
            threshold (float): Threshold value for SSIM score.

        Returns:
            bool: True if images are similar based on SSIM, False otherwise.

        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray1, gray2, full=True)
        return score > threshold

    @staticmethod
    def calculate_dynamic_threshold(images, factor=0.5):
        """
        Calculate dynamic thresholds based on brightness differences between consecutive images.

        Args:
            images (list): List of image arrays.
            factor (float): Factor to adjust the threshold calculation.

        Returns:
            tuple: Lower and higher threshold values.

        """
        brightness_diffs = []
        for i in range(len(images) - 1):
            img1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(img1, img2)
            brightness_diffs.append(np.mean(diff))
        mean_diff = np.mean(brightness_diffs)
        std_diff = np.std(brightness_diffs)
        lower_threshold = max(5, mean_diff - factor * std_diff)
        higher_threshold = min(255, mean_diff + factor * std_diff)
        return lower_threshold, higher_threshold

    def compare_images_bright(self, img1, img2):
        """
        Compare brightness differences between two images using dynamic thresholds.

        Args:
            img1 (numpy.ndarray): First image array.
            img2 (numpy.ndarray): Second image array.

        Returns:
            bool: True if brightness difference falls within calculated thresholds, False otherwise.

        """
        lower_threshold, higher_threshold = self.calculate_dynamic_threshold([img1, img2])
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = np.mean(diff)
        return lower_threshold <= mean_diff <= higher_threshold

    @staticmethod
    def draw_dividing_lines(image, prev_image=None, factor=1.5, min_diff_threshold=30, dynamic_adjustment=True):
        """
        Draw dividing lines on the image based on brightness differences to detect highlights.

        Args:
            image (numpy.ndarray): Image array to draw lines on.
            prev_image (numpy.ndarray): Previous image array for comparison.
            factor (float): Factor for adjusting thresholds.
            min_diff_threshold (int): Minimum threshold for detecting differences.
            dynamic_adjustment (bool): Flag for dynamic adjustment of thresholds.

        Returns:
            numpy.ndarray: Image with drawn dividing lines.

        """
        height, width, _ = image.shape

        highlight_position = ImageProcessor.detect_highlight(image, prev_image, factor, min_diff_threshold,
                                                             dynamic_adjustment)

        left_rect = (0, 2 * height // 7, 2 * width // 8, 3 * height // 8)
        right_rect = (5 * width // 8, 2 * height // 7, width, 3 * height // 8)

        if highlight_position == "left":
            left_color = (0, 0, 255)  # Red for left highlight
            right_color = (0, 255, 0)  # Green for right normal
        elif highlight_position == "right":
            left_color = (0, 255, 0)  # Green for left normal
            right_color = (0, 0, 255)  # Red for right highlight
        else:
            left_color = right_color = (0, 255, 0)  # Green for both normal

        image_with_rect = image.copy()

        cv2.rectangle(image_with_rect, (left_rect[0], left_rect[1]),
                      (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), left_color, 2)
        cv2.rectangle(image_with_rect, (right_rect[0], right_rect[1]),
                      (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), right_color, 2)

        return image_with_rect

    @staticmethod
    def detect_highlight(image, prev_image=None, factor=1.5, min_diff_threshold=30, dynamic_adjustment=True):
        """
        Detect highlights in specific regions of the image based on brightness and differences.

        Args:
            image (numpy.ndarray): Image array to detect highlights in.
            prev_image (numpy.ndarray): Previous image array for comparison.
            factor (float): Factor for adjusting thresholds.
            min_diff_threshold (int): Minimum threshold for detecting differences.
            dynamic_adjustment (bool): Flag for dynamic adjustment of thresholds.

        Returns:
            str: "left" if left side is highlighted, "right" if right side is highlighted, None if no highlight.

        """
        height, width, _ = image.shape
        left_region = image[int(0.25 * height):int(0.5 * height), :int(0.2 * width)]
        right_region = image[int(0.25 * height):int(0.5 * height), int(0.8 * width):]

        left_brightness = np.mean(cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY))
        right_brightness = np.mean(cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY))

        if prev_image is not None:
            prev_left_region = prev_image[int(0.25 * height):int(0.5 * height), :int(0.2 * width)]
            prev_right_region = prev_image[int(0.25 * height):int(0.5 * height), int(0.8 * width):]

            left_diff = np.mean(cv2.absdiff(cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(prev_left_region, cv2.COLOR_BGR2GRAY)))
            right_diff = np.mean(cv2.absdiff(cv2.cvtColor(right_region, cv2.COLOR_BGR2GRAY),
                                             cv2.cvtColor(prev_right_region, cv2.COLOR_BGR2GRAY)))
        else:
            left_diff = right_diff = 0

        brightness_threshold = factor * np.std([left_brightness, right_brightness])
        diff_threshold = factor * np.std([left_diff, right_diff])

        if abs(left_diff) < min_diff_threshold and abs(right_diff) < min_diff_threshold:
            return None
        elif left_brightness > right_brightness and left_diff > diff_threshold:
            if dynamic_adjustment:
                diff_threshold *= 1.5  # Increase threshold for right side to reduce false positives
            return "left"
        elif right_brightness > left_brightness and right_diff > diff_threshold:
            if dynamic_adjustment:
                diff_threshold *= 1.5  # Increase threshold for left side to reduce false positives
            return "right"
        else:
            return None

    def process_images(self):
        """
        Process the loaded images to filter out non-consecutive frames based on brightness differences.

        Returns:
            list: List of filtered image file paths.

        """
        indices_to_remove = []
        for i in range(len(self.image_files) - 1):
            img1 = self.read_and_resize_image(self.image_files[i])
            img2 = self.read_and_resize_image(self.image_files[i + 1])
            if not self.compare_images_bright(img1, img2):
                indices_to_remove.append(i)

        for index in reversed(indices_to_remove):
            del self.image_files[index]

        return self.image_files

    def save_processed_images(self, image_files):
        """
        Save processed images to a destination folder and create JSON file with highlight detection results.

        Args:
            image_files (list): List of image file paths to save.

        """
        destination_folder = './output/'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for img_file in self.image_files:
            shutil.copy(img_file, destination_folder)

        self.image_files = image_files

        leftRight = []
        prev_image = None

        for img_file in self.image_files:
            image = self.read_and_resize_image(img_file)
            leftOrRight = self.detect_highlight(image, prev_image, min_diff_threshold=30)
            leftRight.append({"file_name": os.path.basename(img_file), "left_or_right": leftOrRight})
            prev_image = image

        remaining_files = set(os.path.basename(f) for f in self.image_files)
        for original_file in self.original_image_files:
            original_file_name = os.path.basename(original_file)
            if original_file_name not in remaining_files:
                leftRight.append({
                    "file_name": original_file_name,
                    "left_or_right": None
                })

        leftRight.sort(key=lambda x: int(x["file_name"].split("_")[1].split(".")[0]))

        for i in range(0, len(leftRight) - 5, 5):
            current_value = leftRight[i]["left_or_right"]
            next_value = leftRight[i + 5]["left_or_right"]
            if current_value == next_value:
                for j in range(1, 5):
                    if i + j < len(leftRight) - 1:
                        leftRight[i + j]["left_or_right"] = current_value

        with open(destination_folder + 'left_or_right_temp.json', 'w') as f:
            json.dump(leftRight, f, indent=4)

    def run(self):
        """
        Execute the image processing pipeline: filtering images and saving results.
        """
        self.image_files = self.process_images()
        self.save_processed_images(image_files=self.image_files)


if __name__ == "__main__":
    folder_path = './data/output_deepsort/night_driving-10/4'
    processor = ImageProcessor(folder_path)
    processor.run()
