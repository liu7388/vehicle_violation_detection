import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def inference_and_save_results(image_path):
    """
    Loads a YOLOv5 model, performs inference on the given image, prints the results, and saves the annotated image.

    Args:
        image_path (str): Path to the input image file.

    """
    # Read image
    img = cv2.imread(image_path)

    # Perform inference
    results = model(img)

    # Print detection results
    results.print()

    # Save annotated image with bounding boxes and labels
    results.save()

# Example usage
image_path = 'data/image.jpg'
inference_and_save_results(image_path)
