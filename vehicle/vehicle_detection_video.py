import torch
import cv2
import numpy as np
import os
from sort import Sort

# Set device to MPS if available, otherwise CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize SORT tracker
tracker = Sort()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.to(device)

video_name = '20230309_073056.MOV'
# video_name = 'blinkers-2-1.MOV'

# Set video source; 0 for camera, replace with video file path
video_path = './data/videos/' + video_name
# video_path = 0
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Confidence threshold
confidence_threshold = 0.4

# Class names based on YOLOv5 pretrained model classes
class_names = model.names

# Create base path for storing files
output_base_dir = "./data/output/" + video_name

# Create base directory if it does not exist
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# Object colors (assuming predefined colors for each class based on some rule)
# Assuming 2, 5, 7 are indices for car, bus, truck respectively
colors = {2: 'blue', 5: 'yellow', 7: 'silver'}  # Mapping colors based on class index

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference using YOLO model
    results = model(frame)

    # Get predictions, class labels, and other information
    pred = results.pred[0]

    # Filter out predicted boxes with confidence above threshold
    confident_boxes = pred[pred[:, 4] > confidence_threshold]

    # Filter out boxes classified as 'vehicle' with confidence above threshold
    vehicle_indices = ((confident_boxes[:, -1] == 2) | (confident_boxes[:, -1] == 5) | (confident_boxes[:, -1] == 7))
    vehicle_boxes = confident_boxes[vehicle_indices]

    # Copy the frame for annotation
    annotated_frame = frame.copy()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    central_region_min = 2 * frame_width / 6
    central_region_max = 4 * frame_width / 6

    central_vehicles = []
    detections = []

    for box in vehicle_boxes:
        x1, y1, x2, y2, conf = box[:5]
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    # Perform tracking
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = [int(coord) for coord in obj[:5]]

        # Mark the location of the vehicle
        avg_x = (x1 + x2) / 2
        location = (avg_x, y2)

        # Calculate the width of the vehicle
        car_width = x2 - x1

        # Select only vehicles in the central third of the frame
        if central_region_min <= avg_x <= central_region_max:
            central_vehicles.append((location, car_width, (x1, y1, x2, y2, obj_id)))

    for vehicle in central_vehicles:
        location, car_width, (x1, y1, x2, y2, obj_id) = vehicle
        print('id:', obj_id, 'location:', location)

        # Draw bounding box on the image
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Crop the vehicle area and save the image
        vehicle_crop = frame[y1:y2, x1:x2]
        vehicle_dir = os.path.join(output_base_dir, f"id_{obj_id}")

        # Create directory for vehicle ID if it does not exist
        if not os.path.exists(vehicle_dir):
            os.makedirs(vehicle_dir)

        # Save the image
        cv2.imwrite(os.path.join(vehicle_dir, f"frame_{frame_count}.jpg"), vehicle_crop)

    # Get original YOLO results
    original_results_img = results.render()[0]

    # Combine original YOLO results and annotated frame
    combined_frame = cv2.hconcat([original_results_img, annotated_frame])

    # Display the result image
    cv2.imshow('YOLOv5 Vehicle Detection', combined_frame)

    frame_count += 1

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
