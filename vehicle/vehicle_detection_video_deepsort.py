import cv2
import json
import os
from blinker_detection import ImageProcessor
import argparse



def annotate_video_with_detections(video_path, detections, output_video_path):
    """
    Annotates a video with object detections and saves the annotated video.

    Args:
        video_path (str): Path to the input video file.
        detections (list): List of detections containing frame number, bounding box, and other details.
        output_video_path (str): Path to save the annotated video.

    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for detection in detections:
            if detection['frame_number'] == frame_index:
                left_or_right = detection.get('left_or_right')
                if left_or_right:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) if left_or_right == 'left' else (0, 0, 255)
                    label = 'Left' if left_or_right == 'left' else 'Right'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Annotated video saved to {output_video_path}")


# Parse arguments
parser = argparse.ArgumentParser(description='Annotate video with detections')
parser.add_argument('--video_name', type=str, help='Name of the video file (without extension)')
parser.add_argument('--target_id', type=int, default=5, help='Target ID for annotation (default: 5)')
args = parser.parse_args()

# Check if video name is provided
if not args.video_name:
    print("Error: You need to provide a video name using --video_name argument.")
    exit()

# Get video name and target ID
video = args.video_name[:-4]
target_id = args.target_id

print(video, target_id)

# Paths for video and detections JSON
video_path = '../vehicle/data/videos/train/' + video + '.mp4'
json_path = '../vehicle/data/videos/train/' + video + '.MOV_detections.json'

# Load detections from JSON file
with open(json_path, 'r') as f:
    detections = json.load(f)

video_path = '../vehicle/data/videos/train/' + video + '.mp4'
cap = cv2.VideoCapture(video_path)

# Output directory for annotated frames
output_dir = '../vehicle/data/output_deepsort/' + video
os.makedirs(output_dir, exist_ok=True)

# Loop through detections and process frames
for detection in detections:
    frame_number = detection['frame_number']
    timestamp = detection['timestamp']
    frame_index = int(cap.get(cv2.CAP_PROP_FPS) * timestamp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        break
    obj_id = detection['id']
    bbox = detection['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]
    if cropped_image is None:
        print(f"Cropped image is empty, check cropping coordinates ({x1}, {y1}, {x2}, {y2}).")
        continue
    id_folder = os.path.join(output_dir, str(obj_id))
    os.makedirs(id_folder, exist_ok=True)
    save_path = os.path.join(id_folder, f'frame_{frame_number}_id_{obj_id}.png')
    cv2.imwrite(save_path, cropped_image)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Process images with ImageProcessor
folder_path = '../vehicle/data/output_deepsort/' + video + '/' + str(target_id)
processor = ImageProcessor(folder_path)
processor.run()

# Update detections with left/right information
new_json_path = '../vehicle/data/output_blinker/' + str(target_id) + '/left_or_right_temp.json'
with open(new_json_path, 'r') as f:
    new_data = json.load(f)

left_or_right_dict = {item['file_name']: item['left_or_right'] for item in new_data}

for detection in detections:
    frame_number = detection['frame_number']
    obj_id = detection['id']
    file_name = f'frame_{frame_number}_id_{obj_id}.png'
    detection['left_or_right'] = left_or_right_dict.get(file_name, None)

# Save merged detections to JSON file
merged_json_path = '../vehicle/data/videos/train/' + video + '_merged.json'
with open(merged_json_path, 'w') as f:
    json.dump(detections, f, indent=4)

# Annotate video with detections
output_video_path = '../vehicle/data/output/' + video + '_annotated.mp4'

annotate_video_with_detections('../vehicle/data/videos'
                               '/train/' + video + '.mp4', detections, output_video_path)
