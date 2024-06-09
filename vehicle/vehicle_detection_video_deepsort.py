import cv2
import json
import os


def annotate_video_with_detections(video_path, detections, output_video_path):
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


video = 'night_driving-2'

json_path = './data/videos/train/' + video + '.MOV_detections.json'
with open(json_path, 'r') as f:
    detections = json.load(f)

new_json_path = './data/output_blinker/5-1/left_or_right_temp.json'
with open(new_json_path, 'r') as f:
    new_data = json.load(f)

left_or_right_dict = {item['file_name']: item['left_or_right'] for item in new_data}

for detection in detections:
    frame_number = detection['frame_number']
    obj_id = detection['id']
    file_name = f'frame_{frame_number}_id_{obj_id}.png'
    detection['left_or_right'] = left_or_right_dict.get(file_name, None)

merged_json_path = './data/videos/train/' + video + '_merged.json'
with open(merged_json_path, 'w') as f:
    json.dump(detections, f, indent=4)

output_video_path = './data/output/' + video + '_annotated.mp4'

annotate_video_with_detections('./data/videos/train/' + video + '.mp4', detections, output_video_path)
