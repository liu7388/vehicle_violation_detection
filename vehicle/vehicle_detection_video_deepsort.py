import cv2
import json
import os
from blinker_detection import ImageProcessor


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


target_id = 5
# video = '20230531_152117-1_8s'
# video = 'night_driving-10'
video = '20230309_073056-1'

video_path = './data/videos/train/' + video + '.mp4'
cap = cv2.VideoCapture(video_path)

json_path = './data/videos/train/' + video + '.MOV_detections.json'
with open(json_path, 'r') as f:
    detections = json.load(f)
output_dir = './data/output_deepsort/' + video
os.makedirs(output_dir, exist_ok=True)

for detection in detections:
    frame_number = detection['frame_number']
    timestamp = detection['timestamp']
    # 根据时间戳计算帧索引
    frame_index = int(cap.get(cv2.CAP_PROP_FPS) * timestamp)
    # 设置视频的当前帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        break
    obj_id = detection['id']
    bbox = detection['bbox']
    # 裁剪出标记框的部分
    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]
    # 检查裁剪后的图像是否为空
    if cropped_image is None:
        print(f"裁剪后的图像为空，检查裁剪坐标 ({x1}, {y1}, {x2}, {y2}) 是否正确。")
        continue  # 如果图像为空，跳过保存步骤
    # 创建ID文件夹
    id_folder = os.path.join(output_dir, str(obj_id))
    os.makedirs(id_folder, exist_ok=True)
    # 保存裁剪的图片
    save_path = os.path.join(id_folder, f'frame_{frame_number}_id_{obj_id}.png')
    cv2.imwrite(save_path, cropped_image)
    # print(f'Saved cropped image for ID {obj_id} in frame {frame_number} to {save_path}')

cap.release()
cv2.destroyAllWindows()

folder_path = './data/output_deepsort/' + video + '/' + str(target_id)
processor = ImageProcessor(folder_path)
processor.run()

new_json_path = './data/output_blinker/' + str(target_id) + '/left_or_right_temp.json'
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
