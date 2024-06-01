import cv2
import json
import os

video = '20230309_073056-1'
# video = 'blinkers-1-1'


# 读取视频
video_path = './videos/train/' + video + '.mp4'
cap = cv2.VideoCapture(video_path)

# 读取 JSON 文件
json_path = './videos/train/' + video +'.MOV_detections.json'
with open(json_path, 'r') as f:
    detections = json.load(f)

# 创建输出文件夹
output_dir = './output/' + video
os.makedirs(output_dir, exist_ok=True)

# 读取视频帧并裁剪标记框部分
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
    print(f'Saved cropped image for ID {obj_id} in frame {frame_number} to {save_path}')

cap.release()
cv2.destroyAllWindows()
