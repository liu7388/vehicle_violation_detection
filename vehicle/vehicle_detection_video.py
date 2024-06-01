import os

import torch
import cv2
import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort
from sort import Sort

# 設定裝置為MPS
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 初始化SORT追蹤器
tracker = Sort
# tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3, nn_budget=100)

# 載入YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model.to(device)

# 設定影片來源，0表示從攝像頭讀取，可以替換為影片文件路徑
video_name = 'blinkers-1-1.MOV'
video_path = 'videos/' + video_name
# video_path = 'videos/20230309_073056-1.MOV'
# video_path = 'videos/20230309_073056.MOV'
# video_path = 0
cap = cv2.VideoCapture(video_path)

# 檢查是否成功打開影片
if not cap.isOpened():
    print("錯誤：無法打開影片。")
    exit()

# 設定閥值
confidence_threshold = 0.4

# 定義類別名稱（依據YOLOv5預訓練模型的類別）
class_names = model.names

# 創建存儲文件夾的基礎路徑
output_base_dir = "output-" + video_name

frame_count = 0

# 物體顏色列表（這裡假設根據某些規則，你已經為每個類別定義了顏色）

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO模型進行推論
    results = model(frame)

    # 取得預測框、類別等信息
    pred = results.pred[0]

    # 過濾出置信度大於閥值的預測框
    confident_boxes = pred[pred[:, 4] > confidence_threshold]

    # 過濾出類別為vehicle且置信度大於閥值的預測框
    # 假設2、5、7分別是car, bus, truck
    vehicle_indices = ((confident_boxes[:, -1] == 2) | (confident_boxes[:, -1] == 5) | (confident_boxes[:, -1] == 7))
    vehicle_boxes = confident_boxes[vehicle_indices]

    # 繪製過濾後的結果
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

    # 進行追蹤
    tracked_objects = tracker.update(np.array(detections))

    # 繪製過濾後的結果
    annotated_frame = frame.copy()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    central_region_min = 2 * frame_width / 6
    central_region_max = 4 * frame_width / 6

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = [int(coord) for coord in obj[:5]]

        # 標出車輛位置
        avg_x = (x1 + x2) / 2
        location = (avg_x, y2)

        # 計算車寬
        car_width = x2 - x1

        # 只選擇在畫面中央三分之一位置的車輛
        if central_region_min <= avg_x <= central_region_max:
            central_vehicles.append((location, car_width, (x1, y1, x2, y2, obj_id)))

    # 根據車寬排序並選取值前三之車輛
    central_vehicles = sorted(central_vehicles, key=lambda x: x[1], reverse=True)[:3]

    for vehicle in central_vehicles:
        location, car_width, (x1, y1, x2, y2, obj_id) = vehicle
        print('id:', obj_id, 'location:', location)

        # 在影像上繪製框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 剪裁出車輛範圍並保存圖片
        vehicle_crop = frame[y1:y2, x1:x2]
        vehicle_dir = os.path.join(output_base_dir, f"id_{obj_id}")

        # 創建車輛ID資料夾
        if not os.path.exists(vehicle_dir):
            os.makedirs(vehicle_dir)

        # 保存圖片
        cv2.imwrite(os.path.join(vehicle_dir, f"frame_{frame_count}.jpg"), vehicle_crop)

    # 繪製中央區域矩形框
    cv2.rectangle(annotated_frame, (int(central_region_min), 0), (int(central_region_max), frame_height),
                      (255, 0, 0),
                      2)


    # 取得原始YOLO結果
    original_results_img = results.render()[0]

    # 合併顯示原始YOLO結果與過濾後的結果
    combined_frame = cv2.hconcat([original_results_img, annotated_frame])

    frame_count += 1

    # 顯示結果圖像
    cv2.imshow('YOLOv5 Vehicle Detection', combined_frame)

    # 按q鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
