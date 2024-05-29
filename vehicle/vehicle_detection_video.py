import torch
import cv2
# 設定設備為MPS
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 載入YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.to(device)

# 設定影片來源，0表示從攝像頭讀取，可以替換為影片文件路徑
video_path = 'videos/20230309_073356.MOV'
# video_path = 0
cap = cv2.VideoCapture(video_path)

# 检查是否成功打开影片
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 設定閥值
confidence_threshold = 0.5

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
    vehicle_indices = ((confident_boxes[:, -1] == 2) | (confident_boxes[:, -1] == 5) | (confident_boxes[:, -1] == 7))  # Assuming '2' is the class index for 'vehicle'
    vehicle_boxes = confident_boxes[vehicle_indices]

    # 繪製過濾後的結果
    annotated_frame = frame.copy()
    for box in vehicle_boxes:
        # 獲取框的坐標
        x1, y1, x2, y2 = [int(coord.item()) for coord in box[:4]]
        # 在影像上繪製框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 取得原始YOLO結果
    original_results_img = results.render()[0]

    # 合併顯示原始YOLO結果與過濾後的結果
    combined_frame = cv2.hconcat([original_results_img, annotated_frame])

    # 顯示結果圖像
    cv2.imshow('YOLOv5 Vehicle Detection', combined_frame)
    # cv2.imshow('YOLOv5 Vehicle Detection', combined_frame)

    # cv2.imshow('YOLOv5 Vehicle Detection', results_img)



    # # 取得結果圖像
    # results_img = results.render()[0]

    # 顯示結果圖像
    # cv2.imshow('YOLOv5 Vehicle Detection', results_img)

    # 按q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()

