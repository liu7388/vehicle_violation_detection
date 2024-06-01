import torch
import cv2


# 載入YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 讀取圖像
img = cv2.imread('data/image.jpg')

# 進行推論
results = model(img)

# 取得辨識結果
results.print()  # 印出結果
results.save()   # 保存包含框架和標籤的圖像
