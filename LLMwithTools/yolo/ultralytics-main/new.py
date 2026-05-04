#完整代码
import torch
import cv2
import numpy as np
from ultralytics import YOLO
# 1. 加载模型
model = YOLO('yolov8s.pt') 
#imgae_path改成图⽚的路径
def preprocess_image(image_path):
  image = cv2.imread(image_path)
  return image
image = preprocess_image("ultralytics/assets/bus.jpg")
# 2. ⽬标检测
def detect_objects(model, image):
  results = model(image)
  return results[0] # 取第⼀张图的结果
result = detect_objects(model, image)
# 3. 解析并绘制结果

def draw_detections(img, result):
 # 拷⻉⼀份原图避免在原图上修改
  display_img = img.copy()
 
 # 获取检测框信息
  boxes = result.boxes.cpu().numpy()
 
  for box in boxes:
 # 获取坐标 (x1, y1, x2, y2)
    r = box.xyxy[0].astype(int)
    conf = box.conf[0]
    cls = int(box.cls[0])
    label_name = model.names[cls]
 
 # 绘制矩形框
    cv2.rectangle(display_img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
 
 # 绘制标签
    label_str = f"{label_name} {conf:.2f}"
    cv2.putText(display_img, label_str, (r[0], r[1] - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    return display_img
# 执⾏绘制
detected_image = draw_detections(image, result)
# 4. 显⽰结果
cv2.imshow("YOLOv8 Detection", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()