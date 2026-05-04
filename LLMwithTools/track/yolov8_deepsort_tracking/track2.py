import time
import math
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2 
from ultralytics import YOLO

import sys
import os

# 1. 获取当前代码运行的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出 deep_sort 的父级目录的绝对路径
# 注意：这里请核对一下你的文件夹名字是下划线 (_) 还是中划线 (-)
deepsort_path = os.path.join(current_dir, "track", "yolov8_deepsort_tracking")

# 3. 将这个路径临时加入到 Python 的环境变量中
sys.path.append(deepsort_path)

# 4. 现在 Python 就能成功找到这个包了！
import deep_sort.deep_sort.deep_sort as ds


# 限差参数和目标姓名
cosine_similar_thresh = 0.363
l2norm_similar_thresh = 1.128
target_name = "ZhuochaoWang" # 如果需要动态找人，你可以后期将其作为参数传入

# ... 保留原有的辅助函数 (visualize, write_names, putTextWithBackground, extract_detections) ...
def visualize(input_img, frame, faces, fps, index, thickness=2):
    if faces is None: return
    face = faces[index]
    cv2.rectangle(input_img, (int(face[0]), int(face[1])), (int(face[0] + face[2]), int(face[1] + face[3])), (0, 255, 0), thickness)
    
def write_names(input_img, faces, names, index):
    if faces is None: return
    org = (int(faces[index][0]), int(faces[index][1]))
    cv2.putText(input_img, names, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)
    text_origin = (origin[0], origin[1] - 5)
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def extract_detections(results, detect_class):
    detections = np.empty((0, 4)) 
    confarray = [] 
    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2) 
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf) 
    return detections, confarray 

def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()  
    aligned_frames = align.process(frames)  
    aligned_depth_frame = aligned_frames.get_depth_frame()  
    color_frame = aligned_frames.get_color_frame()  
    intr = color_frame.profile.as_video_stream_profile().intrinsics  
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  
    color_image = np.asanyarray(color_frame.get_data())  
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
 
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate

# 封装找人的前置步骤 (接收外部的 pipeline)
def recognize_people_set_target(pipeline, align, model, tracker, detect_class, stop_event):
    detector = cv2.FaceDetectorYN_create("track/yolov8_deepsort_tracking/face_detect/model/face_detection_yunet_2023mar.onnx", "", (320, 320), 0.9, 0.3, 5000)
    face_recognizer = cv2.FaceRecognizerSF_create("track/yolov8_deepsort_tracking/face_detect/model/face_recognition_sface_2021dec.onnx", "")
    
    print("开始设置跟踪目标...")
    target_id = None
    frame_count = 0
    tm = cv2.TickMeter()
    
    while True:
        if stop_event and stop_event.is_set():
            return None
            
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame: continue
        
        frame = np.asanyarray(color_frame.get_data())
        frame_height, frame_width = frame.shape[:2]
        detector.setInputSize((int(frame_width), int(frame_height)))
        
        people_results = model(frame, stream=True)
        result = frame.copy()
        
        tm.start()
        _, faces = detector.detect(frame)
        tm.stop()
        
        target_face = None
        target_index = None
        if faces is not None:
            for i in range(faces.shape[0]):
                aligned_face = face_recognizer.alignCrop(frame, faces[i])
                feature = face_recognizer.feature(aligned_face)
                try:
                    fs = cv2.FileStorage("track/yolov8_deepsort_tracking/face_detect/feature/vocabulary.xml", cv2.FILE_STORAGE_READ)
                    mat_vocabulary = fs.getNode(target_name).mat()
                    if mat_vocabulary is None: continue
                    cos_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_COSINE)
                    l2_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_NORM_L2)
                    if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
                        target_face = faces[i]
                        target_index = i
                        break
                except Exception as e:
                    pass

        if target_face is not None:
            visualize(result, frame_count, faces, tm.getFPS(), target_index)
            write_names(result, faces, target_name, target_index)
            detections = np.empty((0, 4)) 
            confarray = []
            
            for r in people_results:
                for box in r.boxes:
                    if box.cls[0].int() == detect_class:
                        x1, y1, x2, y2 = box.xywh[0].int().tolist() 
                        if (x1-x2/2) < int(target_face[0]) and (x1+x2/2) > int(target_face[0]+target_face[2]):
                            conf = round(box.conf[0].item(), 2) 
                            detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) 
                            confarray.append(conf)                            
                            resultsTracker = tracker.update(detections, confarray, result)
                            
                            for tx1, ty1, tx2, ty2, Id in resultsTracker:
                                target_id = Id
                                cv2.rectangle(result, (tx1, ty1), (tx2, ty2), (255, 0, 255), 3)
                                putTextWithBackground(result, str(int(Id)), (max(-10, tx1), max(40, ty1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))
                                print(f"✅ 已收集到可追踪信息, ID: {target_id}")
                                return target_id # 直接返回，进入下一步追踪阶段
                                
        cv2.imshow('Tracking Init', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    return target_id

# ================= 核心修改点: 总封装 =================
def track_person_loop(stop_event=None, callback=None):
    """
    人物跟踪主循环
    """
    model = YOLO("yolov8n.pt")
    detect_class = 0 
    tracker = ds.DeepSort("track/yolov8_deepsort_tracking/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    align_to = rs.stream.color  
    align = rs.align(align_to)

    try:
        # 第一阶段：找人
        my_id = recognize_people_set_target(pipeline, align, model, tracker, detect_class, stop_event)
        
        if my_id is None:
            print("未能确定目标ID，退出追踪...")
            return
            
        print(f"🎯 正式开始追踪 ID: {my_id}")
        cv2.destroyWindow('Tracking Init') # 关掉初始化阶段的窗口

        # 第二阶段：追踪
        while True:
            if stop_event and stop_event.is_set():
                print("🛑 收到停止信号，退出追踪进程...")
                break
                
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images(pipeline, align)
            if not depth_image.any() or not color_image.any(): continue
            
            results = model(color_image, stream=True, verbose=False)
            detections, confarray = extract_detections(results, detect_class)
            resultsTracker = tracker.update(detections, confarray, color_image)
            
            for x1, y1, x2, y2, Id in resultsTracker:
                if Id != my_id:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                ux = int((x1 + x2) / 2)
                uy = int((y1 + y2) / 2)
                dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame, depth_intrin)  
                formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"
                
                # 触发回调给大模型
                if callback:
                    callback(dis, formatted_camera_coordinate)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                putTextWithBackground(color_image, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))
                cv2.circle(color_image, (ux, uy), 4, (255, 255, 255), 5)  
                cv2.putText(color_image, formatted_camera_coordinate, (ux + 20, uy + 10), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  

            cv2.imshow('Person Tracking (RealSense)', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("释放 RealSense 摄像头 (Track)...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    track_person_loop()