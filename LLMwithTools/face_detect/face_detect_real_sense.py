import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import pyrealsense2 as rs

# 限差参数
cosine_similar_thresh = 0.363
l2norm_similar_thresh = 1.128
scale = 1.0

def visualize(input_img, frame, faces, fps, thickness=2):
    fps_string = f"FPS : {fps:.2f}"
    # if frame >= 0:
    #     print(f"Frame {frame}, ", end="")
    # print(f"FPS: {fps_string}")
    
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        face = faces[i]
        # 绘制边界框
        cv2.rectangle(input_img, 
                     (int(face[0]), int(face[1])),
                     (int(face[0] + face[2]), int(face[1] + face[3])),
                     (0, 255, 0), thickness)
        
        # 绘制特征点
        cv2.circle(input_img, (int(face[4]), int(face[5])), 2, (255, 0, 0), thickness)
        cv2.circle(input_img, (int(face[6]), int(face[7])), 2, (0, 0, 255), thickness)
        cv2.circle(input_img, (int(face[8]), int(face[9])), 2, (0, 255, 0), thickness)
        cv2.circle(input_img, (int(face[10]), int(face[11])), 2, (255, 0, 255), thickness)
        cv2.circle(input_img, (int(face[12]), int(face[13])), 2, (0, 255, 255), thickness)
    
    cv2.putText(input_img, fps_string, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def write_names(input_img, faces, names):
    if faces is None:
        return
    for i in range(faces.shape[0]):
        org = (int(faces[i][0]), int(faces[i][1]))
        cv2.putText(input_img, names[i], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align

# ================= 核心修改点 =================
def recognize_people(stop_event=None, callback=None):
    """
    人脸识别主循环
    stop_event: 线程停止信号
    callback: 识别到人脸时的回调函数，签名 callback(name)
    """
    pipeline, align = initialize_realsense()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    detector = cv2.FaceDetectorYN_create(
        os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", 
        (320, 320), 0.9, 0.3, 5000)
    
    face_recognizer = cv2.FaceRecognizerSF_create(
        os.path.join(script_dir, "model/face_recognition_sface_2021dec.onnx"), "")
 
    names = []
    try:
        name_file_path = os.path.join(script_dir, "feature/name.txt")
        with open(name_file_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"无法加载名字文件: {e}")
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    try:
        while True:
            # 1. 检查是否收到外部退出信号
            if stop_event and stop_event.is_set():
                print("🛑 收到停止信号，退出人脸识别进程...")
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            frame_height, frame_width = frame.shape[:2]
            scaled_width = int(frame_width * scale)
            scaled_height = int(frame_height * scale)
            
            detector.setInputSize((scaled_width, scaled_height))
            resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
            
            tm.start()
            _, faces = detector.detect(resized_frame)
            tm.stop()
            fps = tm.getFPS()
            
            tar_name = []
            if faces is not None:
                for i in range(faces.shape[0]):
                    tar_name.append("unknown")
                    
                    aligned_face = face_recognizer.alignCrop(resized_frame, faces[i])
                    feature = face_recognizer.feature(aligned_face)
                    
                    try:
                        fs = cv2.FileStorage(os.path.join(script_dir, "feature/vocabulary.xml"), cv2.FILE_STORAGE_READ)
                        for name in names:
                            mat_vocabulary = fs.getNode(name).mat()
                            if mat_vocabulary is None: continue
                                
                            cos_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_COSINE)
                            l2_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_NORM_L2)
                            
                            if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
                                tar_name[i] = name
                                
                                # 2. 触发回调，将名字传给大模型
                                if callback:
                                    callback(name)
                                break
                    except Exception as e:
                        pass
            
            result = resized_frame.copy()
            visualize(result, frame_count, faces, fps)
            write_names(result, faces, tar_name)
            
            cv2.imshow("Live Face Recognition (RealSense)", result)
            if cv2.waitKey(30) == 27:
                break
            frame_count += 1
            
    finally:
        # 3. 必须在 finally 释放摄像头！
        print("释放 RealSense 摄像头 (Face)...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_people()