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

# 画框可视化函数
def visualize(input_img, frame, faces, fps, thickness=2):
    fps_string = f"FPS : {fps:.2f}"
    if frame >= 0:
        print(f"Frame {frame}, ", end="")
    print(f"FPS: {fps_string}")
    
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        face = faces[i]
        print(f"Face {i}, top-left coordinates: ({face[0]}, {face[1]}), "
              f"box width: {face[2]}, box height: {face[3]}, "
              f"score: {face[14]:.2f}")
        
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
    
    # 添加FPS文本
    cv2.putText(input_img, fps_string, (0, 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 在图像上写入识别结果
def write_names(input_img, faces, names):
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        org = (int(faces[i][0]), int(faces[i][1]))
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)
        thickness = 2
        
        cv2.putText(input_img, names[i], org, font_face, 
                   font_scale, color, thickness)

# 保存人脸特征数据
def save_face_data(image, faces, names):
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_detect/model/face_recognition_sface_2021dec.onnx", "")
    
    for i in range(faces.shape[0]):
        aligned_face = face_recognizer.alignCrop(image, faces[i])
        feature = face_recognizer.feature(aligned_face)
        
        # 保存特征到XML
        fs = cv2.FileStorage("./feature/vocabulary.xml", cv2.FILE_STORAGE_APPEND)
        fs.write(names[i], feature)
        fs.release()
        
        # 保存名字到文本文件
        try:
            with open("face_detect/feature/name.txt", "a") as f:
                f.write("\n" + names[i])
        except Exception as e:
            print(f"无法保存名字文件: {e}")

# 从本地数据库识别人脸
def recognize_people():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法初始化视频捕获")
        return
    
    # 获取视频尺寸
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    
    # 初始化人脸检测器
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((frame_width, frame_height))
    
    # 初始化人脸识别器
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_detect/model/face_recognition_sface_2021dec.onnx", "")
    
    # 加载已知人名
    names = []
    try:
        with open("face_detect/feature/name.txt", "r") as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"无法加载名字文件: {e}")
    
    print(f"已加载 {len(names)} 个名字")
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 调整帧大小
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # 人脸检测
        tm.start()
        _, faces = detector.detect(frame)
        tm.stop()
        fps = tm.getFPS()
        
        tar_name = []
        if faces is not None:
            for i in range(faces.shape[0]):
                tar_name.append("unknown")
                print("检测到人脸")
                
                # 人脸对齐和特征提取
                aligned_face = face_recognizer.alignCrop(frame, faces[i])
                feature = face_recognizer.feature(aligned_face)
                
                # 从XML加载特征
                try:
                    fs = cv2.FileStorage("./feature/vocabulary.xml", cv2.FILE_STORAGE_READ)
                    
                    # 与数据库中的每个特征比较
                    for name in names:
                        mat_vocabulary = fs.getNode(name).mat()
                        if mat_vocabulary is None:
                            continue
                            
                        # 计算相似度
                        cos_score = face_recognizer.match(
                            feature, mat_vocabulary, 
                            cv2.FaceRecognizerSF_FR_COSINE)
                        
                        l2_score = face_recognizer.match(
                            feature, mat_vocabulary,
                            cv2.FaceRecognizerSF_FR_NORM_L2)
                        
                        print(f"{name} {cos_score} {l2_score}")
                        
                        # 检查是否匹配
                        if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
                            tar_name[i] = name
                            break
                except Exception as e:
                    print(f"特征匹配错误: {e}")
        
        # 可视化结果
        result = frame.copy()
        visualize(result, frame_count, faces, fps)
        write_names(result, faces, tar_name)
        
        # 显示结果
        cv2.imshow("Live Face Recognition", result)
        
        # 按键处理
        key = cv2.waitKey(30)
        if key == 27:  # ESC键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 摄像头人脸检测
def camera_detect_people():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法初始化视频捕获")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((frame_width, frame_height))
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        tm.start()
        _, faces = detector.detect(frame)
        tm.stop()
        fps = tm.getFPS()
        
        result = frame.copy()
        visualize(result, frame_count, faces, fps)
        cv2.imshow("Face Detection", result)
        
        key = cv2.waitKey(30)
        if key == 32:  # 空格键保存
            names = ["Zhongxingwei"]
            save_face_data(result, faces, names)
        elif key == 27:  # ESC键退出
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 图像文件人脸检测
def image_detect_people(filepath, names):
    image = cv2.imread(filepath)
    if image is None:
        print("图片加载失败")
        return
    
    frame_width = int(image.shape[1] * scale)
    frame_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (frame_width, frame_height))
    
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((frame_width, frame_height))
    
    tm = cv2.TickMeter()
    tm.start()
    _, faces = detector.detect(image)
    tm.stop()
    
    if faces is None or faces.shape[0] < 1:
        print("无法在图像中找到人脸")
        return
    
    print(f"检测到 {faces.shape[0]} 张人脸")
    visualize(image, -1, faces, tm.getFPS())
    
    while True:
        cv2.imshow("Image Face Detection", image)
        key = cv2.waitKey(25)
        
        if key == 32:  # 空格键保存
            save_face_data(image, faces, names)
        elif key == 27:  # ESC键退出
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":


    # 主程序入口
    camera_detect_people()
    # recognize_people()
    
    # 图像检测示例
    # names = ["YanZu"]
    # image_detect_people("./image/yanzu_wu1.png", names)
