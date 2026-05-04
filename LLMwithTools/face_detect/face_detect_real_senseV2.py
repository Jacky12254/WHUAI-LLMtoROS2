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

# ================= 移植的数据保存功能 =================
def save_face_data(image, faces, names):
    """
    保存人脸特征数据到本地XML和txt
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model/face_recognition_sface_2021dec.onnx")
    
    face_recognizer = cv2.FaceRecognizerSF_create(model_path, "")
    
    for i in range(faces.shape[0]):
        aligned_face = face_recognizer.alignCrop(image, faces[i])
        feature = face_recognizer.feature(aligned_face)
        
        # 保存特征到XML
        xml_path = os.path.join(script_dir, "feature/vocabulary.xml")
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_APPEND)
        fs.write(names[i], feature)
        fs.release()
        
        # 保存名字到文本文件
        name_file_path = os.path.join(script_dir, "feature/name.txt")
        try:
            with open(name_file_path, "a") as f:
                f.write("\n" + names[i])
            print(f"✅ 成功录入人脸: {names[i]}")
        except Exception as e:
            print(f"❌ 无法保存名字文件: {e}")

# ================= 核心修改点：识别与录入接口 =================

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
        print("释放 RealSense 摄像头 (Face Recognition)...")
        pipeline.stop()
        cv2.destroyAllWindows()


def camera_detect_people(target_name="NewPerson", stop_event=None):
    """
    通过 RealSense 摄像头实时检测并录入人脸
    target_name: 准备录入的姓名
    stop_event: 线程停止信号
    """
    pipeline, align = initialize_realsense()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # RealSense 默认配置分辨率 1280x720
    scaled_width = int(1280 * scale)
    scaled_height = int(720 * scale)
    
    detector = cv2.FaceDetectorYN_create(
        os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((scaled_width, scaled_height))
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    print(f"👉 准备录入 [{target_name}] 的人脸。请正对摄像头，按【空格键】保存，按【ESC键】退出。")

    try:
        while True:
            if stop_event and stop_event.is_set():
                print("🛑 收到停止信号，退出人脸录入进程...")
                break

            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            frame = np.asanyarray(color_frame.get_data())
            resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
            
            tm.start()
            _, faces = detector.detect(resized_frame)
            tm.stop()
            fps = tm.getFPS()
            
            result = resized_frame.copy()
            visualize(result, frame_count, faces, fps)
            
            # 添加提示文字
            cv2.putText(result, f"Recording: {target_name} | Press SPACE to save", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Camera Face Entry (RealSense)", result)
            
            key = cv2.waitKey(30)
            if key == 32:  # 空格键保存
                if faces is not None and faces.shape[0] > 0:
                    names = [target_name] * faces.shape[0]
                    save_face_data(result, faces, names)
                else:
                    print("⚠️ 未检测到人脸，无法保存！")
            elif key == 27:  # ESC键退出
                break
            
            frame_count += 1
            
    finally:
        print("释放 RealSense 摄像头 (Face Entry)...")
        pipeline.stop()
        cv2.destroyAllWindows()


def image_detect_people(filepath, target_names):
    """
    通过静态图片文件检测并录入人脸
    filepath: 图片绝对或相对路径
    target_names: 录入的人名列表，需与图中检测到的人脸数量对应
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 允许使用相对路径或绝对路径
    full_filepath = filepath if os.path.isabs(filepath) else os.path.join(script_dir, filepath)
    
    image = cv2.imread(full_filepath)
    if image is None:
        print(f"❌ 图片加载失败，请检查路径: {full_filepath}")
        return
    
    frame_width = int(image.shape[1] * scale)
    frame_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (frame_width, frame_height))
    
    detector = cv2.FaceDetectorYN_create(
        os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((frame_width, frame_height))
    
    tm = cv2.TickMeter()
    tm.start()
    _, faces = detector.detect(image)
    tm.stop()
    
    if faces is None or faces.shape[0] < 1:
        print("⚠️ 无法在图像中找到人脸")
        return
    
    print(f"检测到 {faces.shape[0]} 张人脸。按【空格键】保存，按【ESC键】退出。")
    visualize(image, -1, faces, tm.getFPS())
    
    while True:
        cv2.imshow("Image Face Entry", image)
        key = cv2.waitKey(25)
        
        if key == 32:  # 空格键保存
            # 如果提供的名字数量与人脸数量不符，做基础容错处理
            if len(target_names) < faces.shape[0]:
                print(f"⚠️ 提供的名字数量({len(target_names)})少于检测到的人脸数({faces.shape[0]})，部分人脸使用默认名。")
                target_names.extend([f"Unknown_{i}" for i in range(faces.shape[0] - len(target_names))])
            
            save_face_data(image, faces, target_names[:faces.shape[0]])
            break  # 保存后自动退出
            
        elif key == 27:  # ESC键退出
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- 测试接口用例 ---
    
    # 1. 运行实时人脸识别
    recognize_people()
    
    # 2. 运行摄像头人脸录入 (测试时可取消注释)
    # camera_detect_people(target_name="Zhongxingwei")
    
    # 3. 运行静态图片人脸录入 (测试时可取消注释)
    # image_detect_people("./image/yanzu_wu1.png", ["YanZu"])