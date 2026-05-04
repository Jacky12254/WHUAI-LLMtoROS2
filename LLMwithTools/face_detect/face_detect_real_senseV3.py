import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import pyrealsense2 as rs

# ================= 改进点 1：略微放宽限差参数 =================
# 原值: cosine=0.363, l2=1.128
# 微调后：对非完美正脸更宽容，但注意不要放得太宽以免误识率上升
cosine_similar_thresh = 0.340
l2norm_similar_thresh = 1.150
scale = 1.0

def visualize(input_img, frame, faces, fps, thickness=2):
    fps_string = f"FPS : {fps:.2f}"
    
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        face = faces[i]
        # 绘制边界框
        cv2.rectangle(input_img, 
                     (int(face[0]), int(face[1])),
                     (int(face[0] + face[2]), int(face[1] + face[3])),
                     (0, 255, 0), thickness)
        
        # 绘制边界框中心点
        center_x = int(face[0] + face[2] / 2)
        center_y = int(face[1] + face[3] / 2)
        cv2.circle(input_img, (center_x, center_y), 2, (255, 255, 0), thickness)
        

        # 绘制特征点
        cv2.circle(input_img, (int(face[4]), int(face[5])), 2, (255, 0, 0), thickness)
        cv2.circle(input_img, (int(face[6]), int(face[7])), 2, (0, 0, 255), thickness)
        cv2.circle(input_img, (int(face[8]), int(face[9])), 2, (0, 255, 0), thickness)#这是鼻尖，特征颜色是绿色
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
            print(f"✅ 成功录入特征: {names[i]}")
        except Exception as e:
            print(f"❌ 无法保存名字文件: {e}")


def recognize_people(stop_event=None, callback=None):
    """
    人脸识别主循环
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
                        for name_in_db in names:
                            mat_vocabulary = fs.getNode(name_in_db).mat()
                            if mat_vocabulary is None: continue
                                
                            cos_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_COSINE)
                            l2_score = face_recognizer.match(feature, mat_vocabulary, cv2.FaceRecognizerSF_FR_NORM_L2)
                            
                            if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
                                # ================= 改进点 2：剥离后缀，恢复真实姓名 =================
                                # 如果数据库里的名字是 "Zhongxingwei_3"，则只保留 "Zhongxingwei"
                                real_name = name_in_db.rsplit('_', 1)[0] if '_' in name_in_db else name_in_db
                                
                                tar_name[i] = real_name
                                
                                if callback:
                                    callback(real_name)
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
        print("释放 RealSense 摄像头 (Face Recognition)...")
        pipeline.stop()
        cv2.destroyAllWindows()


def camera_detect_people(target_name="NewPerson", stop_event=None):
    """
    通过 RealSense 摄像头实时检测，打开后自动倒计时3秒，并连拍录入人脸
    连拍逻辑：0.2秒/张，共8张，成功获取>=4张即为有效录入
    """
    pipeline, align = initialize_realsense()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scaled_width = int(1280 * scale)
    scaled_height = int(720 * scale)
    
    detector = cv2.FaceDetectorYN_create(
        os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((scaled_width, scaled_height))
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    # === 状态机与控制变量 ===
    state = "COUNTDOWN"          # 初始状态为倒计时
    countdown_duration = 3.0     # 倒计时秒数
    countdown_start_time = time.time()
    
    total_shots = 8              # 总共需要拍摄的张数
    current_shot = 0             # 当前已拍张数
    valid_captures = 0           # 成功捕获到人脸的有效张数
    capture_interval = 0.2       # 抓拍间隔（秒）
    last_capture_time = 0        
    
    retry_delay_start = 0        # 失败后重试的缓冲时间
    
    print(f"\n👉 准备录入 [{target_name}] 的人脸。")
    print(f"⏳ 窗口已开启，正在自动倒计时 3 秒，请正对摄像头准备！\n")

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
            
            current_time = time.time()
            
            # ================= 核心状态机逻辑 =================
            
            if state == "RETRY_DELAY":
                # 录入失败后的短暂缓冲，提示用户调整，然后重新进入倒计时
                cv2.putText(result, "Failed. Adjust pose. Retrying...", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                if current_time - retry_delay_start > 2.0:  # 缓冲 2 秒
                    state = "COUNTDOWN"
                    countdown_start_time = current_time
                    
            elif state == "COUNTDOWN":
                # 计算剩余秒数并向上取整 (例如 2.8秒 显示 3)
                remain = int(np.ceil(countdown_duration - (current_time - countdown_start_time)))
                
                if remain > 0:
                    cv2.putText(result, f"Get Ready: {target_name}", (10, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
                    # 在画面正中央绘制超大字体的倒计时
                    text = str(remain)
                    font_scale = 6.0
                    thickness = 10
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    text_x = (scaled_width - text_size[0]) // 2
                    text_y = (scaled_height + text_size[1]) // 2
                    cv2.putText(result, text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
                else:
                    # 倒计时结束，立刻进入连拍状态
                    state = "CAPTURING"
                    current_shot = 0
                    valid_captures = 0
                    last_capture_time = current_time - capture_interval # 减去间隔以立刻触发第一张
                    print("\n🚀 开始自动连拍，请稍微左右转动头部...")
                    
            elif state == "CAPTURING":
                # 连拍过程中的 UI 提示
                cv2.putText(result, f"Auto Capturing... {current_shot}/{total_shots}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(result, f"Valid Features: {valid_captures} (Need >= 4)", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                # 执行抓拍逻辑
                if current_time - last_capture_time >= capture_interval:
                    current_shot += 1
                    last_capture_time = current_time
                    
                    if faces is not None and faces.shape[0] > 0:
                        valid_captures += 1
                        unique_name = f"{target_name}_{valid_captures}"
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        save_face_data(result, np.array([largest_face]), [unique_name])
                        print(f"📸 抓拍 {current_shot}/{total_shots}: 成功")
                    else:
                        print(f"⚠️ 抓拍 {current_shot}/{total_shots}: 未检测到人脸")
                        
                    # 判断 8 张连拍是否结束
                    if current_shot >= total_shots:
                        print("\n================ 录入结果 ================")
                        if valid_captures >= 4:
                            print(f"✅ 录入成功！共捕获 {valid_captures} 个有效特征。")
                            print("==========================================")
                            
                            # 画面上显示绿色的 SUCCESS! 停留1秒后自动退出
                            cv2.putText(result, "SUCCESS!", (scaled_width//2 - 150, scaled_height//2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
                            cv2.imshow("Camera Face Entry (RealSense)", result)
                            cv2.waitKey(1000) 
                            break # 成功录入，退出循环
                        else:
                            print(f"❌ 录入失败！仅捕获 {valid_captures} 个特征。准备重试...")
                            print("==========================================")
                            # 失败则进入缓冲状态，随后自动重新倒计时
                            state = "RETRY_DELAY"
                            retry_delay_start = current_time

            cv2.imshow("Camera Face Entry (RealSense)", result)
            
            key = cv2.waitKey(10)
            if key == 27:  # 全程支持随时按 ESC 键退出
                print("已手动取消录入。")
                break
            
            frame_count += 1
            
    finally:
        print("释放 RealSense 摄像头 (Face Entry)...")
        pipeline.stop()
        cv2.destroyAllWindows()


def image_detect_people(filepath, target_names):
    """
    通过静态图片文件检测并录入人脸
    """
    # 此函数保持原样即可，静态图片一般只录正脸
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
        
        if key == 32:  
            if len(target_names) < faces.shape[0]:
                target_names.extend([f"Unknown_{i}" for i in range(faces.shape[0] - len(target_names))])
            save_face_data(image, faces, target_names[:faces.shape[0]])
            break 
            
        elif key == 27:  
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    # 强烈建议先运行这一步，多按几次空格录入5个左右的不同角度
    # camera_detect_people(target_name="Zhongxingwei")
    
    # 录入完毕后再运行识别，你会发现侧脸识别率大幅提升！
    recognize_people()