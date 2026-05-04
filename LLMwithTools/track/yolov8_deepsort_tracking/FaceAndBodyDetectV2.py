import tempfile
import time
import math
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2 # opencv-python
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds
import os
# 加载yoloV8模型权重
model = YOLO("yolov8n.pt")
# 设置需要检测和跟踪的目标类别
# yoloV8官方模型的第一个类别为'person'
detect_class = 0
print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别
# 加载DeepSort模型
tracker = ds.DeepSort("track/yolov8_deepsort_tracking/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
script_dir = os.path.dirname(os.path.abspath(__file__))

# 配置 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
 
# 启动相机流
pipeline.start(config)
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)

# 限差参数
cosine_similar_thresh = 0.363
l2norm_similar_thresh = 1.128

# 设置跟踪目标姓名
target_name="ZhuochaoWang"


# ================= 整合的人脸可视化函数 =================
def visualize_face(input_img, frame, faces, fps, index, thickness=2):
    fps_string = f"FPS : {fps:.2f}"
    if faces is None:
        return
    
    face = faces[index]
    # 绘制人脸边界框
    cv2.rectangle(input_img, 
                (int(face[0]), int(face[1])),
                (int(face[0] + face[2]), int(face[1] + face[3])),
                (0, 255, 0), thickness)
        
    # 绘制边界框中心点 (Source 6 新增)
    center_x = int(face[0] + face[2] / 2)
    center_y = int(face[1] + face[3] / 2)
    cv2.circle(input_img, (center_x, center_y), 2, (255, 255, 0), thickness)

    # 绘制5个关键特征点 (右眼、左眼、鼻尖、右嘴角、左嘴角)
    cv2.circle(input_img, (int(face[4]), int(face[5])), 2, (255, 0, 0), thickness)
    cv2.circle(input_img, (int(face[6]), int(face[7])), 2, (0, 0, 255), thickness)
    cv2.circle(input_img, (int(face[8]), int(face[9])), 2, (0, 255, 0), thickness)  # 鼻尖
    cv2.circle(input_img, (int(face[10]), int(face[11])), 2, (255, 0, 255), thickness)
    cv2.circle(input_img, (int(face[12]), int(face[13])), 2, (0, 255, 255), thickness)

def write_names(input_img, faces, name, index):
    if faces is None:
        return
    org = (int(faces[index][0]), int(faces[index][1]) - 10) # 稍微上移避免挡住框
    cv2.putText(input_img, name, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧
    aligned_frames = align.process(frames)  # 获取对齐帧
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
    color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
 
    # 相机参数的获取
    intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile(
    ).intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }'''
 
    # 保存内参到本地
    # with open('./intrinsics.json', 'w') as fp:
    # json.dump(camera_parameters, fp)
    #######################################################
 
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
    depth_image_3d = np.dstack(
        (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
    color_image = np.asanyarray(color_frame.get_data())  # RGB图
 
    # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
 
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate
 
def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """绘制带有背景的文本。

    :param img: 输入图像。
    :param text: 要绘制的文本。
    :param origin: 文本的左上角坐标。
    :param font: 字体类型。
    :param font_scale: 字体大小。
    :param text_color: 文本的颜色。
    :param bg_color: 背景的颜色。
    :param thickness: 文本的线条厚度。
    """
    # 计算文本的尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # 减去5以留出一些边距
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # 在矩形上绘制文本
    text_origin = (origin[0], origin[1] - 5)  # 从左上角的位置减去5来留出一些边距
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    
def extract_detections(results, detect_class):
    """
    从模型结果中提取和处理检测信息。
    - results: YoloV8模型预测结果，包含检测到的物体的位置、类别和置信度等信息。
    - detect_class: 需要提取的目标类别的索引。
    参考: https://docs.ultralytics.com/modes/predict/#working-with-results
    """
    
    # 初始化一个空的二维numpy数组，用于存放检测到的目标的位置信息
    # 如果视频中没有需要提取的目标类别，如果不初始化，会导致tracker报错
    detections = np.empty((0, 4)) 
    
    confarray = [] # 初始化一个空列表，用于存放检测到的目标的置信度。

    # 遍历检测结果
    # 参考：https://docs.ultralytics.com/modes/predict/#working-with-results
    for r in results:
        for box in r.boxes:
            # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist() # 提取目标的位置信息，并从tensor转换为整数列表。
                conf = round(box.conf[0].item(), 2) # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) # 将目标的位置信息添加到detections数组中。
                confarray.append(conf) # 将目标的置信度添加到confarray列表中。
    return detections, confarray # 返回提取出的位置信息和置信度。

# 视频处理
def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    处理视频，检测并跟踪目标。
    - input_path: 输入视频文件的路径。
    - output_path: 处理后视频保存的路径。
    - detect_class: 需要检测和跟踪的目标类别的索引。
    - model: 用于目标检测的模型。
    - tracker: 用于目标跟踪的模型。
    """
    cap = cv2.VideoCapture(input_path)  # 使用OpenCV打开视频文件。

    #cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening video file {input_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）。
    output_video_path = Path(output_path) / "output.avi" # 设置输出视频的保存路径。

    # 设置视频编码格式为XVID格式的avi文件
    # 如果需要使用h264编码或者需要保存为其他格式，可能需要下载openh264-1.8.0
    # 下载地址：https://github.com/cisco/openh264/releases/tag/v1.8.0
    # 下载完成后将dll文件放在当前文件夹内
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size, isColor=True) # 创建一个VideoWriter对象用于写视频。

    # 对每一帧图片进行读取和处理
    while True:
        success, frame = cap.read() # 逐帧读取视频。
        
        # 如果读取失败（或者视频已处理完毕），则跳出循环。
        if not (success):
            break

        # 使用YoloV8模型对当前帧进行目标检测。
        results = model(frame, stream=True)

        # 从预测结果中提取检测信息。
        detections, confarray = extract_detections(results, detect_class)

        # 使用deepsort模型对检测到的目标进行跟踪。
        resultsTracker = tracker.update(detections, confarray, frame)
        
        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # 将位置信息转换为整数。

            # 绘制bounding box和文本
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)  # 将处理后的帧写入到输出视频文件中。
            
    output_video.release()  # 释放VideoWriter对象。
    cap.release()  # 释放视频文件。
    
    print(f'output dir is: {output_video_path}')
    return output_video_path

def detect_and_track_camera(detect_class: int, model, tracker):


    cap = cv2.VideoCapture(0)  # 使用OpenCV打开视频文件。

    #cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # 检查视频文件是否成功打开。
        print(f"Error opening camera 0")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 获取视频的分辨率（宽度和高度）。

    # 对每一帧图片进行读取和处理
    while True:
        success, frame = cap.read() # 逐帧读取视频。
        
        # 如果读取失败（或者视频已处理完毕），则跳出循环。
        if not (success):
            break

        # 使用YoloV8模型对当前帧进行目标检测。
        results = model(frame, stream=True)

        # 从预测结果中提取检测信息。
        detections, confarray = extract_detections(results, detect_class)

        # 使用deepsort模型对检测到的目标进行跟踪。
        resultsTracker = tracker.update(detections, confarray, frame)
        
        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # 将位置信息转换为整数。

            # 绘制bounding box和文本
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))

        #output_video.write(frame)  # 将处理后的帧写入到输出视频文件中。
        cv2.imshow("temp",frame)
        if cv2.waitKey(1) == ord('q'):
          break
            
    #output_video.release()  # 释放VideoWriter对象。
    cap.release()  # 释放视频文件。
    cv2.destroyAllWindows()
    #print(f'output dir is: {output_video_path}')
    #return output_video_path
    return None

    #detect_and_track(input_path, output_path, detect_class, model, tracker)
    detect_and_track_camera(detect_class, model, tracker)

# 从本地数据库识别人脸（使用RealSense）
def recognize_people_set_target():
    target_id=None
    # 初始化人脸检测器（稍后设置尺寸）

    
    detector = cv2.FaceDetectorYN_create(
        os.path.join(script_dir, "model/face_detection_yunet_2023mar.onnx"), "", 
        (320, 320), 0.9, 0.3, 5000)
    
    # 初始化人脸识别器
    face_recognizer = cv2.FaceRecognizerSF_create(
        os.path.join(script_dir, "model/face_recognition_sface_2021dec.onnx"), "")
    
    print(f"开始设置跟踪目标")
    frame_count = 0
    tm = cv2.TickMeter()
    try:
        while True:
            detections = np.empty((0, 4)) 
            confarray = [] # 初始化一个空列表，用于存放检测到的目标的置信度。
            
            # 等待RealSense帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的颜色帧
            color_frame = aligned_frames.get_color_frame()
           
            if not color_frame:
                continue
            
            # 将颜色帧转换为OpenCV格式
            frame = np.asanyarray(color_frame.get_data())
            
            # 获取实际帧尺寸并缩放
            frame_height, frame_width = frame.shape[:2]
            scaled_width = int(frame_width )
            scaled_height = int(frame_height)
            
            # 设置检测器输入尺寸
            detector.setInputSize((scaled_width, scaled_height))
            
            # 调整帧大小
            resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
            people_results = model(resized_frame, stream=True)
            result = resized_frame.copy()
            # 人脸检测
            tm.start()
            _, faces = detector.detect(resized_frame)
            tm.stop()
            fps = tm.getFPS()
            target_face=None
            target_index=None
            if faces is not None:
                print("检测到人脸")
            #     for i in range(faces.shape[0]):
            #         # 人脸对齐和特征提取
            #         aligned_face = face_recognizer.alignCrop(resized_frame, faces[i])
            #         feature = face_recognizer.feature(aligned_face)
                    
            #         # 从XML加载特征
            #         try:
            #             fs = cv2.FileStorage(os.path.join(script_dir, "feature/vocabulary.xml"), cv2.FILE_STORAGE_READ)
                        
            #             # 与数据库中的每个特征比较
                        
            #             mat_vocabulary = fs.getNode(target_name).mat()
            #             if mat_vocabulary is None:
            #                 continue
                                
            #              # 计算相似度
            #             cos_score = face_recognizer.match(
            #                 feature, mat_vocabulary, 
            #                 cv2.FaceRecognizerSF_FR_COSINE)
                            
            #             l2_score = face_recognizer.match(
            #                 feature, mat_vocabulary,
            #                 cv2.FaceRecognizerSF_FR_NORM_L2)
                                                  
                            
            #             # 检查是否匹配
            #             if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
            #                 print("探测到待识别人脸")
            #                 target_face=faces[i]
            #                 target_index=i
            #                 break
            #         except Exception as e:
            #             print(f"特征匹配错误: {e}")


            # if target_face is None:
            #     print("未检测到待识别人脸")

            # else:
                # 可视化结果
                visualize_face(result, frame_count, faces, fps,target_index)
                write_names(result, faces, target_name,target_index)
                print("开始检测人体")
                if target_index is None:
                    break
                      
                for r in people_results:
                    for box in r.boxes:
                    # 如果检测到的目标类别与指定的目标类别相匹配，提取目标的位置信息和置信度
                        if box.cls[0].int() == detect_class:
                            print("检测到人体了")
                            x1, y1, x2, y2 = box.xywh[0].int().tolist() # 提取目标的位置信息，并从tensor转换为整数列表。
                            if target_face is None:
                                break
                            if (x1-x2/2)<int(target_face[0]) and (x1+x2/2)>int(target_face[0]+target_face[2]):
                                print("识别到对应人体")
                                cv2.rectangle(result, 
                                    (int(x1-x2/2), int(y1-y2/2)),
                                    (int(x1+x2/2), int(y2+y2/2)),
                                    (0, 255, 0), 2)
                                # 从预测结果中提取检测信息。
                                conf = round(box.conf[0].item(), 2) # 提取目标的置信度，从tensor中取出浮点数结果，并四舍五入到小数点后两位。
                                detections = np.vstack((detections, np.array([x1, y1, x2, y2]))) # 将目标的位置信息添加到detections数组中。
                                confarray.append(conf) # 将目标的置信度添加到confarray列表中。                            
                                resultsTracker = tracker.update(detections, confarray, result)
                                for x1, y1, x2, y2, Id in resultsTracker:

                                    target_id=Id
                                    print("###################################")
                                    print(target_id)
                                    print("###################################")
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # 将位置信息转换为整数。
                                    # 绘制bounding box和文本             
                                    # 计算步长
                                    xrange = max(1, math.ceil(abs((x1 - x2) / 30)))
                                    yrange = max(1, math.ceil(abs((y1 - y2) / 30)))
                                    # xrange = 1
                                    # yrange = 1
 
                                    point_cloud_data = []
 
                                    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 3)
                                    putTextWithBackground(result, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))
                                    print("已收集到可追踪信息")


                                                     
                frame_count += 1
             # 显示结果
            cv2.imshow('YOLOv8 RealSense', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    
    finally:
        # 确保管道被正确停止
        return target_id
        cv2.destroyAllWindows()

my_id=recognize_people_set_target()

print("###################################")
print(my_id)
print("###################################")
# 初始化 FPS 计算
fps = 0
frame_count = 0
start_time = time.time()
 
try:
    while True:
        # 等待获取一对连续的帧：深度和颜色
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()
 
        if not depth_image.any() or not color_image.any():
            continue
 
        # 获取当前时间
        time1 = time.time()

        # 将图像转为numpy数组
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))


        # 使用YoloV8模型对当前帧进行目标检测。
        results = model(color_image, stream=True)
 
        # 从预测结果中提取检测信息。
        detections, confarray = extract_detections(results, detect_class)
        # 使用deepsort模型对检测到的目标进行跟踪。
        resultsTracker = tracker.update(detections, confarray, color_image)
        
        for x1, y1, x2, y2, Id in resultsTracker:
            if Id != my_id: # 只绘制锁定的目标[cite: 5]
                continue
                
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # 计算目标中心点
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            
            # 解算 3D 相机坐标
            dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame, depth_intrin)
            
            # ================= 整合的人体可视化 =================
            # 1. 绘制人体边界框 (紫红色)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # 2. 绘制带有背景的 Tracking ID 标签
            putTextWithBackground(color_image, f"ID: {int(Id)}", (max(-10, x1), max(40, y1)), 
                                  font_scale=1.0, text_color=(255, 255, 255), bg_color=(255, 0, 255))
            
            # 3. 绘制目标中心点
            cv2.circle(color_image, (ux, uy), 5, (255, 255, 255), -1)
            cv2.circle(color_image, (ux, uy), 7, (0, 0, 255), 2) # 加个红色外圈更醒目
            
            # 4. 绘制 3D 空间坐标
            formatted_camera_coordinate = f"X:{camera_coordinate[0]:.2f} Y:{camera_coordinate[1]:.2f} Z:{camera_coordinate[2]:.2f}m"
            cv2.putText(color_image, formatted_camera_coordinate, (ux + 15, uy + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        # 计算 FPS
        frame_count += 1
        time2 = time.time()
        fps = int(1 / (time2 - time1))
        # 显示 FPS
        cv2.putText(color_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        # 显示结果
        cv2.imshow('YOLOv8 RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite(os.path.join(script_dir, "track.png"), color_image)

 
finally:
    # 停止流
    pipeline.stop()






