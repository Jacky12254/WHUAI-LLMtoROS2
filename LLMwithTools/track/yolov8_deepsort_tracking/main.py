import tempfile
import time
import math
from pathlib import Path
import pyrealsense2 as rs
import numpy as np
import cv2 # opencv-python
from ultralytics import YOLO
import deep_sort.deep_sort.deep_sort as ds

 # 加载yoloV8模型权重
model = YOLO("yolov8n.pt")
# 设置需要检测和跟踪的目标类别
# yoloV8官方模型的第一个类别为'person'
detect_class = 0
print(f"detecting {model.names[detect_class]}") # model.names返回模型所支持的所有物体类别
# 加载DeepSort模型
tracker = ds.DeepSort("/home/jacky/vision/track/yolov8_deepsort_tracking/deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

# 配置 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
 
# 启动相机流
pipeline.start(config)
align_to = rs.stream.color  # 与color流对齐
align = rs.align(align_to)
 
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
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # 将位置信息转换为整数。
            # 绘制bounding box和文本
            
            # 计算步长
            xrange = max(1, math.ceil(abs((x1 - x2) / 30)))
            yrange = max(1, math.ceil(abs((y1 - y2) / 30)))
            # xrange = 1
            # yrange = 1

            point_cloud_data = []

            # 获取范围内点的三维坐标
            for x_position in range(x1, x2, xrange):
                for y_position in range(y1, y2, yrange):
                    depth_pixel = [x_position, y_position]
                    dis, camera_coordinate = get_3d_camera_coordinate(depth_pixel, aligned_depth_frame,
                                                                      depth_intrin)  # 获取对应像素点的三维坐标
                    point_cloud_data.append(f"{camera_coordinate} ")

            # 一次性写入所有数据
            with open("/home/jacky/vision/track/yolov8_deepsort_tracking/data/point_cloud_data.txt", "a") as file:
                file.write(f"\nTime: {time.time()}\n")
                file.write(" ".join(point_cloud_data))
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(color_image, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5, text_color=(255, 255, 255), bg_color=(255, 0, 255))
            # 显示中心点坐标
            ux = int((x1 + x2) / 2)
            uy = int((y1 + y2) / 2)
            dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame,
                                                              depth_intrin)  # 获取对应像素点的三维坐标
            formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"

            cv2.circle(color_image, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
            cv2.putText(color_image, formatted_camera_coordinate, (ux + 20, uy + 10), 0, 1,
                        [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标

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
 
finally:
    # 停止流
    pipeline.stop()
