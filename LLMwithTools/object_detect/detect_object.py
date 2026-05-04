import cv2
import pyrealsense2 as rs
import time
import numpy as np
import math
from ultralytics import YOLO

# 将工具函数改为接收 pipeline 和 align 参数
def get_aligned_images(pipeline, align):
    frames = pipeline.wait_for_frames()  
    aligned_frames = align.process(frames)  
    aligned_depth_frame = aligned_frames.get_depth_frame()  
    color_frame = aligned_frames.get_color_frame()  
 
    intr = color_frame.profile.as_video_stream_profile().intrinsics  
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  
 
    depth_image = np.asanyarray(aligned_depth_frame.get_data())  
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  
    depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  
    color_image = np.asanyarray(color_frame.get_data())  
 
    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame
 
def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)  
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate

# ================= 核心修改点 =================
def object_detection_loop(stop_event=None, callback=None):
    """
    物体检测主循环
    stop_event: 线程停止信号
    callback: 检测到物体时的回调，签名 callback(obj_name, coordinate_str)
    """
    model = YOLO("best.pt") # 如果 best.pt 不在根目录，注意修改路径
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    fps = 0
    frame_count = 0

    try:
        while True:
            # 1. 检查是否收到外部退出信号
            if stop_event and stop_event.is_set():
                print("🛑 收到停止信号，退出物体检测进程...")
                break

            intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images(pipeline, align)
            if not depth_image.any() or not color_image.any():
                continue
     
            time1 = time.time()
            results = model.predict(color_image, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            names_dic = results[0].names
            detected_boxes = results[0].boxes.xyxy
            
            for i, box in enumerate(detected_boxes):
                x1, y1, x2, y2 = map(int, box)
                name = names_dic[int(results[0].boxes.cls[i].item())] # 获取物体名称
                
                ux = int((x1 + x2) / 2)
                uy = int((y1 + y2) / 2)
                dis, camera_coordinate = get_3d_camera_coordinate([ux, uy], aligned_depth_frame, depth_intrin)
                formatted_camera_coordinate = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"
                
                cv2.circle(annotated_frame, (ux, uy), 4, (255, 255, 255), 5)  
                cv2.putText(annotated_frame, formatted_camera_coordinate, (ux + 20, uy + 10), 0, 1,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  
                
                # 2. 触发回调，把物体名称和坐标给大模型
                if callback:
                    callback(name, formatted_camera_coordinate)
     
            frame_count += 1
            time2 = time.time()
            fps = int(1 / (time2 - time1))
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('YOLO Object Detection (RealSense)', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 3. 安全释放摄像头
        print("释放 RealSense 摄像头 (Object)...")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection_loop()