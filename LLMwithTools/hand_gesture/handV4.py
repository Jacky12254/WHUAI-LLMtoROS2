import cv2
import mediapipe as mp
import math
import time
import numpy as np
from collections import deque
import pyrealsense2 as rs # 新增 RealSense 库

# ---------------------------- 角度计算（静态手势）----------------------------
def points_cos_angle(point1, point2):
    try:
        angle_ = math.degrees(math.acos(
            (point1[0] * point2[0] + point1[1] * point2[1]) /
            (((point1[0] ** 2 + point1[1] ** 2) * (point2[0] ** 2 + point2[1] ** 2)) ** 0.5)
        ))
    except:
        angle_ = 65535.0
    if angle_ > 180.0:
        angle_ = 65535.0
    return angle_

def get_fingers_angle(handPoints_list):
    angle_list = []
    # thumb
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[2][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[2][1]))),
        ((int(handPoints_list[3][0]) - int(handPoints_list[4][0])),
         (int(handPoints_list[3][1]) - int(handPoints_list[4][1])))
    )
    angle_list.append(angle_)
    # index
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[6][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[6][1]))),
        ((int(handPoints_list[7][0]) - int(handPoints_list[8][0])),
         (int(handPoints_list[7][1]) - int(handPoints_list[8][1])))
    )
    angle_list.append(angle_)
    # middle
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[10][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[10][1]))),
        ((int(handPoints_list[11][0]) - int(handPoints_list[12][0])),
         (int(handPoints_list[11][1]) - int(handPoints_list[12][1])))
    )
    angle_list.append(angle_)
    # ring
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[14][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[14][1]))),
        ((int(handPoints_list[15][0]) - int(handPoints_list[16][0])),
         (int(handPoints_list[15][1]) - int(handPoints_list[16][1])))
    )
    angle_list.append(angle_)
    # pinky
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[18][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[18][1]))),
        ((int(handPoints_list[19][0]) - int(handPoints_list[20][0])),
         (int(handPoints_list[19][1]) - int(handPoints_list[20][1])))
    )
    angle_list.append(angle_)
    return angle_list

def get_hand_gesture(fingers_angle_List):
    thr_angle_others_bend = 60.0
    thr_angle_thumb_bend = 45.0
    thr_angle_straight = 20.0
    gesture_str = None
    if 65535.0 not in fingers_angle_List:
        if fingers_angle_List[0] > thr_angle_thumb_bend:      
            if (fingers_angle_List[1] > thr_angle_others_bend and
                fingers_angle_List[2] > thr_angle_others_bend and
                fingers_angle_List[3] > thr_angle_others_bend and
                fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "fist"
            elif (fingers_angle_List[1] < thr_angle_straight and
                  fingers_angle_List[2] < thr_angle_straight and
                  fingers_angle_List[3] < thr_angle_straight and
                  fingers_angle_List[4] < thr_angle_straight):
                gesture_str = "four"
            elif (fingers_angle_List[1] < thr_angle_straight and
                  fingers_angle_List[2] < thr_angle_straight and
                  fingers_angle_List[3] < thr_angle_straight and
                  fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "three"
            elif (fingers_angle_List[1] < thr_angle_straight and
                  fingers_angle_List[2] < thr_angle_straight and
                  fingers_angle_List[3] > thr_angle_others_bend and
                  fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "two"
            elif (fingers_angle_List[1] < thr_angle_straight and
                  fingers_angle_List[2] > thr_angle_others_bend and
                  fingers_angle_List[3] > thr_angle_others_bend and
                  fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "one"
        elif fingers_angle_List[0] < thr_angle_straight:     
            if (fingers_angle_List[1] < thr_angle_straight and
                fingers_angle_List[2] < thr_angle_straight and
                fingers_angle_List[3] < thr_angle_straight and
                fingers_angle_List[4] < thr_angle_straight):
                gesture_str = "five"
            elif (fingers_angle_List[1] > thr_angle_others_bend and
                  fingers_angle_List[2] > thr_angle_others_bend and
                  fingers_angle_List[3] > thr_angle_others_bend and
                  fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "thumbUp"
    return gesture_str

def detect_pointing(fingers_angle_List, hand_points):
    thr_straight = 20.0      
    thr_bend = 60.0          

    if 65535.0 in fingers_angle_List:
        return False, (0, 0), 0

    index_straight = fingers_angle_List[1] < thr_straight
    middle_bent = fingers_angle_List[2] > thr_bend
    ring_bent = fingers_angle_List[3] > thr_bend
    pinky_bent = fingers_angle_List[4] > thr_bend

    if index_straight and middle_bent and ring_bent and pinky_bent:
        wrist = hand_points[0]
        tip = hand_points[8]
        dx = tip[0] - wrist[0]
        dy = tip[1] - wrist[1]
        length = math.hypot(dx, dy)
        if length > 0:
            unit_dx = dx / length
            unit_dy = dy / length
            angle_deg = math.degrees(math.atan2(unit_dy, unit_dx))
            return True, (unit_dx, unit_dy), angle_deg
        else:
            return False, (0, 0), 0
    else:
        return False, (0, 0), 0

# ---------------------------- 挥手识别（改进版）----------------------------
def handwave_recognize(prev_list, curr_list, min_fingers=3):
    if prev_list is None or curr_list is None: return None
    needed_ids = [8, 12, 16, 20]
    for idx in needed_ids:
        if idx >= len(prev_list) or idx >= len(curr_list): return None
        
    # 👇 【核心修改 1：动态距离阈值】
    # 提取之前帧中手部的 X 坐标，计算手掌宽度
    xs = [prev_list[i][0] for i in range(len(prev_list))]
    hand_width = max(xs) - min(xs)
    
    # 运动阈值设定为当前手部宽度的 0.8 倍。距离越远手越小，阈值自动跟着变小！
    threshold = max(hand_width * 0.8, 8.0) 

    directions = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
    deltas = []
    for idx in needed_ids:
        x1, y1 = prev_list[idx]
        x2, y2 = curr_list[idx]
        dx = x2 - x1
        dy = y2 - y1
        deltas.append((dx, dy))
        if dx > threshold: directions['right'] += 1
        elif dx < -threshold: directions['left'] += 1
        
        if dy > threshold: directions['down'] += 1
        elif dy < -threshold: directions['up'] += 1
        
    best_dir = max(directions, key=lambda d: directions[d])
    best_votes = directions[best_dir]
    
    if best_votes >= min_fingers:
        if best_dir in ('left', 'right'):
            avg_delta = sum(abs(d[0]) for d in deltas) / len(deltas)
        else:
            avg_delta = sum(abs(d[1]) for d in deltas) / len(deltas)
        if avg_delta > threshold * 0.8:
            return best_dir
    return None

def find_nearest_object_on_ray(depth_frame, origin, direction, tolerance=0.1, min_distance=0.15):
    """
    利用点云计算射线上最近的物体坐标
    - origin: (x, y, z) 射线起点 (指尖)
    - direction: (vx, vy, vz) 射线单位方向向量
    - tolerance: 允许的误差半径 (米)，默认 0.1米 (10厘米)
    - min_distance: 盲区距离 (米)，默认 0.15米，为了防止射线撞到自己的手指或手背
    """
    # 1. 让 RealSense 引擎直接生成当前帧的三维点云
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    
    # 将点云数据转换为安全的 (N, 3) NumPy 数组
    verts = np.empty((len(vtx), 3), dtype=np.float32)
    verts[:, 0] = vtx['f0']
    verts[:, 1] = vtx['f1']
    verts[:, 2] = vtx['f2']
    
    # 过滤掉深度无效的黑洞点 (Z <= 0)
    valid_mask = (verts[:, 2] > 0)
    valid_points = verts[valid_mask]
    
    if len(valid_points) == 0: return None
        
    O = np.array(origin)
    V = np.array(direction)
    
    # 2. 计算所有有效点到射线起点(指尖)的向量 W
    W = valid_points - O
    
    # 3. 计算点在射线方向上的投影长度 t (通过点积 W·V 获得)
    # 物理意义：这个点在射线前方多远的地方
    t = np.dot(W, V)
    
    # 4. 过滤掉在手指后方的点，以及离手指太近的点（把手自己的点裁掉）
    front_mask = t > min_distance
    W_front = W[front_mask]
    t_front = t[front_mask]
    points_front = valid_points[front_mask]
    
    if len(points_front) == 0: return None
        
    # 5. 计算点到射线的垂直距离 d (利用勾股定理: d^2 = |W|^2 - t^2)
    W_sq = np.sum(W_front**2, axis=1)
    d_sq = W_sq - t_front**2
    d_sq = np.maximum(d_sq, 0) # 消除浮点数精度带来的极小负数
    d = np.sqrt(d_sq)
    
    # 6. 找出在圆柱体误差范围内 (d < tolerance) 的点
    cylinder_mask = d < tolerance
    final_points = points_front[cylinder_mask]
    final_t = t_front[cylinder_mask]
    
    if len(final_points) == 0: return None
        
    # 7. 在所有命中的点中，找出 t 最小的（也就是离指尖最近的）那个点！
    closest_idx = np.argmin(final_t)
    return final_points[closest_idx]

# ---------------------------- 主程序 ----------------------------
def start_gesture_recognition(on_wave_callback=None, 
                              on_point_callback=None, 
                              stop_event=None, 
                              local_display=True):
    mp_holistic = mp.solutions.holistic
    
    # 1. 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置深度和色彩流
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"打开 RealSense 相机失败，请检查连接: {e}")
        return

    # 创建对齐对象，将深度图对齐到色彩图
    align_to = rs.stream.color
    align = rs.align(align_to)

    holistic = mp_holistic.Holistic(model_complexity=0)

    previous_time_fps = 0
    cooling_time = 0.5
    last_wave_time = 0
    last_wave_dir = None
    frame_buffer = deque(maxlen=10)
    wave_history = []
    gesture_str = None          
    pointing_active = False     
    pointing_vector = (0, 0)    
    pointing_angle = 0  
    
    # 新增：保存当前的距离        
    current_distance = 0.0
    last_point_time = 0
    try:
        while not (stop_event and stop_event.is_set()):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
                
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            img = np.asanyarray(color_frame.get_data())
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape
            
            results = holistic.process(img_rgb)
            current_time = time.time()
            hand_points = []
            
            base_3d = None # 记录食指根部的 3D 坐标
            tip_3d = None  # 记录食指指尖的 3D 坐标

            active_hand_landmarks = None
            if results.right_hand_landmarks:
                active_hand_landmarks = results.right_hand_landmarks
            elif results.left_hand_landmarks:
                active_hand_landmarks = results.left_hand_landmarks

            # 只要有任何一只手存在，就开始处理
            if active_hand_landmarks:
                for id, lm in enumerate(active_hand_landmarks.landmark):
                    cx = min(max(int(lm.x * w), 0), w - 1)
                    cy = min(max(int(lm.y * h), 0), h - 1)
                    hand_points.append((cx, cy))
                    
                    # 【核心 1】：获取食指根部 (id=5) 的 3D 坐标作为起点
                    if id == 5:
                        dist_5 = depth_frame.get_distance(cx, cy)
                        if dist_5 > 0:
                            base_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], dist_5)

                    # 【核心 2】：获取食指指尖 (id=8) 的 3D 坐标作为终点
                    if id == 8:
                        cv2.circle(img_rgb, (cx, cy), 5, (0, 0, 255), -1)
                        dist_8 = depth_frame.get_distance(cx, cy)
                        if dist_8 > 0:
                            tip_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], dist_8)
                        current_distance = dist_8

                if len(hand_points) >= 21:
                    angle_list = get_fingers_angle(hand_points)
                    gesture_str = get_hand_gesture(angle_list)
                    pointing_active, vec2d, ang2d = detect_pointing(angle_list, hand_points)
                    pointing_vector = vec2d
                    pointing_angle = ang2d
                    # 【核心 3】：如果检测到指向手势，并且起点终点都有深度数据
                    # 【核心 3】：如果检测到指向手势...
                    if pointing_active and base_3d and tip_3d:
                        vx = tip_3d[0] - base_3d[0]
                        vy = tip_3d[1] - base_3d[1]
                        vz = tip_3d[2] - base_3d[2]
                        
                        length = math.sqrt(vx**2 + vy**2 + vz**2)
                        if length > 0:
                            uvx, uvy, uvz = vx / length, vy / length, vz / length
                            
                            now = time.time()
                            if now - last_point_time > 2.0:
                                # 👇 新增：发射射线，寻找容错范围 10cm 内最近的物体坐标！
                                hit_point = find_nearest_object_on_ray(
                                    depth_frame=depth_frame, 
                                    origin=tip_3d, 
                                    direction=(uvx, uvy, uvz), 
                                    tolerance=0.10,     # 半径 10 厘米的容错圆柱
                                    min_distance=0.15   # 忽略指尖前方 15 厘米内的点（避开手指）
                                )
                                
                                if hit_point is not None:
                                    print(f"👉 击中物体! 物体三维坐标: ({hit_point[0]:.2f}, {hit_point[1]:.2f}, {hit_point[2]:.2f})")
                                    # 触发回调，把目标点坐标也一起传出去
                                    if on_point_callback:
                                        on_point_callback(tip_3d, (uvx, uvy, uvz), hit_point)
                                else:
                                    print(f"👉 指向了虚空，前方无物体。")
                                    
                                last_point_time = now

                frame_buffer.append(hand_points.copy())
                if len(frame_buffer) == 10:
                    # 拿 0.3 秒前的帧和当前帧对比
                    wave_dir = handwave_recognize(frame_buffer[0], frame_buffer[-1], min_fingers=2)
                    
                    now = time.time()
                    # 1. 随时清理 2 秒之前的老旧动作记录（挥手中断太久就作废）
                    wave_history = [w for w in wave_history if now - w[0] < 2.0]
                    
                    if wave_dir:
                        # 2. 如果历史为空，或者当前动作方向和上次记录的方向不同（例如：左 -> 右）才算一次有效累积
                        if not wave_history or wave_history[-1][1] != wave_dir:
                            wave_history.append((now, wave_dir))
                            
                        # 3. 如果2秒内积攒了至少 2 个不同的方向 (一来一回)，则认定是真实挥手！
                        if len(wave_history) >= 2:
                            if (now - last_wave_time >= cooling_time):
                                # 👇 【核心新增】：获取挥手者(以手腕 id=0 为基准)的三维空间坐标
                                wrist_x, wrist_y = hand_points[0]
                                wrist_dist = depth_frame.get_distance(wrist_x, wrist_y)
                                
                                person_3d = None
                                if wrist_dist > 0:
                                    # 将二维像素+深度，反投影为真实世界的三维坐标 (X, Y, Z)
                                    person_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [wrist_x, wrist_y], wrist_dist)
                                
                                if person_3d:
                                    px, py, pz = person_3d
                                    print(f"👋 确认有效挥手！动作: {[w[1] for w in wave_history]}, 客人三维坐标: ({px:.2f}, {py:.2f}, {pz:.2f})")
                                    
                                    # 触发回调给 LLM，将距离和精确三维坐标一起传出去
                                    if on_wave_callback:
                                        on_wave_callback(wave_dir, wrist_dist, person_3d)
                                else:
                                    print("👋 确认有效挥手！但未获取到稳定的深度坐标。")
                                    if on_wave_callback:
                                        on_wave_callback(wave_dir, current_distance, None)
                                last_wave_time = now
                                wave_history.clear() # 触发后清空动作积攒槽
                                frame_buffer.clear() # 清空缓存，防止连续疯狂触发
            else:
                frame_buffer.clear()
                gesture_str = None
                pointing_active = False

            if previous_time_fps > 0:
                fps = 1.0 / (current_time - previous_time_fps)
            else:
                fps = 0
            previous_time_fps = current_time

            if local_display:
            # 翻转画面做镜像显示
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)# 注意：RealSense 的色彩流默认是 BGR 格式的，所以我们直接转换回 BGR 就好，不需要再翻转了。
                # img_bgr = cv2.flip(img_bgr, 1)

                if pointing_active and len(hand_points) >= 21:
                    wrist_x, wrist_y = hand_points[0]
                    tip_x, tip_y = hand_points[8]
                    wrist_x_m = w - wrist_x
                    tip_x_m = w - tip_x
                    cv2.arrowedLine(img_bgr, (int(wrist_x_m), int(wrist_y)), (int(tip_x_m), int(tip_y)),
                                    (0, 255, 255), 3, tipLength=0.3)
                    vec_text = f"Point: ({pointing_vector[0]:.2f}, {pointing_vector[1]:.2f}) ang={int(pointing_angle)} d={current_distance:.2f}m"
                    cv2.putText(img_bgr, vec_text, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                cv2.putText(img_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                if gesture_str:
                    cv2.putText(img_bgr, f"Static: {gesture_str} Dist: {current_distance:.2f}m", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                if last_wave_dir and (time.time() - last_wave_time < 0.5):
                    cv2.putText(img_bgr, f"Wave: {last_wave_dir}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                cv2.imshow("RealSense Hand Gesture", img_bgr)
                if cv2.waitKey(2) & 0xFF == 27:
                    break
    finally:
        # 安全退出，释放资源
        pipeline.stop()
        if local_display:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    start_gesture_recognition()