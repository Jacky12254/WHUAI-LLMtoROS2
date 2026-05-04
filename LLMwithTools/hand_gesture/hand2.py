import cv2
import mediapipe as mp
import math
import time
from collections import deque

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
        if fingers_angle_List[0] > thr_angle_thumb_bend:      # 拇指弯曲
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
        elif fingers_angle_List[0] < thr_angle_straight:      # 拇指伸直
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

# ---------------------------- 指向手势识别 ----------------------------
def detect_pointing(fingers_angle_List, hand_points):
    """
    判断是否指向手势，并返回指向单位向量 (dx, dy) 和角度（度数）。
    条件：食指角度 < 20°，中指、无名指、小指角度 > 60°。
    向量方向：从手腕(0)指向食指指尖(8)。
    """
    thr_straight = 20.0      # 伸直阈值
    thr_bend = 60.0          # 弯曲阈值

    if 65535.0 in fingers_angle_List:
        return False, (0, 0), 0

    # 检查手指弯曲状态
    index_straight = fingers_angle_List[1] < thr_straight
    middle_bent = fingers_angle_List[2] > thr_bend
    ring_bent = fingers_angle_List[3] > thr_bend
    pinky_bent = fingers_angle_List[4] > thr_bend

    if index_straight and middle_bent and ring_bent and pinky_bent:
        # 获取手腕（id=0）和食指指尖（id=8）
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
def handwave_recognize(prev_list, curr_list, img_width, min_fingers=3):
    if prev_list is None or curr_list is None:
        return None
    needed_ids = [8, 12, 16, 20]
    for idx in needed_ids:
        if idx >= len(prev_list) or idx >= len(curr_list):
            return None
    threshold = max(0.12 * img_width, 25)
    directions = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
    deltas = []
    for idx in needed_ids:
        x1, y1 = prev_list[idx]
        x2, y2 = curr_list[idx]
        dx = x2 - x1
        dy = y2 - y1
        deltas.append((dx, dy))
        if dx > threshold:
            directions['right'] += 1
        elif dx < -threshold:
            directions['left'] += 1
        if dy > threshold:
            directions['down'] += 1
        elif dy < -threshold:
            directions['up'] += 1
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

## --- hand2.py ---

# ... [保留文件上面的所有 import 和手势计算函数 (points_cos_angle, get_fingers_angle 等)] ...

# ---------------------------- 主程序 ----------------------------
# 新增 callback 参数，默认为 None
def start_gesture_recognition(on_wave_callback=None):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备连接和权限")
        return

    holistic = mp_holistic.Holistic(model_complexity=0)

    previous_time_fps = 0
    cooling_time = 0.3
    last_wave_time = 0
    last_wave_dir = None
    frame_buffer = deque(maxlen=3)

    gesture_str = None          
    pointing_active = False     
    pointing_vector = (0, 0)    
    pointing_angle = 0          

    while True:
        success, img = cap.read()
        if not success:
            print("读取帧失败，跳过")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        results = holistic.process(img_rgb)

        current_time = time.time()
        hand_points = []

        if results.right_hand_landmarks:
            for id, lm in enumerate(results.right_hand_landmarks.landmark):
                cx, cy = lm.x * w, lm.y * h
                hand_points.append((cx, cy))
                if id == 8:
                    cv2.circle(img_rgb, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            if len(hand_points) >= 21:
                angle_list = get_fingers_angle(hand_points)
                gesture_str = get_hand_gesture(angle_list)

                pointing, vec, ang = detect_pointing(angle_list, hand_points)
                pointing_active = pointing
                pointing_vector = vec
                pointing_angle = ang

            frame_buffer.append(hand_points.copy())
            if len(frame_buffer) == 3:
                wave_dir = handwave_recognize(frame_buffer[0], frame_buffer[2], w, min_fingers=2)
                if wave_dir:
                    now = time.time()
                    if (now - last_wave_time >= cooling_time) or (wave_dir != last_wave_dir):
                        print(f"挥手动作：{wave_dir}")
                        # ==========================================
                        # 【核心修改点】：触发回调函数通知外部
                        if on_wave_callback:
                            on_wave_callback(wave_dir)
                        # ==========================================
                        last_wave_time = now
                        last_wave_dir = wave_dir
        else:
            frame_buffer.clear()
            gesture_str = None
            pointing_active = False

        if previous_time_fps > 0:
            fps = 1.0 / (current_time - previous_time_fps)
        else:
            fps = 0
        previous_time_fps = current_time

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.flip(img_bgr, 1)

        if pointing_active and len(hand_points) >= 21:
            wrist_x, wrist_y = hand_points[0]
            tip_x, tip_y = hand_points[8]
            wrist_x_m = w - wrist_x
            tip_x_m = w - tip_x
            cv2.arrowedLine(img_bgr, (int(wrist_x_m), int(wrist_y)), (int(tip_x_m), int(tip_y)),
                            (0, 255, 255), 3, tipLength=0.3)
            vec_text = f"Pointing: ({pointing_vector[0]:.2f}, {pointing_vector[1]:.2f})  angle={int(pointing_angle)} deg"
            cv2.putText(img_bgr, vec_text, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.putText(img_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        if gesture_str:
            cv2.putText(img_bgr, f"Static: {gesture_str}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        if last_wave_dir and (time.time() - last_wave_time < 0.5):
            cv2.putText(img_bgr, f"Wave: {last_wave_dir}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture & Pointing", img_bgr)
        if cv2.waitKey(2) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 为了保证该脚本独立运行时仍能工作
    start_gesture_recognition()