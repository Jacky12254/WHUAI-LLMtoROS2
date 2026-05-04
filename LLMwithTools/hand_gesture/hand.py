import cv2
import mediapipe as mp
import math
import time
 
 
def points_cos_angle(point1, point2):
    # 计算两个坐标点的余弦值
    try:
        angle_ = math.degrees(math.acos(
            (point1[0] * point2[0] + point1[1] * point2[1]) / (
                    ((point1[0] ** 2 + point1[1] ** 2) * (point2[0] ** 2 + point2[1] ** 2)) ** 0.5)))
        # math.acos返回一个数的反余弦值（单位为弧度）此处为向量a、b之积(x1*x2+y1*y2)除以向量a、b模的积；math.degrees将弧度值转换为角度
    except:
        angle_ = 65535.  # 将未检测到角度时(数据溢出)的情况排除,容错处理
    if angle_ > 180.:
        angle_ = 65535.
    return angle_
 
 
def get_fingers_angle(handPoints_list):
    # 利用mediapipe的手部关键点数组构建相关向量并传入函数中计算两个二维向量之间的角度,最后将结果置入列表中
    angle_list = []
    # ---------------------------- thumb 大拇指角度
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[2][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[2][1]))),
        ((int(handPoints_list[3][0]) - int(handPoints_list[4][0])),
         (int(handPoints_list[3][1]) - int(handPoints_list[4][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- index 食指角度
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[6][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[6][1]))),
        ((int(handPoints_list[7][0]) - int(handPoints_list[8][0])),
         (int(handPoints_list[7][1]) - int(handPoints_list[8][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- middle 中指角度
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[10][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[10][1]))),
        ((int(handPoints_list[11][0]) - int(handPoints_list[12][0])),
         (int(handPoints_list[11][1]) - int(handPoints_list[12][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- ring 无名指角度
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[14][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[14][1]))),
        ((int(handPoints_list[15][0]) - int(handPoints_list[16][0])),
         (int(handPoints_list[15][1]) - int(handPoints_list[16][1])))
    )
    angle_list.append(angle_)
    # ---------------------------- pink 小拇指角度
    angle_ = points_cos_angle(
        ((int(handPoints_list[0][0]) - int(handPoints_list[18][0])),
         (int(handPoints_list[0][1]) - int(handPoints_list[18][1]))),
        ((int(handPoints_list[19][0]) - int(handPoints_list[20][0])),
         (int(handPoints_list[19][1]) - int(handPoints_list[20][1])))
    )
    angle_list.append(angle_)
    return angle_list
 
 
def get_hand_gesture(fingers_angle_List):
    # 利用二维约束的方法定义手势
    thr_angle_others_bend = 60.  # 规定该角度为其余四个手指弯曲时的角度
    thr_angle_thumb_bend = 45.  # 规定该角度为拇指弯曲时的角度
    thr_angle_straight = 20.  # 规定该角度为手指伸直时的角度
    gesture_str = None
    if 65535. not in fingers_angle_List:
        if (fingers_angle_List[0] > thr_angle_thumb_bend):  # 拇指弯曲时
            if (fingers_angle_List[1] > thr_angle_others_bend) and (fingers_angle_List[2] > thr_angle_others_bend) and (
                    fingers_angle_List[3] > thr_angle_others_bend) and (fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "fist"  # 拳头(四指聚拢)
            elif (fingers_angle_List[1] < thr_angle_straight) and (fingers_angle_List[2] < thr_angle_straight) and (
                    fingers_angle_List[3] < thr_angle_straight) and (fingers_angle_List[4] < thr_angle_straight):
                gesture_str = "four"  # 四
            elif (fingers_angle_List[1] < thr_angle_straight) and (fingers_angle_List[2] < thr_angle_straight) and (
                    fingers_angle_List[3] < thr_angle_straight) and (fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "three"  # 三
            elif (fingers_angle_List[1] < thr_angle_straight) and (fingers_angle_List[2] < thr_angle_straight) and (
                    fingers_angle_List[3] > thr_angle_others_bend) and (fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "two"  # 二
            elif (fingers_angle_List[1] < thr_angle_straight) and (fingers_angle_List[2] > thr_angle_others_bend) and (
                    fingers_angle_List[3] > thr_angle_others_bend) and (fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "one"  # 一
        elif (fingers_angle_List[0] < thr_angle_straight):  # 拇指伸直时
            if (fingers_angle_List[1] < thr_angle_straight) and (fingers_angle_List[2] < thr_angle_straight) and (
                    fingers_angle_List[3] < thr_angle_straight) and (fingers_angle_List[4] < thr_angle_straight):
                gesture_str = "five"  # 五
            elif (fingers_angle_List[1] > thr_angle_others_bend) and (
                    fingers_angle_List[2] > thr_angle_others_bend) and (
                    fingers_angle_List[3] > thr_angle_others_bend) and (fingers_angle_List[4] > thr_angle_others_bend):
                gesture_str = "thumbUp"  # 点赞
    return gesture_str
 
 
def handwave_recognize(list1, list2):
    # 计算两组关键点的坐标差与规定间隔'ds'进行比对来区分手势
    ds = 0.2 * 640  # 规定坐标差比对距离,可根据工作位置进行调整,第一个乘数不建议改,第二个乘数为摄像头像素宽度
    x1_8, y1_8 = list1[8][0], list1[8][1]
    x1_12, y1_12 = list1[12][0], list1[12][1]
    x1_16, y1_16 = list1[16][0], list1[16][1]
    x1_20, y1_20 = list1[20][0], list1[20][1]
    x2_8, y2_8 = list2[8][0], list2[8][1]
    x2_12, y2_12 = list2[12][0], list2[12][1]
    x2_16, y2_16 = list2[16][0], list2[16][1]
    x2_20, y2_20 = list2[20][0], list2[20][1]
    gesture_str = None
    if x2_8 - x1_8 > ds and x2_12 - x1_12 > ds and x2_16 - x1_16 > ds and x2_20 - x1_20 > ds:
        gesture_str = "right"  # 向右挥手
        return gesture_str
    elif x1_8 - x2_8 > ds and x1_12 - x2_12 > ds and x1_16 - x2_16 > ds and x1_20 - x2_20 > ds:
        gesture_str = "left"  # 向左挥手
        return gesture_str
    elif y1_8 - y2_8 > ds and y1_12 - y2_12 > ds and y1_16 - y2_16 > ds and y1_20 - y2_20 > ds:
        gesture_str = "up"  # 向上挥手
        return gesture_str
    elif y2_8 - y1_8 > ds and y2_12 - y1_12 > ds and y2_16 - y1_16 > ds and y2_20 - y1_20 > ds:
        gesture_str = "down"  # 向下挥手
        return gesture_str
    else:
        return gesture_str
 
 
def main():
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    holistic = mp_holistic.Holistic(model_complexity=0)  # 设置模型复杂度为最小,减小机器性能消耗
    cooling_time = 1
    previous_time_fps = 0
    previous_time_cooling = 0
    gesture_str = None
    while True:
        success, img = cap.read()
        if(success==False):print("false")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为rgb,提高识别性能
        h, w, c = img.shape  # 摄像头所拍摄图像的高,宽,通道
        results = holistic.process(img)
        hand_clmList = []
        current_time = time.time()  # 获取当前时间
        if results.right_hand_landmarks:
            for id, lm in enumerate(results.right_hand_landmarks.landmark):  # 获取手指关节点
                cx, cy = lm.x * w, lm.y * h  # 计算关键点在图像上对应的像素坐标
                hand_clmList.append((cx, cy))
                if id == 8:
                    cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), 20)  # 标记食指位置
            angle_list = get_fingers_angle(hand_clmList)  # 获取手指弯曲角度
            
            #gesture_str = get_hand_gesture(angle_list)  # 获取当前手势
            gesture_str=handwave_recognize()
            if current_time - previous_time_cooling >= cooling_time:  # 冷却机制
                if gesture_str == "fist" : 
                    print("hello!!")  # 以fist手势为例，如果为fist手势时执行某个任务，这里用print代指该任务
                    previous_time_cooling = current_time
 
        fps = 1 / (current_time - previous_time_fps)
        previous_time_fps = current_time
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 1)  # 转换回BGR进行显示并镜像翻转
        cv2.putText(
            img, f"{int(fps)} FPS", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2,
        )  # 显示fps
        if gesture_str is not None:
            cv2.putText(
                img, gesture_str, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2,
            )  # 显示手势
        cv2.imshow("image", img)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    cap.release()
 
 
if __name__ == '__main__':
    main()