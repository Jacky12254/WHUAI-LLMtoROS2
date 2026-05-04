import numpy as np
from scipy.spatial.transform import Rotation as R

# 手眼标定参数（相机 → 末端）


#1号机械臂
rotation_matrix = np.array([
    [0.8531200610232105, -0.5199266080840086, 0.04315650224486889], 
    [0.5207851190970867, 0.8536171550091686, -0.01098236773589317], 
    [-0.031129105460757586, 0.031844542394432254, 0.999007959884656]]
)

translation_vector = np.array(
    [-0.001304723740993616, 
    -0.06523536037218251, 
    0.02073011752961186]
)

# 二号机械臂
# rotation_matrix = np.array([
#    [0.8471709906058602, -0.5313053444284251, 0.003992950998684923],
#    [0.5313203402680643, 0.8471483729808938, -0.006191136668881793],
#    [-9.323794166467302e-05, 0.007366487468047835, 0.9999728627163186],
# ])

# translation_vector = np.array([
#    -0.01065101332564819,
#    -0.0549892824033035,
#    0.020366931754922737,
# ])


def camera_to_base(obj_cam, end_effector_pose):
    """
    obj_cam: [x, y, z] 相机坐标系
    end_effector_pose: [x, y, z, rx, ry, rz] 机械臂末端位姿（base坐标系）
    """

    # 齐次：相机 -> 末端
    T_cam_to_ee = np.eye(4)
    T_cam_to_ee[:3, :3] = rotation_matrix
    T_cam_to_ee[:3, 3] = translation_vector

    # 末端 -> base
    pos = end_effector_pose[:3]
    rot = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()

    T_base_to_ee = np.eye(4)
    T_base_to_ee[:3, :3] = rot
    T_base_to_ee[:3, 3] = pos

    # 点
    obj_cam_h = np.append(obj_cam, 1)

    # 相机 → 末端 → base
    obj_ee = T_cam_to_ee @ obj_cam_h
    obj_base = T_base_to_ee @ obj_ee

    return obj_base[:3]
