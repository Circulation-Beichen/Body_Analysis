import numpy as np
import math
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """计算三个 3D 点之间的角度（以点 b 为顶点）。"""
    a = np.array(a) # 第一个点
    b = np.array(b) # 中间点 (顶点)
    c = np.array(c) # 结束点

    # 计算从中间点出发的向量
    ba = a - b
    bc = c - b

    # 计算点积
    dot = np.dot(ba, bc)

    # 计算向量模长
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # 避免除零错误
    if norm_ba == 0 or norm_bc == 0:
        return 0

    # 计算角度的余弦值
    cosine_angle = dot / (norm_ba * norm_bc)

    # 由于潜在的浮点精度问题，将值限制在 [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # 计算角度 (单位：度)
    angle = np.degrees(np.arccos(cosine_angle))

    return angle

def analyze_pose(points_3d_world):
    """
    分析 3D 姿态关键点，计算关节角度并对简单姿态进行分类。

    Args:
        points_3d_world (dict): 映射 PoseLandmark 枚举到 3D 坐标的字典。

    Returns:
        tuple: 包含以下元素的元组:
            - dict: 计算出的角度字典 (例如: {'left_elbow': 160.5, ...})。
            - str: 描述分类姿态的字符串 (例如: "Standing Straight")。
    """
    angles = {}
    pose_classification = "未知" # 姿态分类初始化为未知
    landmark_coords = points_3d_world # 直接使用输入的3D坐标字典

    # --- 计算角度 --- (检查关键点是否存在以避免错误)
    try:
        # 左臂肘部角度
        if all(lm in landmark_coords for lm in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]):
            angles['left_elbow'] = calculate_angle(landmark_coords[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                  landmark_coords[mp_pose.PoseLandmark.LEFT_ELBOW],
                                                  landmark_coords[mp_pose.PoseLandmark.LEFT_WRIST])
        # 右臂肘部角度
        if all(lm in landmark_coords for lm in [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]):
            angles['right_elbow'] = calculate_angle(landmark_coords[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                   landmark_coords[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                                   landmark_coords[mp_pose.PoseLandmark.RIGHT_WRIST])
        # 左腿膝盖角度
        if all(lm in landmark_coords for lm in [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]):
            angles['left_knee'] = calculate_angle(landmark_coords[mp_pose.PoseLandmark.LEFT_HIP],
                                                 landmark_coords[mp_pose.PoseLandmark.LEFT_KNEE],
                                                 landmark_coords[mp_pose.PoseLandmark.LEFT_ANKLE])
        # 右腿膝盖角度
        if all(lm in landmark_coords for lm in [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]):
             angles['right_knee'] = calculate_angle(landmark_coords[mp_pose.PoseLandmark.RIGHT_HIP],
                                                  landmark_coords[mp_pose.PoseLandmark.RIGHT_KNEE],
                                                  landmark_coords[mp_pose.PoseLandmark.RIGHT_ANKLE])
    except Exception as e:
         print(f"计算角度时出错: {e}") # 打印计算角度时的错误信息

    # --- 简单的姿态分类 --- (基于计算出的角度)
    left_knee_angle = angles.get('left_knee', 180) # 获取左膝角度，若不存在则默认为180度（伸直）
    right_knee_angle = angles.get('right_knee', 180) # 获取右膝角度，若不存在则默认为180度
    left_elbow_angle = angles.get('left_elbow', 180) # 获取左肘角度，若不存在则默认为180度
    right_elbow_angle = angles.get('right_elbow', 180) # 获取右肘角度，若不存在则默认为180度

    if not angles: # 如果没有任何角度被成功计算
        pose_classification = "数据不完整"
    elif left_knee_angle < 130 and right_knee_angle < 130: # 判断是否膝盖显著弯曲
        pose_classification = "下蹲 / 坐姿"
    elif (left_elbow_angle < 100 or right_elbow_angle < 100) and left_knee_angle > 150 and right_knee_angle > 150: # 判断是否手臂弯曲且腿伸直
        pose_classification = "手臂弯曲"
    elif left_knee_angle > 160 and right_knee_angle > 160 and left_elbow_angle > 160 and right_elbow_angle > 160: # 判断是否四肢大部分伸直
        pose_classification = "站直"
    else: # 其他情况
        pose_classification = "中间状态 / 其他"

    return angles, pose_classification

def draw_pose_analysis(vis_frame, angles, classification):
    """将计算出的角度和分类信息绘制到给定的图像帧上。"""
    y_offset = 30 # 初始 Y 坐标偏移量，避免遮挡其他信息 (如 FPS)
    # 显示姿态分类结果
    cv2.putText(vis_frame, f"姿态: {classification}", (10, y_offset), # 文本内容和位置
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # 字体、大小、颜色、粗细
    y_offset += 25 # 增加 Y 坐标偏移，为下一行文本留出空间

    # 显示计算出的角度
    for name, angle in angles.items():
        # 将变量名转换为更易读的显示名称 (例如 'left_elbow' -> 'Left Elbow')
        display_name = name.replace('_', ' ').title()
        # 绘制角度文本
        cv2.putText(vis_frame, f"{display_name}: {angle:.1f} deg", (10, y_offset), # 保留一位小数
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20 # 增加 Y 坐标偏移
        if y_offset > vis_frame.shape[0] - 10: # 检查是否超出图像底部，防止绘制到屏幕外
            break # 如果超出则停止绘制更多角度

    return vis_frame # 返回绘制了信息的图像帧