import numpy as np
import math
import mediapipe as mp
import cv2
# 导入 PIL 相关库
from PIL import ImageDraw, ImageFont, Image

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
    分析 3D 姿态关键点 (期望输入为 Mediapipe 世界坐标系，单位：米)，
    计算关节角度并对简单姿态进行分类。

    Args:
        points_3d_world (dict): 映射 PoseLandmark 枚举到 3D 世界坐标 (米) 的字典。

    Returns:
        tuple: 包含以下元素的元组:
            - dict: 计算出的角度字典 (例如: {'left_elbow': 160.5, ...})。
            - str: 描述分类姿态的字符串 (例如: "站直")。
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

    # 注意：当前的分类逻辑仅基于角度，不受坐标系尺度和原点影响。
    # 如果未来添加基于绝对位置或距离的判断，需要考虑坐标系(米，相对臀部中心)。
    return angles, pose_classification

def draw_pose_analysis(vis_frame, angles, classification):
    """将计算出的角度和分类信息绘制到给定的图像帧上 (支持中文)。"""
    y_offset = 30 # 初始 Y 坐标偏移量

    # --- 使用 PIL 绘制中文 --- (需要 Pillow 库: pip install Pillow)
    try:
        # 1. 选择一个支持中文的字体文件路径 (你需要根据你的系统修改此路径)
        # 常见的 Windows 字体路径:
        # font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
        # font_path = "C:/Windows/Fonts/msyh.ttc"   # 微软雅黑 (可能需要指定字体索引, 如 font = ImageFont.truetype(font_path, size, index=0))
        font_path = "C:/Windows/Fonts/simhei.ttf"  # 使用黑体作为示例

        # 2. 加载字体和大小
        font_size_cn = 20 # 中文字体大小
        font_cn = ImageFont.truetype(font_path, font_size_cn)

        # 3. 将 OpenCV 图像 (BGR) 转换为 PIL 图像 (RGB)
        #   必须先确保 vis_frame 不是 None 且是有效的图像
        if vis_frame is None or vis_frame.size == 0:
             print("错误: 输入到 draw_pose_analysis 的 vis_frame 无效")
             return vis_frame # 或者返回一个错误指示?

        img_pil = Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))

        # 4. 创建一个 Draw 对象
        draw = ImageDraw.Draw(img_pil)

        # 5. 绘制中文文本 (姿态分类)
        text_cn = f"姿态: {classification}"
        #   注意 PIL 的坐标原点和颜色格式 (RGBA 或 RGB)
        #   为了便于定位，文本的 Y 坐标通常是基线位置，稍微调整一下
        draw.text((10, y_offset - font_size_cn // 2), text_cn, font=font_cn, fill=(0, 255, 255, 255)) # 黄色 (RGBA)

        # 6. 将 PIL 图像转换回 OpenCV 图像 (BGR)
        vis_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        y_offset += 30 # 更新偏移量 (给中文留出比英文稍多的空间)

    except IOError:
        print(f"警告: 无法加载字体 {font_path}。将使用 OpenCV 默认字体绘制中文（可能显示为问号）。")
        # 如果加载字体失败，回退到 OpenCV 的 putText (会显示问号)
        cv2.putText(vis_frame, f"Pose: {classification} (Font Error)", (10, y_offset), # 添加提示
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
    except Exception as e:
         print(f"绘制中文时出错: {e}")
         # 保险起见，还是用 OpenCV 绘制一下，即使是问号
         cv2.putText(vis_frame, f"Pose: {classification} (Draw Error)", (10, y_offset),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
         y_offset += 25

    # --- 使用 OpenCV 绘制英文角度信息 (保持不变) ---
    font_size_en = 0.5
    for name, angle in angles.items():
        # 将变量名转换为更易读的显示名称 (例如 'left_elbow' -> 'Left Elbow')
        display_name = name.replace('_', ' ').title()
        # 绘制角度文本
        cv2.putText(vis_frame, f"{display_name}: {angle:.1f} deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size_en, (255, 255, 0), 1) # 青色
        y_offset += 20 # 增加 Y 坐标偏移
        if y_offset > vis_frame.shape[0] - 10: # 检查是否超出图像底部，防止绘制到屏幕外
            break # 如果超出则停止绘制更多角度

    return vis_frame # 返回绘制了信息的图像帧