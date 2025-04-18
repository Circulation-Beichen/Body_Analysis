import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import ttk # Optional: For themed widgets
import pose_analysis # Import the new analysis module
import visualization_3d # Import the new 3D visualization module
import threading # 新增：导入 threading 模块
import json # 新增：导入 json 库
import subprocess # 新增：导入 subprocess 库

# --- 全局配置和常量 ---
# 摄像头和模式设置
CAMERA_INDEX = 1
STITCHING_MODE = 'SxS'
# REQUESTED_WIDTH = 2560
# REQUESTED_HEIGHT = 720
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 540
# 3D 重建假设参数
BASELINE_MM = 25.0
#CALIBRATION_FILE = 'calibration_approx.npz'
CALIBRATION_FILE = 'stereo_calibration.npz'
# 视频文件处理
VIDEO_FILE_PATH = 'example_hp2.mp4' # 注意：使用 .mp4 扩展名
# Mediapipe 设置
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
VISIBILITY_THRESHOLD = 0.1

# --- 新增: EMA 平滑参数 和 全局状态 ---
ALPHA = 0.3 # 平滑因子
smoothed_mp_world_landmarks = {} # 平滑后的点
pause_event = threading.Event() # 新增：全局暂停事件
vis_thread = None # 新增：全局可视化线程引用

# --- 新增：全局变量，用于数据记录 ---
recorded_pose_data = [] # 存储所有帧的姿态数据

# --- 全局变量 (在主程序块中初始化) ---
mp_pose = None
mp_drawing = None
pose = None
K1, dist1, K2, dist2 = None, None, None, None
P1, P2 = None, None
frame_width, frame_height = 0, 0 # 单目尺寸
combined_width, combined_height = 0, 0 # 合并尺寸

# --- 核心处理函数 ---
def process_frame(frame_left, frame_right, K1, dist1, K2, dist2, P1, P2, pose):
    """处理左右帧，进行姿态估计，以MediaPipe世界坐标为主，三角测量Z值为辅。"""
    # print("--- 开始处理帧 ---")
    stereo_z_values_mm = {} # 存储当前帧通过三角测量计算出的有效 Z 值 (mm)
    mp_world_points = {} # 存储当前帧有效的 MediaPipe 世界坐标点 (m)
    vis_frame_left = frame_left.copy()
    vis_frame_right = frame_right.copy()

    # --- Mediapipe 处理 --- (获取 2D 和 World Landmarks)
    image_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    image_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    image_left_rgb.flags.writeable = False
    image_right_rgb.flags.writeable = False
    results_left = pose.process(image_left_rgb)
    results_right = pose.process(image_right_rgb)

    # --- 提取 主要3D数据: MediaPipe World Landmarks (来自左视图结果) ---
    if results_left.pose_world_landmarks:
        landmarks = results_left.pose_world_landmarks.landmark
        num_landmarks = len(landmarks)
        print(f"  [Mediapipe World] 检测到 {num_landmarks} 个世界关键点")
        for i in range(num_landmarks):
            # 理论上 world landmark 没有 visibility，但 2D landmark 有，可以用来筛选
            if results_left.pose_landmarks and i < len(results_left.pose_landmarks.landmark) and results_left.pose_landmarks.landmark[i].visibility > VISIBILITY_THRESHOLD:
                lm = landmarks[i]
                mp_world_points[mp_pose.PoseLandmark(i)] = np.array([lm.x, lm.y, lm.z])
        print(f"    - 存储了 {len(mp_world_points)} 个可见的世界关键点")
    # else:
        # print("  [Mediapipe World] 左视图未检测到世界关键点")

    # --- 计算 辅助数据: Stereo Z (mm) ---
    # 这部分代码与之前类似，但只为了获取 Z 值
    if results_left.pose_landmarks and results_right.pose_landmarks:
        landmarks_left = results_left.pose_landmarks.landmark
        landmarks_right = results_right.pose_landmarks.landmark
        num_landmarks_stereo = min(len(landmarks_left), len(landmarks_right))

        points_2d_left_raw = []
        points_2d_right_raw = []
        landmark_indices_stereo = []

        for i in range(num_landmarks_stereo):
            lm_l = landmarks_left[i]
            lm_r = landmarks_right[i]
            if lm_l.visibility > VISIBILITY_THRESHOLD and lm_r.visibility > VISIBILITY_THRESHOLD:
                x_l, y_l = int(lm_l.x * frame_width), int(lm_l.y * frame_height)
                x_r, y_r = int(lm_r.x * frame_width), int(lm_r.y * frame_height)
                # (边界检查可以省略，因为我们只关心Z值)
                points_2d_left_raw.append([x_l, y_l])
                points_2d_right_raw.append([x_r, y_r])
                landmark_indices_stereo.append(i)

        if landmark_indices_stereo: # 只有同时可见的点才继续
            points_2d_left_undistorted = []
            points_2d_right_undistorted = []
            np_points_2d_left_raw = np.array(points_2d_left_raw, dtype=np.float32).reshape(-1, 1, 2)
            np_points_2d_right_raw = np.array(points_2d_right_raw, dtype=np.float32).reshape(-1, 1, 2)
            try:
                undistorted_left_raw = cv2.undistortPoints(np_points_2d_left_raw, K1, dist1, None, K1)
                undistorted_right_raw = cv2.undistortPoints(np_points_2d_right_raw, K2, dist2, None, K2)
                points_2d_left_undistorted = undistorted_left_raw.reshape(-1, 2)
                points_2d_right_undistorted = undistorted_right_raw.reshape(-1, 2)
            except Exception as e:
                 print(f"错误: [Stereo Z] 去畸变点时出错: {e}")
                 landmark_indices_stereo = []

            if landmark_indices_stereo and len(points_2d_left_undistorted) == len(landmark_indices_stereo):
                try:
                    points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_left_undistorted.T, points_2d_right_undistorted.T)
                    valid_w_indices = np.where(np.abs(points_4d_hom[3]) > 1e-6)[0]
                    if len(valid_w_indices) > 0:
                        points_4d_hom_valid = points_4d_hom[:, valid_w_indices]
                        points_3d_stereo = points_4d_hom_valid[:3] / points_4d_hom_valid[3]
                        points_3d_stereo = points_3d_stereo.T
                        landmark_indices_valid_stereo = [landmark_indices_stereo[i] for i in valid_w_indices]
                        count_valid_stereo_z = 0
                        for idx, point_3d in zip(landmark_indices_valid_stereo, points_3d_stereo):
                            if point_3d[2] > 0 and point_3d[2] < 10000: # 检查深度有效性
                                stereo_z_values_mm[mp_pose.PoseLandmark(idx)] = point_3d[2]
                                count_valid_stereo_z += 1
                        # print(f"  [Stereo Z] 计算并存储了 {count_valid_stereo_z} 个有效 Stereo Z 值 (mm)")
                except Exception as e:
                    print(f"    - [Stereo Z] 三角测量错误: {e}")

    # --- EMA 平滑处理 (作用于 MediaPipe World Landmarks) ---
    global smoothed_mp_world_landmarks # 使用全局变量存储平滑状态
    current_smoothed_mp_world_points = {} # 存储当前帧平滑后的结果

    if mp_world_points: # 仅当当前帧有世界坐标点时才进行平滑
        for landmark_enum, current_point in mp_world_points.items():
            if landmark_enum in smoothed_mp_world_landmarks:
                previous_point = smoothed_mp_world_landmarks[landmark_enum]
                smoothed_point = ALPHA * current_point + (1 - ALPHA) * previous_point
            else:
                smoothed_point = current_point # 初始平滑值
            current_smoothed_mp_world_points[landmark_enum] = smoothed_point

        # 更新全局平滑状态
        smoothed_mp_world_landmarks.clear()
        smoothed_mp_world_landmarks.update(current_smoothed_mp_world_points)
    # else:
        # 如果当前帧没有世界坐标点，可以保留上一帧的平滑结果或清空
        # pass # 保留旧状态可能导致"冻结"，暂时先这样

    # --- 可视化 --- (结合 2D, Smoothed World 3D, Stereo Z)
    # 绘制 2D 检测结果 (与之前相同)
    if results_left.pose_landmarks:
        mp_drawing.draw_landmarks(vis_frame_left, results_left.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    if results_right.pose_landmarks:
         mp_drawing.draw_landmarks(vis_frame_right, results_right.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2))

    # 绘制辅助 Stereo Z 值 (mm)
    if stereo_z_values_mm and results_left.pose_landmarks:
        num_landmarks_to_draw = min(len(results_left.pose_landmarks.landmark), len(mp_pose.PoseLandmark))
        for i in range(num_landmarks_to_draw):
            landmark_enum = mp_pose.PoseLandmark(i)
            # 仅当该点在本帧成功计算出 Stereo Z 值时才绘制
            if landmark_enum in stereo_z_values_mm:
                lm_2d = results_left.pose_landmarks.landmark[i]
                if lm_2d.visibility > VISIBILITY_THRESHOLD:
                    x_pixel = int(lm_2d.x * frame_width)
                    y_pixel = int(lm_2d.y * frame_height)
                    # 获取 Stereo Z 值
                    stereo_z = stereo_z_values_mm[landmark_enum]
                    coord_text = f"Z:{stereo_z:.0f}mm"
                    cv2.putText(vis_frame_left, coord_text, (x_pixel + 5, y_pixel - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA) # 用橙色区分

    # --- Perform Pose Analysis and Draw Results (使用平滑后的 MediaPipe World Landmarks) ---
    analysis_angles = {}
    # 使用平滑后的世界坐标点进行姿态分析
    analysis_points = current_smoothed_mp_world_points if current_smoothed_mp_world_points else {}
    if analysis_points:
        analysis_angles, analysis_classification = pose_analysis.analyze_pose(analysis_points)
        vis_frame_left = pose_analysis.draw_pose_analysis(vis_frame_left, analysis_angles, analysis_classification)
    else:
        analysis_classification = "No 3D Points" # 如果没有世界坐标点
        vis_frame_left = pose_analysis.draw_pose_analysis(vis_frame_left, {}, analysis_classification)

    # --- 重新组合帧 ---
    output_frame = None
    try:
        if STITCHING_MODE == 'SxS':
            output_frame = np.hstack((vis_frame_left, vis_frame_right))
        elif STITCHING_MODE == 'TxB':
            output_frame = np.vstack((vis_frame_left, vis_frame_right))
    except Exception as e:
        print(f"错误: 拼接帧时出错: {e}")
        output_frame = None

    # 返回: OpenCV 显示帧, 平滑后的世界坐标点 (用于3D可视化), 辅助 Stereo Z 值 (虽然目前没用到)
    return output_frame, current_smoothed_mp_world_points, stereo_z_values_mm

# --- 分析函数 --- (修改暂停逻辑和线程启动)
def run_realtime_analysis():
    """从摄像头进行实时分析"""
    global frame_width, frame_height, combined_width, combined_height # 允许修改全局尺寸变量
    global vis_thread # 引用全局线程变量
    smoothed_mp_world_landmarks = {} # 重置

    print("--- 开始实时分析 ---")
    # --- 视角处理 (实时模式: 加载或使用默认，不保存) ---
    print("[实时分析] 尝试加载视角...")
    loaded_angle = visualization_3d.load_view_angle()
    if loaded_angle:
        elev, azim = loaded_angle
        print(f"  - 成功加载视角: elev={elev}, azim={azim}")
    else:
        elev, azim = visualization_3d.DEFAULT_ELEV, visualization_3d.DEFAULT_AZIM
        print(f"  - 加载失败或文件无效，使用默认视角: elev={elev}, azim={azim}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 (索引 {CAMERA_INDEX})。")
        return

    # --- 设置视频格式为 MJPEG --- (重要：通常在设置分辨率之前)
    try:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            print(f"成功请求 MJPEG 格式。")
        else:
            print(f"警告：无法设置 MJPEG 格式 (可能不支持或已被覆盖)。")
    except Exception as e:
        print(f"警告: 设置 FOURCC 时出错: {e}")

    # 设置分辨率
    if 'REQUESTED_WIDTH' in globals() and 'REQUESTED_HEIGHT' in globals():
        print(f"尝试设置分辨率为 {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
        time.sleep(0.5)

    # 读取实际尺寸
    combined_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    combined_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头实际输出分辨率 (合并): {combined_width}x{combined_height}")
    
    #读取实际FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头实际FPS: {actual_fps}")

    # 计算单目尺寸和分割点
    split_point_h, split_point_v = 0, 0
    if STITCHING_MODE == 'SxS':
        frame_width = combined_width // 2
        frame_height = combined_height
        split_point_h = frame_width
        print(f"模式: SxS, 单目分辨率: {frame_width}x{frame_height}")
    elif STITCHING_MODE == 'TxB':
        frame_width = combined_width
        frame_height = combined_height // 2
        split_point_v = frame_height
        print(f"模式: TxB, 单目分辨率: {frame_width}x{frame_height}")
    else:
        print(f"错误: 无效的 STITCHING_MODE '{STITCHING_MODE}'")
        cap.release()
        return

    # FPS 计算器
    frame_count_display = 0
    start_time_display = time.time()
    fps_display = "Calculating..."

    # 初始化并启动 3D 可视化线程 (传入视角参数和暂停事件)
    print(f"[实时分析] 初始化 3D 可视化线程，传入视角: elev={elev}, azim={azim}")
    vis_thread = visualization_3d.VisualizationThread(
        init_elev=elev, init_azim=azim, pause_event=pause_event
    )
    vis_thread.start()

    is_paused = False # 新增：控制主循环暂停状态

    while True:
        # --- 键盘事件处理 --- (移到循环顶部，优先处理暂停/退出)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n收到退出指令 'q'。")
            # --- 新增：退出前检查是否暂停，若是则保存视角 ---
            print(f"  - 检查退出时状态：is_paused = {is_paused}") # 调试打印当前暂停状态
            if is_paused:
                print("[主循环] 退出时检测到已暂停，尝试保存当前视角...")
                if vis_thread and vis_thread.is_alive():
                    print("  - 可视化线程存在且活跃，调用 get_current_view_and_save()...") # 调试打印
                    saved_successfully = vis_thread.get_current_view_and_save()
                    if saved_successfully:
                        print("[主循环] 退出前视角已保存。")
                    else:
                        print("[主循环] 退出前视角保存失败或无法执行 (vis_thread.get_current_view_and_save 返回 False)。") # 调试打印
                else:
                    print("[主循环] 无法保存视角：可视化线程未运行或不存在。") # 调试打印
            else:
                print("  - 退出时未处于暂停状态，不尝试保存视角。") # 调试打印
            # --- 结束新增部分 ---
            break # 原有的退出循环
        elif key == ord('p'): # 切换暂停状态
            is_paused = not is_paused
            if is_paused:
                pause_event.set() # 通知可视化线程暂停
                print("\n[主循环] 已暂停。请在 3D 窗口调整视角，按 P 键恢复并保存视角，或按 Q 退出并保存视角。")
            else:
                if vis_thread and vis_thread.is_alive():
                    print("[主循环] 恢复中，尝试触发视角保存...") # 调试打印
                    saved_successfully = vis_thread.get_current_view_and_save() # 调用保存函数
                    if saved_successfully:
                         print("[主循环] 可视化线程报告视角已保存。") # 调试打印
                    else:
                         print("[主循环] 可视化线程报告视角保存失败或无法执行。") # 调试打印
                else:
                     print("[主循环] 无法保存视角：可视化线程未运行。") # 调试打印
                pause_event.clear() # 通知可视化线程恢复
                print("[主循环] 已恢复。")

        # --- 检查暂停状态 ---
        if is_paused:
            # 如果暂停，显示提示信息，但跳过大部分处理
            # 我们需要一个"背景板"来绘制暂停提示，如果 output_frame 可用就用它
            # 否则可能需要创建一个空的
            # （简化：如果 output_frame 正好是 None，这帧就不显示 PAUSED）
            if 'output_frame' in locals() and output_frame is not None:
                cv2.putText(output_frame, "PAUSED (Press P to Resume & Save View, Q to Quit & Save View)",
                            (output_frame.shape[1] // 2 - 350, output_frame.shape[0] - 30), # 调整文本位置和内容
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Real-time Pose Analysis", output_frame)
            time.sleep(0.05) # 避免 CPU 空转
            continue # 跳过本轮循环的处理
        # --- 暂停处理结束 ---

        ret, combined_frame = cap.read()
        if not ret:
            print("错误：无法读取摄像头帧。")
            break
        if combined_frame is None or combined_frame.shape[0] == 0 or combined_frame.shape[1] == 0:
            continue
        if combined_frame.shape[1] != combined_width or combined_frame.shape[0] != combined_height:
            continue # 尺寸不匹配，跳过

        frame_count_display += 1

        # 分割帧
        frame_left, frame_right = None, None
        try:
            if STITCHING_MODE == 'SxS':
                if split_point_h > 0 and split_point_h < combined_width:
                    frame_left = combined_frame[:, :split_point_h]
                    frame_right = combined_frame[:, split_point_h:]
            elif STITCHING_MODE == 'TxB':
                 if split_point_v > 0 and split_point_v < combined_height:
                    frame_left = combined_frame[:split_point_v, :]
                    frame_right = combined_frame[split_point_v:, :]

            if frame_left is None or frame_right is None or frame_left.shape[0] == 0 or frame_right.shape[0] == 0: continue
            if frame_left.shape[0] != frame_height or frame_left.shape[1] != frame_width or \
               frame_right.shape[0] != frame_height or frame_right.shape[1] != frame_width: continue
        except Exception as e:
            continue # 分割错误，跳过

        # 处理帧并获取 平滑世界点 和 辅助Z值
        output_frame, current_smoothed_points, _ = process_frame(frame_left, frame_right, K1, dist1, K2, dist2, P1, P2, pose)

        # 显示帧和 FPS
        if output_frame is not None:
            current_time_display = time.time()
            elapsed_time_display = current_time_display - start_time_display
            if elapsed_time_display >= 1.0:
                fps = frame_count_display / elapsed_time_display
                fps_display = f"{fps:.2f}"
                frame_count_display = 0
                start_time_display = current_time_display

            # --- Draw FPS on the combined frame (bottom left) ---
            # Ensure output_frame is valid before drawing FPS
            if output_frame is not None and output_frame.shape[0] > 0 and output_frame.shape[1] > 0:
                cv2.putText(output_frame, f"FPS: {fps_display}", (10, output_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Real-time Pose Analysis", output_frame) # Renamed window
            else:
                print("Skipping display: Invalid output frame.") # Debug message

        # 更新 3D 可视化线程的数据
        if vis_thread.is_alive():
            vis_thread.update_data(current_smoothed_points)

    # 停止可视化线程
    if vis_thread and vis_thread.is_alive():
        vis_thread.stop()
        vis_thread.join(timeout=1)

    cap.release()
    cv2.destroyAllWindows()
    print("实时分析结束。")

# --- 新增：数据后处理函数，进行筛选和插值 ---
def filter_and_interpolate_data(raw_data, min_landmarks=30, min_gap_to_remove=10):
    """
    对录制的原始姿态数据进行筛选和插值。

    Args:
        raw_data (list): 包含每帧 landmark 字典的列表。
                         每个字典的 key 是 landmark 枚举，value 是 numpy array。
        min_landmarks (int): 保留帧所需的最小关键点数。
        min_gap_to_remove (int): 连续劣质帧达到此长度时将其移除。

    Returns:
        list: 处理后的数据列表，每帧保证包含所有 33 个关键点。
    """
    print(f"--- 开始后处理数据：共 {len(raw_data)} 帧 ---")
    if not raw_data:
        return []

    processed_data = []
    frame_status = [] # 0: 劣质 (<min), 1: 部分 (>=min, <33), 2: 完整 (==33)
    landmark_keys = list(mp_pose.PoseLandmark) # 获取所有33个关键点的枚举列表

    # --- Pass 1: 初始状态标记 ---
    print("Pass 1: 标记帧状态...")
    valid_frame_indices = [] # 存储所有初始看起来可能有效的帧的索引
    for i, frame_dict in enumerate(raw_data):
        count = len(frame_dict)
        if count == 33:
            frame_status.append(2)
            processed_data.append(frame_dict.copy()) # 先复制一份完整数据
            valid_frame_indices.append(i)
        elif count >= min_landmarks:
            frame_status.append(1)
            processed_data.append(frame_dict.copy()) # 复制部分数据，待插值
            valid_frame_indices.append(i)
        else:
            frame_status.append(0)
            processed_data.append(None) # 劣质帧先置空
    print(f"  - 标记完成：完整帧 {frame_status.count(2)}，部分帧 {frame_status.count(1)}，劣质帧 {frame_status.count(0)}")

    # --- Pass 2: 移除长劣质帧间隙 ---
    print(f"Pass 2: 移除长度 >= {min_gap_to_remove} 的劣质帧间隙...")
    indices_to_remove = set()
    i = 0
    while i < len(frame_status):
        if frame_status[i] == 0:
            j = i
            while j < len(frame_status) and frame_status[j] == 0:
                j += 1
            gap_length = j - i
            if gap_length >= min_gap_to_remove:
                print(f"  - 发现长劣质间隙：索引 {i} 到 {j-1} (长度 {gap_length})，将被移除。")
                for k in range(i, j):
                    indices_to_remove.add(k)
            i = j # 跳过已检查的间隙
        else:
            i += 1

    # 执行移除 (通过构建新的列表)
    temp_processed_data = []
    temp_frame_status = []
    original_indices_map = [] # 记录新索引对应的原始索引
    current_original_index = 0
    for original_idx, data in enumerate(processed_data):
         if original_idx not in indices_to_remove:
             temp_processed_data.append(data)
             temp_frame_status.append(frame_status[original_idx])
             original_indices_map.append(original_idx) # 记录原始索引
         current_original_index += 1

    processed_data = temp_processed_data
    frame_status = temp_frame_status
    print(f"  - 移除长间隙后剩余 {len(processed_data)} 帧。")

    if not processed_data:
        print("警告：移除长间隙后没有剩余数据。")
        return []

    # --- Pass 3: 插值部分帧 ---
    print("Pass 3: 插值部分帧 (状态 1)...")
    interpolated_count = 0
    failed_interpolation_indices = set()

    # 需要再次迭代，因为移除操作改变了索引
    for i in range(len(processed_data)):
        if frame_status[i] == 1: # 需要插值的帧
            # 寻找最近的前后完整帧 (状态 2)
            prev_valid_idx = -1
            for k in range(i - 1, -1, -1):
                if frame_status[k] == 2: # 找到前面的完整帧
                    prev_valid_idx = k
                    break

            next_valid_idx = -1
            for k in range(i + 1, len(processed_data)):
                 if frame_status[k] == 2: # 找到后面的完整帧
                    next_valid_idx = k
                    break

            if prev_valid_idx != -1 and next_valid_idx != -1:
                # --- 执行插值 ---
                prev_frame_dict = processed_data[prev_valid_idx]
                next_frame_dict = processed_data[next_valid_idx]
                current_frame_dict = processed_data[i]

                # 计算插值比例 (使用调整后的索引)
                prev_original_idx = original_indices_map[prev_valid_idx]
                next_original_idx = original_indices_map[next_valid_idx]
                current_original_idx = original_indices_map[i]
                
                if next_original_idx <= prev_original_idx: # 避免除零或负数
                     print(f"  - 警告: 帧 {i} (原始 {current_original_idx}) 的前后有效帧索引无效 ({prev_original_idx}, {next_original_idx})，跳过插值。")
                     failed_interpolation_indices.add(i)
                     continue # 跳过这个插值

                ratio = (current_original_idx - prev_original_idx) / (next_original_idx - prev_original_idx)

                missing_landmarks = []
                for landmark_key in landmark_keys:
                    if landmark_key not in current_frame_dict:
                         missing_landmarks.append(landmark_key)
                         # 检查前后帧是否有该点（理论上状态2应该有）
                         if landmark_key not in prev_frame_dict or landmark_key not in next_frame_dict:
                              print(f"  - 警告: 帧 {i} (原始 {current_original_idx}) 缺失关键点 {landmark_key.name}，但其前后完整帧之一也缺失该点，无法插值！")
                              failed_interpolation_indices.add(i)
                              break # 中断当前帧的插值
                         
                         prev_coord = prev_frame_dict[landmark_key]
                         next_coord = next_frame_dict[landmark_key]
                         
                         # 线性插值
                         interpolated_coord = prev_coord + (next_coord - prev_coord) * ratio
                         current_frame_dict[landmark_key] = interpolated_coord
                
                if i not in failed_interpolation_indices:
                    if len(current_frame_dict) == 33:
                         # print(f"  - 成功插值帧 {i} (原始 {current_original_idx})，补全了 {len(missing_landmarks)} 个点。")
                         frame_status[i] = 2 # 标记为完整
                         interpolated_count += 1
                    else:
                         # 如果插值后仍然不完整（例如因为前后帧缺失数据），标记为失败
                         print(f"  - 警告: 帧 {i} (原始 {current_original_idx}) 插值后关键点数量仍不足 33 ({len(current_frame_dict)})，标记为失败。")
                         failed_interpolation_indices.add(i)

            else:
                 print(f"  - 警告: 帧 {i} (原始 {original_indices_map[i]}) 找不到足够的前后完整帧进行插值，标记为失败。")
                 failed_interpolation_indices.add(i)

    print(f"  - 插值完成：成功插值 {interpolated_count} 帧。")

    # --- Pass 4: 移除插值失败和剩余劣质帧 ---
    print("Pass 4: 移除插值失败和剩余的劣质帧...")
    final_data = []
    removed_count = 0
    for i in range(len(processed_data)):
        # 保留状态为 2 (原始完整或成功插值) 且未标记为插值失败的帧
        if frame_status[i] == 2 and i not in failed_interpolation_indices:
            # 确认一下最终数量
            if len(processed_data[i]) == 33:
                final_data.append(processed_data[i])
            else:
                 print(f"  - 内部错误：帧 {i} 状态为2但关键点数不为33 ({len(processed_data[i])})，已丢弃。")
                 removed_count += 1
        else:
            removed_count += 1

    print(f"  - 移除完成：最终保留 {len(final_data)} 帧，移除了 {removed_count} 帧。")
    print(f"--- 数据后处理完成 ---")
    return final_data

def run_video_file_analysis(video_path):
    """从视频文件进行分析，并在结束后导出数据。"""
    global frame_width, frame_height, combined_width, combined_height # 允许修改全局尺寸变量
    global vis_thread
    # global recorded_pose_data # 不再直接使用全局变量存储最终结果
    # recorded_pose_data = [] # 每次运行时清空旧数据
    raw_recorded_data = [] # 新增：用于存储原始的、未经过滤的数据

    print(f"--- 开始处理视频文件: {video_path} ---")
    # --- 视角处理 (视频模式: 加载，若无则使用默认并保存) ---
    loaded_angle = visualization_3d.load_view_angle()
    if loaded_angle:
        elev, azim = loaded_angle
    else:
        elev, azim = visualization_3d.DEFAULT_ELEV, visualization_3d.DEFAULT_AZIM
        print("未找到视角文件或文件无效，使用默认视角并保存。")
        visualization_3d.save_view_angle(elev, azim) # 保存默认值

    if not os.path.exists(video_path):
        print(f"错误: 视频文件 '{video_path}' 不存在。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'。")
        return

    # 读取视频尺寸
    combined_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    combined_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) # 用于控制播放速度 (waitKey 延时)
    print(f"视频分辨率 (合并): {combined_width}x{combined_height}, FPS: {video_fps:.2f}")

    wait_time = int(1000 / video_fps) if video_fps > 0 else 1 # waitKey 延时

    # 计算单目尺寸和分割点
    split_point_h, split_point_v = 0, 0
    if STITCHING_MODE == 'SxS':
        frame_width = combined_width // 2
        frame_height = combined_height
        split_point_h = frame_width
        print(f"模式: SxS, 单目分辨率: {frame_width}x{frame_height}")
    elif STITCHING_MODE == 'TxB':
        frame_width = combined_width
        frame_height = combined_height // 2
        split_point_v = frame_height
        print(f"模式: TxB, 单目分辨率: {frame_width}x{frame_height}")
    else:
        print(f"错误: 无效的 STITCHING_MODE '{STITCHING_MODE}'")
        cap.release()
        return

    # 初始化并启动 3D 可视化线程 (传入视角参数和暂停事件)
    vis_thread = visualization_3d.VisualizationThread(
        init_elev=elev, init_azim=azim, pause_event=pause_event
    )
    vis_thread.start()

    is_paused = False
    output_frame = None

    while cap.isOpened():
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("\n收到退出指令。")
            # --- 新增：退出前检查是否暂停，若是则保存视角 ---
            if is_paused:
                print("[主循环] 退出时检测到已暂停，尝试保存当前视角...")
                if vis_thread and vis_thread.is_alive():
                    saved_successfully = vis_thread.get_current_view_and_save()
                    if saved_successfully:
                        print("[主循环] 退出前视角已保存。")
                    else:
                        print("[主循环] 退出前视角保存失败或无法执行。")
                else:
                    print("[主循环] 无法保存视角：可视化线程未运行。")
            # --- 结束新增部分 ---
            break # 原有的退出循环
        elif key == ord('p'):
            is_paused = not is_paused
            if is_paused:
                pause_event.set()
                print("\n[主循环] 已暂停。请在 3D 窗口调整视角，按 P 键恢复并保存视角。")
            else: # 恢复运行
                if vis_thread and vis_thread.is_alive():
                    print("[主循环] 恢复中，尝试触发视角保存...") # 调试打印
                    saved_successfully = vis_thread.get_current_view_and_save()
                    if saved_successfully:
                         print("[主循环] 可视化线程报告视角已保存。") # 调试打印
                    else:
                         print("[主循环] 可视化线程报告视角保存失败或无法执行。") # 调试打印
                else:
                    print("[主循环] 无法保存视角：可视化线程未运行。") # 调试打印
                pause_event.clear()
                print("[主循环] 已恢复。")

        if is_paused:
            # 暂停时仍需显示最后一帧并绘制提示
            if output_frame is not None:
                 cv2.putText(output_frame, "PAUSED (Press P to Resume & Save View)",
                             (output_frame.shape[1] // 2 - 250, output_frame.shape[0] - 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 cv2.imshow("Real-time Pose Analysis", output_frame)
            continue # 跳过后续处理

        ret, combined_frame = cap.read()
        if not ret:
            print("视频播放完毕或读取错误。")
            break
        if combined_frame is None or combined_frame.shape[0] == 0 or combined_frame.shape[1] == 0:
            continue

        # 分割帧 (与实时逻辑相同)
        frame_left, frame_right = None, None
        try:
            if STITCHING_MODE == 'SxS':
                if split_point_h > 0 and split_point_h < combined_width:
                    frame_left = combined_frame[:, :split_point_h]
                    frame_right = combined_frame[:, split_point_h:]
            elif STITCHING_MODE == 'TxB':
                 if split_point_v > 0 and split_point_v < combined_height:
                    frame_left = combined_frame[:split_point_v, :]
                    frame_right = combined_frame[split_point_v:, :]

            if frame_left is None or frame_right is None or frame_left.shape[0] == 0 or frame_right.shape[0] == 0: continue
            if frame_left.shape[0] != frame_height or frame_left.shape[1] != frame_width or \
               frame_right.shape[0] != frame_height or frame_right.shape[1] != frame_width: continue
        except Exception as e:
            continue

        # 处理帧并获取 平滑世界点 和 辅助Z值
        output_frame, current_smoothed_points, _ = process_frame(frame_left, frame_right, K1, dist1, K2, dist2, P1, P2, pose)

        # --- 修改：记录原始平滑数据 (字典 key 是 Landmark 枚举) ---
        # 不再进行if判断，记录所有非空结果，后续处理
        if current_smoothed_points:
            raw_recorded_data.append(current_smoothed_points.copy()) # 存储原始字典
        else:
            raw_recorded_data.append({}) # 如果没有检测到，存一个空字典
        # -------------------------------------------------------

        # 显示帧
        if output_frame is not None:
            cv2.imshow("Real-time Pose Analysis", output_frame)

        # 更新 3D 可视化线程的数据 (仍然使用当前帧数据)
        if vis_thread.is_alive():
            vis_thread.update_data(current_smoothed_points if current_smoothed_points else {}) # 传空字典如果无数据

    # 停止可视化线程
    if vis_thread and vis_thread.is_alive():
        vis_thread.stop()
        vis_thread.join(timeout=1)

    cap.release()
    cv2.destroyAllWindows()

    # --- 新增：调用后处理函数 ---
    print("--- 视频处理循环结束，开始数据后处理 ---")
    final_pose_data_for_export = filter_and_interpolate_data(raw_recorded_data)

    # --- 修改：导出处理后的数据，并转换格式 ---
    if final_pose_data_for_export:
        # 转换数据格式：从 {Landmark: array} 转换到 {landmark_name: list}
        exportable_data = []
        for frame_dict in final_pose_data_for_export:
            frame_data_exportable = {}
            for landmark_enum, point_array in frame_dict.items():
                 # 确保 landmark_enum 是 PoseLandmark 类型
                 if isinstance(landmark_enum, mp_pose.PoseLandmark):
                     frame_data_exportable[landmark_enum.name] = point_array.tolist()
                 else:
                      print(f"警告：在导出转换时遇到无效的 key 类型: {type(landmark_enum)}，跳过。")
            if len(frame_data_exportable) == 33: # 最后确认一次
                 exportable_data.append(frame_data_exportable)
            else:
                 print(f"警告：后处理后的帧数据在导出转换时关键点数不为 33 ({len(frame_data_exportable)})，已丢弃。")


        if exportable_data:
            export_filename = "pose_data.json"
            export_to_json(exportable_data, export_filename)
            # 尝试自动打开 Blender
            launch_blender_with_script(export_filename)
        else:
             print("错误：数据后处理和转换后没有可导出的有效数据。")

    else:
        print("数据后处理后没有录制到有效的姿态数据，跳过导出。")
    # -----------------------------

    print("视频文件处理结束。")

# --- 新增：导出到 JSON 的函数 ---
def export_to_json(all_frames_data, filename="pose_data.json"):
    """将记录的所有帧数据导出到 JSON 文件。"""
    print(f"准备导出 {len(all_frames_data)} 帧数据到 {filename}...")
    try:
        with open(filename, 'w') as f:
            json.dump(all_frames_data, f) # 可以不用 indent 节省空间
        print(f"姿态数据已成功导出到: {filename}")
    except Exception as e:
        print(f"导出到 JSON 时出错: {e}")

# --- 新增：尝试启动 Blender 的函数 ---
def launch_blender_with_script(json_file_path):
    """尝试找到 Blender 并使用导入脚本启动它。"""
    blender_executable_path = None
    # 尝试查找 Blender 的常见路径 (Windows 示例)
    possible_paths = [
        r"C:\Application_SourseFile\blender\blender-3.1.2-windows-x64\blender.exe", # 使用原始字符串处理反斜杠
    ]
    for path in possible_paths:
        if os.path.exists(path):
            blender_executable_path = path
            break

    if not blender_executable_path:
        print("警告: 未能在常见位置找到 Blender 可执行文件。请手动打开 Blender。")
        return

    # 确保 Blender 导入脚本存在于 Python 脚本旁边
    blender_script_path = os.path.join(os.path.dirname(__file__), "import_pose_json.py")
    if not os.path.exists(blender_script_path):
        print(f"错误: 找不到 Blender 导入脚本: {blender_script_path}")
        print("请确保 import_pose_json.py 文件与 body_analysis.py 在同一目录下。")
        return

    print(f"尝试使用脚本 '{blender_script_path}' 启动 Blender ({blender_executable_path})...")
    try:
        # 使用 Popen 启动 Blender，让 Python 脚本可以继续
        # 传递 JSON 文件路径作为脚本参数 (通过环境变量)
        env = os.environ.copy()
        env['POSE_JSON_FILE'] = os.path.abspath(json_file_path) # 传递绝对路径

        subprocess.Popen([blender_executable_path, "-P", blender_script_path], env=env)
        print("Blender 启动命令已发送。如果 Blender 未启动，请检查路径或手动打开。")
    except Exception as e:
        print(f"启动 Blender 时出错: {e}")

# --- GUI 函数 --- (移除暂停按钮)
def create_gui():
    """创建并运行 Tkinter GUI 选择界面"""
    # global vis_thread # 不再需要访问 vis_thread
    root = tk.Tk()
    root.title("选择分析模式")
    # root.geometry("450x150") # 恢复原始宽度或根据按钮调整
    root.geometry("350x150")

    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
    style.configure("TLabel", padding=6, font=('Helvetica', 11))

    label = ttk.Label(root, text="请选择要运行的分析模式:")
    label.pack(pady=10)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    btn_realtime = ttk.Button(button_frame, text="实时分析 (摄像头)", command=lambda: start_analysis(root, run_realtime_analysis))
    btn_realtime.pack(side=tk.LEFT, padx=10) # 调整回间距

    btn_video = ttk.Button(button_frame, text=f"处理文件 ({os.path.basename(VIDEO_FILE_PATH)})", command=lambda: start_analysis(root, run_video_file_analysis, VIDEO_FILE_PATH))
    btn_video.pack(side=tk.LEFT, padx=10) # 调整回间距

    # --- 移除：暂停/调整按钮及其逻辑 ---
    # btn_pause_text = tk.StringVar(value="暂停 & 调整视角")
    # is_paused = False
    # def toggle_pause_adjust(): ...
    # btn_pause_adjust = ttk.Button(...) ...

    # 辅助函数，用于启动分析并在之后运行 mainloop
    def start_analysis(root_window, analysis_func, *args):
        root_window.destroy() # 关闭选择窗口
        analysis_func(*args)  # 运行选择的分析函数

    root.mainloop()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 初始化 Mediapipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=MODEL_COMPLEXITY, smooth_landmarks=True,
        enable_segmentation=False, min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF)

    # 加载标定数据
    if os.path.exists(CALIBRATION_FILE):
        try:
            data = np.load(CALIBRATION_FILE)
            K1 = data['mtx_left']
            dist1 = data['dist_left']
            K2 = data['mtx_right']
            dist2 = data['dist_right']
            print(f"从 '{CALIBRATION_FILE}' 加载了标定参数。")
        except Exception as e:
            print(f"错误: 加载标定文件 '{CALIBRATION_FILE}' 时出错: {e}")
            exit()
    else:
        print(f"错误: 标定文件 '{CALIBRATION_FILE}' 不存在!")
        print("请先运行 calibrate_camera.py。")
        exit()

    if K1 is None or K2 is None or dist1 is None or dist2 is None:
        print("标定参数加载不完整，无法继续。")
        exit()

    # 计算假设的投影矩阵 P1, P2
    R_guess = np.eye(3, dtype=np.float32)
    T_guess = np.array([[BASELINE_MM], [0], [0]], dtype=np.float32)
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R_guess, T_guess))

    # 启动 GUI
    create_gui()

    # 清理 Mediapipe 资源
    if pose:
        pose.close()
    # 确保关闭 Matplotlib 窗口 (如果主循环未正常结束)
    # visualization_3d.close_plot() # 不再需要，线程会自行处理
    print("程序完全结束。")
