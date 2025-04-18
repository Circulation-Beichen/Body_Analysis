import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tkinter as tk
from tkinter import ttk # Optional: For themed widgets
import pose_analysis # Import the new analysis module

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
VIDEO_FILE_PATH = 'example_1.mp4' # 注意：使用 .mp4 扩展名
# Mediapipe 设置
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
VISIBILITY_THRESHOLD = 0.1

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
    """处理左右帧，进行姿态估计和3D三角测量（近似）"""
    # print("--- 开始处理帧 ---") # 添加帧处理开始标记
    points_3d_world = {}
    vis_frame_left = frame_left.copy()
    vis_frame_right = frame_right.copy()

    # --- Mediapipe 处理 ---
    image_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    image_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    image_left_rgb.flags.writeable = False
    image_right_rgb.flags.writeable = False
    results_left = pose.process(image_left_rgb)
    results_right = pose.process(image_right_rgb)

    # --- 提取 2D 点, 去畸变, 三角测量 ---
    points_2d_left_raw = []
    points_2d_right_raw = []
    landmark_indices = []

    num_landmarks_left = len(results_left.pose_landmarks.landmark) if results_left.pose_landmarks else 0
    num_landmarks_right = len(results_right.pose_landmarks.landmark) if results_right.pose_landmarks else 0
    print(f"  [Mediapipe] 检测到 L:{num_landmarks_left}, R:{num_landmarks_right} 关键点") # 打印检测数量 (中文)

    if results_left.pose_landmarks and results_right.pose_landmarks:
        landmarks_left = results_left.pose_landmarks.landmark
        landmarks_right = results_right.pose_landmarks.landmark
        num_landmarks = min(num_landmarks_left, num_landmarks_right)

        # 提取原始像素坐标并检查可见性
        visibility_threshold = VISIBILITY_THRESHOLD # 使用全局变量
        for i in range(num_landmarks):
            lm_l = landmarks_left[i]
            lm_r = landmarks_right[i]
            if lm_l.visibility > visibility_threshold and lm_r.visibility > visibility_threshold:
                x_l, y_l = int(lm_l.x * frame_width), int(lm_l.y * frame_height)
                x_r, y_r = int(lm_r.x * frame_width), int(lm_r.y * frame_height)
                x_l = max(0, min(x_l, frame_width - 1))
                y_l = max(0, min(y_l, frame_height - 1))
                x_r = max(0, min(x_r, frame_width - 1))
                x_r = max(0, min(x_r, frame_width - 1))
                points_2d_left_raw.append([x_l, y_l])
                points_2d_right_raw.append([x_r, y_r])
                landmark_indices.append(i)

        # print(f"  [可见性筛选] 找到 {len(landmark_indices)} 对满足阈值 {visibility_threshold} 的匹配点") # 打印可见点对数量 (中文)

        # --- 去畸变 ---
        points_2d_left_undistorted = []
        points_2d_right_undistorted = []
        if points_2d_left_raw and points_2d_right_raw:
            np_points_2d_left_raw = np.array(points_2d_left_raw, dtype=np.float32).reshape(-1, 1, 2)
            np_points_2d_right_raw = np.array(points_2d_right_raw, dtype=np.float32).reshape(-1, 1, 2)
            try:
                undistorted_left_raw = cv2.undistortPoints(np_points_2d_left_raw, K1, dist1, None, K1)
                undistorted_right_raw = cv2.undistortPoints(np_points_2d_right_raw, K2, dist2, None, K2)
                points_2d_left_undistorted = undistorted_left_raw.reshape(-1, 2)
                points_2d_right_undistorted = undistorted_right_raw.reshape(-1, 2)
                # print(f"  [去畸变] 完成 {len(points_2d_left_undistorted)} 对点的去畸变") # 打印去畸变后数量 (中文)
            except Exception as e:
                 print(f"错误: 去畸变点时出错: {e}")
                 landmark_indices = []
        # else:
             # print("  [去畸变] 没有点需要去畸变")

        # --- 三角测量 ---
        num_points_for_triangulation = len(landmark_indices)
        print(f"  [三角测量] 尝试对 {num_points_for_triangulation} 对点进行三角测量") # 打印尝试三角测量的数量 (中文)
        if num_points_for_triangulation > 0 and len(points_2d_left_undistorted) == num_points_for_triangulation and len(points_2d_right_undistorted) == num_points_for_triangulation:
            try:
                points_4d_hom = cv2.triangulatePoints(P1, P2, points_2d_left_undistorted.T, points_2d_right_undistorted.T)
                print(f"    - triangulatePoints 输出形状: {points_4d_hom.shape}") # 打印三角测量输出形状 (中文)
                valid_w_indices = np.where(np.abs(points_4d_hom[3]) > 1e-6)[0]
                print(f"    - 找到 {len(valid_w_indices)} 个具有有效W分量的点") # 打印 W 有效数量 (中文)
                if len(valid_w_indices) > 0:
                    points_4d_hom_valid = points_4d_hom[:, valid_w_indices]
                    points_3d_valid = points_4d_hom_valid[:3] / points_4d_hom_valid[3]
                    points_3d_valid = points_3d_valid.T
                    landmark_indices_valid = [landmark_indices[i] for i in valid_w_indices]
                    count_valid_depth = 0
                    for idx, point_3d in zip(landmark_indices_valid, points_3d_valid):
                        if point_3d[2] > 0 and point_3d[2] < 20000:
                            points_3d_world[mp_pose.PoseLandmark(idx)] = point_3d
                            count_valid_depth += 1
                            landmark_name = mp_pose.PoseLandmark(idx).name # 获取关键点名称
                            print(f"      * 存储点: {landmark_name}, Z = {point_3d[2]:.1f} mm") # 打印具体Z值
                    print(f"    - 存储了 {count_valid_depth} 个深度有效(0<Z<20000)的点") # 打印深度有效数量 (中文) - **启用**
            except Exception as e:
                print(f"    - 三角测量错误: {e}") # 打印三角测量错误 (中文)
                points_3d_world = {}
        # else:
            # print("  [三角测量] 输入数组不匹配或为空，跳过三角测量")

    # else: # 如果左右视图有一个未检测到
        # print("  [Mediapipe] 未在两个视图中都检测到关键点")

    # --- 可视化 ---
    if results_left.pose_landmarks:
        mp_drawing.draw_landmarks(vis_frame_left, results_left.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    if results_right.pose_landmarks:
         mp_drawing.draw_landmarks(vis_frame_right, results_right.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 128, 0), thickness=2, circle_radius=2))
    if points_3d_world and results_left.pose_landmarks:
        for idx, lm in enumerate(results_left.pose_landmarks.landmark):
            landmark_enum = mp_pose.PoseLandmark(idx)
            if landmark_enum in points_3d_world:
                 x_pixel = int(lm.x * frame_width)
                 y_pixel = int(lm.y * frame_height)
                 point_3d = points_3d_world[landmark_enum]
                 coord_text = f"Z:{point_3d[2]:.0f}mm"
                 cv2.putText(vis_frame_left, coord_text, (x_pixel + 5, y_pixel - 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    # --- Perform Pose Analysis and Draw Results ---
    analysis_angles = {}
    analysis_classification = "No 3D Points"
    if points_3d_world: # Only analyze if we have 3D points
        analysis_angles, analysis_classification = pose_analysis.analyze_pose(points_3d_world)
        vis_frame_left = pose_analysis.draw_pose_analysis(vis_frame_left, analysis_angles, analysis_classification)
    else:
        # Optionally draw a message if no points were found
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
        output_frame = None # 确保返回 None

    return output_frame

# --- 分析函数 ---
def run_realtime_analysis():
    """从摄像头进行实时分析"""
    global frame_width, frame_height, combined_width, combined_height # 允许修改全局尺寸变量

    print("--- 开始实时分析 ---")
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

    while True:
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

        # 处理帧
        output_frame = process_frame(frame_left, frame_right, K1, dist1, K2, dist2, P1, P2, pose)

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n收到退出指令。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("实时分析结束。")


def run_video_file_analysis(video_path):
    """从视频文件进行分析"""
    global frame_width, frame_height, combined_width, combined_height # 允许修改全局尺寸变量

    print(f"--- 开始处理视频文件: {video_path} ---")
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

    while cap.isOpened():
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

        # 处理帧
        output_frame = process_frame(frame_left, frame_right, K1, dist1, K2, dist2, P1, P2, pose)

        # 显示帧
        if output_frame is not None:
            # 视频文件处理不需要实时显示 FPS，但保持窗口名一致
            cv2.imshow("Real-time Pose Analysis", output_frame) # Renamed window

        if cv2.waitKey(wait_time) & 0xFF == ord('q'): # 使用 wait_time 控制播放速度
            print("\n收到退出指令。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("视频文件处理结束。")


# --- GUI 函数 ---
def create_gui():
    """创建并运行 Tkinter GUI 选择界面"""
    root = tk.Tk()
    root.title("选择分析模式")
    root.geometry("350x150") # 设置窗口大小

    # 设置样式 (可选)
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
    style.configure("TLabel", padding=6, font=('Helvetica', 11))

    label = ttk.Label(root, text="请选择要运行的分析模式:")
    label.pack(pady=10)

    def start_realtime():
        print("选择: 实时分析")
        root.destroy() # 关闭 GUI 窗口
        run_realtime_analysis()

    def start_video_file():
        print(f"选择: 处理视频文件 ({VIDEO_FILE_PATH})")
        root.destroy() # 关闭 GUI 窗口
        run_video_file_analysis(VIDEO_FILE_PATH)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    btn_realtime = ttk.Button(button_frame, text="实时分析 (摄像头)", command=start_realtime)
    btn_realtime.pack(side=tk.LEFT, padx=10)

    btn_video = ttk.Button(button_frame, text=f"处理文件 ({os.path.basename(VIDEO_FILE_PATH)})", command=start_video_file)
    btn_video.pack(side=tk.LEFT, padx=10)

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
    print("程序完全结束。")
