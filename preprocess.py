import cv2
import numpy as np
import time

# --- 配置参数 ---
# !! 重要 !!: 修改这个索引以选择你的双目摄像头 (通常是 0, 1, 2...)
CAMERA_INDEX = 1

# !! 重要 !!: 确认你的摄像头是左右拼接还是上下拼接
# 'SxS' for Side-by-Side (左右拼接)
# 'TxB' for Top-and-Bottom (上下拼接)
STITCHING_MODE = 'SxS' # 或者 'TxB'

# (可选) 尝试设置期望的 *合并后* 分辨率
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 540
# REQUESTED_WIDTH = 2560
# REQUESTED_HEIGHT = 720

# --- 光流法参数 ---
# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 100,      # 最多检测角点数
                       qualityLevel = 0.3,   # 角点质量阈值 (0-1)
                       minDistance = 7,      # 角点间最小距离
                       blockSize = 7 )       # 计算协方差矩阵的邻域大小
# Lucas-Kanade 光流参数
lk_params = dict( winSize  = (15, 15),       # 跟踪窗口大小
                  maxLevel = 2,            # 最大金字塔层数
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # 停止迭代标准

# --- 初始化摄像头 ---
print(f"尝试打开摄像头 (索引 {CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# --- 检查摄像头是否成功打开 ---
if not cap.isOpened():
    print(f"错误：无法打开摄像头 (索引 {CAMERA_INDEX})。")
    print("请检查摄像头连接、驱动和索引号。")
    exit()
else:
    print(f"摄像头 {CAMERA_INDEX} 打开成功。")

# --- (可选) 尝试设置分辨率 ---
if 'REQUESTED_WIDTH' in locals() and 'REQUESTED_HEIGHT' in locals():
    print(f"尝试设置分辨率为 {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    time.sleep(0.5) # 给点时间让设置生效

# --- 读取实际分辨率并计算分割点 ---
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头实际输出分辨率: {actual_width}x{actual_height}")

# --- FPS 测量阶段 ---
print("开始测量实际 FPS (请稍候)...")
warmup_duration = 1.0 # 预热时间 (秒)
measurement_duration = 2.0 # 测量时间 (秒)
total_measurement_phase_duration = warmup_duration + measurement_duration

fps_measure_start_time = time.time()
fps_measure_frame_count = 0
measured_fps = 30.0 # 默认值

while True:
    current_time = time.time()
    elapsed_time = current_time - fps_measure_start_time

    # 检查是否超出总测量阶段时间
    if elapsed_time >= total_measurement_phase_duration:
        break

    ret, _ = cap.read() # 只读取帧，不处理
    if not ret:
        print("警告: 在 FPS 测量期间无法读取帧。")
        # 可以选择退出或继续使用默认 FPS
        break

    # 只在测量阶段计数 (跳过预热阶段)
    if elapsed_time >= warmup_duration:
        fps_measure_frame_count += 1

# 计算测量的 FPS
if measurement_duration > 0 and fps_measure_frame_count > 0:
    measured_fps = fps_measure_frame_count / measurement_duration
    print(f"测量完成。估算 FPS: {measured_fps:.2f}")
else:
    print(f"警告: 未能在测量期间有效计数帧，将使用默认 FPS: {measured_fps}")


# --- 视频输出设置 ---
output_filename = 'output_motion_estimation.mp4'
# # 尝试获取摄像头FPS，如果失败则使用默认值30 # 注释掉旧逻辑
# target_fps = cap.get(cv2.CAP_PROP_FPS)
# if target_fps <= 0:
#     print("警告: 无法获取摄像头FPS，将使用默认值 30 FPS 进行视频录制。")
#     target_fps = 30.0
# else:
#     print(f"检测到摄像头FPS: {target_fps:.2f} (将用于视频录制)")
target_fps = measured_fps # 使用测量得到的 FPS
# 使用 MP4V 编解码器 (常见的 .mp4 格式)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 输出帧尺寸与摄像头实际输出一致
frame_size = (actual_width, actual_height)
# 初始化 VideoWriter
video_writer = cv2.VideoWriter(output_filename, fourcc, target_fps, frame_size)
if not video_writer.isOpened():
    print(f"错误: 无法打开视频文件 '{output_filename}' 进行写入。请检查权限或编解码器支持。")
    # 可以在这里决定是否退出，或者只是禁用写入功能
    video_writer = None # 禁用写入
else:
    # print(f"将把带有运动估计的视频保存到: {output_filename}") # 旧信息
    print(f"将使用 {target_fps:.2f} FPS 保存视频到: {output_filename}") # 新信息

split_point_v = actual_height // 2 # 用于上下分割
split_point_h = actual_width // 2  # 用于左右分割

if STITCHING_MODE == 'SxS':
    print(f"模式: 左右拼接 (SxS), 在宽度 {split_point_h} 处分割。")
    if actual_width < actual_height * 1.5:
         print("警告：分辨率看起来不像典型的左右拼接模式。")
elif STITCHING_MODE == 'TxB':
    print(f"模式: 上下拼接 (TxB), 在高度 {split_point_v} 处分割。")
    if actual_height < actual_width * 1.5: # 反过来检查
         print("警告：分辨率看起来不像典型的上下拼接模式。")
else:
    print("错误：无效的 STITCHING_MODE。请设置为 'SxS' 或 'TxB'。")
    cap.release()
    exit()

# --- FPS 计算初始化 ---
frame_count = 0
start_time = time.time()
fps_display = "Calculating..."

# --- 运动估计初始化 ---
prev_gray_left = None
prev_points_left = None
prev_gray_right = None
prev_points_right = None
motion_mask_left = None # 用于绘制轨迹
motion_mask_right = None # 用于绘制轨迹

print("\n按 'q' 键退出...")

# --- 主循环 ---
while True:
    # 读取合并后的帧
    ret, combined_frame = cap.read()

    # 检查是否成功读取
    if not ret:
        print("错误：无法读取摄像头帧。")
        break

    frame_count += 1

    # 确保帧有效
    if combined_frame is None:
        print("错误：读取到的帧为空。")
        continue

    # --- 分割图像 ---
    frame_left = None
    frame_right = None # 或者 frame_top, frame_bottom

    try:
        if STITCHING_MODE == 'SxS':
            if combined_frame.shape[1] == actual_width:
                frame_left = combined_frame[:, :split_point_h]
                frame_right = combined_frame[:, split_point_h:]
            else:
                 print(f"错误：帧宽度 {combined_frame.shape[1]} 与预期 {actual_width} 不符，跳过分割。")
                 continue # 跳过这一帧的处理

        elif STITCHING_MODE == 'TxB':
             if combined_frame.shape[0] == actual_height:
                # 为了保持命名一致性，即使是上下，我们也叫 left/right 代表第一个/第二个视图
                frame_left = combined_frame[:split_point_v, :] # Top view
                frame_right = combined_frame[split_point_v:, :] # Bottom view
             else:
                 print(f"错误：帧高度 {combined_frame.shape[0]} 与预期 {actual_height} 不符，跳过分割。")
                 continue # 跳过这一帧的处理

    except Exception as e:
        print(f"分割图像时出错: {e}")
        continue # 跳过这一帧

    # --- 处理和可视化分割后的帧 ---
    if frame_left is not None and frame_right is not None:
        # 转换为灰度图
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # 初始化运动轨迹绘制掩码
        if motion_mask_left is None:
             motion_mask_left = np.zeros_like(frame_left)
        if motion_mask_right is None:
             motion_mask_right = np.zeros_like(frame_right)

        # --- 处理左眼/上眼图像 ---
        estimated_motion_left = (0, 0)
        if prev_gray_left is not None:
            # 如果没有特征点或者特征点太少，重新检测
            if prev_points_left is None or len(prev_points_left) < 5:
                prev_points_left = cv2.goodFeaturesToTrack(prev_gray_left, mask=None, **feature_params)
                # 如果还是没检测到，重置掩码并跳过光流计算
                if prev_points_left is None:
                    motion_mask_left = np.zeros_like(frame_left)


            # 只有在有足够特征点时才计算光流
            if prev_points_left is not None and len(prev_points_left) > 0:
                # 计算光流
                new_points_left, status_left, error_left = cv2.calcOpticalFlowPyrLK(
                    prev_gray_left, gray_left, prev_points_left, None, **lk_params)

                # 选择好的点 (status=1 的点)
                if new_points_left is not None:
                    good_new_left = new_points_left[status_left == 1]
                    good_old_left = prev_points_left[status_left == 1]

                    # --- 计算平均运动 ---
                    if len(good_new_left) > 0 and len(good_old_left) > 0:
                        dx_left = np.mean(good_new_left[:, 0] - good_old_left[:, 0])
                        dy_left = np.mean(good_new_left[:, 1] - good_old_left[:, 1])
                        estimated_motion_left = (dx_left, dy_left)

                        # --- 绘制轨迹 ---
                        for i, (new, old) in enumerate(zip(good_new_left, good_old_left)):
                            a, b = new.ravel().astype(int) # 使用 astype(int)
                            c, d = old.ravel().astype(int) # 使用 astype(int)
                            # 在掩码上绘制轨迹线
                            motion_mask_left = cv2.line(motion_mask_left, (a, b), (c, d), (0, 255, 0), 2)
                            # 在当前帧上绘制特征点
                            frame_left = cv2.circle(frame_left, (a, b), 5, (0, 0, 255), -1)

                    # 更新下一帧要跟踪的点
                    prev_points_left = good_new_left.reshape(-1, 1, 2)
                else:
                    # 如果光流输出为空，重置跟踪点
                    prev_points_left = None
                    motion_mask_left = np.zeros_like(frame_left) # 清空轨迹

            else: # 如果没有旧的点可跟踪 (可能是第一帧之后或者检测失败后)
                 prev_points_left = cv2.goodFeaturesToTrack(gray_left, mask=None, **feature_params)
                 motion_mask_left = np.zeros_like(frame_left) # 清空轨迹

        # 将轨迹绘制到当前帧上
        # img_left_with_motion = cv2.add(frame_left, motion_mask_left) # 注释掉叠加轨迹
        img_left_with_motion = frame_left # 直接使用原始分割帧
        # 显示估计的运动
        # cv2.putText(img_left_with_motion, f"Motion L(dx,dy): ({estimated_motion_left[0]:.1f}, {estimated_motion_left[1]:.1f})",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA) # 注释掉显示运动文本


        # --- 处理右眼/下眼图像 (逻辑与左眼类似) ---
        estimated_motion_right = (0, 0)
        if prev_gray_right is not None:
            if prev_points_right is None or len(prev_points_right) < 5:
                prev_points_right = cv2.goodFeaturesToTrack(prev_gray_right, mask=None, **feature_params)
                if prev_points_right is None:
                     motion_mask_right = np.zeros_like(frame_right)


            if prev_points_right is not None and len(prev_points_right) > 0:
                new_points_right, status_right, error_right = cv2.calcOpticalFlowPyrLK(
                    prev_gray_right, gray_right, prev_points_right, None, **lk_params)

                if new_points_right is not None:
                    good_new_right = new_points_right[status_right == 1]
                    good_old_right = prev_points_right[status_right == 1]

                    if len(good_new_right) > 0 and len(good_old_right) > 0:
                        dx_right = np.mean(good_new_right[:, 0] - good_old_right[:, 0])
                        dy_right = np.mean(good_new_right[:, 1] - good_old_right[:, 1])
                        estimated_motion_right = (dx_right, dy_right)

                        for i, (new, old) in enumerate(zip(good_new_right, good_old_right)):
                            a, b = new.ravel().astype(int)
                            c, d = old.ravel().astype(int)
                            motion_mask_right = cv2.line(motion_mask_right, (a, b), (c, d), (0, 255, 0), 2)
                            frame_right = cv2.circle(frame_right, (a, b), 5, (0, 0, 255), -1)

                    prev_points_right = good_new_right.reshape(-1, 1, 2)
                else:
                     prev_points_right = None
                     motion_mask_right = np.zeros_like(frame_right) # 清空轨迹

            else:
                 prev_points_right = cv2.goodFeaturesToTrack(gray_right, mask=None, **feature_params)
                 motion_mask_right = np.zeros_like(frame_right) # 清空轨迹


        # img_right_with_motion = cv2.add(frame_right, motion_mask_right) # 注释掉叠加轨迹
        img_right_with_motion = frame_right # 直接使用原始分割帧
        # cv2.putText(img_right_with_motion, f"Motion R(dx,dy): ({estimated_motion_right[0]:.1f}, {estimated_motion_right[1]:.1f})",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA) # 注释掉显示运动文本


        # --- 重新拼接带运动估计的帧 --- # 现在是不带运动估计的帧
        combined_frame_clean = None # 重命名变量以反映内容
        try:
            if STITCHING_MODE == 'SxS':
                # combined_frame_with_motion = np.hstack((img_left_with_motion, img_right_with_motion))
                combined_frame_clean = np.hstack((frame_left, frame_right)) # 使用未修改的帧
            elif STITCHING_MODE == 'TxB':
                # combined_frame_with_motion = np.vstack((img_left_with_motion, img_right_with_motion))
                combined_frame_clean = np.vstack((frame_left, frame_right)) # 使用未修改的帧

            # 确保拼接后的尺寸正确 (以防万一)
            # if combined_frame_with_motion is not None and combined_frame_with_motion.shape[1] == actual_width and combined_frame_with_motion.shape[0] == actual_height:
            if combined_frame_clean is not None and combined_frame_clean.shape[1] == actual_width and combined_frame_clean.shape[0] == actual_height:
                 # --- 可视化合并帧 (不带运动估计) --- # 修改窗口标题
                 # cv2.imshow(f'Combined Frame (Motion Est.) - Index {CAMERA_INDEX} - FPS: {fps_display}', combined_frame_with_motion)
                 cv2.imshow(f'Combined Frame (Clean) - Index {CAMERA_INDEX} - FPS: {fps_display}', combined_frame_clean)

                 # --- 写入视频帧 --- (写入干净的帧)
                 if video_writer is not None:
                    # video_writer.write(combined_frame_with_motion)
                    video_writer.write(combined_frame_clean)
            else:
                 # 如果拼接结果尺寸不对，显示原始帧并打印错误
                 print(f"错误: 拼接后的帧尺寸 {combined_frame_clean.shape[:2][::-1]} 与预期 {frame_size} 不符。")
                 cv2.imshow(f'Combined Frame (Raw - Stitching Size Error) - Index {CAMERA_INDEX} - FPS: {fps_display}', combined_frame_clean)

        except Exception as e:
            print(f"拼接或显示/写入运动估计帧时出错: {e}")
            # 出错时回退显示原始帧
            cv2.imshow(f'Combined Frame (Raw - Error) - Index {CAMERA_INDEX} - FPS: {fps_display}', combined_frame_clean)


        # --- 注释掉独立窗口显示 ---
        # cv2.imshow(f'Left Eye/Top Eye - Motion Est. - FPS: {fps_display}', img_left_with_motion)
        # cv2.imshow(f'Right Eye/Bottom Eye - Motion Est. - FPS: {fps_display}', img_right_with_motion)

        # 更新前一帧和前一点用于下一次迭代 (这部分仍然需要，用于光流计算本身)
        prev_gray_left = gray_left.copy()
        prev_gray_right = gray_right.copy()

    else:
        # 如果分割失败，只显示原始的合并帧
        # 修改窗口标题以区分状态
        cv2.imshow(f'Combined Frame (Raw - Splitting Error) - Index {CAMERA_INDEX} - FPS: {fps_display}', combined_frame)

        # --- 不再需要销毁独立窗口，因为它们不再被创建 ---
        # try:
        #     if cv2.getWindowProperty(f'Left Eye/Top Eye - Motion Est. - FPS: {fps_display}', cv2.WND_PROP_VISIBLE) >= 1:
        #          cv2.destroyWindow(f'Left Eye/Top Eye - Motion Est. - FPS: {fps_display}')
        #     if cv2.getWindowProperty(f'Right Eye/Bottom Eye - Motion Est. - FPS: {fps_display}', cv2.WND_PROP_VISIBLE) >= 1:
        #          cv2.destroyWindow(f'Right Eye/Bottom Eye - Motion Est. - FPS: {fps_display}')
        # except cv2.error:
        #     pass

        # 重置运动估计状态
        prev_gray_left = None
        prev_points_left = None
        prev_gray_right = None
        prev_points_right = None
        motion_mask_left = None
        motion_mask_right = None


    # --- 计算并更新 FPS ---
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= 1.0: # 每秒更新一次
        fps = frame_count / elapsed_time
        fps_display = f"{fps:.2f}"
        # print(f"Actual FPS: {fps_display}") # 减少终端输出

        # --- 更新窗口标题以显示 FPS (可选) ---
        try:
             # 更新所有可能存在的合并帧窗口的标题 (根据不同状态)
             # cv2.setWindowTitle(f'Combined Frame (Motion Est.) - Index {CAMERA_INDEX} - FPS: {fps_display}',
             #                   f'Combined Frame (Motion Est.) - Index {CAMERA_INDEX} - FPS: {fps_display}')
             cv2.setWindowTitle(f'Combined Frame (Clean) - Index {CAMERA_INDEX} - FPS: {fps_display}',
                                f'Combined Frame (Clean) - Index {CAMERA_INDEX} - FPS: {fps_display}')
             # ... (other setWindowTitle calls for error states remain similar, just update the base title if needed) ...
        except cv2.error:
             pass # 忽略窗口可能已关闭的错误

        # 重置计数器和计时器
        frame_count = 0
        start_time = current_time

    # --- 等待按键事件 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("收到退出指令。")
        break

# --- 清理资源 ---
print("正在释放摄像头资源...")
cap.release()
# --- 释放 VideoWriter ---
if video_writer is not None:
    print(f"正在关闭视频文件 '{output_filename}'...")
    video_writer.release()
print("正在关闭所有OpenCV窗口...")
cv2.destroyAllWindows()
print("程序结束。")