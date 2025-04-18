import cv2
import numpy as np
import os
import time
import glob

# --- 标定板参数 ---
# 棋盘格内部角点的数量 (width, height) - **必须与你的标定板完全一致**
pattern_size = (9, 6)
# 每个方格的物理尺寸 (毫米) - **必须与你打印并测量的尺寸完全一致**
square_size_mm = 22.3

# --- 摄像头和捕捉参数 ---
CAMERA_INDEX = 1
STITCHING_MODE = 'SxS' # 或者 'TxB'，根据你的摄像头设置
# 使用你进行分析时相同的分辨率和格式
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 540

MIN_IMAGES_NEEDED = 20 # 需要捕捉的最小有效图像数量
CAPTURE_DELAY_SEC = 0.5 # 捕捉一张图像后的短暂延迟，避免重复捕捉

# --- 输出文件 ---
OUTPUT_CALIBRATION_FILE = 'stereo_calibration.npz' # 保存双目标定结果

# --- 初始化摄像头 ---
print(f"尝试打开摄像头 (索引 {CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"错误：无法打开摄像头 (索引 {CAMERA_INDEX})。")
    exit()
else:
    print(f"摄像头 {CAMERA_INDEX} 打开成功。")

# --- 设置视频格式为 MJPEG ---
try:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"成功请求 MJPEG 格式。")
    else:
        print(f"警告：无法设置 MJPEG 格式。")
except Exception as e:
    print(f"警告: 设置 FOURCC 时出错: {e}")

# --- 设置分辨率 ---
print(f"尝试设置分辨率为 {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
time.sleep(0.5) # 等待设置生效

# --- 读取实际分辨率并计算单目尺寸 ---
combined_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
combined_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头实际输出分辨率 (合并): {combined_width}x{combined_height}")

# 计算单目尺寸和分割点
frame_width, frame_height, split_point_h, split_point_v = 0, 0, 0, 0
if STITCHING_MODE == 'SxS':
    frame_width = combined_width // 2; frame_height = combined_height; split_point_h = frame_width
elif STITCHING_MODE == 'TxB':
    frame_width = combined_width; frame_height = combined_height // 2; split_point_v = frame_height
else: print("无效模式"); cap.release(); exit()
print(f"模式: {STITCHING_MODE}, 单目分辨率: {frame_width}x{frame_height}")

# --- 准备标定数据 ---
# 3D 世界坐标点 ( Z=0 )
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp = objp * square_size_mm # 转换到毫米单位

# 存储所有图像的 3D 世界点和 2D 图像点
objpoints = [] # 存储世界坐标系中的点 (对于所有图像都相同)
imgpoints_left = [] # 存储左侧图像检测到的角点像素坐标
imgpoints_right = [] # 存储右侧图像检测到的角点像素坐标

# 优化 findChessboardCorners 的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("\n--- 开始双目标定图像捕捉 ---")
print(f"请将 {pattern_size[0]}x{pattern_size[1]} 的棋盘格在左右摄像头视野内移动。")
print(f"当左右两侧都检测到角点时，按【空格键】捕捉图像对。")
print(f"需要捕捉至少 {MIN_IMAGES_NEEDED} 张有效图像对。")
print("按 'q' 键退出。")

capture_message_timer = 0
last_capture_time = 0

while True:
    ret, combined_frame = cap.read()
    if not ret:
        print("错误：无法读取摄像头帧。")
        break
    if combined_frame is None or combined_frame.shape[1] != combined_width or combined_frame.shape[0] != combined_height:
        continue

    # --- 分割帧 ---
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
        if frame_left is None or frame_right is None or \
           frame_left.shape[0] != frame_height or frame_left.shape[1] != frame_width or \
           frame_right.shape[0] != frame_height or frame_right.shape[1] != frame_width:
            continue
    except Exception as e:
        print(f"帧分割错误: {e}")
        continue

    display_frame_left = frame_left.copy() # 用于显示的左帧
    display_frame_right = frame_right.copy() # 用于显示的右帧
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # --- 查找棋盘格角点 (左右分别查找) ---
    # 设置了 CALIB_CB_ADAPTIVE_THRESH 等标志以提高检测率
    ret_corners_left, corners_left = cv2.findChessboardCorners(
        gray_left, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    ret_corners_right, corners_right = cv2.findChessboardCorners(
        gray_right, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found_corners_both = False # 标记本帧是否左右都成功找到角点
    corners_left_subpix = None
    corners_right_subpix = None

    # 如果左右都找到角点
    if ret_corners_left and ret_corners_right:
        found_corners_both = True
        # 优化角点位置 (亚像素精度)
        corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        # --- 绘制角点 ---
        cv2.drawChessboardCorners(display_frame_left, pattern_size, corners_left_subpix, ret_corners_left)
        cv2.drawChessboardCorners(display_frame_right, pattern_size, corners_right_subpix, ret_corners_right)

        # --- 显示提示信息 ---
        cv2.putText(display_frame_left, "Found! Press SPACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame_right, "Found! Press SPACE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif ret_corners_left:
         cv2.drawChessboardCorners(display_frame_left, pattern_size, corners_left, ret_corners_left)
         cv2.putText(display_frame_left, "Found Left Only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif ret_corners_right:
         cv2.drawChessboardCorners(display_frame_right, pattern_size, corners_right, ret_corners_right)
         cv2.putText(display_frame_right, "Found Right Only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # --- 显示已捕捉数量 ---
    captured_count = len(objpoints) # 使用 objpoints 的长度，因为左右是同步添加的
    cv2.putText(display_frame_left, f"Captured: {captured_count}/{MIN_IMAGES_NEEDED}", (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame_right, f"Captured: {captured_count}/{MIN_IMAGES_NEEDED}", (10, frame_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- 显示短暂的 "Captured!" 消息 ---
    if capture_message_timer > 0:
        cv2.putText(display_frame_left, "Captured!", (frame_width // 2 - 50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(display_frame_right, "Captured!", (frame_width // 2 - 50, frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        capture_message_timer -= 1

    # --- 并排显示左右视图 ---
    display_combined = np.hstack((display_frame_left, display_frame_right))
    cv2.imshow('Stereo Calibration Capture - Press SPACE / q', display_combined)

    # --- 键盘事件 ---
    key = cv2.waitKey(1) & 0xFF

    # 按 'q' 退出
    if key == ord('q'):
        print("\n用户中断捕捉。")
        break

    # 按空格键捕捉 (只有在左右都找到角点且距离上次捕捉超过延迟时才生效)
    current_time = time.time()
    if found_corners_both and key == 32 and (current_time - last_capture_time > CAPTURE_DELAY_SEC):
        print(f"捕捉图像对 #{captured_count + 1}")
        objpoints.append(objp)
        imgpoints_left.append(corners_left_subpix)
        imgpoints_right.append(corners_right_subpix)
        capture_message_timer = 10 # 显示 "Captured!" 消息 10 帧
        last_capture_time = current_time

        # 检查是否已捕捉足够图像
        if len(objpoints) >= MIN_IMAGES_NEEDED:
            print(f"\n已成功捕捉 {len(objpoints)} 张图像对，达到目标数量。")
            break # 退出捕捉循环

# --- 清理摄像头和窗口 ---
print("\n正在释放摄像头资源...")
cap.release()
cv2.destroyAllWindows()

# --- 执行相机标定 ---
if len(objpoints) >= MIN_IMAGES_NEEDED:
    print("\n--- 开始进行左右摄像头标定计算 ---")

    # --- 标定左摄像头 ---
    print("标定左摄像头...")
    # 注意：传入的是 gray_left.shape[::-1] 即 (width, height)
    ret_calib_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None
    )
    if ret_calib_left:
        print("左摄像头标定成功！")
        print("相机内参矩阵 (mtx_left):")
        print(mtx_left)
        print("畸变系数 (dist_left):")
        print(dist_left)
        # 计算重投影误差
        mean_error_left = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left)
            error = cv2.norm(imgpoints_left[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error_left += error
        print(f"左摄像头平均重投影误差: {mean_error_left / len(objpoints)}")
    else:
        print("左摄像头标定计算失败。")
        mtx_left, dist_left = None, None # 标记失败

    # --- 标定右摄像头 ---
    print("\n标定右摄像头...")
    # 注意：传入的是 gray_right.shape[::-1] 即 (width, height)
    ret_calib_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None
    )
    if ret_calib_right:
        print("右摄像头标定成功！")
        print("相机内参矩阵 (mtx_right):")
        print(mtx_right)
        print("畸变系数 (dist_right):")
        print(dist_right)
        # 计算重投影误差
        mean_error_right = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right)
            error = cv2.norm(imgpoints_right[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error_right += error
        print(f"右摄像头平均重投影误差: {mean_error_right / len(objpoints)}")
    else:
        print("右摄像头标定计算失败。")
        mtx_right, dist_right = None, None # 标记失败

    # --- 保存标定结果 (仅当左右都成功时) ---
    if mtx_left is not None and dist_left is not None and mtx_right is not None and dist_right is not None:
        try:
            np.savez(OUTPUT_CALIBRATION_FILE,
                     mtx_left=mtx_left, dist_left=dist_left,
                     mtx_right=mtx_right, dist_right=dist_right)
            print(f"\n双目标定结果已保存到: {OUTPUT_CALIBRATION_FILE}")
        except Exception as e:
            print(f"\n错误: 保存标定文件 '{OUTPUT_CALIBRATION_FILE}' 时出错: {e}")
    else:
        print("\n由于至少一个摄像头标定失败，未保存标定结果。")

else:
    print(f"\n未能捕捉到足够数量 ({MIN_IMAGES_NEEDED}) 的有效图像对 ({len(objpoints)} captured)，无法进行标定。")

print("\n脚本执行完毕。")
