# calibrate_camera.py
import cv2
import numpy as np
import os
import time

# --- 配置 ---
INPUT_VIDEO = 'preprocess_show.mp4' # 包含直线的视频
OUTPUT_FILE = 'calibration_approx.npz'
STITCHING_MODE = 'SxS' # 与视频源一致
FOCAL_LENGTH_GUESS_FACTOR = 1.0 # 用于猜测初始 K
# NUM_FRAMES_TO_PROCESS = 100 # 不再需要处理帧来估计畸变
# MIN_LINE_LENGTH = 50
# MAX_LINE_GAP = 10
# CANNY_THRESHOLD1 = 50
# CANNY_THRESHOLD2 = 150

# --- 函数：从视频分割帧 (仅用于获取尺寸) ---
def get_dimensions_from_video(video_path, mode):
    cap_temp = cv2.VideoCapture(video_path)
    if not cap_temp.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}' 以获取尺寸。")
        return None, None, None
    combined_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    combined_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_temp.release()

    frame_width, frame_height = None, None
    if mode == 'SxS':
        frame_width = combined_width // 2
        frame_height = combined_height
    elif mode == 'TxB':
        frame_width = combined_width
        frame_height = combined_height // 2
    else:
        print(f"错误: 无效的 STITCHING_MODE '{mode}'")
        return None, None, None

    if frame_width is None or frame_height is None or frame_width <= 0 or frame_height <= 0:
         print("错误：未能从视频计算有效的单目尺寸。")
         return None, None, None

    return combined_width, combined_height, frame_width, frame_height

# --- 主逻辑 ---
print(f"读取视频信息: {INPUT_VIDEO}")
combined_width, combined_height, frame_width, frame_height = get_dimensions_from_video(INPUT_VIDEO, STITCHING_MODE)

if frame_width is None:
    exit()

print(f"视频分辨率 (合并): {combined_width}x{combined_height}")
print(f"计算出的单目分辨率: {frame_width}x{frame_height}")

# --- 使用基于规格书的估计值 ---
fx_estimate = 1000.0
fy_estimate = 1000.0
# 确保使用正确的单目尺寸计算 cx, cy
if frame_width == 1280 and frame_height == 720:
    cx_estimate = 640.0
    cy_estimate = 360.0
else:
    # 如果视频尺寸不是预期的 1280x720 (单目)，则回退到中心点
    print(f"警告: 检测到的单目尺寸 ({frame_width}x{frame_height}) 与预期 (1280x720) 不符，将使用图像中心作为主点估计。")
    cx_estimate = frame_width / 2.0
    cy_estimate = frame_height / 2.0

K_estimate = np.array([[fx_estimate, 0, cx_estimate],
                       [0, fy_estimate, cy_estimate],
                       [0, 0, 1]], dtype=np.float32)
print(f"\n使用的基于规格估计的 K 矩阵 (将用于左右相机):")
print(K_estimate)

# --- 假设零畸变 --- 
dist_guess = np.zeros((4, 1), dtype=np.float32) # k1, k2, p1, p2 = 0
print(f"\n假设的畸变系数 (k1,k2,p1,p2): {dist_guess.T}")

print("\n警告：此脚本使用基于规格书估计的内参K，并假设零畸变。")
# print("它不进行任何基于内容的实际校准。") # 这句不再完全准确，因为用了规格
print("如果需要更精确的标定，请使用棋盘格或其他标定模式。")

# --- 保存结果 --- 
try:
    np.savez(OUTPUT_FILE,
             mtx1=K_estimate, dist1=dist_guess,
             mtx2=K_estimate, dist2=dist_guess) # 假设左右相同
    print(f"\n已将基于规格估计的参数保存到: {OUTPUT_FILE}")
    print("内容: mtx1, dist1, mtx2, dist2")
except Exception as e:
    print(f"\n错误: 保存文件 '{OUTPUT_FILE}' 时出错: {e}")

print("参数生成脚本结束。")
