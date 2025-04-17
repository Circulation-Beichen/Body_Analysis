import cv2
import numpy as np
import time

# --- 配置参数 ---
CAMERA_INDEX = 1
# REQUESTED_WIDTH = 2560 # 改为 2560
# REQUESTED_HEIGHT = 720 # 改为 720
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 540
STITCHING_MODE = 'SxS'

# ORB 参数 (只需要检测器)
N_FEATURES = 1000     # 检测的最大特征点数

# 新匹配策略参数
SEARCH_RADIUS = 40.0   # 在右图中搜索对应点的半径 (像素)

# 预处理参数
# APPLY_LAPLACIAN_PREPROCESSING = True # 是否启用拉普拉斯预处理
# LAPLACIAN_THRESHOLD_VALUE = 20     # 拉普拉斯图像阈值化 - 可能需要调整
APPLY_MEDIAN_BLUR = True # 是否启用中值滤波
MEDIAN_BLUR_KSIZE = 3    # 中值滤波核大小 (必须是奇数)

# 相机参数 (用于三角测量)
BASELINE_MM = 25.0

# 视差过滤参数
MIN_DISPARITY = 1 # 最小允许视差 (像素) - 至少差 1 个像素
MAX_DISPARITY = 250 # 最大允许视差 (像素) - 恢复一个合理的值

# --- 初始化摄像头 ---
print(f"尝试打开摄像头 (索引 {CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"错误：无法打开摄像头 (索引 {CAMERA_INDEX})。")
    exit()
else:
    print(f"摄像头 {CAMERA_INDEX} 打开成功。")

# --- 设置视频格式为 MJPEG --- (重要：通常在设置分辨率之前)
try:
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
        print(f"成功请求 MJPEG 格式。")
    else:
        print(f"警告：无法设置 MJPEG 格式 (可能不支持或已被覆盖)。")
except Exception as e:
    print(f"警告: 设置 FOURCC 时出错: {e}")

# --- 设置分辨率 ---
if 'REQUESTED_WIDTH' in locals() and 'REQUESTED_HEIGHT' in locals():
    print(f"尝试设置分辨率为 {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    time.sleep(0.5)

# --- 读取实际分辨率并计算单目尺寸 ---
combined_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
combined_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头实际输出分辨率 (合并): {combined_width}x{combined_height}")

# 计算单目尺寸和分割点
frame_width, frame_height, split_point_h, split_point_v = 0, 0, 0, 0
if STITCHING_MODE == 'SxS':
    frame_width = combined_width // 2
    frame_height = combined_height
    split_point_h = frame_width
    # --- 重要: 确认单目尺寸是否为 1280x720 ---
    if frame_width != 1280 or frame_height != 720:
        print(f"警告: 当前 SxS 单目分辨率 ({frame_width}x{frame_height}) 与规格书估计参数所基于的 (1280x720) 不符！K 矩阵可能不准。")
elif STITCHING_MODE == 'TxB':
    frame_width = combined_width
    frame_height = combined_height // 2
    split_point_v = frame_height
    # --- 重要: 确认单目尺寸是否为 1280x720 ---
    if frame_width != 1280 or frame_height != 720:
         print(f"警告: 当前 TxB 单目分辨率 ({frame_width}x{frame_height}) 与规格书估计参数所基于的 (1280x720) 不符！K 矩阵可能不准。")
else:
    print(f"错误: 无效的 STITCHING_MODE '{STITCHING_MODE}'")
    cap.release()
    exit()
print(f"模式: {STITCHING_MODE}, 单目分辨率: {frame_width}x{frame_height}")

# --- 使用基于规格书的相机内参 K --- (不再猜测)
fx_estimate = 1000.0
fy_estimate = 1000.0
cx_estimate = 640.0  # 基于 1280 宽度
cy_estimate = 360.0  # 基于 720 高度
K_estimate = np.array([[fx_estimate, 0, cx_estimate],
                       [0, fy_estimate, cy_estimate],
                       [0, 0, 1]], dtype=np.float32)

# --- 假设相机外参 R, T ---
R_guess = np.eye(3, dtype=np.float32)
T_guess = np.array([[BASELINE_MM], [0], [0]], dtype=np.float32)

# --- 计算假设的投影矩阵 P1, P2 --- (使用 K_estimate)
P1 = K_estimate @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K_estimate @ np.hstack((R_guess, T_guess))

print("\n--- 使用的相机参数 (用于三角测量) ---")
print(f"基于规格估计的 K:\n{K_estimate}")
print(f"假设 R: 单位矩阵")
print(f"假设 T: {T_guess.T} mm (基线={BASELINE_MM}mm)")
print("------------------------------------------\n")

# --- 初始化 ORB 检测器 (仅用于检测角点) ---
orb = cv2.ORB_create(nfeatures=N_FEATURES)
# --- 移除 BFMatcher ---
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

print("\n开始基于邻近度的角点匹配和近似距离计算...")
print("按 'q' 键退出...")

# --- 实时处理循环 ---
while True:
    ret, combined_frame = cap.read()
    if not ret:
        print("错误：无法读取摄像头帧。")
        break
    if combined_frame is None or combined_frame.shape[0] == 0 or combined_frame.shape[1] == 0:
        continue
    if combined_frame.shape[1] != combined_width or combined_frame.shape[0] != combined_height:
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

        if frame_left is None or frame_right is None or frame_left.shape[0] == 0 or frame_right.shape[0] == 0: continue
        if frame_left.shape[0] != frame_height or frame_left.shape[1] != frame_width or \
           frame_right.shape[0] != frame_height or frame_right.shape[1] != frame_width: continue
    except Exception as e:
        continue

    # 转换为灰度图
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # --- 可选：应用拉普拉斯 + 阈值化预处理 (注释掉) ---
    processed_left = gray_left
    processed_right = gray_right
    # if APPLY_LAPLACIAN_PREPROCESSING:
    #     try:
    #         # 应用拉普拉斯算子
    #         laplacian_left = cv2.Laplacian(gray_left, cv2.CV_64F)
    #         laplacian_right = cv2.Laplacian(gray_right, cv2.CV_64F)
    #         abs_laplacian_left = np.absolute(laplacian_left)
    #         abs_laplacian_right = np.absolute(laplacian_right)
    #         laplacian_8u_left = np.uint8(np.clip(abs_laplacian_left, 0, 255)) # 限制范围并转uint8
    #         laplacian_8u_right = np.uint8(np.clip(abs_laplacian_right, 0, 255))
    #
    #         # 阈值化
    #         _, thresh_left = cv2.threshold(laplacian_8u_left, LAPLACIAN_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    #         _, thresh_right = cv2.threshold(laplacian_8u_right, LAPLACIAN_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    #         processed_left = thresh_left
    #         processed_right = thresh_right
    #         # 可选：显示预处理结果用于调试
    #         # cv2.imshow('Processed Left', processed_left)
    #         # cv2.imshow('Processed Right', processed_right)
    #     except Exception as e:
    #         print(f"预处理错误: {e}")
    #         # 如果预处理出错，则回退到使用原始灰度图
    #         processed_left = gray_left
    #         processed_right = gray_right

    # --- 可选：应用中值滤波 --- 
    if APPLY_MEDIAN_BLUR:
        try:
            # 确保 ksize 是奇数
            ksize = MEDIAN_BLUR_KSIZE if MEDIAN_BLUR_KSIZE % 2 == 1 else MEDIAN_BLUR_KSIZE + 1
            processed_left = cv2.medianBlur(gray_left, ksize)
            processed_right = cv2.medianBlur(gray_right, ksize)
        except Exception as e:
             print(f"中值滤波错误: {e}")
             # 出错则回退
             processed_left = gray_left
             processed_right = gray_right

    # --- 在处理后的图像上检测关键点 ---
    kp_left = orb.detect(processed_left, None)
    kp_right = orb.detect(processed_right, None)

    # --- 基于邻近度的新匹配逻辑 ---
    proximity_matches = [] # 存储匹配对信息 {'pt_l': (x,y), 'pt_r': (x,y)}
    points_l_for_triangulation = []
    points_r_for_triangulation = []

    if kp_left and kp_right:
        # 为了快速查找右侧点，可以构建一个数据结构，但对于几百个点，直接遍历也还行
        right_points = {i: kp.pt for i, kp in enumerate(kp_right)} # 字典: 索引 -> (x, y)

        for kp_l in kp_left:
            pt_l = kp_l.pt
            candidates_in_radius = [] # 存储 (距离平方, 右点坐标)

            # 在右图中搜索半径内的点
            for idx_r, pt_r in right_points.items():
                dist_sq = (pt_l[0] - pt_r[0])**2 + (pt_l[1] - pt_r[1])**2
                if dist_sq < SEARCH_RADIUS**2:
                    # 应用基础视差过滤
                    disparity = pt_l[0] - pt_r[0]
                    if MIN_DISPARITY < disparity < MAX_DISPARITY:
                        candidates_in_radius.append({'dist_sq': dist_sq, 'pt_r': pt_r})

            # 如果在半径内找到了候选点
            if candidates_in_radius:
                # 选择距离最近的点
                candidates_in_radius.sort(key=lambda x: x['dist_sq'])
                best_match_pt_r = candidates_in_radius[0]['pt_r']
                proximity_matches.append({'pt_l': pt_l, 'pt_r': best_match_pt_r})
                points_l_for_triangulation.append(pt_l)
                points_r_for_triangulation.append(best_match_pt_r)


    # --- 三角测量 (逻辑不变, 使用 points_l/r_for_triangulation) ---
    points_3d_world = {} # 字典，键是 points_l_for_triangulation 的索引，值是 3D 坐标
    if points_l_for_triangulation and points_r_for_triangulation:
        np_points_l = np.array(points_l_for_triangulation, dtype=np.float32)
        np_points_r = np.array(points_r_for_triangulation, dtype=np.float32)

        try:
            # P1, P2 仍然使用基于规格的 K 和假设的 R, T 计算得到
            points_4d_hom = cv2.triangulatePoints(P1, P2, np_points_l.T, np_points_r.T)
            valid_w_indices = np.where(np.abs(points_4d_hom[3]) > 1e-6)[0]
            if len(valid_w_indices) > 0:
                points_4d_hom_valid = points_4d_hom[:, valid_w_indices]
                points_3d_valid = points_4d_hom_valid[:3] / points_4d_hom_valid[3]
                points_3d_valid = points_3d_valid.T

                # 注意：这里的索引对应 points_l/r_for_triangulation 列表的索引
                for i, point_3d in enumerate(points_3d_valid):
                    original_match_index = valid_w_indices[i] # 获取有效点在原始列表中的索引
                    if point_3d[2] > 0: # 仅过滤 Z > 0
                        points_3d_world[original_match_index] = point_3d

        except Exception as e:
            # print(f"Triangulation Error: {e}")
            pass # 三角测量失败则忽略


    # --- 可视化 ---
    vis_combined = np.hstack((frame_left, frame_right))

    # 绘制最终通过邻近度匹配且成功三角测量的匹配
    count_drawn = 0
    # 遍历成功三角测量的点
    for match_idx, point_3d in points_3d_world.items():
        # 从 points_l/r_for_triangulation 获取对应的 2D 点
        if match_idx < len(points_l_for_triangulation):
            pt_l = tuple(map(int, points_l_for_triangulation[match_idx]))
            pt_r = tuple(map(int, points_r_for_triangulation[match_idx]))
            pt_r_shifted = (pt_r[0] + frame_width, pt_r[1]) # 转换到合并图像坐标
            z_dist_mm = point_3d[2]

            # 绘制线和点
            cv2.line(vis_combined, pt_l, pt_r_shifted, (0, 255, 0), 1)
            cv2.circle(vis_combined, pt_l, 3, (0, 0, 255), -1)
            cv2.circle(vis_combined, pt_r_shifted, 3, (0, 0, 255), -1)

            # 绘制距离文本
            dist_text = f"{z_dist_mm:.0f}mm"
            cv2.putText(vis_combined, dist_text, (pt_l[0] + 5, pt_l[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)
            count_drawn += 1

    # --- 更新显示统计信息 ---
    # 显示检测到的角点总数
    cv2.putText(vis_combined, f"Detected L:{len(kp_left or [])}, R:{len(kp_right or [])}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    # 显示通过邻近度+视差过滤找到的初始匹配数
    cv2.putText(vis_combined, f"Proximity Matches (Radius={SEARCH_RADIUS:.0f}, {MIN_DISPARITY}<d<{MAX_DISPARITY}): {len(proximity_matches)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    # Ratio Test 和 Y Filter 不再适用
    cv2.putText(vis_combined, f"Triangulated (Z>0) & Drawn: {count_drawn}", (10, 70), # 调整行号
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    # 显示结果 (英文标题)
    cv2.imshow('Proximity Matching Test (Filtered + Triangulated)', vis_combined) # 更新窗口标题

    # --- 按 'q' 退出 ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n收到退出指令。")
        break

# --- 清理资源 ---
print("\n正在释放资源...")
cap.release()
cv2.destroyAllWindows()
print("测试脚本结束。")
