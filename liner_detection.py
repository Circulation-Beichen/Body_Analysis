import cv2
import numpy as np
import os
import math

# --- 配置参数 ---
input_video_path = 'example.mp4'
# 输出文件名调整，表明是多通道对比
output_video_path = 'example_liner_channels_compare.mp4'

# Canny 边缘检测参数 (对各通道可能需要不同值，但先用统一的)
canny_threshold1 = 50
canny_threshold2 = 150

# 霍夫直线变换 (Probabilistic Hough Transform) 参数
rho = 1
theta = np.pi / 180
threshold = 50
min_line_length = 50
max_line_gap = 10

# 斜率调整参数
ADJUST_SLOPES = True # 是否启用斜率调整
angle_tolerance_degrees = 5.0 # 分组的角度容差 (度)

# --- 检查输入文件 ---
if not os.path.exists(input_video_path):
    print(f"错误：输入视频文件 '{input_video_path}' 不存在。")
    exit()

# --- 打开视频文件 ---
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 '{input_video_path}'。")
    exit()

# --- 获取视频属性 ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"输入视频属性: {frame_width}x{frame_height} @ {fps:.2f} FPS")

# --- 定义视频写入器 (尺寸加倍) ---
output_width = frame_width * 2
output_height = frame_height * 2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
if not out.isOpened():
    print(f"错误: 无法创建视频写入对象 '{output_video_path}'")
    cap.release()
    exit()

# --- 辅助函数：处理单个通道/灰度图 ---
def process_channel(image_channel, channel_name=""):
    """对单通道图像进行模糊、边缘检测、直线检测和调整"""
    # 1. 高斯模糊
    blurred = cv2.GaussianBlur(image_channel, (5, 5), 0)
    # 2. Canny 边缘检测
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    # 3. 霍夫直线变换
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            np.array([]),
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # 4. 准备绘制结果的图像 (转为 BGR 以便绘制彩色线条和文字)
    output_image_bgr = cv2.cvtColor(image_channel, cv2.COLOR_GRAY2BGR)
    adjusted_lines_coords = [] # 存储最终绘制的线段坐标

    if lines is not None:
        # 4.1 计算角度、中点、长度
        line_data = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length == 0: continue # 避免除零错误
            line_data.append({
                'id': i, 'line': line[0], 'angle_rad': angle_rad, 'angle_deg': angle_deg,
                'midpoint': (mid_x, mid_y), 'length': length
            })

        # 4.2 (可选) 调整斜率
        if ADJUST_SLOPES and line_data:
            grouped = [False] * len(line_data)
            groups = []
            for i in range(len(line_data)):
                if grouped[i]: continue
                current_group = [line_data[i]]
                grouped[i] = True
                for j in range(i + 1, len(line_data)):
                    if grouped[j]: continue
                    angle_diff = abs(line_data[i]['angle_deg'] - line_data[j]['angle_deg'])
                    angle_diff = min(angle_diff, 180.0 - angle_diff) # 处理 0-180 环绕
                    if angle_diff < angle_tolerance_degrees:
                        current_group.append(line_data[j])
                        grouped[j] = True
                groups.append(current_group)

            for group in groups:
                if not group: continue
                avg_cos = np.mean([math.cos(item['angle_rad']) for item in group])
                avg_sin = np.mean([math.sin(item['angle_rad']) for item in group])
                avg_angle_rad = math.atan2(avg_sin, avg_cos)

                for item in group:
                    mid_x, mid_y = item['midpoint']
                    length = item['length']
                    half_len = length / 2.0
                    dx = half_len * math.cos(avg_angle_rad)
                    dy = half_len * math.sin(avg_angle_rad)
                    new_x1 = int(round(mid_x - dx)); new_y1 = int(round(mid_y - dy))
                    new_x2 = int(round(mid_x + dx)); new_y2 = int(round(mid_y + dy))
                    adjusted_lines_coords.append((new_x1, new_y1, new_x2, new_y2))
        else:
            # 不调整，使用原始线段
            adjusted_lines_coords = [tuple(l['line']) for l in line_data]

    # 5. 绘制直线
    line_color = (0, 0, 255) # 红色
    line_thickness = 2
    for line_coords in adjusted_lines_coords:
        x1, y1, x2, y2 = line_coords
        cv2.line(output_image_bgr, (x1, y1), (x2, y2), line_color, line_thickness)

    # 6. 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0) # 绿色
    text_thickness = 1
    cv2.putText(output_image_bgr, channel_name, (10, 25), font, font_scale, font_color, text_thickness, cv2.LINE_AA)

    return output_image_bgr

# --- 逐帧处理 ---
frame_count = 0
print("开始处理视频，对比各通道直线检测...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n视频处理完毕或读取错误。")
        break

    frame_count += 1

    # 1. 拆分通道 + 计算灰度图
    b_channel, g_channel, r_channel = cv2.split(frame)
    gray_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 分别处理四个图像
    processed_b = process_channel(b_channel, "Blue Channel")
    processed_g = process_channel(g_channel, "Green Channel")
    processed_r = process_channel(r_channel, "Red Channel")
    processed_gray = process_channel(gray_channel, "Grayscale")

    # 3. 拼接成 2x2 网格
    top_row = np.hstack((processed_b, processed_g))
    bottom_row = np.hstack((processed_r, processed_gray))
    combined_output = np.vstack((top_row, bottom_row))

    # 4. 写入输出视频
    out.write(combined_output)

    # 5. 显示处理过程
    cv2.imshow('Channel Comparison', combined_output)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 按 'q' 提前退出
        print("\n用户中断处理。")
        break

    # 打印进度
    if frame_count % 50 == 0: # 减少打印频率
        print(f"  已处理 {frame_count} 帧...", end='\r')

# --- 清理资源 ---
print(f"\n总共处理了 {frame_count} 帧。")
print("正在释放资源...")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"多通道直线检测对比结果已保存到: {output_video_path}")
print("脚本执行完毕。")
