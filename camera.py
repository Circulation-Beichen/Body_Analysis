import cv2
import numpy as np
import time

# --- 配置参数 ---
# !! 重要 !!: 你的双目摄像头现在被识别为一个设备，你需要找到它的索引号。
# 尝试 0, 1, 2 ... 直到找到正确的那个。内置摄像头通常是 0。
CAMERA_INDEX = 1  # 尝试修改这个值 (例如 0, 1, 2...)

# (可选) 尝试设置期望的分辨率 (这是 *合并后* 的分辨率)
# 例如，如果每个眼睛是 1280x720，那么左右拼接后可能是 2560x720
# 如果不确定，可以注释掉这两行，先看摄像头默认输出什么
REQUESTED_WIDTH = 2560
REQUESTED_HEIGHT = 720

# --- 初始化摄像头 ---
print(f"尝试打开摄像头 (索引 {CAMERA_INDEX})...")
# 仍然建议使用 CAP_DSHOW 后端在 Windows 上提高兼容性
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# --- 检查摄像头是否成功打开 ---
if not cap.isOpened():
    print(f"错误：无法打开摄像头 (索引 {CAMERA_INDEX})。")
    print("请检查：")
    print("1. 摄像头是否已连接并被 Windows 识别？")
    print("2. 索引号是否正确？尝试 0, 1, 2 ...")
    print("3. 是否有其他程序正在使用该摄像头？")
    print("4. 摄像头驱动是否正确安装？")
    exit()
else:
    print(f"摄像头 {CAMERA_INDEX} 打开成功。")

#--- (可选) 尝试设置分辨率 ---
if 'REQUESTED_WIDTH' in locals() and 'REQUESTED_HEIGHT' in locals():
    print(f"尝试设置分辨率为 {REQUESTED_WIDTH}x{REQUESTED_HEIGHT}...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
    # 等待一小段时间让设置生效
    time.sleep(0.5)


# 读取实际生效的分辨率 (这非常重要，因为它决定了如何分割图像)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"摄像头实际输出分辨率: {actual_width}x{actual_height}")

# 检查分辨率是否合理（例如，宽度是否大约是高度的两倍，如果左右拼接的话）
if actual_width < actual_height * 1.5: # 粗略检查，不适用于上下拼接
    print("警告：摄像头输出的宽度看起来不像典型的左右拼接双目图像。请检查输出图像。")
    print("如果图像是上下拼接的，你需要修改下面的分割逻辑。")

# --- 计算分割点 ---
# 假设是左右拼接 (Side-by-Side, SxS)
split_point = actual_width // 2
print(f"假设左右拼接，分割点将在宽度 = {split_point} 处。")

print("\n按 'q' 键退出...")

# --- 主循环：读取、分割并显示图像 ---
while True:
    # 读取合并后的帧
    ret, combined_frame = cap.read()

    # 检查是否成功读取
    if not ret:
        print("错误：无法读取摄像头帧。")
        break

    # --- 分割图像 ---
    # 确保帧不是 None 并且宽度足够分割
    if combined_frame is not None and combined_frame.shape[1] == actual_width:
        # 使用 NumPy 切片分割图像
        # frame_left = combined_frame[行范围, 列范围]
        frame_left = combined_frame[:, :split_point]  # 从最左边到分割点
        frame_right = combined_frame[:, split_point:] # 从分割点到最右边

        # --- 可视化 ---
        # 1. 显示原始的合并帧
        cv2.imshow(f'Combined Frame (Index {CAMERA_INDEX}, {actual_width}x{actual_height})', combined_frame)

        # 2. 显示分割后的左右帧
        cv2.imshow('Left Eye', frame_left)
        cv2.imshow('Right Eye', frame_right)

    else:
        print("错误：读取到的帧为空或宽度与预期不符。")
        # 如果帧有问题，只显示原始的（可能损坏的）帧
        if combined_frame is not None:
             cv2.imshow(f'Combined Frame (Index {CAMERA_INDEX}, Error?)', combined_frame)
        # 尝试关闭可能存在的左右眼窗口
        try:
            if cv2.getWindowProperty('Left Eye', cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow('Left Eye')
            if cv2.getWindowProperty('Right Eye', cv2.WND_PROP_VISIBLE) >= 1:
                 cv2.destroyWindow('Right Eye')
        except cv2.error:
            pass


    # --- 等待按键事件 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("收到退出指令。")
        break

    # 添加一点点延时，避免CPU占用过高 (可选)
    # time.sleep(0.01)

# --- 清理资源 ---
print("正在释放摄像头资源...")
cap.release()
print("正在关闭所有OpenCV窗口...")
cv2.destroyAllWindows()
print("程序结束。")