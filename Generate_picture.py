import cv2
import numpy as np
import os

# --- 配置参数 ---
# 棋盘格内部角点数量 (width, height)
pattern_size = (9, 6)  # 例如 9x6 的内部角点
# 每个方格的物理尺寸 (毫米) - 重要：打印时必须保证这个尺寸
square_size_mm = 20.0

# A4 纸张尺寸 (毫米) - 用于计算边距和画布大小
a4_width_mm = 210
a4_height_mm = 297

# 图像分辨率 (像素/毫米) - 影响打印清晰度 (10 像素/毫米 ~= 254 DPI)
pixels_per_mm = 10

# 边距 (毫米) - 棋盘格距离图像边缘的距离
margin_mm = 20

# 输出文件名
output_filename = f'chessboard_{pattern_size[0]}x{pattern_size[1]}_{int(square_size_mm)}mm.png'

# --- 计算尺寸 (像素) ---
square_size_px = int(square_size_mm * pixels_per_mm)
margin_px = int(margin_mm * pixels_per_mm)

# 棋盘格的总宽度和高度 (像素)
board_width_px = pattern_size[0] * square_size_px
board_height_px = pattern_size[1] * square_size_px

# 完整棋盘格绘制区域的宽度和高度 (包括边缘方块)
full_board_width_px = (pattern_size[0] + 1) * square_size_px
full_board_height_px = (pattern_size[1] + 1) * square_size_px

# 图像的总宽度和高度 (像素)
img_width_px = full_board_width_px + 2 * margin_px
img_height_px = full_board_height_px + 2 * margin_px

# --- 检查尺寸是否适合 A4 --- (可选，给用户提示)
print(f"棋盘格尺寸: {pattern_size[0]+1} x {pattern_size[1]+1} 个方块")
print(f"方块尺寸: {square_size_mm} mm")
print(f"计算出的图像总尺寸: {img_width_px / pixels_per_mm:.1f} mm x {img_height_px / pixels_per_mm:.1f} mm")
if img_width_px / pixels_per_mm > a4_width_mm or img_height_px / pixels_per_mm > a4_height_mm:
    print("\n警告: 生成的图像尺寸可能超出 A4 纸张范围！")
    print("请考虑减小方块尺寸 (square_size_mm) 或边距 (margin_mm)。")
    # 可以选择在这里退出或继续生成
    # exit()
print("-" * 30)

# --- 创建白色背景图像 ---
# 使用单通道灰度图即可
image = np.ones((img_height_px, img_width_px), dtype=np.uint8) * 255

# --- 绘制棋盘格 ---
black = 0 # 黑色

for r in range(pattern_size[1] + 1):  # 行
    for c in range(pattern_size[0] + 1):  # 列
        if (r + c) % 2 == 0: # 交替颜色
            # 计算当前方块的左上角坐标 (像素)
            top_left_x = margin_px + c * square_size_px
            top_left_y = margin_px + r * square_size_px

            # 计算右下角坐标
            bottom_right_x = top_left_x + square_size_px
            bottom_right_y = top_left_y + square_size_px

            # 绘制黑色填充矩形
            cv2.rectangle(image, (top_left_x, top_left_y),
                          (bottom_right_x, bottom_right_y),
                          black, thickness=cv2.FILLED)

# --- 添加信息文本 (可选) ---
font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.6
text_color = 0 # 黑色
text_thickness = 1

text1 = f"Pattern: {pattern_size[0]}x{pattern_size[1]} corners"
text2 = f"Square Size: {square_size_mm:.1f} mm"

# 获取文本尺寸以确定位置
(text1_width, text1_height), _ = cv2.getTextSize(text1, font, text_scale, text_thickness)
(text2_width, text2_height), _ = cv2.getTextSize(text2, font, text_scale, text_thickness)

# 在底部中心附近添加文本
text1_x = (img_width_px - text1_width) // 2
text1_y = img_height_px - margin_px // 2 # 放在下边距中间
text2_x = (img_width_px - text2_width) // 2
text2_y = text1_y + text1_height + 5 # 在第一行文本下方

cv2.putText(image, text1, (text1_x, text1_y), font, text_scale, text_color, text_thickness, cv2.LINE_AA)
cv2.putText(image, text2, (text2_x, text2_y), font, text_scale, text_color, text_thickness, cv2.LINE_AA)

# --- 保存图像 ---
try:
    cv2.imwrite(output_filename, image)
    print(f"\n成功生成标定板图像: {output_filename}")
    print(f"图像尺寸 (像素): {img_width_px} x {img_height_px}")
    print("\n打印说明:")
    print("1. 请使用 A4 纸张打印此图像。")
    print("2. 打印时请务必选择 '实际大小' 或 '100%' 缩放，【不要】勾选 '缩放以适应页面' 或 'Fit to page'。")
    print(f"3. 打印后请用尺子测量黑色方块的边长，确保其【精确等于】{square_size_mm:.1f} mm。")
except Exception as e:
    print(f"\n错误：保存图像时出错: {e}")

# --- 显示图像 (可选) ---
# cv2.imshow('Generated Chessboard', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\n脚本执行完毕。")
