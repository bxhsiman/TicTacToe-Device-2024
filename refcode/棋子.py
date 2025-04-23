import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(1)

# 存储棋子位置的列表
chess_positions = []

# 定义位置合并的阈值
merge_threshold = 40  # 假设阈值为20像素

while True:
    # 读取视频流帧
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去除噪声
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 使用霍夫圆变换检测圆
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=35)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # 提取圆的区域ROI
            roi = frame[y - r:y + r, x - r:x + r]
            # 计算ROI的灰度平均值
            mean = cv2.mean(roi)[0]

            # 判断圆的颜色并标注
            if mean < 130:  # 假设阈值100为黑色
                color = (0, 0, 255)  # 红色标注黑色圆
                label = "Black"
            else:
                color = (255, 0, 0)  # 蓝色标注白色圆
                label = "White"

            # 判断是否合并位置
            merge = False
            for idx, (cx, cy, _) in enumerate(chess_positions):
                dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if dist < merge_threshold:
                    # 更新位置为平均值
                    chess_positions[idx] = ((cx + x) // 2, (cy + y) // 2, label)
                    merge = True
                    break

            if not merge:
                # 如果不合并，则添加新位置
                chess_positions.append((x, y, label))

            # 在帧上绘制圆和标签
            cv2.circle(frame, (x, y), r, color, 4)
            cv2.putText(frame, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # 显示结果帧和灰度图像
    cv2.imshow("Detected Circles", frame)
    cv2.imshow("Gray", blurred)  # 显示灰度图像

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 输出所有检测到的棋子位置信息
print("Detected Chess Positions:")
for idx, (x, y, label) in enumerate(chess_positions):
    print(f"Chess {idx + 1}: Position ({x}, {y}), Color: {label}")

# 释放摄像头资源和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
