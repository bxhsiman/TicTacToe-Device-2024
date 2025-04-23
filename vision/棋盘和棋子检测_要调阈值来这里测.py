'''
写在前面，这个代码必须在识别到棋盘的情况下才可以运行，否则会报错。
'''
import cv2
import numpy as np

def find_qipan(frame):
    """
    检测棋盘的位置并返回棋盘的角点。

    参数:
    - frame: 输入图像帧

    返回:
    - rect: 棋盘的9个角点坐标数组，如果未检测到棋盘则返回None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图像

    edged = cv2.Canny(gray, 50, 150)  # 使用Canny边缘检测算法检测边缘
    # 这里的50和150是Canny边缘检测的低阈值和高阈值，可以根据需要调整
    kernel = np.ones((5, 5), np.uint8)  # 定义一个5x5的结构元素用于膨胀操作
    dilated = cv2.dilate(edged, kernel, iterations=1)  # 膨胀操作增强边缘
    # 膨胀的结构元素大小和迭代次数可以调整

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)  # 选择最大轮廓
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # 计算轮廓的周长
        # epsilon 是多边形近似的精度，0.02 是一个常见值，可以调整以更精确或更粗略的轮廓

        approx = cv2.approxPolyDP(largest_contour, epsilon, True)  # 多边形近似

        if len(approx) == 4:  # 如果检测到4个角点（可能是棋盘）
            corners = approx.reshape((4, 2))  # 将角点重塑为4x2数组
            tl, bl, br, tr = corners  # 左上、左下、右下、右上角点

            cross_points = []
            for i in range(4):
                for j in range(4):
                    cross_x = int((tl[0] * (3 - i) + tr[0] * i) * (3 - j) / 9 +
                                  (bl[0] * (3 - i) + br[0] * i) * j / 9)
                    cross_y = int((tl[1] * (3 - i) + tr[1] * i) * (3 - j) / 9 +
                                  (bl[1] * (3 - i) + br[1] * i) * j / 9)
                    cross_points.append((cross_x, cross_y))

            centers = []
            for i in range(3):
                for j in range(3):
                    center_x = int((cross_points[i * 4 + j][0] + cross_points[i * 4 + j + 1][0] +
                                    cross_points[(i + 1) * 4 + j][0] + cross_points[(i + 1) * 4 + j + 1][0]) / 4)
                    center_y = int((cross_points[i * 4 + j][1] + cross_points[i * 4 + j + 1][1] +
                                    cross_points[(i + 1) * 4 + j][1] + cross_points[(i + 1) * 4 + j + 1][1]) / 4)
                    centers.append((center_x, center_y))

            if len(centers) == 9:  # 确保检测到9个棋盘交点
                centers = np.array(centers)
                rect = np.zeros((9, 2), dtype="float32")
                s = centers.sum(axis=1)  # 计算每个点的总和
                idx_0 = np.argmin(s)  # 左上角点索引
                idx_8 = np.argmax(s)  # 右下角点索引
                diff = np.diff(centers, axis=1)  # 计算x和y坐标的差异
                idx_2 = np.argmin(diff)  # 右上角点索引
                idx_6 = np.argmax(diff)  # 左下角点索引
                rect[0] = centers[idx_0]
                rect[2] = centers[idx_2]
                rect[6] = centers[idx_6]
                rect[8] = centers[idx_8]

                calc_center = (rect[0] + rect[2] + rect[6] + rect[8]) / 4  # 计算棋盘中心
                mask = np.zeros(centers.shape[0], dtype=bool)
                idxes = [1, 3, 4, 5, 7]  # 选择非角点的索引
                mask[idxes] = True
                others = centers[mask]
                idx_l = others[:, 0].argmin()
                idx_r = others[:, 0].argmax()
                idx_t = others[:, 1].argmin()
                idx_b = others[:, 1].argmax()
                found = np.array([idx_l, idx_r, idx_t, idx_b])
                mask = np.isin(range(len(others)), found, invert=False)
                idx_c = np.where(mask == False)[0]
                if len(idx_c) == 1:
                    rect[1] = others[idx_t]  # 上中
                    rect[3] = others[idx_l]  # 左中
                    rect[4] = others[idx_c]  # 中心
                    rect[5] = others[idx_r]  # 右中
                    rect[7] = others[idx_b]  # 下中
                    return rect  # 返回棋盘角点坐标

    return None  # 未检测到棋盘

def detect_chess_positions(frame):
    """
    检测棋盘上的棋子位置。

    参数:
    - frame: 输入图像帧

    返回:
    - chess_positions: 棋子的位置和颜色信息
    - frame: 原图像帧
    """
    chess_positions = []  # 存储棋子位置的列表

    merge_threshold = 40  # 定义位置合并的阈值
    # 这个阈值用于合并位置，避免同一位置检测到多个棋子，可以根据需要调整

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)  # 高斯模糊去除噪声
    # (11, 11) 是高斯模糊的核大小，可以根据图像噪声情况调整

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=20, maxRadius=35)
    # 使用霍夫圆变换检测圆
    # dp: 分辨率累加器的反比例
    # minDist: 检测圆之间的最小距离
    # param1: Canny边缘检测的高阈值
    # param2: 检测圆心的累加器阈值
    # minRadius 和 maxRadius: 圆的最小半径和最大半径
    # 这些参数根据实际棋子的大小和图像情况进行调整

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # 四舍五入并转为整数

        for (x, y, r) in circles:
            roi = frame[y - r:y + r, x - r:x + r]  # 提取圆的区域ROI
            mean = cv2.mean(roi)[0]  # 计算ROI的灰度平均值

            if mean < 130:  # 假设阈值130为黑色
                color = (0, 0, 255)  # 红色标注黑色圆
                label = "Black"
            else:
                color = (255, 0, 0)  # 蓝色标注白色圆
                label = "White"
            # 130 是用于区分黑白棋子的灰度阈值，可以根据实际棋子颜色进行调整

            merge = False
            for idx, (cx, cy, _) in enumerate(chess_positions):
                dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)  # 计算距离
                if dist < merge_threshold:
                    chess_positions[idx] = ((cx + x) // 2, (cy + y) // 2, label)  # 更新位置为平均值
                    merge = True
                    break

            if not merge:
                chess_positions.append((x, y, label))  # 如果不合并，则添加新位置

    return chess_positions, frame  # 返回检测到的棋子位置信息和帧

# 打开摄像头
cam = cv2.VideoCapture(1)  # 使用摄像头1，根据实际情况修改
if not cam.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cam.read()
    # 调用棋盘检测函数
    rect = find_qipan(frame)
    # 显示棋盘检测结果
    if rect is not None:
        print("Detected Chessboard Corners:")
        print(rect)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        # 调用函数检测棋子位置
        chess_positions, frame = detect_chess_positions(frame)
        # 在帧上绘制检测到的圆和标签
        for (x, y, label) in chess_positions:
            if label == "Black":
                color = (0, 0, 255)  # 红色标注黑色圆
            else:
                color = (255, 0, 0)  # 蓝色标注白色圆

            cv2.circle(frame, (x, y), 20, color, 4)  # 绘制圆
            cv2.putText(frame, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)  # 绘制标签
        for i in range(9):
            cv2.circle(frame, (int(rect[i][0]), int(rect[i][1])), 3, (0, 255, 0), -1)  # 绘制棋盘角点
            cv2.putText(frame, f"{i + 1}", (int(rect[i][0]), int(rect[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # 标注角点编号
        # 显示图像帧
        cv2.imshow('Chessboard Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # 按 'q' 键退出

cam.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
