import cv2
import numpy as np
image = cv2.imread('image.jpg')
threshold = 30  # 可以根据实际情况调整阈值

lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([threshold, threshold, threshold], dtype=np.uint8)

mask = cv2.inRange(image, lower_black, upper_black)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # 检查边界框内的像素是否满足黑色的标准
    if np.mean(image[y:y + h, x:x + w, :]) < threshold:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形边界框

# 显示结果
cv2.imshow('Detected Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
