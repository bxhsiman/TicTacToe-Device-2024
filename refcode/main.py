import cv2
import numpy as np

def find_qipan():
    find_center_method = 1  # 1, 2

    cam = cv2.VideoCapture(1)  # 使用摄像头0，可以根据实际情况修改
    if not cam.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edged = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) == 4:
                corners = approx.reshape((4, 2))
                tl, bl, br, tr = corners

                cross_points = []
                for i in range(4):
                    for j in range(4):
                        cross_x = int((tl[0] * (3 - i) + tr[0] * i) * (3 - j) / 9 +
                                      (bl[0] * (3 - i) + br[0] * i) * j / 9)
                        cross_y = int((tl[1] * (3 - i) + tr[1] * i) * (3 - j) / 9 +
                                      (bl[1] * (3 - i) + br[1] * i) * j / 9)
                        cross_points.append((cross_x, cross_y))
                        cv2.circle(img, (cross_x, cross_y), 3, (0, 255, 0), -1)

                centers = []
                if find_center_method == 1:
                    for i in range(3):
                        for j in range(3):
                            center_x = int((cross_points[i * 4 + j][0] + cross_points[i * 4 + j + 1][0] +
                                            cross_points[(i + 1) * 4 + j][0] + cross_points[(i + 1) * 4 + j + 1][
                                                0]) / 4)
                            center_y = int((cross_points[i * 4 + j][1] + cross_points[i * 4 + j + 1][1] +
                                            cross_points[(i + 1) * 4 + j][1] + cross_points[(i + 1) * 4 + j + 1][
                                                1]) / 4)
                            centers.append((center_x, center_y))
                            cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)


                if len(centers) == 9:
                    centers = np.array(centers)
                    rect = np.zeros((9, 2), dtype="float32")
                    s = centers.sum(axis=1)
                    idx_0 = np.argmin(s)
                    idx_8 = np.argmax(s)
                    diff = np.diff(centers, axis=1)
                    idx_2 = np.argmin(diff)
                    idx_6 = np.argmax(diff)
                    rect[0] = centers[idx_0]
                    rect[2] = centers[idx_2]
                    rect[6] = centers[idx_6]
                    rect[8] = centers[idx_8]

                    calc_center = (rect[0] + rect[2] + rect[6] + rect[8]) / 4
                    mask = np.zeros(centers.shape[0], dtype=bool)
                    idxes = [1, 3, 4, 5, 7]
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
                        rect[1] = others[idx_t]
                        rect[3] = others[idx_l]
                        rect[4] = others[idx_c]
                        rect[5] = others[idx_r]
                        rect[7] = others[idx_b]
                        for i in range(9):
                            cv2.putText(img, f"{i + 1}", (int(rect[i][0]), int(rect[i][1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        print("> 45 degree")

        cv2.imshow('Chessboard Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

find_qipan()
