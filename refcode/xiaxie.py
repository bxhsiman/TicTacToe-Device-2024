import cv2
import numpy as np
import serial
import time

ser = serial.Serial('COM3', 9600)
ser.flushInput()

move_table = {
    (1,1): "", "b1": 2, "c1": 3,
    "a2": 4, "b2": 5, "c2": 6,
    "a3": 7, "b3": 8, "c3": 9
}

# 第一问 放入5号
def move_q1(move_from="c1"):
    command1 = move_from
    return(command1 + "b5")

# 第二问 先从c移动到任意位置 再从d移动到任意位置
def move_q2(move_from1="c1", move_from2="c2", move_from3="c3", move_from4="c4", move_to1, move_to2, move_to3, move_to4):
    return(move_from1 + move_to1 + move_from2 + move_to2 + move_from3 + move_to3 + move_from4 + move_to4)

# 第三问 旋转45度
def move_q3():
    pass

# 第四问 执行器
def move_q4():


# 执行器 先取棋子 再放棋子
def send_move_command(command):
    ser.write(command)
    ser.flushInput()


# 初始化棋盘和参数
board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
sideAI = 1
player = -1
current_side = sideAI  # 电脑先手
isFirstMove = True     # 是否电脑的第一步

def find_qipan(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            centers = []
            for i in range(3):
                for j in range(3):
                    center_x = int((cross_points[i * 4 + j][0] + cross_points[i * 4 + j + 1][0] +
                                    cross_points[(i + 1) * 4 + j][0] + cross_points[(i + 1) * 4 + j + 1][0]) / 4)
                    center_y = int((cross_points[i * 4 + j][1] + cross_points[i * 4 + j + 1][1] +
                                    cross_points[(i + 1) * 4 + j][1] + cross_points[(i + 1) * 4 + j + 1][1]) / 4)
                    centers.append((center_x, center_y))

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
                    return rect

    return None

def detect_chess_positions(frame, rect):
    chess_positions = []
    merge_threshold = 40

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=15, maxRadius=40)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            roi = frame[y - r:y + r, x - r:x + r]
            mean = cv2.mean(roi)[0]

            if mean < 130:
                color = (0, 0, 255)
                label = "Black"
            else:
                color = (255, 0, 0)
                label = "White"

            merge = False
            for idx, (cx, cy, _) in enumerate(chess_positions):
                dist = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if dist < merge_threshold:
                    chess_positions[idx] = ((cx + x) // 2, (cy + y) // 2, label)
                    merge = True
                    break

            if not merge:
                chess_positions.append((x, y, label))
                for i in range(9):
                    if abs(x - rect[i][0]) < 20 and abs(y - rect[i][1]) < 20:
                        if label == "Black":
                            board[i // 3][i % 3] = 1
                        else:
                            board[i // 3][i % 3] = -1

    return chess_positions, board

def checkWinner(board):
    for row in board:
        if sum(row) == 3:
            return 1
        if sum(row) == -3:
            return -1

    for col in range(3):
        if board[0][col] + board[1][col] + board[2][col] == 3:
            return 1
        if board[0][col] + board[1][col] + board[2][col] == -3:
            return -1

    if board[0][0] + board[1][1] + board[2][2] == 3 or board[0][2] + board[1][1] + board[2][0] == 3:
        return 1
    if board[0][0] + board[1][1] + board[2][2] == -3 or board[0][2] + board[1][1] + board[2][0] == -3:
        return -1

    return 0

def isMovesLeft(board):
    for row in board:
        if 0 in row:
            return True
    return False

def checkImmediateWin(board, side):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = side
                if checkWinner(board) == side:
                    board[i][j] = 0
                    return (i, j)
                board[i][j] = 0
    return None

def findBestMove(board):
    global isFirstMove
    if isFirstMove:
        while True:
            try:
                print("请选择电脑的第一步落子位置:")
                row = int(input("请输入行号 (0, 1, 2): "))
                col = int(input("请输入列号 (0, 1, 2): "))
                if board[row][col] == 0:
                    isFirstMove = False
                    return (row, col)
                else:
                    print("该位置已经有棋子，请重新选择。")
            except ValueError:
                print("请输入有效的数字。")
            except IndexError:
                print("请输入范围内的数字。")

    win_move = checkImmediateWin(board, sideAI)
    if win_move:
        return win_move

    block_move = checkImmediateWin(board, player)
    if block_move:
        return block_move

    strategic_moves = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]

    for move in strategic_moves:
        if board[move[0]][move[1]] == 0:
            return move

# 打开摄像头
cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error: Camera not found!")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    rect = find_qipan(frame)
    if rect is not None:
        chess_positions, board = detect_chess_positions(frame, rect)

        if current_side == sideAI:
            print("电脑回合:")
            print(board)
            move = findBestMove(board)
            print(f"电脑落子位置: {move}")
            #sentdata
            chess_positions, board = detect_chess_positions(frame, rect)
            board[move[0]][move[1]] = sideAI
            if checkWinner(board) == sideAI:
                print("电脑获胜!")
                break
            if not isMovesLeft(board):
                print("平局!")
                break
            current_side = player

        else:
            print("玩家回合:")

            _ = int(input("下完棋子请输入0: "))
            chess_positions, board = detect_chess_positions(frame, rect)

            if checkWinner(board) == player:
                print("玩家获胜!")
            if not isMovesLeft(board):
                print("平局!")
            current_side = sideAI

        # 显示检测结果
        for (x, y, label) in chess_positions:
            color = (0, 0, 255) if label == "Black" else (255, 0, 0)
            cv2.circle(frame, (x, y), 20, color, 4)
            cv2.putText(frame, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for i in range(9):
            cv2.circle(frame, (int(rect[i][0]), int(rect[i][1])), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"{i + 1}", (int(rect[i][0]), int(rect[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Chessboard Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
