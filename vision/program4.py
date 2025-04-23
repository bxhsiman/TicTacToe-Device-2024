import cv2
import numpy as np
import serial
import tkinter as tk

def col_to_num(col, row):
    if col == 0:
        return 2*row + 3
    elif col == 1:
        return 2*row + 2
    elif col == 2:
        return 2*row + 1

#开局黑子选择位置GUI
class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("井字棋棋盘")
        self.create_widgets()
        self.selected_cell = None  # 用于存储用户选择的单元格
        self.result = None  # 存储选择的结果

    def create_widgets(self):
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.root, text="", font=('Arial', 40), width=5, height=2,
                                   command=lambda row=i, col=j: self.select_cell(row, col))
                button.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = button

        # 添加确认按钮
        self.confirm_button = tk.Button(self.root, text="确认", font=('Arial', 20), command=self.confirm_selection)
        self.confirm_button.grid(row=3, column=0, columnspan=2, pady=10)

        # 添加关闭按钮
        self.close_button = tk.Button(self.root, text="关闭", font=('Arial', 20), command=self.close_gui)
        self.close_button.grid(row=3, column=2, pady=10)

    def select_cell(self, row, col):
        """选择单元格"""
        if self.selected_cell is None:
            self.selected_cell = (row, col)
            self.buttons[row][col].config(bg="lightblue")
        else:
            prev_row, prev_col = self.selected_cell
            self.buttons[prev_row][prev_col].config(bg="SystemButtonFace")
            self.selected_cell = None

    def confirm_selection(self):
        """确认选择的单元格并关闭GUI"""
        if self.selected_cell is not None:
            row, col = self.selected_cell
            self.result = (row, col)  # 保存结果
            self.buttons[row][col]["text"] = f"({row}, {col})"
            print(f"选择的单元格是: ({row}, {col})")

            for i in range(3):
                for j in range(3):
                    self.buttons[i][j].config(bg="SystemButtonFace")

            self.selected_cell = None
            self.root.destroy()  # 关闭 GUI
        else:
            print("请先选择一个单元格")

    def close_gui(self):
        """直接关闭GUI"""
        self.root.destroy()
#玩家暂停按钮
class PauseDialog:
    def __init__(self, master):
        self.master = master
        self.master.title("确认")
        self.master.geometry("300x150")  # 设置窗口大小
        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.master, text="下完棋子后点击确认继续。")
        self.label.pack(pady=20)

        # 创建一个大一点的按钮
        self.confirm_button = tk.Button(self.master, text="确认", command=self.close, font=('Arial', 16), width=30, height=5)
        self.confirm_button.pack(pady=20)

    def close(self):
        self.master.destroy()
# 初始化棋盘和参数
board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
board_copy = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
sideAI = 1
player = -1
current_side = sideAI  # 电脑先手
isFirstMove = True     # 是否电脑的第一步

count = 0  # 用于统计电脑落子次数

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
    global count

    count += 1

    if isFirstMove:
        root = tk.Tk()
        app = TicTacToe(root)
        root.mainloop()
        (row, col)=app.result
        if board[row][col] == 0:
            isFirstMove = False
            return (row, col)

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

ser = serial.Serial('COM6', 9600)

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


            chess_positions, board = detect_chess_positions(frame, rect)



            #这里处理第6问

            if checkWinner(board) == sideAI:
                print("电脑获胜!")
                break
            if not isMovesLeft(board):
                print("平局!")
                break

            print(f"电脑落子位置: {move}")
            #sentdata

            if current_side == 1:
                command1 = 'C'
            elif sideAI == -1:
                command1 = 'D'

            ser.write((command1+str(count)+'B'+str(col_to_num(move[1], move[0]))).encode('utf-8'))
            print("发送数据：")
            print(command1+str(count)+'b'+str(col_to_num(move[1], move[0])))

            current_side = player

        else:
            print("玩家回合:")

            # 创建并运行 GUI 界面，等待用户点击确认
            root = tk.Tk()
            dialog = PauseDialog(root)
            root.mainloop()
            chess_positions, board = detect_chess_positions(frame, rect)

            if checkWinner(board) == player:
                print("玩家获胜!")
                break
            if not isMovesLeft(board):
                print("平局!")
                break
            current_side = sideAI

cam.release()
cv2.destroyAllWindows()
