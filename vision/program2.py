import tkinter as tk
import serial

def col_to_num(col, row):
    if col == 0:
        return 2*row + 3
    elif col == 1:
        return 2*row + 2
    elif col == 2:
        return 2*row + 1


class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("井字棋游戏")
        self.create_widgets()

        self.ser = serial.Serial('COM6', 9600) # 串口初始化

    def __del__(self):
        self.ser.close() # 串口关闭

    def create_widgets(self):
        self.buttons = [[None for _ in range(3)] for _ in range(3)]

        # 创建 3x3 棋盘
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.root, text="", font=('Arial', 40), width=5, height=2,
                                   command=lambda row=i, col=j: self.select_position(row, col))
                button.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = button

        # 创建颜色选择按钮
        self.color_buttons = {
            "black1": tk.Button(self.root, text="黑色1", width=8, height=2, command=lambda: self.set_color("black1")),
            "black2": tk.Button(self.root, text="黑色2", width=8, height=2, command=lambda: self.set_color("black2")),
            "white1": tk.Button(self.root, text="白色1", width=8, height=2, command=lambda: self.set_color("white1")),
            "white2": tk.Button(self.root, text="白色2", width=8, height=2, command=lambda: self.set_color("white2"))
        }

        # 放置颜色按钮
        self.color_buttons["black1"].grid(row=3, column=0, padx=5, pady=5)
        self.color_buttons["black2"].grid(row=3, column=1, padx=5, pady=5)
        self.color_buttons["white1"].grid(row=3, column=2, padx=5, pady=5)
        self.color_buttons["white2"].grid(row=3, column=3, padx=5, pady=5)

        # 创建功能按钮
        self.next_button = tk.Button(self.root, text="重置", width=8, height=2, command=self.reset_board)
        self.next_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

        self.exit_button = tk.Button(self.root, text="退出", width=8, height=2, command=self.exit_application)
        self.exit_button.grid(row=4, column=2, columnspan=2, padx=5, pady=5)

        self.color = None
        self.color_used = {
            "black1": False,
            "black2": False,
            "white1": False,
            "white2": False
        }

    def select_position(self, row, col):
        if self.color:
            if self.buttons[row][col]["text"] == "":
                # 串口控制逻辑

                if self.color == "black1":
                    command1 = "c1"
                elif self.color == "black2":
                    command1 = "c2"
                elif self.color == "white1":
                    command1 = "d1"
                elif self.color == "white2":
                    command1 = "d2"


                self.ser.write(f"{command1}" + 'b' + str(col_to_num(col, row)) ) # 发送数据到串口

                self.buttons[row][col]["text"] = "B" if self.color.startswith("black") else "A"
                print(f"点击的位置: ({row}, {col}), 颜色: {self.color}")
                self.color_used[self.color] = True
                self.color = None

    def set_color(self, color):
        if not self.color_used[color]:
            self.color = color

    def reset_board(self):
        for row in range(3):
            for col in range(3):
                self.buttons[row][col]["text"] = ""
        self.color = None
        for key in self.color_buttons:
            self.color_used[key] = False

    def exit_application(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TicTacToe(root)
    root.mainloop()
