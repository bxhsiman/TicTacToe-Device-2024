import tkinter as tk
import subprocess
import sys
import os

def start_program1():
    subprocess.Popen(['python', 'program1.py'])

def start_program2():
    subprocess.Popen(['python', 'program2.py'])

def start_program3():
    subprocess.Popen(['python', 'program3.py'])

def start_program4():
    subprocess.Popen(['python', 'program4.py'])

def start_program5():
    subprocess.Popen(['python', 'program5.py'])

def restart_application():
    """重新启动应用程序并关闭当前进程"""
    python = sys.executable  # 获取当前的Python解释器路径
    # 启动新的Python进程
    subprocess.Popen([python] + sys.argv)
    # 关闭当前进程
    sys.exit()

# 创建主窗口
root = tk.Tk()
root.title("程序启动器")

# 创建并布局按钮
button1 = tk.Button(root, text="启动程序 1", command=start_program1, width=50, height=3)
button1.pack(pady=5)

button2 = tk.Button(root, text="启动程序 2", command=start_program2, width=50, height=3)
button2.pack(pady=5)

button3 = tk.Button(root, text="启动程序 3", command=start_program3, width=50, height=3)
button3.pack(pady=5)

button4 = tk.Button(root, text="启动程序 4", command=start_program4, width=50, height=3)
button4.pack(pady=5)

button5 = tk.Button(root, text="启动程序 5", command=start_program5, width=50, height=3)
button5.pack(pady=5)

# 创建重启按钮
restart_button = tk.Button(root, text="重启", command=restart_application, width=50, height=3)
restart_button.pack(pady=10)

# 运行主循环
root.mainloop()
