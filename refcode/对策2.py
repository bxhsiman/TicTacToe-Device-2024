# 初始化棋盘, 黑1, 白-1, 空0
board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
sideAI = None
player = None

# 检查是否有胜利方
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

# 检查棋盘是否满
def isMovesLeft(board):
    for row in board:
        if 0 in row:
            return True
    return False

# 检查是否有即将胜利的位置，并返回该位置
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

# 寻找电脑落子最优位置
def findBestMove(board, isFirstMove):
    if isFirstMove:
        while True:
            try:
                print("请选择电脑的第一步落子位置:")
                row = int(input("请输入行号 (0, 1, 2): "))
                col = int(input("请输入列号 (0, 1, 2): "))
                if board[row][col] == 0:
                    return (row, col)
                else:
                    print("该位置已经有棋子，请重新选择。")
            except ValueError:
                print("请输入有效的数字。")
            except IndexError:
                print("请输入范围内的数字。")

    # 4.1 检查电脑是否可以直接获胜
    win_move = checkImmediateWin(board, sideAI)
    if win_move:
        return win_move

    # 4.2 检查玩家是否可以直接获胜，若可以则拦截
    block_move = checkImmediateWin(board, player)
    if block_move:
        return block_move

    # 4.3 按策略选择有利的位置
    # 优先级: 中心 -> 四个角 -> 边中心
    strategic_moves = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]

    for move in strategic_moves:
        if board[move[0]][move[1]] == 0:
            return move

# 主程序
def play_game():
    global sideAI, player
    # 选择对弈模式
    while True:
        try:
            mode = int(input("请选择对弈模式 (1: 电脑先手, 2: 玩家先手): "))
            if mode == 1:
                sideAI = 1
                player = -1
                break
            elif mode == 2:
                sideAI = -1
                player = 1
                break
            else:
                print("请输入有效的数字 (1 或 2)。")
        except ValueError:
            print("请输入有效的数字。")

    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    current_side = sideAI  # 默认电脑先手
    isFirstMove = True    # 是否电脑的第一步
    while True:
        # 电脑回合
        if current_side == sideAI:
            print("电脑回合:")
            move = findBestMove(board, isFirstMove)
            isFirstMove = False  # 第一步之后设置为 False
            board[move[0]][move[1]] = sideAI
            print_board(board)
            if checkWinner(board) == sideAI:
                print("电脑获胜!")
                break
            if not isMovesLeft(board):
                print("平局!")
                break
            current_side = player

        # 玩家回合
        else:
            print("玩家回合:")
            while True:
                try:
                    row = int(input("请输入行号 (0, 1, 2): "))
                    col = int(input("请输入列号 (0, 1, 2): "))
                    if board[row][col] == 0:
                        board[row][col] = player
                        break
                    else:
                        print("该位置已经有棋子，请重新输入。")
                except ValueError:
                    print("请输入有效的数字。")
                except IndexError:
                    print("请输入范围内的数字。")
            print_board(board)
            if checkWinner(board) == player:
                print("玩家获胜!")
                break
            if not isMovesLeft(board):
                print("平局!")
                break
            current_side = sideAI

# 打印棋盘
def print_board(board):
    for row in board:
        print(row)

# 启动游戏
play_game()
