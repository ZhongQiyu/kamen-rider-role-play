# zero_sum.py

import numpy as np

def evaluate(position):
    # 简化评估：随机返回评分
    return np.random.randint(-10, 10)

def is_terminal(position):
    # 简化终端检查：随机决定游戏是否结束
    return np.random.choice([True, False], p=[0.1, 0.9])

def get_children(position):
    # 简化子节点生成：生成随机的子位置
    return [position + np.random.randint(-1, 2) for _ in range(3)]

def minimax(position, depth, maximizingPlayer):
    if depth == 0 or is_terminal(position):
        return evaluate(position)
    
    if maximizingPlayer:
        maxEval = float('-inf')
        for child in get_children(position):
            eval = minimax(child, depth-1, False)
            maxEval = max(maxEval, eval)
        return maxEval
    else:
        minEval = float('inf')
        for child in get_children(position):
            eval = minimax(child, depth-1, True)
            minEval = min(minEval, eval)
        return minEval

# 初始位置
initial_position = 0

# 深度限制
max_depth = 3

# 调用 minimax 算法，初始为最大化玩家
best_score = minimax(initial_position, max_depth, True)
print("Best score estimated by minimax:", best_score)
