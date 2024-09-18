# strategy.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# 多臂老虎机模拟多参数选择
class MultiArmedBandit:
    def __init__(self, arms):
        self.arms = arms
        self.n_arms = len(arms)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
    
    def select_arm(self):
        # 使用epsilon贪心算法选择参数
        epsilon = 0.1
        if np.random.rand() > epsilon:
            return np.argmax(self.values)
        else:
            return np.random.randint(self.n_arms)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

# 初始化随机森林的超参数集合
parameters = [
    {'n_estimators': 10, 'max_depth': 5},
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 10, 'max_depth': 10},
    {'n_estimators': 50, 'max_depth': 10}
]

bandit = MultiArmedBandit(parameters)

for _ in range(100):
    arm = bandit.select_arm()
    params = parameters[arm]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    accuracy = cross_val_score(model, X_test, y_test, cv=3).mean()
    bandit.update(arm, accuracy)

best_parameters = parameters[np.argmax(bandit.values)]
print("Best Parameters:", best_parameters)

# 零和博弈中的 Minimax 算法实现
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

# 零和博弈的初始位置和深度
initial_position = 0
max_depth = 3

# 调用 minimax 算法，初始为最大化玩家
best_score = minimax(initial_position, max_depth, True)
print("Best score estimated by minimax:", best_score)
