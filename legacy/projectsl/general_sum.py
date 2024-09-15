# general_sum.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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

# 初始化参数
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
