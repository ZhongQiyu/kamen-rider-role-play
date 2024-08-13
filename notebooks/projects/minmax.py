# minmax.py

import numpy as np

class Agent:
    def __init__(self, name, strategy=None):
        self.name = name
        self.strategy = strategy
    
    def set_strategy(self, strategy):
        self.strategy = strategy

class Game:
    def __init__(self, payoff_matrices):
        self.agents = []
        self.payoff_matrices = payoff_matrices

    def add_agent(self, agent):
        self.agents.append(agent)

    def worst_case_analysis(self):
        worst_case_gains = []
        worst_case_losses = []

        for i, agent in enumerate(self.agents):
            if i % 2 == 0:  # Player I type agents (even index)
                min_values = np.min(self.payoff_matrices[i], axis=1)
                worst_case_gains.append(np.max(min_values))
            else:  # Player II type agents (odd index)
                max_values = np.max(self.payoff_matrices[i], axis=0)
                worst_case_losses.append(np.min(max_values))

        return worst_case_gains, worst_case_losses

    def expected_outcome(self):
        expected_gains = []
        for i, agent in enumerate(self.agents):
            strategy = agent.strategy
            if strategy is None:
                raise ValueError(f"Agent {agent.name} has not set a strategy.")
            
            other_agents_strategies = [self.agents[j].strategy for j in range(len(self.agents)) if j != i]
            combined_strategy = np.prod(other_agents_strategies, axis=0)
            expected_gain = strategy.T @ self.payoff_matrices[i] @ combined_strategy
            expected_gains.append(expected_gain)
        
        return expected_gains

    def minimax_theorem_check(self):
        worst_case_gains, worst_case_losses = self.worst_case_analysis()
        return np.max(worst_case_gains), np.min(worst_case_losses)

# 定义支付矩阵
payoff_matrix1 = np.array([
    [3, 1],
    [0, 2]
])

payoff_matrix2 = np.array([
    [4, 2],
    [1, 3]
])

payoff_matrix3 = np.array([
    [2, 4],
    [3, 1]
])

payoff_matrix4 = np.array([
    [1, 5],
    [2, 3]
])

# 创建四个智能体
agent1 = Agent(name="Player I - 1")
agent2 = Agent(name="Player II - 1")
agent3 = Agent(name="Player I - 2")
agent4 = Agent(name="Player II - 2")

# 创建游戏实例并添加智能体
game = Game(payoff_matrices=[payoff_matrix1, payoff_matrix2, payoff_matrix3, payoff_matrix4])
game.add_agent(agent1)
game.add_agent(agent2)
game.add_agent(agent3)
game.add_agent(agent4)

# 设置混合策略
agent1.set_strategy(np.array([0.5, 0.5]))
agent2.set_strategy(np.array([0.5, 0.5]))
agent3.set_strategy(np.array([0.5, 0.5]))
agent4.set_strategy(np.array([0.5, 0.5]))

# 执行最坏情况分析
worst_case_gains, worst_case_losses = game.worst_case_analysis()
print("Worst-case gains for Player I type agents:", worst_case_gains)
print("Worst-case losses for Player II type agents:", worst_case_losses)

# 计算期望收益
expected_gains = game.expected_outcome()
print("\nExpected gains for all agents:", expected_gains)

# 检查 Minimax 定理
V_max, V_min = game.minimax_theorem_check()
print(f"\nMinimax Theorem Check: max(min(X^T*A*y)) = {V_max}, min(max(X^T*A*y)) = {V_min}")
print("Is max(min(X^T*A*y)) equal to min(max(X^T*A*y))?", V_max == V_min)
