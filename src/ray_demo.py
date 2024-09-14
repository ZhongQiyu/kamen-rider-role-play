import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

ray.init()

tune.run(
    PPOTrainer,
    config={
        "env": "CartPole-v0",  # 环境名称
        "num_workers": 4,  # 并行 worker 数量
        "train_batch_size": 4000,
    }
)
