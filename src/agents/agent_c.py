# agent_c.py

from stable_baselines3 import PPO

class AgentC(BaseAgent):
    def __init__(self, redis_client):
        super().__init__(redis_client)
        self.model = PPO('MlpPolicy', 'CartPole-v1', verbose=1)
        self.model.learn(total_timesteps=10000)

    def run(self):
        while True:
            state = int(self.redis_client.get_value('state'))
            user_action = self.redis_client.get_value('user_action').decode('utf-8')
            task = f"Task for state {state} and action {user_action}"
            self.redis_client.set_value('task', task)
            print(f"Agent C assigned task: {task}")
            time.sleep(3)
