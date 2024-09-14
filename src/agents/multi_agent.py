# multi_agent.py

import threading

class MultiAgent:
    def __init__(self):
        self.redis_client = RedisClient()

    def start_agents(self):
        agent_a = AgentA(self.redis_client)
        agent_b = AgentB(self.redis_client)
        agent_c = AgentC(self.redis_client)

        threading.Thread(target=agent_a.run).start()
        threading.Thread(target=agent_b.run).start()
        threading.Thread(target=agent_c.run).start()

    def run_all(self):
        self.start_agents()
