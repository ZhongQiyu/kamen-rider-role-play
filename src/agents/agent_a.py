# agent_a.py

class AgentA(BaseAgent):
    def run(self):
        while True:
            state = random.randint(0, 100)
            self.redis_client.set_value('state', state)
            print(f"Agent A sensed state: {state}")
            time.sleep(1)
