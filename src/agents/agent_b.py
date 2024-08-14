# agent_b.py

class AgentB(BaseAgent):
    def run(self):
        while True:
            state = int(self.redis_client.get_value('state'))
            user_action = f"User action based on state {state}"
            self.redis_client.set_value('user_action', user_action)
            print(f"Agent B processed user action: {user_action}")
            time.sleep(2)
