# base_agent.py

class BaseAgent:
    def __init__(self, redis_client):
        self.redis_client = redis_client
