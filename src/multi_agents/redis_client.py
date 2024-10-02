# redis_client.py

class RedisClient:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def set_value(self, key, value):
        self.client.set(key, value)

    def get_value(self, key):
        return self.client.get(key)
