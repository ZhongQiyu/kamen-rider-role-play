# knowledge_base.py

class KnowledgeBase:
    def __init__(self, model_size):
        self.model_size = model_size
        self.database = {}

    def add_entry(self, key, value):
        if self.model_size not in self.database:
            self.database[self.model_size] = {}
        self.database[self.model_size][key] = value

    def get_entry(self, key):
        return self.database.get(self.model_size, {}).get(key)

# 测试用例
if __name__ == "__main__":
    # 测试知识库
    kb = KnowledgeBase('2B')
    kb.add_entry('问候', '你好！')
    assert kb.get_entry('问候') == '你好！'
