# test_dialogues.py

import random
from gensim.models import FastText
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import deque
import numpy as np
import spacy

class DialogueGenerator:
    def __init__(self, dialogue_data):
        self.dialogues = [d["text"] for d in dialogue_data["dialogues"]]
        self.fasttext_model = None
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        self.nlp = spacy.load('ja_core_news_sm')

    # FastText 增强
    def train_fasttext(self):
        self.fasttext_model = FastText(self.dialogues, vector_size=100, window=3, min_count=1, sg=1)
    
    def generate_fasttext_variants(self, text, topn=3):
        words = text.split()
        variants = []
        for word in words:
            similar_words = [w for w, _ in self.fasttext_model.wv.most_similar(word, topn=topn)]
            for similar in similar_words:
                variant = text.replace(word, similar)
                variants.append(variant)
        return variants

    # 对话树生成
    class DialogueNode:
        def __init__(self, text, options=None):
            self.text = text
            self.options = options if options else []

    def create_dialogue_tree(self):
        root_node = self.DialogueNode(self.dialogues[0])
        node1 = self.DialogueNode(self.dialogues[1])
        node2 = self.DialogueNode(self.dialogues[2])
        root_node.options = [{"text": "继续对话1", "next": node1}, {"text": "继续对话2", "next": node2}]
        return root_node

    def traverse_dialogue(self, node):
        print(node.text)
        if not node.options:
            return
        choice = random.choice(node.options)
        self.traverse_dialogue(choice['next'])

    # 对话模板创建填充
    def fill_templates(self):
        templates = ["{dialogue}。", "说话人：{dialogue}。"]
        filled = []
        for dialogue in self.dialogues:
            for template in templates:
                filled.append(template.format(dialogue=dialogue))
        return filled

    # 基于语言模型微调
    def generate_language_model_text(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    # 强化学习生成
    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=2000)
            self.gamma = 0.95
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.learning_rate = 0.001

        def act(self, state):
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)

    def grammar_check(self, text):
        doc = self.nlp(text)
        errors = [token.text for token in doc if token.pos_ == "X"]  # 假设 'X' 为错误标签
        return len(errors) == 0

    def validate_generated_text(self, generated_texts):
        valid_texts = [text for text in generated_texts if self.grammar_check(text)]
        return valid_texts

# 测试用例
def test_dialogue_generator():
    # 加载数据
    with open("../data/text/episodes/json/ep48.json", "r") as f:
        data = eval(f.read())
    
    # 初始化生成器
    generator = DialogueGenerator(data)

    # FastText 增强测试
    generator.train_fasttext()
    fasttext_variants = generator.generate_fasttext_variants("ただいま通報場所に到着しました。")
    print("FastText 生成的变体：", fasttext_variants)

    # 对话树生成测试
    dialogue_tree_root = generator.create_dialogue_tree()
    print("对话树测试：")
    generator.traverse_dialogue(dialogue_tree_root)

    # 对话模板填充测试
    filled_dialogues = generator.fill_templates()
    print("填充的对话模板：", filled_dialogues)

    # 基于语言模型微调测试
    generated_text = generator.generate_language_model_text("剣崎君が見つけたと言った時")
    print("基于语言模型生成的文本：", generated_text)

    # 强化学习生成测试
    agent = generator.DQNAgent(state_size=4, action_size=3)
    print("强化学习代理生成的动作：", agent.act(None))

    # 语法检查和验证测试
    all_generated_text = filled_dialogues + fasttext_variants + [generated_text]
    valid_texts = generator.validate_generated_text(all_generated_text)
    print("所有生成且通过语法检查的对话：", valid_texts)

# 运行测试用例
test_dialogue_generator()
