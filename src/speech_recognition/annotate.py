# annotate.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils._pytree import _register_pytree_node
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import zhipuai
import os
import random
import numpy as np

# 加载自定义 CUDA 扩展
custom_cuda = load(name="custom_cuda", sources=["cuda_add.cpp", "cuda_add_kernel.cu"])

# 设置 ZhipuAI API Key 和设备
os.environ["zhipuai_api_key"] = "your_zhipuai_api_key_here"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 pytree 数据结构
class JapanesePrompt:
    def __init__(self, text, emotion):
        self.text = text
        self.emotion = emotion

def flatten_japanese_prompt(prompt):
    return [prompt.text, prompt.emotion], None

def unflatten_japanese_prompt(_, children):
    return JapanesePrompt(children[0], children[1])

# 注册 pytree 数据结构
_register_pytree_node(JapanesePrompt, flatten_japanese_prompt, unflatten_japanese_prompt)

# ZhipuAI 日语模型加载
def load_japanese_model():
    model_id = 'zhipuai-japanese-large'
    response = zhipuai.Model.get(model_id=model_id)

    if response.get("success"):
        print("模型加载成功")
    else:
        print(f"加载模型失败: {response.get('message')}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model

# 定义自定义的 Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.x = np.random.rand(1000, 3)
        self.y = np.random.randint(low=0, high=2, size=(1000,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# collate_fn 定义
def collate_fn(batch):
    data_list, label_list = zip(*batch)
    return torch.tensor(data_list, dtype=torch.float32), torch.tensor(label_list, dtype=torch.long)

# 自定义 CUDA 反向传播操作
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = custom_cuda.forward(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = custom_cuda.backward(input, grad_output)
        return grad_input

# 定义 SimpleModel 使用自定义 CUDA 操作
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        weight = torch.ones_like(x).cuda()
        out = custom_cuda.forward(x, weight)[0]
        return self.fc(out)

# 滑动窗口函数
def sliding_window(text, window_size, step_size):
    return [text[i:i+window_size] for i in range(0, len(text), step_size) if i+window_size <= len(text)]

# 扩展对话数据集
def prepare_prompts_with_window(prompts, window_size, step_size):
    extended_prompts = []
    for prompt in prompts:
        windows = sliding_window(prompt.text, window_size, step_size)
        extended_prompts.extend([JapanesePrompt(window, prompt.emotion) for window in windows])
    return extended_prompts

# 数据批处理函数
def custom_collate_fn_with_emotion(batch):
    texts = [item.text for item in batch]
    emotions = [item.emotion for item in batch]
    return texts, emotions

# JapaneseModel 定义
class JapaneseModel(nn.Module):
    def __init__(self, tokenizer, model):
        super(JapaneseModel, self).__init__()
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        outputs = self.model(**inputs)
        return outputs.logits

# 带有情感标注的训练过程
def train_with_emotion(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        texts, emotions = batch
        optimizer.zero_grad()
        logits = model(texts)
        loss = F.cross_entropy(logits, torch.ones_like(logits))  # 需要调整标签
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 超参数随机搜索
def random_search_hyperparams(n_trials, model, dataloader):
    best_loss = float('inf')
    best_params = {}

    for trial in range(n_trials):
        lr = random.uniform(1e-5, 1e-3)
        batch_size = random.choice([2, 4, 8])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print(f"试验 {trial+1}: 学习率 = {lr}, 批次大小 = {batch_size}")
        dataloader.batch_size = batch_size
        loss = train_with_emotion(model, dataloader, optimizer)
        if loss < best_loss:
            best_loss = loss
            best_params = {'learning_rate': lr, 'batch_size': batch_size}

    print(f"最佳参数: 学习率 = {best_params['learning_rate']}, 批次大小 = {best_params['batch_size']}")
    return best_params

# 主程序执行逻辑
if __name__ == "__main__":
    # 使用滑动窗口扩展对话数据
    dialogues_with_emotions = [
        JapanesePrompt("剣崎: 俺はもう止まらない！ 橘: そんなことはさせない！", "怒り"),
        JapanesePrompt("剣崎: 世界を救うために、この戦いを続けるんだ！", "決意"),
        JapanesePrompt("相川: 俺のせいで、みんなに迷惑をかけてしまった……", "悲しみ"),
        JapanesePrompt("天音: 私、ずっとあなたのことを信じてた！", "喜び"),
        # 添加更多对话...
    ]

    window_size = 20
    step_size = 10
    extended_prompts = prepare_prompts_with_window(dialogues_with_emotions, window_size, step_size)

    # 批处理数据加载
    dataloader = DataLoader(extended_prompts, batch_size=2, collate_fn=custom_collate_fn_with_emotion)

    # 加载日语模型
    tokenizer, model = load_japanese_model()
    japanese_model = JapaneseModel(tokenizer, model).to(device)

    # 超参数随机搜索
    best_params = random_search_hyperparams(5, japanese_model, dataloader)

    # 使用最佳参数重新训练模型
    optimizer = torch.optim.Adam(japanese_model.parameters(), lr=best_params['learning_rate'])
    train_with_emotion(japanese_model, dataloader, optimizer)

# bert_qa.py

import json
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline

def generate_qa_pairs():
    """生成 1000 个问答对并保存到 JSON 文件。"""
    qa_pairs = [{"question": "什么是Python?", "answer": "Python是一种编程语言。"} for _ in range(1000)]
    with open('qa_pairs.jsonl', 'w') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + '\n')

def setup_bert_model():
    model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return nlp