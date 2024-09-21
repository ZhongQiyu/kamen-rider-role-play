# test_dynamic_batching.py

import os
import torch
import random
from torch.utils._pytree import _register_pytree_node
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader
import zhipuai
import torch.nn.functional as F

# 加载自定义 CUDA 扩展 (cpp_extension)
custom_cuda = load(name="custom_cuda", sources=["custom_op.cpp", "custom_op_kernel.cu"])

# 设置 ZhipuAI 的 API Key
os.environ["zhipuai_api_key"] = "your_zhipuai_api_key_here"

# 使用 CUDA 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义一个用于演示的简单 pytree 数据结构
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

# 使用 ZhipuAI API 获取大型日语语言模型
def load_japanese_model():
    model_id = 'zhipuai-japanese-large'  # 假设 ZhipuAI 提供了这个模型
    response = zhipuai.Model.get(model_id=model_id)

    if response.get("success"):
        print("Model loaded successfully from ZhipuAI.")
    else:
        print(f"Failed to load model from ZhipuAI: {response.get('message')}")

    # 初始化日语模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return tokenizer, model

# 定义反向传播的自定义 CUDA 操作
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

# 定义提示词生成的模型部分
class JapaneseModel(torch.nn.Module):
    def __init__(self, tokenizer, model):
        super(JapaneseModel, self).__init__()
        self.tokenizer = tokenizer
        self.model = model

    def forward(self, prompt):
        inputs = self.tokenizer(prompt.text, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        return outputs.logits

# 自定义 collate_fn 函数用于 DataLoader，可以处理情感标注
def custom_collate_fn_with_emotion(batch):
    texts = [item.text for item in batch]
    emotions = [item.emotion for item in batch]
    max_length = max(len(text) for text in texts)
    padded_texts = [text.ljust(max_length) for text in texts]  # 使用空格进行 padding
    return padded_texts, emotions

# 滑动窗口函数
def sliding_window(text, window_size, step_size):
    return [text[i:i+window_size] for i in range(0, len(text), step_size) if i+window_size <= len(text)]

# 使用滑动窗口扩展对话数据集
def prepare_prompts_with_window(prompts, window_size, step_size):
    extended_prompts = []
    for prompt in prompts:
        windows = sliding_window(prompt.text, window_size, step_size)
        extended_prompts.extend([JapanesePrompt(window, prompt.emotion) for window in windows])
    return extended_prompts

# 定义超参数调整逻辑（随机搜索）
def random_search_hyperparams(n_trials, model, dataloader):
    best_loss = float('inf')
    best_params = {}
    
    for trial in range(n_trials):
        lr = random.uniform(1e-5, 1e-3)
        batch_size = random.choice([2, 4, 8])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print(f"Trial {trial+1}: Learning Rate = {lr}, Batch Size = {batch_size}")
        dataloader.batch_size = batch_size
        
        loss = train_with_emotion(model, dataloader, optimizer)
        if loss < best_loss:
            best_loss = loss
            best_params = {'learning_rate': lr, 'batch_size': batch_size}

    print(f"Best Parameters: Learning Rate = {best_params['learning_rate']}, Batch Size = {best_params['batch_size']}")
    return best_params

# 定义带有情感标注的训练过程
def train_with_emotion(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        texts, emotions = batch
        optimizer.zero_grad()
        logits = model(texts)
        loss = F.cross_entropy(logits, torch.ones_like(logits))  # 示例损失
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 定义带有情感标注的对话数据集，风格基于《假面骑士剑》
dialogues_with_emotions = [
    JapanesePrompt("剣崎: 俺はもう止まらない！ 橘: そんなことはさせない！", "怒り"),
    JapanesePrompt("剣崎: 世界を救うために、この戦いを続けるんだ！", "決意"),
    JapanesePrompt("相川: 俺のせいで、みんなに迷惑をかけてしまった……", "悲しみ"),
    JapanesePrompt("天音: 私、ずっとあなたのことを信じてた！", "喜び"),
    JapanesePrompt("橘: 負けるわけにはいかない、この力を使うしかない！", "決意"),
    JapanesePrompt("始: なぜ俺はこの世界に存在するのか……", "疑問"),
    JapanesePrompt("剣崎: この力、使えばどうなるか分かっているんだ……", "恐れ"),
    JapanesePrompt("相川: もう一度、戦ってみせる！", "決意"),
    JapanesePrompt("剣崎: 君が信じる道を行け！俺もそうする！", "支援"),
    JapanesePrompt("始: 俺には帰る場所がない……", "孤独")
]

# 使用滑动窗口扩展对话
window_size = 20
step_size = 10
extended_prompts = prepare_prompts_with_window(dialogues_with_emotions, window_size, step_size)

# 使用 DataLoader 加载批处理数据
dataloader = DataLoader(extended_prompts, batch_size=2, collate_fn=custom_collate_fn_with_emotion)

# 日语模型加载
tokenizer, model = load_japanese_model()
japanese_model = JapaneseModel(tokenizer, model)

# 使用随机搜索进行超参数调整
best_params = random_search_hyperparams(5, japanese_model, dataloader)

# 使用最佳超参数重新训练模型
optimizer = torch.optim.Adam(japanese_model.parameters(), lr=best_params['learning_rate'])
train_with_emotion(japanese_model, dataloader, optimizer)
