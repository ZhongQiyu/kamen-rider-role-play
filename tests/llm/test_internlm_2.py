# test_internlm_2.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_and_chat(model_name: str, prompt: str, initial_history=None):
    """
    加载模型并与之进行对话。

    参数:
    - model_name (str): 模型的名称。
    - prompt (str): 用户的输入文本。
    - initial_history (list, optional): 对话的历史记录。如果是新的对话，使用空列表。

    返回:
    - str: 模型的响应。
    - list: 更新的历史记录。
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载模型并设置为浮点16，以防止内存溢出
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()
    model = model.eval()

    # 设置初始对话历史记录
    if initial_history is None:
        initial_history = []

    # 生成响应
    response, history = model.chat(tokenizer, prompt, history=initial_history)
    return response, history

# 模型名称列表
models = ["internlm/internlm2-chat-1_8b", "internlm/internlm2-chat-20b", "internlm/internlm2-chat-7b"]

# 初始提示
initial_prompt = "hello"
management_prompt = "please provide three suggestions about time management"

# 对每个模型进行对话
for model_name in models:
    print(f"Using model: {model_name}")
    
    # 第一次对话
    response, history = load_and_chat(model_name, initial_prompt)
    print(response)
    
    # 第二次对话
    response, history = load_and_chat(model_name, management_prompt, history)
    print(response)
