# test_zhipuai_api_key.py

import torch
import torch.nn as nn
from torch.utils._pytree import _register_pytree_node
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import requests

# 配置 Zhipu AI 的 API 密钥
zhipuai_api_key = "your_zhipuai_api_key"  # 替换为实际的 API 密钥
os.environ['ZHIPUAI_API_KEY'] = zhipuai_api_key

# 定义 CUDA 同步函数
def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        print("CUDA is not available, skipping synchronization.")

# 自定义 pytree 节点类
class CustomNode:
    def __init__(self, data):
        self.data = data

# 注册自定义 pytree 节点
_register_pytree_node(CustomNode,
                      lambda x: (x.data, None),  # Flatten function
                      lambda data, context: CustomNode(data)  # Unflatten function
                      )

# 初始化日语模型和分词器
def load_japanese_model():
    # 从 Hugging Face 的模型库中加载预训练的日语模型
    tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    
    # 使用 CUDA（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_japanese_model()

def query_zhipuai(prompt, max_length=100):
    """
    通过 Zhipu AI API 进行查询的函数
    """
    headers = {
        "Authorization": f"Bearer {zhipuai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_length
    }
    
    response = requests.post("https://api.zhipu.ai/v1/complete", json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('text', '')
    else:
        print(f"Error querying Zhipu AI: {response.status_code}")
        return ""

def generate_response(prompt):
    """
    使用本地语言模型生成响应
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    synchronize_cuda()  # 同步 CUDA 操作

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 主函数
if __name__ == "__main__":
    # 使用日语作为输入提示
    prompt = "今日はどんな気持ちですか？"  # 日语的提示语句
    
    # 从 Zhipu AI 获取结果
    zhipuai_result = query_zhipuai(prompt)
    print(f"Zhipu AI Result: {zhipuai_result}")

    # 使用本地日语模型生成响应
    local_model_result = generate_response(prompt)
    print(f"Local Model Result: {local_model_result}")
