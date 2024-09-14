# cli_demo.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型本地路径和 Hugging Face 模型名称
local_model_path = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
huggingface_model_name = "internlm/internlm2-chat-1_8b"
access_token = "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"  # 替换为你的 Hugging Face 访问令牌

try:
    # 优先尝试加载本地模型
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, device_map='cuda:0')
    model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
    print("Successfully loaded model from local path.")
except Exception as e:
    print(f"Local model loading failed: {e}. Attempting to load from Hugging Face...")
    # 如果本地加载失败，则尝试从 Hugging Face 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        huggingface_model_name,
        trust_remote_code=True,
        use_auth_token=access_token,
        device_map='cuda:0'
    )
    model = AutoModelForCausalLM.from_pretrained(
        huggingface_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_auth_token=access_token,
        device_map='cuda:0'
    )
    print("Successfully loaded model from Hugging Face.")

model = model.eval()

# 定义系统提示
system_prompt = """
You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
