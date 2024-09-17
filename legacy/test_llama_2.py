import os
from transformers import AutoTokenizer, pipeline
import torch

# 安装必要的库
os.system("apt-get -y install -qq aria2")

# 克隆所需的存储库
os.system("git clone -b v2.5 https://github.com/camenduru/text-generation-webui")
os.chdir("text-generation-webui")

# 安装依赖
os.system("pip install -q -r requirements.txt")
os.system("pip install transformers torch accelerate")

# 下载模型文件
model_url_base = "https://huggingface.co/4bit/Llama-2-7b-chat-hf"
model_dir = "./models/Llama-2-7b-chat-hf"
os.makedirs(model_dir, exist_ok=True)

files_to_download = [
    ("resolve/main/model-00001-of-00002.safetensors", "model-00001-of-00002.safetensors"),
    ("resolve/main/model-00002-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    ("raw/main/model.safetensors.index.json", "model.safetensors.index.json"),
    ("raw/main/special_tokens_map.json", "special_tokens_map.json"),
    ("resolve/main/tokenizer.model", "tokenizer.model"),
    ("raw/main/tokenizer_config.json", "tokenizer_config.json"),
    ("raw/main/config.json", "config.json"),
    ("raw/main/generation_config.json", "generation_config.json"),
]

for file_url, file_name in files_to_download:
    os.system(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {model_url_base}/{file_url} -d {model_dir} -o {file_name}")

# 创建设置文件
with open("./settings.yaml", "w") as f:
    f.write("dark_theme: true\n")
    f.write("chat_style: wpp\n")

# 启动服务器
os.system("python server.py --share --settings ./settings.yaml --model ./models/Llama-2-7b-chat-hf")

# 通过 Hugging Face CLI 登录
os.system("huggingface-cli login")
os.system("huggingface-cli whoami")

# 加载模型和分词器
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

# 设置文本生成流水线
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def get_llama_response(prompt: str) -> None:
    """
    生成Llama模型的响应。
    
    参数：
        prompt (str): 用户输入或问题。
    
    返回：
        None: 打印模型的响应。
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])

# 示例用法
prompts = [
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    """I'm a programmer and Python is my favorite language because of its simple syntax and variety of applications I can build with it.
    Based on that, what language should I learn next?
    Give me 5 recommendations""",
    'How to learn fast?\n',
    'I love basketball. Do you have any recommendations of team sports I might like?\n',
    'How to get rich?\n'
]

for prompt in prompts:
    get_llama_response(prompt)

# 交互式聊天循环
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    get_llama_response(user_input)
