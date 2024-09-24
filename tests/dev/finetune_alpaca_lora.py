import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 依赖项安装提示：
# pip install torch transformers datasets peft

# 加载 GPT-J 模型和分词器
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 加载数据集
train_data_path = "path_to_train_data.txt"  # 训练集路径
val_data_path = "path_to_validation_data.txt"  # 验证集路径

dataset = load_dataset('text', data_files={'train': train_data_path, 'validation': val_data_path})

# 数据集分词
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,  # 低秩分解矩阵的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
    lora_dropout=0.05,  # Dropout 概率
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./gptj-alpaca-lora",
    evaluation_strategy="steps",
    eval_steps=400,  # 每400步评估一次
    per_device_train_batch_size=2,  # 根据你的硬件调整
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=800,
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,  # 使用混合精度训练
    optim="adamw_torch"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("./gptj-alpaca-lora")
tokenizer.save_pretrained("./gptj-alpaca-lora")
