# qlora.py

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

# 定义 QLoRA 配置
qlora_config = LoraConfig(
    r=4,  # QLoRA 低秩矩阵的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
    lora_dropout=0.1,  # Dropout 概率
    bias="none",  # 无偏置
    task_type="CAUSAL_LM"  # 任务类型为自回归语言模型
)

# 应用 QLoRA 到模型
model = get_peft_model(model, qlora_config)

# 训练参数设置
training_args = TrainingArguments(
    output_dir="./gptj-qlora",  # 输出目录
    evaluation_strategy="steps",
    eval_steps=400,  # 每400步评估一次
    per_device_train_batch_size=2,  # 根据硬件调整
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # 训练轮数
    save_steps=800,
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,  # 使用混合精度训练
    optim="adamw_torch"  # 优化器
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
model.save_pretrained("./gptj-qlora")
tokenizer.save_pretrained("./gptj-qlora")


{
    "episode": 1,
    "scene": "学校",
    "dialogue": [
        {
            "turn": 1,
            "speaker": "A",
            "text": "今日はどうしたの？",
            "timestamp": "00:01:23",
            "context": "A进入教室，看到B看起来有些不开心。",
            "emotion": "concerned"
        },
        {
            "turn": 2,
            "speaker": "B",
            "text": "ちょっと疲れたんだ。",
            "timestamp": "00:01:30",
            "context": "A刚问了B。",
            "emotion": "tired"
        }
    ]
}



{
    "episode": 1,
    "scene": "学校",
    "dialogues": [
        {
            "turn": 1,
            "speaker": "A",
            "text": "今日はどうしたの？",
            "context": {
                "previous_dialogue": "None",
                "scene_description": "A进入教室，看到B看起来有些不开心。",
                "emotion": "concerned"
            }
        },
        {
            "turn": 2,
            "speaker": "B",
            "text": "ちょっと疲れたんだ。",
            "context": {
                "previous_dialogue": "A: 今日はどうしたの？",
                "scene_description": "B坐在窗边的桌子上，头靠在手上。",
                "emotion": "tired"
            }
        }
    ]
}



{
    "scene_id": "episode1_scene10",
    "timestamp_start": "00:10:23",
    "timestamp_end": "00:15:45",
    "dialogue": [
        {
            "speaker": "角色A",
            "role_info": {
                "personality": "乐观开朗",
                "background": "来自东京的大学生"
            },
            "text": "今日はいい天気ですね。",
            "timestamp": "00:10:23"
        },
        {
            "speaker": "角色B",
            "role_info": {
                "personality": "认真严格",
                "background": "高校教师"
            },
            "text": "そうですね、でも私は今日は家で仕事があります。",
            "timestamp": "00:10:35"
        },
        {
            "speaker": "角色A",
            "role_info": {
                "personality": "乐观开朗",
                "background": "来自东京的大学生"
            },
            "text": "それは残念です。",
            "timestamp": "00:10:50"
        }
    ]
}



视频转文本qlora

怎么配置jan.ai的token

