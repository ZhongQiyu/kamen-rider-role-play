import torch
from transformers import GPTJForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load your dataset (adjust to your specific dataset)
dataset = load_dataset('text', data_files={'train': 'path_to_train_data.txt', 'validation': 'path_to_validation_data.txt'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8, # Rank of the update matrices
    lora_alpha=16, # Scaling factor
    target_modules=["q_proj", "v_proj"], # Modules where LoRA is applied
    lora_dropout=0.05, # Dropout rate
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gptj-alpaca-lora",
    evaluation_strategy="steps",
    eval_steps=400,  # Evaluate every 400 steps
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=800,
    logging_dir="./logs",
    logging_steps=200,
    fp16=True,  # Mixed precision training
    optim="adamw_torch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./gptj-alpaca-lora")
tokenizer.save_pretrained("./gptj-alpaca-lora")
