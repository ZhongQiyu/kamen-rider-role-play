import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-1_8b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-20b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-20b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
# Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "hello", history=[])
print(response)
# Hello! How can I help you today?
response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=history)
print(response)

# -*- coding: utf-8 -*-


from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)

# Creating the Llama Pipeline
# Set up a pipeline for text generation.
# This pipeline simplifies the process of feeding prompts to the model and receiving generated text as output.
from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Getting Responses
# Function to get responses from the Llama model

def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
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

# Example usage
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
get_llama_response(prompt)

# More Queries
prompt = """I'm a programmer and Python is my favorite language because of its simple syntax and variety of applications I can build with it.
Based on that, what language should I learn next?
Give me 5 recommendations"""
get_llama_response(prompt)

prompt = 'How to learn fast?\n'
get_llama_response(prompt)

prompt = 'I love basketball. Do you have any recommendations of team sports I might like?\n'
get_llama_response(prompt)

prompt = 'How to get rich?\n'
get_llama_response(prompt)

# Conversational Loop
# Create an interactive chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "quit", "exit"]:
        print("Chatbot: Goodbye!")
        break
    get_llama_response(user_input)

# Conclusion
# Thanks to the Hugging Face Library, creating a pipeline to chat with Llama 2 (or any other open-source LLM) is quite easy.
