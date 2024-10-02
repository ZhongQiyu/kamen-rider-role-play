# sem_align.py

import torch
from monotonic_align import monotonic_align

# 示例输入
input_sequence = torch.randn(10, 256)  # 10个时间步，特征维度256
target_sequence = torch.randn(5, 256)   # 5个时间步，特征维度256

# 对齐操作
alignments = monotonic_align(input_sequence, target_sequence)

# 使用对齐结果
print(alignments)

# alignment

import os
import re

def clean_text(text):
    """
    清理文本中的非对话信息（如时间戳和说话人标签），只保留纯对话内容。
    """
    # 去除时间戳和说话人标签 (e.g., "说话人1", "00:05")
    cleaned_text = re.sub(r"说话人\d+\s*\d{2}:\d{2}", "", text)
    # 移除多余的空行和空白字符
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def generate_conversation_pairs(lines):
    """
    生成滑窗式对话对：两个连续的句子作为一个对话对，并滑动一行。
    """
    pairs = []
    for i in range(len(lines) - 1):  # 滑窗，步长为1
        pair = f"{lines[i]} {lines[i+1]}"
        pairs.append(pair)
    return pairs

def preprocess_file(input_file, output_file):
    """
    读取对话文件，生成滑窗式对话对，并保存。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 清理文本
    cleaned_text = clean_text(raw_text)
    # 将清理后的文本按行分割
    lines = cleaned_text.split('. ')  # 假设句子以“.”为结束标记，按此分割

    # 生成滑窗式对话对
    conversation_pairs = generate_conversation_pairs(lines)

    # 将处理好的对话对保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in conversation_pairs:
            f.write(pair + '\n')

def preprocess_directory(input_dir, output_dir):
    """
    批量处理目录中的所有文本文件，生成滑窗式对话对并保存到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"pairs_{filename}")
            preprocess_file(input_path, output_path)
            print(f"Processed {filename} -> pairs_{filename}")

if __name__ == "__main__":
    # 输入文件夹路径：包含需要处理的对话文件
    input_directory = "/path/to/input_files"
    # 输出文件夹路径：用于保存处理后的文件
    output_directory = "/path/to/output_files"
    
    # 预处理目录下的所有文本文件
    preprocess_directory(input_directory, output_directory)
