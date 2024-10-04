# parser.py

import json
import os
import re
from typing import List
from datetime import datetime
import logging
import requests
import torch
from monotonic_align import monotonic_align
import boto3

class UnifiedProcessor:
    def __init__(self, api_key):
        # 初始化所需组件，包括日志、AWS Comprehend 和翻译API
        self.api_key = api_key
        self.url = "https://translation.googleapis.com/language/translate/v2"
        self.logger = self.setup_logging()
        self.comprehend = boto3.client('comprehend')

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger(__name__)

    # 对话处理相关方法
    def handle_dialog_from_file(self, file_path):
        """处理从文件加载的对话"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            current_speaker = None
            current_time = None
            dialog = []
            for line in file:
                speaker_match = re.match(r'^说话人(\d+) (\d{2}:\d{2})', line)
                if speaker_match:
                    if current_speaker is not None:
                        data.append({
                            'speaker': current_speaker,
                            'time': current_time,
                            'text': ' '.join(dialog).strip()
                        })
                        dialog = []
                    current_speaker, current_time = speaker_match.groups()
                else:
                    dialog.append(line.strip())
            if current_speaker and dialog:
                data.append({
                    'speaker': current_speaker,
                    'time': current_time,
                    'text': ' '.join(dialog).strip()
                })
        return data

    def process_transcript(self, file_path, output_path):
        """处理转录文本，将对话格式化"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        content = content.replace("None None", "").strip()

        pattern = r'(说话人\d+ \d{2}:\d{2})'
        split_content = re.split(pattern, content)

        result = []
        for i in range(1, len(split_content), 2):
            speaker_and_time = split_content[i].strip()
            dialogue = split_content[i + 1].strip()
            if dialogue:
                result.append(f"{speaker_and_time} {dialogue}")

        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n".join(result))
        self.logger.info(f"Processed file saved to: {output_path}")

    def process_all_transcripts(self, input_dir, output_dir):
        """批量处理多个转录文件"""
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)

                self.logger.info(f"Processing file: {filename}")
                self.process_transcript(input_file, output_file)
                self.logger.info(f"Processed file saved to: {output_file}")

    # 翻译相关方法
    def translate(self, text, target_lang):
        """调用 Google 翻译 API 翻译文本"""
        params = {
            'q': text,
            'target': target_lang,
            'key': self.api_key
        }
        response = requests.get(self.url, params=params)
        return response.json().get('data', {}).get('translations', [{}])[0].get('translatedText')

    # 语义对齐相关方法
    def align_sequences(self, input_sequence, target_sequence):
        """调用 monotonic_align 对输入序列和目标序列进行对齐"""
        alignments = monotonic_align(input_sequence, target_sequence)
        return alignments

    # 文本清理和对齐相关方法
    @staticmethod
    def clean_text(text):
        """清理文本中的非对话信息"""
        cleaned_text = re.sub(r"说话人\d+\s*\d{2}:\d{2}", "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    @staticmethod
    def generate_conversation_pairs(lines):
        """生成滑窗式对话对"""
        pairs = []
        for i in range(len(lines) - 1):
            pair = f"{lines[i]} {lines[i+1]}"
            pairs.append(pair)
        return pairs

    def preprocess_file(self, input_file, output_file):
        """预处理单个文件，生成滑窗式对话对"""
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        cleaned_text = self.clean_text(raw_text)
        lines = cleaned_text.split('. ')  # 假设句子以"."为结束标记
        conversation_pairs = self.generate_conversation_pairs(lines)

        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in conversation_pairs:
                f.write(pair + '\n')

    def preprocess_directory(self, input_dir, output_dir):
        """批量处理目录中的所有文本文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"pairs_{filename}")
                self.preprocess_file(input_path, output_path)
                print(f"Processed {filename} -> pairs_{filename}")

# 主程序执行逻辑
if __name__ == "__main__":
    api_key = 'YOUR_API_KEY'  # 替换为你的API密钥
    processor = UnifiedProcessor(api_key)

    # 对话处理示例
    input_dir = 'input/transcripts/'
    output_dir = 'output/transcripts/'
    
    # 批量处理文本文件
    processor.process_all_transcripts(input_dir, output_dir)

    # 翻译测试
    translated_text = processor.translate("你好", "en")
    print("翻译结果:", translated_text)

    # 语义对齐测试
    input_sequence = torch.randn(10, 256)  # 10个时间步，特征维度256
    target_sequence = torch.randn(5, 256)  # 5个时间步，特征维度256
    alignments = processor.align_sequences(input_sequence, target_sequence)
    print(alignments)

    # 文本对齐预处理
    processor.preprocess_directory(input_dir, output_dir)
