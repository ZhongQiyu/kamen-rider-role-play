# parser.py

import json
import os
import re
from pathlib import Path
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
import logging
from fugashi import Tagger as FugashiTagger
import boto3

# 定义 NLP 类用于处理对话和文本格式化
class NLP:
    class Dialogue(BaseModel):
        dialogue_id: str
        utterances: List[str] = Field(default_factory=list)
        created_at: datetime = Field(default=datetime.now)
        participants: List[str] = []

        @property
        def number_of_utterances(self):
            return len(self.utterances)

    def __init__(self):
        self.logger = self.setup_logging()
        self.comprehend = boto3.client('comprehend')

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger(__name__)

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

        # 使用正则表达式处理文本格式
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

# 主程序执行逻辑
if __name__ == "__main__":
    nlp_pipeline = NLP()

    input_dir = 'input/transcripts/'
    output_dir = 'output/transcripts/'

    # 批量处理文本文件
    nlp_pipeline.process_all_transcripts(input_dir, output_dir)
