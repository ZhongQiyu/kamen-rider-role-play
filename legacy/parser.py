# parser.py

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List
from datetime import datetime
from collections import Counter
from pydantic import BaseModel, Field
import logging
from fugashi import Tagger as FugashiTagger
import boto3

class Parser:
    def __init__(self, project_structure_path, dialogue_data_path, command=None, nodes_array=None, head_node_ip=None, port=None, video_dir=None, split_size_mb=None):
        self.project_structure_path = Path(project_structure_path)
        self.dialogue_data_path = Path(dialogue_data_path)
        self.project_structure = self.load_json(self.project_structure_path)
        self.dialogues = self.load_json(self.dialogue_data_path)
        self.command = command
        self.nodes_array = nodes_array
        self.head_node_ip = head_node_ip
        self.port = port
        self.video_dir = video_dir
        self.split_size = split_size_mb * 1024 * 1024 if split_size_mb else None

    def load_json(self, path):
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)

    def save_json(self, data, path):
        with path.open('w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def run_command(self, command):
        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Command '{command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command '{command}'. Error: {e}")

    def manage_cluster(self, action):
        if action == "start":
            print(f"Starting Ray head node on {self.head_node_ip}...")
            subprocess.Popen([
                'ray', 'start', '--head',
                f'--node-ip-address={self.head_node_ip}',
                f'--port={self.port}', '--block'
            ])

            for node in self.nodes_array[1:]:
                print(f"Starting Ray worker on {node}...")
                subprocess.Popen([
                    'ray', 'start', f'--address={self.head_node_ip}:{self.port}', '--block'
                ])
            time.sleep(30)  # Wait for nodes to start
            print("Ray cluster started.")
        elif action == "stop":
            print("Stopping Ray cluster...")
            subprocess.run(['ray', 'stop'], check=True)

    def copy_files(self, src_path, dest_path, file_extension):
        print(f"Copying files from {src_path} to {dest_path}...")
        os.makedirs(dest_path, exist_ok=True)
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith(file_extension):
                    shutil.copy(os.path.join(root, file), dest_path)
        print("All files have been copied.")

    def split_large_files(self):
        print(f"Splitting large files in {self.video_dir}...")
        for file_path in Path(self.video_dir).glob("*.mov"):
            file_size = file_path.stat().st_size
            if file_size > self.split_size:
                print(f"Splitting file: {file_path}")
                subprocess.run([
                    'split', '-b', f"{self.split_size}m", str(file_path),
                    f"{file_path.stem}_part_"
                ], check=True)
            else:
                print(f"File {file_path} is smaller than the split size threshold, skipping.")
        print("All file splits completed.")

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
        self.rekognition = boto3.client('rekognition')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger(__name__)

    def handle_dialog_from_file(self, file_path):
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

    def tokenize_japanese(self, text: str) -> List[str]:
        tagger = FugashiTagger('-Owakati')
        return tagger.parse(text).strip().split()

    def analyze_frequency(self, dialogues: List[str]) -> List[tuple]:
        word_counts = Counter()
        for dialogue in dialogues:
            words = self.tokenize_japanese(dialogue)
            word_counts.update(words)
        return word_counts.most_common(10)

    def process_text_file(self, file_path, output_dir):
        dialogues = self.handle_dialog_from_file(file_path)

        final_output = {
            "title": os.path.basename(file_path).replace('.txt', ''),
            "dialogues": dialogues
        }

        output_file_path = os.path.join(output_dir, f"{final_output['title']}.json")
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(final_output, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Data saved to {output_file_path}")

    def process_all_files(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for txt_file in os.listdir(input_dir):
            if txt_file.endswith('.txt'):
                file_path = os.path.join(input_dir, txt_file)
                self.process_text_file(file_path, output_dir)

    def save_episode_dialogues(self, episode, output_dir):
        output_file = os.path.join(output_dir, f"{episode['title']}.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            for dialogue in episode['dialogues']:
                f.write(f"{dialogue['speaker']} {dialogue['time']}: {dialogue['text']}\n\n")
        
        self.logger.info(f"Episode {episode['title']} saved to {output_file}")

    def process_dialogues_json(self, file_path, output_dir):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for episode in data[:49]:  # Process the first 49 episodes
                self.save_episode_dialogues(episode, output_dir)

    def process_transcript(self, file_path, output_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 消除 None None
        content = content.replace("None None", "").strip()

        # 利用正则表达式进行拆分，找到说话人的信息并保留它
        pattern = r'(说话人\d+ \d{2}:\d{2})'
        split_content = re.split(pattern, content)

        # 组合成所需格式的每一行
        result = []
        for i in range(1, len(split_content), 2):
            speaker_and_time = split_content[i].strip()
            dialogue = split_content[i + 1].strip()
            if dialogue:
                result.append(f"{speaker_and_time} {dialogue}")

        # 将处理后的结果写入新文件
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n".join(result))

    def process_all_transcripts(self, input_folder, output_folder):
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 遍历输入文件夹中的所有 .txt 文件
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                input_file = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, filename)

                self.logger.info(f"Processing file: {filename}")
                self.process_transcript(input_file, output_file)
                self.logger.info(f"Processed file saved to: {output_file}")

    # 使用AWS Comprehend进行情感分析
    def analyze_sentiment(self, text):
        response = self.comprehend.detect_sentiment(
            Text=text,
            LanguageCode='ja'
        )
        self.logger.info(f"文本情感分析结果: {response}")
        return response

    # 使用AWS Comprehend进行关键短语提取
    def extract_key_phrases(self, text):
        response = self.comprehend.detect_key_phrases(
            Text=text,
            LanguageCode='ja'
        )
        self.logger.info(f"关键短语提取结果: {response}")
        return response

    # 使用AWS Comprehend进行实体识别
    def identify_entities(self, text):
        response = self.comprehend.detect_entities(
            Text=text,
            LanguageCode='ja'
        )
        self.logger.info(f"实体识别结果: {response}")
        return response

    # 使用AWS Comprehend进行语言检测
    def detect_language(self, text):
        response = self.comprehend.detect_dominant_language(
            Text=text
        )
        self.logger.info(f"语言检测结果: {response}")
        return response

    # 使用AWS SageMaker进行推理
    def invoke_sagemaker_model(self, endpoint_name, payload):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        result = response['Body'].read().decode('utf-8')
        self.logger.info(f"SageMaker模型推理结果: {result}")
        return result

    # 使用AWS Rekognition进行图像分析
    def analyze_image_with_rekognition(self, image_bytes):
        response = self.rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=80
        )
        self.logger.info(f"图像标签分析结果: {response}")
        return response

def main():
    # Initialize ProgramManager and NLP
    manager = Parser('path/to/project_structure.json', 'path/to/dialogues.json', nodes_array=["node1", "node2"], head_node_ip="192.168.0.1", port=6379, video_dir="data/video", split_size_mb=50)
    nlp_pipeline = NLP()

    # Example: Manage Ray cluster
    manager.manage_cluster("start")
    manager.run_command("your_command_here")
    manager.copy_files("/source/path", "/destination/path", ".pdf")
    manager.split_large_files()
    manager.save_json(manager.dialogues, manager.dialogue_data_path)

    # Example: Analyze frequency of words in dialogues
    dialogues = [
        "今日は天気がいいから、公園に行きましょう。",
        "明日は雨が降るそうです。",
        "週末に映画を見に行く予定です。",
        "最近、仕事が忙しいですね。"
    ]
    frequencies = nlp_pipeline.analyze_frequency(dialogues)
    nlp_pipeline.logger.info(f"Top 10 word frequencies: {frequencies}")

    # AWS Comprehend example
    sentiment_result = nlp_pipeline.analyze_sentiment("これは素晴らしい映画です")

    # Directories for processing
    input_dir = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\text\\txt'
    output_dir_json = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\text\\json'
    output_dir_episodes = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\text\\episodes'
    output_dir_trimmed = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\text\\txt\\trimmed'

    # Process text files to JSON
    nlp_pipeline.process_all_files(input_dir, output_dir_json)

    # Convert dialogues JSON to episode text files
    dialogues_json_path = os.path.join(output_dir_json, 'dialogues.json')
    nlp_pipeline.process_dialogues_json(dialogues_json_path, output_dir_episodes)

    # Process and trim transcripts
    nlp_pipeline.process_all_transcripts(input_dir, output_dir_trimmed)

if __name__ == "__main__":
    main()
