# asr_pipeline.py

import os
import re
import logging
import boto3
from pydub import AudioSegment
from datetime import datetime
from collections import Counter
from typing import List
from pydantic import BaseModel, Field
from fugashi import Tagger as FugashiTagger
import argparse

class ASRPipeline:
    class Dialogue(BaseModel):
        dialogue_id: str
        utterances: List[str] = Field(default_factory=list)
        created_at: datetime = Field(default=datetime.now)
        participants: List[str] = []

        @property
        def number_of_utterances(self):
            return len(self.utterances)

    def __init__(self, s3_bucket, aws_access_key_id, aws_secret_access_key, region_name, config_file="model_config.json"):
        self.s3_bucket = s3_bucket
        self.config_file = config_file
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def run_commands(self, args):
        if args.download:
            s3_folder, local_folder = args.download
            self.download_folder_from_s3(s3_folder, local_folder)
        if args.convert:
            input_folder, output_folder = args.convert
            self.convert_audio(input_folder, output_folder)

    def download_folder_from_s3(self, s3_folder, local_folder):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_folder):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                file_name = os.path.basename(s3_key)
                local_path = os.path.join(local_folder, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                self.logger.info(f"Downloaded {s3_key} from S3 to {local_path}")

    def convert_audio(self, input_folder, output_folder, output_format='wav'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for filename in os.listdir(input_folder):
            if filename.endswith(".m4a"):
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.{output_format}")
                if not os.path.exists(output_file_path):
                    try:
                        audio = AudioSegment.from_file(input_file_path, format="m4a")
                        audio.export(output_file_path, format=output_format)
                        self.logger.info(f"Converted {filename} to {output_file_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to convert {filename}: {str(e)}")

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Manage AWS S3 audio files and handle dialogues for ASR.")
        parser.add_argument('--download', nargs=2, metavar=('S3_FOLDER', 'LOCAL_FOLDER'),
                            help='Download all files from an S3 folder to a local folder')
        parser.add_argument('--convert', nargs=2, metavar=('INPUT_FOLDER', 'OUTPUT_FOLDER'),
                            help='Convert all audio files in a folder to a different format')
        return parser.parse_args()

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

if __name__ == "__main__":
    args = ASRPipeline.parse_arguments()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = 'us-east-2'
    s3_bucket = 'kamen-rider-blade-roleplay-sv'
    
    pipeline = ASRPipeline(s3_bucket, aws_access_key_id, aws_secret_access_key, region_name)
    pipeline.run_commands(args)

    dialogues = [
        "今日は天気がいいから、公園に行きましょう。",
        "明日は雨が降るそうです。",
        "週末に映画を見に行く予定です。",
        "最近、仕事が忙しいですね。"
    ]
    
    frequencies = pipeline.analyze_frequency(dialogues)
    pipeline.logger.info(f"Top 10 word frequencies: {frequencies}")

    sample_dialogue = pipeline.Dialogue(
        dialogue_id="dlg_1001",
        utterances=["今日は天気がいいから、公園に行きましょう。", "週末に映画を見に行く予定です。"],
        participants=["User1", "User2"]
    )
