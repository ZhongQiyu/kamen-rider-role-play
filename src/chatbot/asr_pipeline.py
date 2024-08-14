# asr_pipeline.py

import os
import re
import json
import boto3
import logging
import argparse
from typing import List
from datetime import datetime
from pydub import AudioSegment
from collections import Counter
from pydantic import BaseModel, Field
from fugashi import Tagger as FugashiTagger

class ASRPipeline:
    def __init__(self, s3_bucket, aws_access_key_id, aws_secret_access_key, region_name):
        self.s3_bucket = s3_bucket
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

if __name__ == "__main__":
    args = ASRPipeline.parse_arguments()
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = 'us-east-2'
    s3_bucket = 'kamen-rider-blade-roleplay-sv'

    asr_pipeline = ASRPipeline(s3_bucket, aws_access_key_id, aws_secret_access_key, region_name)
    asr_pipeline.run_commands(args)
