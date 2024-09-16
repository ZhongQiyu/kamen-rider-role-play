# asr.py

import os
import re
import json
import boto3
import logging
import time
import argparse
from typing import List
from datetime import datetime
from pydub import AudioSegment
from collections import Counter
from pydantic import BaseModel, Field
from fugashi import Tagger as FugashiTagger

class ASR:
    def __init__(self, s3_bucket, aws_access_key_id, aws_secret_access_key, region_name):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.transcribe_client = boto3.client(
            'transcribe',
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

    def upload_folder_to_s3(self, local_folder_path, s3_folder_key):
        for root, dirs, files in os.walk(local_folder_path):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                s3_file_key = os.path.join(s3_folder_key, relative_path)
                self.s3_client.upload_file(local_file_path, self.s3_bucket, s3_file_key)
                self.logger.info(f"File {local_file_path} uploaded to S3 at {s3_file_key}")

    def download_and_concatenate_audio(self, file_keys):
        input_files = []
        for file_key in file_keys:
            local_file_path = '/tmp/' + os.path.basename(file_key)
            self.s3_client.download_file(self.s3_bucket, file_key, local_file_path)
            input_files.append(local_file_path)
        
        output_file = '/tmp/output_audio.mp3'
        ffmpeg.input('concat:' + '|'.join(input_files)).output(output_file).run()
        return output_file

    def concatenate_and_upload_audio(self, files_to_concatenate):
        output_file = self.download_and_concatenate_audio(files_to_concatenate)
        s3_output_key = 'concatenated/concatenated_audio.mp3'
        self.s3_client.upload_file(output_file, self.s3_bucket, s3_output_key)
        self.logger.info(f"Concatenated audio uploaded to S3 at {s3_output_key}")
        return f's3://{self.s3_bucket}/{s3_output_key}'

    def transcribe_audio(self, s3_uri):
        job_name = f'transcription-job-{int(time.time())}'
        self.transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp3',
            LanguageCode='ja-JP'
        )
        
        self.logger.info(f"Transcription job {job_name} started for {s3_uri}")
        
        while True:
            status = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            if job_status in ['COMPLETED', 'FAILED']:
                self.logger.info(f"Transcription job {job_name} completed with status {job_status}")
                break
            self.logger.info(f"Transcription job {job_name} status: {job_status}")
            time.sleep(15)
        
        return status

    def process_event(self, event):
        try:
            # Extract file information from the event
            bucket_name = event['Records'][0]['s3']['bucket']['name']
            file_key = event['Records'][0]['s3']['object']['key']
            
            self.logger.info(f"Processing file: {file_key} from bucket: {bucket_name}")
            
            # If there are multiple files to concatenate, define the file_keys list here
            file_keys = [file_key]  # Example only processes a single file; you can extend to multiple files

            # Concatenate and upload audio
            s3_uri = self.concatenate_and_upload_audio(file_keys)

            # Perform audio transcription
            transcription_status = self.transcribe_audio(s3_uri)
            
            self.logger.info("Audio processing and transcription completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_key}: {e}")
            raise e

        return {
            'statusCode': 200,
            'body': 'Audio processed and transcribed.'
        }

    def start_aws_workflow(self, local_folder, s3_folder, event):
        # Upload local folder to S3
        self.upload_folder_to_s3(local_folder, s3_folder)
        
        # Process S3 event to perform audio transcription
        return self.process_event(event)


if __name__ == "__main__":
    bucket_name = 'kamen-rider-blade-roleplay-sv'
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = 'us-east-2'

    asr_pipeline = ASR(bucket_name, aws_access_key_id, aws_secret_access_key, region_name)
    
    local_folder = 'C:\\Users\\xiaoy\\Documents\\backup\\audio\\krb'
    s3_folder = 'audio_files'
    
    # Simulated event
    mock_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': bucket_name},
                    'object': {'key': 'example_audio_file.mp3'}
                }
            }
        ]
    }
    
    # Start AWS workflow
    asr_pipeline.start_aws_workflow(local_folder, s3_folder, mock_event)

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    # This Lambda function is triggered by file uploads
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        print(f"A new file {object_key} was uploaded in bucket {bucket_name}")

        # Additional logic could be added here, such as updating a GitHub repository or other processing

    return {
        'statusCode': 200,
        'body': json.dumps('Process completed successfully!')
    }
