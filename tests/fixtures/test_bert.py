# test_torch.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForNextSentencePrediction
import re
import ast  # 用于安全地解析字符串为列表
import boto3
import logging
import time
from pydub import AudioSegment
from typing import List


class BERTModelProcessor:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练的BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        
        # 启用FP16
        self.model.half()
        
        # 使用多GPU支持
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)
        self.model.eval()

    def clean_content(self, content):
        # 去除 "回复 xxx :" 的前缀
        return re.sub(r'^回复\s.*?:', '', content).strip()

    def process_batch(self, batch_data):
        related_labels = []
        topk_weighted = []
        for ask_content, answer_content, topk in batch_data:
            # 清理内容
            ask_content_cleaned = self.clean_content(ask_content)
            answer_content_cleaned = self.clean_content(answer_content)
            
            encodings = self.tokenizer(ask_content_cleaned, answer_content_cleaned, truncation=True, padding=True, return_tensors="pt").to(self.device)
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**encodings)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    next_sentence_score = probs[0][0].item()  # 获取NSP相关性的得分
                    
            # 根据阈值判断相关性（高于0.5为0，低于0.5为1）
            label = 0 if next_sentence_score > 0.5 else 1
            related_labels.append(label)

            # 计算加权后的topk
            try:
                topk_values = ast.literal_eval(topk) if isinstance(topk, str) else topk  # 安全地解析字符串为列表
                topk_values = [float(value) for value in topk_values]  # 确保所有值都是浮点数
            except (ValueError, SyntaxError):
                topk_values = []  # 如果解析失败，将topk设为空列表
            
            if topk_values:
                weighted_score = sum([score * next_sentence_score for score in topk_values]) / sum(topk_values)
                topk_weighted.append(weighted_score)
            else:
                topk_weighted.append(0.0)  # 如果topk为空，则默认加权得分为0

        return related_labels, topk_weighted

    def process_file(self, input_file, output_file, batch_size=2):
        data = pd.read_csv(input_file, sep='\t', names=["ask", "ask_content", "answer", "answer_content", "topk"])
        total_rows = len(data)
        related_results = []
        topk_weighted_results = []

        for i in tqdm(range(0, total_rows, batch_size)):
            batch_data = data.iloc[i:i + batch_size]
            ask_answer_pairs = batch_data[['ask_content', 'answer_content', 'topk']].values
            batch_related, batch_topk_weighted = self.process_batch(ask_answer_pairs)
            related_results.extend(batch_related)
            topk_weighted_results.extend(batch_topk_weighted)

        # 将计算的related和topk_weighted列添加到原始数据中
        data['related'] = related_results
        data['topk_weighted'] = topk_weighted_results

        # 保存包含新列的结果到新的文件中
        data.to_csv(output_file, sep='\t', index=False)


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
    # 主程序：处理BERT文件并保存结果
    input_file = 'C:\\Users\\xiaoy\\Documents\\Yuki\\pc.txt'
    output_file = 'C:\\Users\\xiaoy\\Documents\\Yuki\\fasttext_pc_bert.txt'

    bert_processor = BERTModelProcessor(model_name='bert-base-chinese')
    bert_processor.process_file(input_file, output_file, batch_size=2)

    # AWS ASR 处理
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
