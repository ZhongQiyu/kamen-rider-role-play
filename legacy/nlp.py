# nlp.py

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
from flask import Flask, request, jsonify
from textblob import TextBlob
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
from sklearn.linear_model import SGDClassifier
import tensorflow as tf
import horovod.tensorflow as hvd
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer


# 初始化 Flask 应用程序
app = Flask(__name__)


# ASR 处理类
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

            # Concatenate and upload audio
            file_keys = [file_key]  # Example only processes a single file; extend for multiple files
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


# NLP 情感分析类
class NlpAgent:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def feedback_loop(self, user_feedback, text):
        print(f"Received user feedback: {user_feedback} for text: {text}")


# 配置 Flask 路由处理函数
@app.route('/agent-a', methods=['POST'])
def agent_a():
    question = request.json['question']
    answer = call_agent_b(question)
    return jsonify({'answer': answer})


@app.route('/agent-b', methods=['POST'])
def agent_b():
    question = request.json['question']
    answer = generate_answer(question)
    return jsonify({'answer': answer})


def call_agent_b(question):
    return "这是对问题的回答"  # 模拟调用 Agent B


def generate_answer(question):
    return "这是生成的答案"  # 模拟生成答案逻辑


# 初始化和配置 TensorFlow 分布式训练
def setup_tensorflow_distributed():
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.optimizers.Adam(0.001 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([1000, 10]), tf.random.uniform([1000], maxval=10, dtype=tf.int32)))
    dataset = dataset.batch(32 * hvd.size())
    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    model.fit(dataset, epochs=5, callbacks=callbacks, verbose=2 if hvd.rank() == 0 else 0)


# 配置 Ray RLlib 强化学习
def setup_ray_rllib():
    ray.init(ignore_reinit_error=True)
    tune.run(
        PPOTrainer,
        config={
            "env": "CartPole-v0",
            "num_workers": 4,
            "framework": "tf",
            "train_batch_size": 4000,
        }
    )


# 配置和运行在线学习模型
def train_online_model():
    model = SGDClassifier()
    X_train, y_train = np.random.randn(100, 10), np.random.randint(0, 2, 100)
    for _ in range(5):
        X_partial, y_partial = np.random.randn(10, 10), np.random.randint(0, 2, 10)
        model.partial_fit(X_partial, y_partial, classes=np.unique(y_train))


# 主程序入口
if __name__ == '__main__':
    # 示例音频处理
    bucket_name = 'kamen-rider-blade-roleplay-sv'
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region_name = 'us-east-2'
    asr_pipeline = ASR(bucket_name, aws_access_key_id, aws_secret_access_key, region_name)
    local_folder = 'C:\\Users\\xiaoy\\Documents\\backup\\audio\\krb'
    s3_folder = 'audio_files'
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
    asr_pipeline.start_aws_workflow(local_folder, s3_folder, mock_event)

    # TensorFlow 分布式训练
    setup_tensorflow_distributed()
    # Ray RLlib 强化学习
    setup_ray_rllib()
    # 启动 Flask 服务器
    app.run(debug=True)

    # 示例文本
    text = "I love this movie. It's amazing!"
    agent = NlpAgent()
    sentiment = agent.analyze_sentiment(text)
    print(f"Sentiment polarity: {sentiment}")

    # 模拟用户反馈
    user_feedback = "positive" if sentiment > 0 else "negative"
    agent.feedback_loop(user_feedback, text)
