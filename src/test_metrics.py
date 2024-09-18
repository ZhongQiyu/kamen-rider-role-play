# test_metrics.py

import os
import re
import json
import csv
import logging
import random
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import MeCab
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datasets import Dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW,
                          BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer,
                          pipeline, set_seed)
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# 文本和图像评估器
class TextMetricsEvaluator:
    def __init__(self):
        self.mecab = MeCab.Tagger("-Owakati")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def mecab_tokenize(self, text):
        return self.mecab.parse(text).strip()

    def calculate_metrics(self, reference, candidate):
        cer_value = cer(reference, candidate)
        bleu_score = sentence_bleu([reference.split()], candidate.split())
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        meteor = meteor_score([reference], candidate)
        embeddings1 = self.sentence_model.encode(reference, convert_to_tensor=True)
        embeddings2 = self.sentence_model.encode(candidate, convert_to_tensor=True)
        semantic_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

        return {
            "CER": cer_value,
            "BLEU": bleu_score,
            "ROUGE": rouge_scores,
            "METEOR": meteor,
            "Semantic Similarity": semantic_similarity
        }


class ImageMetricsEvaluator:
    def __init__(self):
        pass

    def calculate_ssim(self, image_path1, image_path2):
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        ssim_value = cv2.quality.QualitySSIM_compute(image1, image2)[0]
        return ssim_value

    def calculate_image_metrics(self, image_path1, image_path2):
        ssim_value = self.calculate_ssim(image_path1, image_path2)
        return {
            "SSIM": ssim_value
        }


class MetricsEvaluatorFactory:
    @staticmethod
    def get_evaluator(modality):
        if modality == 'text':
            return TextMetricsEvaluator()
        elif modality == 'audio':
            return AudioMetricsEvaluator()
        elif modality == 'image':
            return ImageMetricsEvaluator()
        else:
            raise ValueError("Unsupported modality")


# 数据处理器
class DataProcessor:
    def __init__(self, directory_path, config_file):
        self.directory_path = directory_path
        self.data = []
        self.dialog = []
        self.current_time = None
        self.current_episode = {'episode': 'Unknown', 'dialogs': []}
        self.current_speaker = None
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def parse_train_data(self, data):
        input_key = self.config['train_data']['input_key']
        output_key = self.config['train_data']['output_key']
        parsed_data = [{'input': entry[input_key], 'ideal_response': entry[output_key]} for entry in data]
        return parsed_data

    def parse_asr_data(self, data):
        text_key = self.config['asr_data']['text_key']
        parsed_data = [{'text': entry[text_key]} for entry in data]
        return parsed_data

    def process_line(self, line):
        speaker_match = re.match(r'^話者(\d+)\s+(\d{2}:\d{2})\s+(.*)$', line)  # 日语版本的正则表达式
        if speaker_match:
            if self.dialog:  # 如果有未完成的对话，先完成它
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.current_speaker, self.current_time, text = speaker_match.groups()
            self.dialog = [text]
        else:
            self.dialog.append(line)

    def process_all_files(self):
        files = [f for f in os.listdir(self.directory_path) if f.endswith('.txt')]
        files = sorted(files, key=self.sort_files)
        for filename in files:
            file_path = os.path.join(self.directory_path, filename)
            self.process_file(file_path)

    def finalize_episode(self):
        if self.current_episode:
            if self.dialog:
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.data.append(self.current_episode)
            print(f"Finalized episode: {self.current_episode}")
            self.current_episode = {'episode': 'Unknown', 'dialogs': []}

    def export_to_txt(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for content in self.data:
                file.write(json.dumps(content, ensure_ascii=False) + '\n')

    def save_as_json(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)

    def save_as_csv(self, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['episode', 'time', 'speaker', 'text']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for episode in self.data:
                if 'dialogs' not in episode:
                    continue
                for dialog in episode['dialogs']:
                    writer.writerow({
                        'episode': episode['episode'],
                        'time': dialog['time'],
                        'speaker': dialog['speaker'],
                        'text': dialog['text']
                    })

    @staticmethod
    def sort_files(filename):
        part = filename.split('.')[0]
        try:
            return int(part)
        except ValueError:
            return float('inf')


# 多模态处理器
class MultiModalProcessor:
    def __init__(self, config_file, device='cpu'):
        self.config = self.load_config(config_file)
        self.device = device
        self.initialize_models()
        set_seed(42)  # 设置随机种子，以确保文本生成的可重复性
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def initialize_models(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.config.get('tokenizer_model', 'bert-base-multilingual-cased'))
            self.classification_model = BertForSequenceClassification.from_pretrained(
                self.config.get('classification_model', 'bert-base-multilingual-cased'),
                num_labels=self.config.get('num_labels', 2)
            ).to(self.device)
            
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def train_seq2seq_model(self, model_name, train_data, lr=5e-5, num_epochs=3):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = AdamW(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            random.shuffle(train_data)
            total_loss = 0
            total_reward = 0
            for data in train_data:
                input_text = data['input']
                ideal_response = data['ideal_response']
                loss, reward = self.train_step(model, tokenizer, optimizer, input_text, ideal_response)
                total_loss += loss
                total_reward += reward
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}, Reward: {total_reward/len(train_data)}")

        return model, tokenizer

    def main(self, args):
        if args.command == 'train':
            self.train_model(args)
        elif args.command == 'process':
            self.process_data(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Processing Script")

    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    train_parser.add_argument('--train_data_file', type=str, required=True, help="Path to the training data file")
    train_parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    train_parser.add_argument('--model_type', type=str, required=True, choices=['seq2seq', 'classification'], help="Type of the model")
    train_parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")

    process_parser = subparsers.add_parser('process')
    process_parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    process_parser.add_argument('--asr_data_file', type=str, required=True, help="Path to the ASR data file")

    args = parser.parse_args()

    processor = MultiModalProcessor(args.config_file)
    processor.main(args)
