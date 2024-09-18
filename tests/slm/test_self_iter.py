# bert_ft.py

import os
import gc
import glob
import torch
import fasttext
import numpy as np
import pandas as pd
import psutil
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForNextSentencePrediction

class ModelTrainer:
    def __init__(self, source_path, target_path, log_dir, model_path, bert_checkpoint_path):
        # 初始化路径和日志
        self.source_path = source_path
        self.target_path = target_path
        self.log_dir = log_dir
        self.model_path = model_path
        self.bert_checkpoint_path = bert_checkpoint_path
        os.makedirs(self.target_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.log_dir, 'processing.log'), level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Script started")

        # 初始化 BERT 模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')
        self.device = torch.device("cpu")
        self.model.to(self.device)

        if os.path.exists(self.bert_checkpoint_path):
            self.model.load_state_dict(torch.load(self.bert_checkpoint_path, map_location=self.device))
            logging.info(f"BERT model checkpoint loaded from {self.bert_checkpoint_path}")

    def load_and_merge_data(self):
        # 读取所有标记过的txt文件并合并
        files = glob.glob(os.path.join(self.source_path, '*_labeled.txt'))
        data_frames = []

        for file in files:
            df = pd.read_csv(file, sep='\t', names=['ask_content', 'answer_content', 'related', 'related_classes'])
            basename = os.path.basename(file)
            tieba_name = basename.split('_')[1]
            source = basename.split('_')[0]
            df['tieba_name'] = tieba_name
            df['source'] = source
            new_file_name = os.path.join(self.target_path, basename.replace('.txt', '_n.txt'))
            df.to_csv(new_file_name, index=False)
            data_frames.append(df)

        all_labelled = pd.concat(data_frames, ignore_index=True)
        all_labelled.to_csv(os.path.join(self.target_path, 'all_labelled.csv'), index=False)
        print("文件处理完成，并已保存到指定路径。")

    def train_bert_model(self, file_path):
        # 读取新的数据集并训练 BERT 模型
        df = pd.read_csv(file_path, sep='\t\t', names=['ask_content', 'answer_content'], engine='python')
        logging.info(f"Loaded {len(df)} rows of data")

        dataset = TextPairDataset(df, self.tokenizer, self.device)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        logging.info(f"Dataloader configured with batch size 256")

        self.model.eval()
        related_scores = []

        for i, (batch, ask_batch, answer_batch) in enumerate(tqdm(dataloader, desc="BERT inference")):
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                batch_scores = logits[:, 0].cpu().numpy()
                related_scores.extend(batch_scores)

                batch_df = pd.DataFrame(batch_scores, columns=['related'])
                batch_df['related_classes'] = batch_df['related'].apply(lambda x: 0 if x > 0.7 else 1)
                batch_label_distribution = Counter(batch_df['related_classes'])
                logging.info(f"Batch {i + 1} - BERT Label Distribution: {batch_label_distribution}")
                print(f"Batch {i + 1} - BERT Label Distribution: {batch_label_distribution}")

        torch.save(self.model.state_dict(), self.bert_checkpoint_path)
        logging.info(f"BERT model checkpoint saved to {self.bert_checkpoint_path}")

    def train_fasttext_model(self, data_path, iterations=20):
        # FastText 自我训练迭代
        def monitor_memory():
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss

        def load_data(file_path):
            print(f"Loading data from {file_path}...")
            try:
                df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8', usecols=['ask_content', 'answer_content', 'related_classes'])
                print("Data loaded successfully.")
                return df
            except Exception as e:
                print(f"Failed to load data: {e}")
                return None

        def prepare_fasttext_data(df, output_file):
            print("Preparing data for FastText...")
            df.dropna(subset=['ask_content', 'answer_content', 'related_classes'], inplace=True)
            df['related_classes'] = pd.to_numeric(df['related_classes'], errors='coerce').fillna(0).astype(int)
            with open(output_file, 'w', encoding='utf-8') as file:
                for _, row in df.iterrows():
                    label = f"__label__{int(row['related_classes'])} "
                    content = f"{row['ask_content']} {row['answer_content']}\n"
                    file.write(label + content)
            print(f"Data prepared and written to {output_file}")

        def train_and_evaluate(train_file, model_path):
            model = fasttext.train_supervised(
                input=train_file,
                lr=0.05,
                dim=25,
                epoch=2,
                wordNgrams=1,
                minCount=5,
                bucket=50000,
                loss='softmax',
                verbose=2,
                maxn=0,
                thread=8
            )
            model.save_model(model_path)
            print(f"Model trained and saved to {model_path}")
            return model

        df = load_data(data_path)

        for i in range(iterations):
            mem_before = monitor_memory()
            print(f"Starting iteration {i+1} with initial memory usage: {mem_before} bytes")

            train_file = os.path.join(self.log_dir, f'fasttext_train_{i}.txt')
            prepare_fasttext_data(df, train_file)
            model = train_and_evaluate(train_file, os.path.join(self.model_path, f'fasttext_model_iteration_{i+1}.bin'))

            df['data'] = df['ask_content'].astype(str) + ' ' + df['answer_content'].astype(str)
            predictions = [model.predict(text)[0][0] for text in df['data']]
            df['predicted_label'] = [int(pred.replace('__label__', '')) for pred in predictions]

            df['related_classes'] = df['predicted_label']

            del predictions, model
            gc.collect()

            mem_after = monitor_memory()
            print(f"Iteration {i+1} complete with final memory usage: {mem_after} bytes, difference: {mem_after - mem_before} bytes")

            df.drop(columns=['data', 'predicted_label'], inplace=True)
            gc.collect()

class TextPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, device):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ask_content = self.data.iloc[idx, 0]
        answer_content = self.data.iloc[idx, 1]
        inputs = self.tokenizer(ask_content, answer_content, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs = {key: val.squeeze(0).to(self.device) for key, val in inputs.items()}
        return inputs, ask_content, answer_content

def main():
    trainer = ModelTrainer(
        source_path='../data/ft_data/labelled',
        target_path='../data/ft_data/labelled',
        log_dir='../logs',
        model_path='../data/ft_data/labelled/models',
        bert_checkpoint_path='../data/bert_model_checkpoint.pth'
    )

    trainer.load_and_merge_data()
    trainer.train_bert_model('../data/sample/txt/paper/pc_aiouniya_pc_ft_trunc0.txt')
    trainer.train_fasttext_model('../data/sample/txt/paper/pc_aiouniya_pc_ft_trunc0.txt', iterations=20)

if __name__ == "__main__":
    main()
