# bert_ft_all.py

import gc
import os
import sys
import json
import mmap
import dask
import jieba
import torch
import psutil
import fasttext
import platform
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dask import delayed
from datetime import datetime
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor

# 自动选择调度器和线程数
if platform.system() == 'Windows':
    dask.config.set(scheduler='processes', num_workers=os.cpu_count())
else:
    dask.config.set(scheduler='threads', num_workers=os.cpu_count())

def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def clean_up_memory(*args):
    for arg in args:
        del arg
    gc.collect()

def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            os.chmod(path, 0o777)  # 设置目录权限为所有用户可读、写和执行
            log_info(f"Directory {path} created successfully with 777 permissions.")
        except Exception as e:
            log_info(f"Failed to create directory {path}: {e}")
            return False
    elif not os.access(path, os.W_OK):
        try:
            os.chmod(path, 0o777)  # 尝试修改现有目录的权限
            log_info(f"Directory {path} permissions changed to 777.")
        except Exception as e:
            log_info(f"Failed to set permissions for {path}: {e}")
            return False
    return True

def monitor_memory():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss
    log_info(f"Current memory usage: {mem_usage / (1024**2):.2f} MB")
    return mem_usage

def apply_hyper_resolution(text_chunk):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_tensor = torch.tensor([ord(c) for c in text_chunk], dtype=torch.int32).to(device)
    enhanced_text_tensor = torch.where((text_tensor >= ord('a')) & (text_tensor <= ord('z')), 
                                       text_tensor - ord('a') + ord('A'), text_tensor)
    enhanced_text = ''.join(chr(c) for c in enhanced_text_tensor.cpu().numpy())
    return enhanced_text

def clean_data(df):
    log_info("Cleaning data...")
    df = df.dropna(subset=['ask_content', 'answer_content', 'related'])
    df['related'] = pd.to_numeric(df['related'], errors='coerce').fillna(0)
    log_info("Data cleaned successfully.")
    return df

def save_training_state(state_file, state_data):
    with open(state_file, 'w') as f:
        json.dump(state_data, f)
    log_info(f"Training state saved to {state_file}")

def load_training_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state_data = json.load(f)
        log_info(f"Training state loaded from {state_file}")
        return state_data
    else:
        return {'iteration': 0, 'model_path': None}

def prepare_data_for_fasttext(ddf, output_file):
    log_info("Preparing data for FastText...")
    ensure_directory_exists(os.path.dirname(output_file))

    with ProgressBar():
        for i, partition in enumerate(ddf.to_delayed()):
            partition = partition.compute()
            partition = clean_data(partition)

            log_info("Starting parallel jieba segmentation...")
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                partition['ask_content'] = list(executor.map(jieba_segmentation, partition['ask_content']))
                partition['answer_content'] = list(executor.map(jieba_segmentation, partition['answer_content']))
            log_info("Finished parallel jieba segmentation.")

            partition = partition[partition['related'].notna()]
            partition = partition[partition['ask_content'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
            partition = partition[partition['answer_content'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]

            partition['formatted'] = partition.apply(
                lambda x: f"__label__{int(round(x['related'] * 10))} {x['ask_content']} {x['answer_content']}",
                axis=1
            )

            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for line in partition['formatted']:
                        f.write(line + '\n')
                log_info(f"Processed partition {i + 1} and written to {output_file}")

                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        first_line = lines[0].strip()
                        log_info(f"First line of batch {i + 1}: {first_line}")

            except PermissionError as e:
                log_info(f"Permission denied while writing to {output_file}: {e}")
                return

            clean_up_memory(partition)
            gc.collect()

    log_info(f"Data prepared and written to {output_file}")

def jieba_segmentation(text):
    return '^'.join(jieba.cut(text))

def process_and_predict(row, model, delimiter='^'):
    if pd.isna(row['ask_content']) or pd.isna(row['answer_content']):
        return -1
    segmented_text = delimiter.join(jieba.cut(str(row['ask_content']))) + ' ' + delimiter.join(jieba.cut(str(row['answer_content'])))
    prediction = int(model.predict(segmented_text)[0][0].replace('__label__', ''))
    return prediction / 10.0

def apply_predictions(ddf, model_path, delimiter='^'):
    def predict_partition(df, model_path):
        if model_path is None or not os.path.exists(model_path):
            log_info(f"Model file does not exist or model path is None at {model_path}")
            return df

        try:
            model = fasttext.load_model(model_path)
        except Exception as e:
            log_info(f"Error loading model: {e}")
            return df

        texts = df.apply(lambda x: delimiter.join(jieba.cut(str(x['ask_content']))) + ' ' + 
                         delimiter.join(jieba.cut(str(x['answer_content']))), axis=1).tolist()
        
        batch_predictions = model.predict(texts)  # 使用FastText的批量预测
        
        df['related'] = [int(pred[0].replace('__label__', '')) / 10.0 for pred in batch_predictions[0]]
        log_info(f"Sample data after prediction: {df.head()}")

        del model
        gc.collect()  
        return df

    meta = ddf.head(0).copy()
    meta['related'] = 0.0
    ddf = ddf.map_partitions(predict_partition, model_path=model_path, meta=meta)
    
    return ddf

def load_data_with_dask(file_path):
    log_info(f"Loading data from {file_path} using Dask...")
    block_size = "512KB"
    ddf = dd.read_csv(file_path, delimiter='\t', assume_missing=True, blocksize=block_size)
    log_info("Data loaded successfully using Dask.")
    return ddf

def train_incremental_with_fasttext(file_path, model_path, existing_model=None):
    log_info(f"Training using data from: {file_path}")
    log_info(f"Model path: {model_path}")

    model_dir = os.path.dirname(model_path)
    if not ensure_directory_exists(model_dir):
        log_info(f"Model directory {model_dir} does not exist or is not writable.")
        return None

    try:
        model = fasttext.train_supervised(
            input=file_path,
            lr=0.01,
            dim=5,
            epoch=1,
            wordNgrams=1,
            minCount=10,
            bucket=10000,
            loss='softmax',
            thread=os.cpu_count()
        )

        model.save_model(model_path)
        log_info(f"Model trained and saved to {model_path}")

        return model_path

    except Exception as e:
        log_info(f"Error during model training with fasttext: {e}")
        return None

def plot_label_distribution(df, iteration, log_dir):
    ensure_directory_exists(log_dir)

    if 'dask' in str(type(df)):
        df = df.compute()

    if 'related' not in df.columns or not pd.api.types.is_numeric_dtype(df['related']):
        log_info("Column 'related' is missing or not numeric.")
        return

    bins = np.linspace(0, 1, 11)

    df['related_binned'] = pd.cut(
        df['related'], bins, include_lowest=True,
        labels=[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    )

    try:
        label_counts = df['related_binned'].value_counts(sort=False)
        label_counts.index = label_counts.index.astype(str)
        log_info(f"label_counts type: {type(label_counts)}, head: {label_counts.head()}")
    except Exception as e:
        log_info(f"Error during value_counts computation: {e}")
        return

    if not isinstance(label_counts, pd.Series):
        log_info(f"label_counts is not a valid Pandas Series: {type(label_counts)}")
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    try:
        label_counts.plot(kind='bar', color=colors)
    except Exception as e:
        log_info(f"Error during plotting: {e}")
        return

    plt.title(f'Label Distribution at Iteration {iteration}')
    plt.xlabel('Related Classes (Binned)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.tight_layout()

    for index, value in enumerate(label_counts):
        plt.text(index, value, str(value), ha='center', va='bottom')

    save_path = os.path.join(log_dir, f'label_distribution_{iteration}.png')
    
    try:
        plt.savefig(save_path)
        log_info(f"Plot saved for iteration {iteration}.")
    except Exception as e:
        log_info(f"Error saving plot: {e}")
    finally:
        plt.close()

def self_train_iterations(data_path, model_path, log_dir, iterations=1):  # 设置迭代次数为1
    ensure_directory_exists(log_dir)
    state_file = os.path.join(log_dir, 'training_state.json')

    state_data = load_training_state(state_file)
    start_iteration = state_data.get('iteration', 0)
    model_incremental_path = state_data.get('model_path', None)

    ddf = load_data_with_dask(data_path)
    ddf = ddf.persist()

    if start_iteration == 0:
        plot_label_distribution(ddf.compute(), 0, log_dir)

    for i in range(start_iteration, iterations):
        log_info(f"\n--- Iteration {i+1} ---")
        
        mem_before = monitor_memory()

        train_file = os.path.join(log_dir, f'fasttext_train_{i}.txt')
        prepare_data_for_fasttext(ddf, train_file)

        with ProgressBar():
            trained_model_path = delayed(train_incremental_with_fasttext)(
                train_file,
                os.path.join(model_path, f'fasttext_model_iteration_{i+1}.bin'),
                model_incremental_path
            )
            model_incremental_path = trained_model_path.compute()

        if model_incremental_path is None:
            log_info(f"Training failed for iteration {i+1}.")
            break

        state_data['iteration'] = i + 1
        state_data['model_path'] = model_incremental_path
        save_training_state(state_file, state_data)

        ddf = apply_predictions(ddf, model_path=model_incremental_path)
        df = ddf.compute()

        plot_label_distribution(df, i + 1, log_dir)

        mem_after = monitor_memory()
        log_info(f"Iteration {i+1} complete. Memory usage difference: {(mem_after - mem_before) / (1024**2):.2f} MB")

        gc.collect()

def main():
    data_path = 'all_labelled.txt'
    model_path = 'models'
    log_dir = 'logs'

    if not ensure_directory_exists(model_path):
        log_info("Model directory is not properly set.")
        return

    if not ensure_directory_exists(log_dir):
        log_info("Log directory is not properly set.")
        return

    iterations = 1
    self_train_iterations(data_path, model_path, log_dir, iterations)

if __name__ == "__main__":
    main()
