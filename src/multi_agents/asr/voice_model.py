# voice_model.py

import os
import time
import logging
import boto3
from pydub import AudioSegment
import torch
from torch.multiprocessing import Pool
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import numpy as np
from speechbrain.pretrained import EncoderClassifier


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


# 使用Wav2Vec2生成角色声模
def process_wav_file_wav2vec2(wav_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to('cuda')

    # 加载语音文件
    signal, fs = torchaudio.load(wav_path)

    # 重采样到16kHz
    if fs != 16000:
        transform = torchaudio.transforms.Resample(fs, 16000)
        signal = transform(signal)

    # 提取特征
    input_values = processor(signal.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to('cuda')
    with torch.no_grad():
        features = model(input_values).last_hidden_state.mean(dim=1)

    return features.cpu().squeeze().numpy()


# 使用SpeechBrain生成角色声模
def process_wav_file_speechbrain(wav_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

    # 加载语音文件
    signal, fs = torchaudio.load(wav_path)

    # 提取嵌入（声学特征）
    embeddings = classifier.encode_batch(signal)

    return embeddings.squeeze().detach().numpy()


# 使用多线程/多卡并行生成角色声模
def generate_voice_model_parallel(wav_folder, output_dir, num_workers=4, method="wav2vec2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历角色文件夹
    for character_folder in os.listdir(wav_folder):
        character_path = os.path.join(wav_folder, character_folder)

        if os.path.isdir(character_path):
            print(f"处理角色: {character_folder}")
            wav_files = [os.path.join(character_path, wav) for wav in os.listdir(character_path) if wav.endswith(".wav")]

            # 选择使用的方法
            process_func = process_wav_file_wav2vec2 if method == "wav2vec2" else process_wav_file_speechbrain

            # 使用多线程/多卡并行处理音频文件
            with Pool(num_workers) as pool:
                embeddings_list = pool.map(process_func, wav_files)

            # 保存角色的声模
            if embeddings_list:
                character_model = sum(embeddings_list) / len(embeddings_list)
                model_path = os.path.join(output_dir, f"{character_folder}_voice_model.npy")
                np.save(model_path, character_model)
                print(f"保存声模到: {model_path}")


if __name__ == "__main__":
    # 语音文件存放的主文件夹
    wav_directory = "/path/to/tokusatsu/wav_files"  # 每个角色一个文件夹
    # 保存声模的输出目录
    model_output_directory = "/path/to/output/models"

    # 使用多线程/多卡生成声模，选择使用Wav2Vec2或SpeechBrain方法
    generate_voice_model_parallel(wav_directory, model_output_directory, num_workers=4, method="wav2vec2")
    generate_voice_model_parallel(wav_directory, model_output_directory, num_workers=4, method="speechbrain")
