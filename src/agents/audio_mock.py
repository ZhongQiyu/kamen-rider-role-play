import torch
import os
from torch.multiprocessing import Pool
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import numpy as np

def process_file(wav_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to('cuda')
    
    # 加载语音文件
    signal, fs = torchaudio.load(wav_path)
    
    # 重采样到 16kHz
    if fs != 16000:
        transform = torchaudio.transforms.Resample(fs, 16000)
        signal = transform(signal)
    
    # 提取特征
    input_values = processor(signal.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values.to('cuda')
    with torch.no_grad():
        features = model(input_values).last_hidden_state.mean(dim=1)
    
    return features.cpu().squeeze().numpy()

def generate_voice_model_parallel(wav_folder, output_dir, num_workers=4):
    """
    使用多线程/多卡并行方式生成每个角色的声模。
    
    Args:
        wav_folder (str): 存放角色语音 .wav 文件的文件夹，每个角色一个子文件夹。
        output_dir (str): 保存生成的声模的目录。
        num_workers (int): 使用的并行工作进程数（线程或 GPU 卡数）。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历角色文件夹
    for character_folder in os.listdir(wav_folder):
        character_path = os.path.join(wav_folder, character_folder)
        
        if os.path.isdir(character_path):
            print(f"处理角色: {character_folder}")
            wav_files = [os.path.join(character_path, wav) for wav in os.listdir(character_path) if wav.endswith(".wav")]
            
            # 使用多线程/多卡并行处理音频文件
            with Pool(num_workers) as pool:
                embeddings_list = pool.map(process_file, wav_files)
            
            # 保存角色的声模
            if embeddings_list:
                character_model = sum(embeddings_list) / len(embeddings_list)
                model_path = os.path.join(output_dir, f"{character_folder}_voice_model.npy")
                np.save(model_path, character_model)
                print(f"保存声模到: {model_path}")

if __name__ == "__main__":
    wav_directory = "/path/to/tokusatsu/wav_files"  # 每个角色一个文件夹
    model_output_directory = "/path/to/output/models"
    
    # 使用多线程/多卡生成声模
    generate_voice_model_parallel(wav_directory, model_output_directory, num_workers=4)



import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier

def generate_voice_model(wav_folder, output_dir):
    """
    生成每个角色的声模（Voice Model）。
    
    Args:
        wav_folder (str): 存放角色语音 .wav 文件的文件夹，每个角色一个子文件夹。
        output_dir (str): 保存生成的声模的目录。
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载预训练的声纹模型
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

    # 遍历每个角色的文件夹
    for character_folder in os.listdir(wav_folder):
        character_path = os.path.join(wav_folder, character_folder)
        
        if os.path.isdir(character_path):
            print(f"处理角色: {character_folder}")
            
            embeddings_list = []
            
            # 遍历角色的所有 .wav 文件
            for wav_file in os.listdir(character_path):
                if wav_file.endswith(".wav"):
                    wav_path = os.path.join(character_path, wav_file)
                    # 加载语音文件
                    signal, fs = torchaudio.load(wav_path)
                    
                    # 提取嵌入（声学特征）
                    embeddings = classifier.encode_batch(signal)
                    
                    # 保存嵌入
                    embeddings_list.append(embeddings.squeeze().detach().numpy())
                    
            # 保存角色的声模（多个嵌入的平均值作为声模）
            if embeddings_list:
                character_model = sum(embeddings_list) / len(embeddings_list)
                model_path = os.path.join(output_dir, f"{character_folder}_voice_model.npy")
                np.save(model_path, character_model)
                print(f"保存声模到: {model_path}")
    
    print("所有角色的声模生成完成！")

if __name__ == "__main__":
    # 语音文件存放的主文件夹
    wav_directory = "/path/to/tokusatsu/wav_files"  # 每个角色一个文件夹
    # 保存声模的输出目录
    model_output_directory = "/path/to/output/models"
    
    # 开始生成声模
    generate_voice_model(wav_directory, model_output_directory)



