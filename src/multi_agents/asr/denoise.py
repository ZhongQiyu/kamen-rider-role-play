# denoise.py

import os
import subprocess
import glob

# 定义输入、中间和输出文件夹
input_folder = r"C:\Users\xiaoy\Downloads\wav"
intermediate_folder = r"C:\Users\xiaoy\Downloads\pcm"
output_folder = r"C:\Users\xiaoy\Downloads\m4a"

# 确保中间和输出文件夹存在
os.makedirs(intermediate_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 检查是否有音频文件
audio_files = glob.glob(os.path.join(input_folder, '*.[mM][4aA]')) + glob.glob(os.path.join(input_folder, '*.mp3'))

if len(audio_files) == 0:
    print(f"No audio files found in {input_folder}")
    exit(1)

# 绝对路径的 rnnoise_demo
rnnoise_path = r"/absolute/path/to/rnnoise_demo"

for input_file in audio_files:
    # 提取文件扩展名和基本名称
    base_name, extension = os.path.splitext(os.path.basename(input_file))
    extension = extension.lower().strip('.')

    print(f"Processing {input_file} with extension {extension}")

    # 转换到 PCM 格式
    pcm_path = os.path.join(intermediate_folder, f"{base_name}.pcm")
    try:
        subprocess.run(['ffmpeg', '-i', input_file, '-f', 's16le', '-acodec', 'pcm_s16le', pcm_path], check=True)
        print(f"Converted to PCM: {pcm_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to convert {input_file} to PCM")
        continue

    # 应用 rnnoise 降噪
    denoised_pcm_path = os.path.join(intermediate_folder, f"{base_name}_denoised.pcm")
    if os.path.isfile(pcm_path):
        subprocess.run([rnnoise_path, pcm_path, denoised_pcm_path], check=True)
        print(f"Noise reduction applied: {denoised_pcm_path}")

    # 将降噪后的 PCM 文件转换回 M4A 格式
    if os.path.isfile(denoised_pcm_path):
        output_path = os.path.join(output_folder, f"{base_name}_denoised.m4a")
        subprocess.run(['ffmpeg', '-f', 's16le', '-ar', '44100', '-ac', '1', '-i', denoised_pcm_path, output_path], check=True)
        print(f"Converted {denoised_pcm_path} to {output_path} as .m4a")

    # 如果原文件是 MP3，直接转换为 M4A
    if extension == "mp3":
        direct_output_path = os.path.join(output_folder, f"{base_name}.m4a")
        subprocess.run(['ffmpeg', '-i', input_file, direct_output_path], check=True)
        print(f"Directly converted {input_file} to {direct_output_path} as .m4a")

    # 删除中间文件
    if os.path.isfile(pcm_path):
        os.remove(pcm_path)
    if os.path.isfile(denoised_pcm_path):
        os.remove(denoised_pcm_path)

print(f"All files processed and saved to {output_folder}")

# 清理残留的中间文件
for pcm_file in glob.glob(os.path.join(intermediate_folder, '*.pcm')):
    os.remove(pcm_file)
