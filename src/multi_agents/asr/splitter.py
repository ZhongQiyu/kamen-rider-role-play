# splitter.py

import os
import subprocess
import json

def extract_tracks(input_file, output_dir):
    """
    分离并提取视频、音频和字幕轨道
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分离视频轨道
    video_output = os.path.join(output_dir, "video_only.mp4")
    video_command = f"ffmpeg -i {input_file} -an -sn -vn {video_output}"
    subprocess.run(video_command, shell=True)
    print(f"视频轨道已提取到: {video_output}")

    # 分离音轨
    audio_output = os.path.join(output_dir, "audio_only.aac")
    audio_command = f"ffmpeg -i {input_file} -vn -sn -c:a copy {audio_output}"
    subprocess.run(audio_command, shell=True)
    print(f"音轨已提取到: {audio_output}")

    # 分离字幕轨道
    subtitle_output = os.path.join(output_dir, "subtitles_only.srt")
    subtitle_command = f"ffmpeg -i {input_file} -vn -an -c:s copy {subtitle_output}"
    subprocess.run(subtitle_command, shell=True)
    print(f"字幕轨道已提取到: {subtitle_output}")

def extract_chapters(input_file, output_dir):
    """
    提取章节信息并保存为文本文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chapters_output = os.path.join(output_dir, "chapters.txt")
    command = f"ffmpeg -i {input_file} -f ffmetadata {chapters_output}"
    subprocess.run(command, shell=True)
    print(f"章节信息已提取到: {chapters_output}")

def extract_metadata(input_file, output_dir):
    """
    使用 ffmpeg 提取文件的元数据信息并保存为 JSON 文件
    """
    metadata_output = os.path.join(output_dir, "metadata.json")
    command = f"ffmpeg -i {input_file} -f ffmetadata -"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    with open(metadata_output, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    print(f"元数据已提取到: {metadata_output}")

def extract_data_tracks(input_file, output_dir):
    """
    使用 mediainfo 提取数据轨道信息并保存为 JSON 文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_output = os.path.join(output_dir, "data_tracks.json")
    command = f"mediainfo --Output=JSON {input_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # 解析 JSON 并提取数据轨道信息
    if result.returncode == 0:
        data_info = json.loads(result.stdout)
        with open(data_output, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=4)
        print(f"数据轨道信息已提取到: {data_output}")
    else:
        print(f"提取数据轨道时出错: {result.stderr}")

if __name__ == "__main__":
    input_m2ts_or_mkv = "/path/to/your/input_file.m2ts"  # 输入文件路径
    output_directory = "/path/to/output_dir"  # 输出文件夹路径

    # 提取视频、音轨和字幕
    extract_tracks(input_m2ts_or_mkv, output_directory)
    
    # 提取章节
    extract_chapters(input_m2ts_or_mkv, output_directory)
    
    # 提取元数据
    extract_metadata(input_m2ts_or_mkv, output_directory)
    
    # 提取数据轨道
    extract_data_tracks(input_m2ts_or_mkv, output_directory)
