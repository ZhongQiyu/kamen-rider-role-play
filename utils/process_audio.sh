#!/bin/bash

# 定义输入、中间和输出文件夹
input_folder="C:\Users\xiaoy\Downloads\wav"
intermediate_folder="C:\Users\xiaoy\Downloads\pcm"
output_folder="C:\Users\xiaoy\Downloads\m4a"

# 确保中间和输出文件夹存在
mkdir -p "$intermediate_folder"
mkdir -p "$output_folder"

# 检查是否有音频文件
shopt -s nullglob
files=("$input_folder"/*.{m4a,mp3})
if [ ${#files[@]} -eq 0 ]; then
    echo "No audio files found in $input_folder"
    exit 1
fi

# 绝对路径的rnnoise_demo
rnnoise_path="/absolute/path/to/rnnoise_demo"

for input_file in "${files[@]}"; do
    # 提取文件扩展名和基本名称
    extension="${input_file##*.}"
    base_name=$(basename "$input_file" .$extension)

    echo "Processing $input_file with extension $extension"

    # 转换到 PCM 格式
    ffmpeg -i "$input_file" -f s16le -acodec pcm_s16le "$intermediate_folder/$base_name.pcm" || { echo "Failed to convert $input_file to PCM"; continue; }
    echo "Converted to PCM: $intermediate_folder/$base_name.pcm"

    # 应用 rnnoise 降噪
    if [ -f "$intermediate_folder/$base_name.pcm" ]; then
        "$rnnoise_path" "$intermediate_folder/$base_name.pcm" "$intermediate_folder/${base_name}_denoised.pcm"
        echo "Noise reduction applied: $intermediate_folder/${base_name}_denoised.pcm"
    fi

    # 将降噪后的 PCM 文件转换回 M4A 格式
    if [ -f "$intermediate_folder/${base_name}_denoised.pcm" ]; then
        output_path="$output_folder/${base_name}_denoised.m4a"
        ffmpeg -f s16le -ar 44100 -ac 1 -i "$intermediate_folder/${base_name}_denoised.pcm" "$output_path"
        echo "Converted ${base_name}_denoised.pcm to $output_path as .m4a"
    fi

    # 如果原文件是MP3，直接转换为M4A
    if [ "$extension" = "mp3" ]; then
        direct_output_path="$output_folder/${base_name}.m4a"
        ffmpeg -i "$input_file" "$direct_output_path"
        echo "Directly converted $input_file to $direct_output_path as .m4a"
    fi

    # 删除中间文件
    rm -f "$intermediate_folder/$base_name.pcm" "$intermediate_folder/${base_name}_denoised.pcm"
done

echo "All files processed and saved to $output_folder"

# 清理残留的中间文件
trap 'rm -f "$intermediate_folder"/*.pcm' EXIT
