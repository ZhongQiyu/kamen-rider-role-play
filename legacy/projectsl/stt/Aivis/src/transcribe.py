import deepspeech
import wave

# 加载预训练的模型
model_file_path = 'path_to_deepspeech_model.pbmm'
model = deepspeech.Model(model_file_path)

# 读取音频文件
audio_file_path = 'path_to_your_audio_file.wav'
with wave.open(audio_file_path, 'rb') as audio_file:
    frames = audio_file.getnframes()
    buffer = audio_file.readframes(frames)
    # 确保音频文件是16kHz
    assert audio_file.getframerate() == 16000

# 执行语音识别
text = model.stt(buffer)

# 输出转录文本
print(text)
