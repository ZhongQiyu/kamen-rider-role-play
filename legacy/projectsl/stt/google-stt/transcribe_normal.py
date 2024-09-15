# 同步语音识别
import io
import argparse
from google.cloud import speech

def transcribe_audio(speech_file=None, gcs_uri=None, language_code='ja-JP', sample_rate_hertz=16000, encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16):
    """Transcribe the given audio file or GCS URI."""
    client = speech.SpeechClient()
    
    if speech_file:
        with open(speech_file, "rb") as audio_file:
            content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
    elif gcs_uri:
        audio = speech.RecognitionAudio(uri=gcs_uri)
    else:
        raise ValueError("Either a speech_file or gcs_uri must be provided.")
    
    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
    )
    
    response = client.recognize(config=config, audio=audio)
    
    return [result.alternatives[0].transcript for result in response.results]

# 定义语言和音频源文件
langs = ['zh-CN','en-US','ja-JP']
audio = '/Users/qaz1214/Downloads/kamen-rider-blade-roleplay-sv/notebooks/audio-transcription/google-stt/ep23.m4a'

# 使用函数示例
print(transcribe_audio(speech_file=audio))

# 初始化客户端
client = speech.SpeechClient()

# 从本地文件加载音频
with io.open(audio, 'rb') as audio_file:
    content = audio_file.read()
    audio = types.RecognitionAudio(content=content)

# 设置识别配置
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US'
)

# 调用Google Speech-to-Text API进行语音识别
response = client.recognize(config=config, audio=audio)

# 打印识别结果
for result in response.results:
    print('Transcript: {}'.format(result.alternatives[0].transcript))

# 使用方法示例
# print(transcribe_audio(speech_file="audio_file.wav"))
# print(transcribe_audio(gcs_uri="gs://bucket_name/audio_file.flac"))

# https://cloud.google.com/speech-to-text/docs/sync-recognize?hl=zh-cn
