# video_processor.py

import os
import json
import subprocess
import requests
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import joblib
from fugashi import Tagger

class VideoAnalysisTool:
    def __init__(self, video_path=None, url=None):
        self.video_path = video_path
        self.url = url
        self.frames_folder = 'frames'
        self.audio_output = 'output.mp3'
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize_japanese)
        self.model = SGDClassifier()
        self.initial_accuracy = None
        self.tagger = Tagger('-Owakati')
        
        if video_path:
            self.clip = VideoFileClip(video_path)

    def run_command(self, command):
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def extract_frames(self, fps=1):
        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)
        for i, frame in enumerate(self.clip.iter_frames(fps=fps)):
            frame_image = Image.fromarray(frame)
            frame_image.save(f"{self.frames_folder}/frame_{i+1:03d}.png")

    def extract_audio(self):
        audio = self.clip.audio
        audio.write_audiofile(self.audio_output)

    def convert_video_format(self, output_path, output_format='mp4'):
        command = ['ffmpeg', '-i', self.video_path, f'{output_path}.{output_format}']
        self.run_command(command)

    def tokenize_japanese(self, text):
        return self.tagger.parse(text).strip().split()

    def load_additional_dataset(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_dataset(self, text_data, additional_data, dataset_output):
        tokenized_text_data = [self.tokenize_japanese(sentence) for sentence in text_data]
        combined_data = {
            'frames': [os.path.join(self.frames_folder, frame) for frame in os.listdir(self.frames_folder) if frame.endswith('.png')],
            'audio': self.audio_output,
            'text': tokenized_text_data,
            'additional_data': additional_data
        }
        with open(dataset_output, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)

    def process_video(self, fps=1):
        self.extract_frames(fps=fps)
        self.extract_audio()

    def process_text_data(self, text_data):
        # 假设这是一个文本处理的函数
        return text_data

    def online_learning(self, stream, threshold=0.1):
        for i, (documents, labels) in enumerate(stream):
            X_new = self.vectorizer.transform(documents)
            if i == 0:
                self.model.fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)
                print(f"Initial Batch - Accuracy: {self.initial_accuracy}")
                continue
            predictions = self.model.predict(X_new)
            new_accuracy = self.model.score(X_new, labels)
            print(f"Batch {i+1} - New Data Accuracy: {new_accuracy}")
            print(classification_report(labels, predictions))

            if new_accuracy < self.initial_accuracy - threshold:
                print("Performance decreased, updating the model.")
                self.model.partial_fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)

    def save_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def get_video_info(self):
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.content, 'html.parser')

            title = soup.find('h1').text.strip()
            description = soup.find('div', class_='video-description').text.strip()
            duration = soup.find('span', class_='video-duration').text.strip()
            video_url = soup.find('video')['src']

            return {
                'title': title,
                'description': description,
                'duration': duration,
                'video_url': video_url
            }
        except Exception as e:
            print(f"无法从 {self.url} 获取视频信息: {e}")
            return None

    def close(self):
        if self.clip:
            self.clip.close()


# 使用示例
if __name__ == "__main__":
    # 初始化工具
    video_tool = VideoAnalysisTool(video_path='path_to_video.mp4', url="https://www.example.com/video-page")
    
    # 视频处理
    video_tool.process_video(fps=1)
    # 假设文本数据和附加数据
    text_data = ["Example sentence 1", "Example sentence 2"]
    additional_data = {"key": "value"}
    video_tool.create_dataset(text_data, additional_data, dataset_output='dataset.json')

    # 模拟在线学习的数据流
    stream = (({"dummy_text": ["Example sentence"]}, [0]) for _ in range(100))
    video_tool.online_learning(stream)
    video_tool.save_model()

    video_tool.close()

    # 获取视频信息
    video_info = video_tool.get_video_info()
    if video_info:
        print("视频标题:", video_info['title'])
        print("视频描述:", video_info['description'])
        print("视频时长:", video_info['duration'])
        print("视频播放链接:", video_info['video_url'])
