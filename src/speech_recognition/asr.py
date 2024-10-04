#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : asr.py
# @Author: Qiyu (Allen) Zhong
# @Date  : 2024/10/4
# @Desc  : 合并文件，包含音频、视频处理及聊天接口

import os
import gradio as gr
import subprocess
import glob
from typing import List, Iterable
from loguru import logger
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 配置类定义
class Config:
    def __init__(self, appid: str = None, apikey: str = None, apisecret: str = None):
        self.XF_APPID = appid or os.environ.get("SPARKAI_APP_ID")
        self.XF_APIKEY = apikey or os.environ.get("SPARKAI_API_KEY")
        self.XF_APISECRET = apisecret or os.environ.get("SPARKAI_API_SECRET")

# ChatModel 定义
class ChatModel:
    def __init__(self, config: Config, domain: str = 'generalv3.5', model_url: str = 'wss://spark-api.xf-yun.com/v3.5/chat', stream: bool = False):
        self.spark = ChatSparkLLM(
            spark_api_url=model_url,
            spark_app_id=config.XF_APPID,
            spark_api_key=config.XF_APIKEY,
            spark_api_secret=config.XF_APISECRET,
            spark_llm_domain=domain,
            streaming=stream,
        )
        self.stream = stream

    def generate(self, msgs: str | List[ChatMessage]) -> str:
        if self.stream:
            raise Exception('模型初始化为流式输出，请调用generate_stream方法')
        messages = self.__trans_msgs(msgs)
        resp = self.spark.generate([messages])
        return resp.generations[0][0].text

    def generate_stream(self, msgs: str | List[ChatMessage]) -> Iterable[str]:
        if not self.stream:
            raise Exception('模型初始化为批式输出，请调用generate方法')
        messages = self.__trans_msgs(msgs)
        resp_iterable = self.spark.stream(messages)
        for resp in resp_iterable:
            yield resp.content

    def __trans_msgs(self, msg: str):
        if isinstance(msg, str):
            return [ChatMessage(role="user", content=msg)]
        return msg

# 视频处理功能
def process_videos(input_dir, output_dir):
    """
    转换m2ts文件到mkv格式
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in glob.glob(os.path.join(input_dir, '*.m2ts')):
        if os.path.isfile(file):
            base_name = os.path.basename(file).replace('.m2ts', '')
            output_file = os.path.join(output_dir, base_name + '.mkv')
            print(f"正在处理文件: {file} -> {output_file}")
            subprocess.run(['ffmpeg', '-i', file, '-c', 'copy', output_file], check=True)
            print(f"文件 {output_file} 转换成功！")

# 降噪处理功能
def denoise_audio(input_folder, intermediate_folder, output_folder, rnnoise_path):
    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = glob.glob(os.path.join(input_folder, '*.m4a')) + glob.glob(os.path.join(input_folder, '*.mp3'))
    if len(audio_files) == 0:
        print(f"No audio files found in {input_folder}")
        return

    for input_file in audio_files:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        pcm_path = os.path.join(intermediate_folder, f"{base_name}.pcm")
        denoised_pcm_path = os.path.join(intermediate_folder, f"{base_name}_denoised.pcm")
        output_path = os.path.join(output_folder, f"{base_name}_denoised.m4a")

        # 转换音频文件为PCM格式
        subprocess.run(['ffmpeg', '-i', input_file, '-f', 's16le', '-acodec', 'pcm_s16le', pcm_path], check=True)
        # 应用rnnoise降噪
        subprocess.run([rnnoise_path, pcm_path, denoised_pcm_path], check=True)
        # 将降噪后的PCM文件转换回M4A
        subprocess.run(['ffmpeg', '-f', 's16le', '-ar', '44100', '-ac', '1', '-i', denoised_pcm_path, output_path], check=True)

        # 删除中间文件
        os.remove(pcm_path)
        os.remove(denoised_pcm_path)

# Chat功能
class SparkApp:
    def __init__(self, config: Config):
        self.config = config
        self.model = ChatModel(config)

    def chat_interface(self):
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天历史")
            chat_query = gr.Textbox(label="输入问题", placeholder="输入需要咨询的问题")
            llm_submit_tab = gr.Button("发送", visible=True)
            chat_query.submit(fn=self.chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])
            llm_submit_tab.click(fn=self.chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])

        demo.queue().launch()

    def chat(self, chat_query, chat_history):
        bot_message = self.model.generate(chat_query)
        chat_history.append((chat_query, bot_message))
        return "", chat_history

# 主函数入口
if __name__ == '__main__':
    config = Config()

    # 1. 启动聊天界面
    app = SparkApp(config)
    app.chat_interface()

    # 2. 处理音频降噪任务
    input_folder = "./input_wav"
    intermediate_folder = "./intermediate_pcm"
    output_folder = "./output_m4a"
    rnnoise_path = "/path/to/rnnoise_demo"
    denoise_audio(input_folder, intermediate_folder, output_folder, rnnoise_path)

    # 3. 处理视频转换任务
    input_video_dir = "./videos"
    output_video_dir = "./output_mkv"
    process_videos(input_video_dir, output_video_dir)
