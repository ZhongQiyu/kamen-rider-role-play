#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : asr.py
# @Author: Qiyu (Allen) Zhong
# @Date  : 2024/10/3
# @Desc  :

import os
import gradio as gr
from typing import List, Iterable
from loguru import logger
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

class Config:
    def __init__(self, appid: str = None, apikey: str = None, apisecret: str = None):
        """
        初始化讯飞API的环境配置
        :param appid: 讯飞API的App ID
        :param apikey: 讯飞API的API Key
        :param apisecret: 讯飞API的API Secret
        """
        self.XF_APPID = appid or os.environ.get("SPARKAI_APP_ID")
        self.XF_APIKEY = apikey or os.environ.get("SPARKAI_API_KEY")
        self.XF_APISECRET = apisecret or os.environ.get("SPARKAI_API_SECRET")

class ChatModel:
    def __init__(self, config: Config, domain: str = 'generalv3.5', model_url: str = 'wss://spark-api.xf-yun.com/v3.5/chat', stream: bool = False):
        """
        初始化聊天模型
        :param config: 项目配置文件
        :param domain: 模型域名
        :param model_url: 模型地址
        :param stream: 是否启用流式调用
        """
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
        """
        批量生成对话
        :param msgs: 消息列表
        :return: 生成的对话文本
        """
        if self.stream:
            raise Exception('模型初始化为流式输出，请调用generate_stream方法')

        messages = self.__trans_msgs(msgs)
        resp = self.spark.generate([messages])
        return resp.generations[0][0].text

    def generate_stream(self, msgs: str | List[ChatMessage]) -> Iterable[str]:
        """
        流式生成对话
        :param msgs: 消息列表
        :return: 生成的对话文本流
        """
        if not self.stream:
            raise Exception('模型初始化为批式输出，请调用generate方法')
        messages = self.__trans_msgs(msgs)
        resp_iterable = self.spark.stream(messages)
        for resp in resp_iterable:
            yield resp.content

    def __trans_msgs(self, msg: str):
        """
        内部方法，将字符串转换为消息对象
        :param msg: 字符串或消息列表
        :return: 消息列表
        """
        if isinstance(msg, str):
            return [ChatMessage(role="user", content=msg)]
        return msg

class SparkApp:
    def __init__(self, config: Config):
        """
        初始化应用程序
        :param config: 配置文件对象
        """
        self.config = config
        self.model = ChatModel(config)
    
    def chat_interface(self):
        """
        定义聊天界面
        """
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天历史")
            chat_query = gr.Textbox(label="输入问题", placeholder="输入需要咨询的问题")
            llm_submit_tab = gr.Button("发送", visible=True)
            gr.Examples(["请介绍一下Datawhale。", "如何在大模型应用比赛中突围并获奖？", "请介绍一下基于Gradio的应用开发"], chat_query)
            chat_query.submit(fn=self.chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])
            llm_submit_tab.click(fn=self.chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])

        demo.queue().launch()

    def chat(self, chat_query, chat_history):
        """
        处理聊天请求
        :param chat_query: 用户输入的聊天内容
        :param chat_history: 聊天历史记录
        :return: 更新后的聊天历史记录
        """
        bot_message = self.model.generate(chat_query)
        chat_history.append((chat_query, bot_message))
        return "", chat_history

    def run_text_to_audio(self, text: str, audio_path: str):
        """
        将文本转换为语音
        :param text: 输入的文本
        :param audio_path: 生成的音频文件路径
        """
        t2a = Text2Audio(self.config)
        t2a.gen_audio(text, audio_path)

    def run_audio_to_text(self, audio_path: str):
        """
        将语音转换为文本
        :param audio_path: 输入的音频文件路径
        :return: 转换后的文本
        """
        a2t = Audio2Text(self.config)
        audio_text = a2t.gen_text(audio_path)
        return audio_text

    def run_text_to_img(self, prompt: str, img_path: str):
        """
        根据文本生成图片
        :param prompt: 输入的提示文本
        :param img_path: 生成的图片文件路径
        """
        t2i = Text2Img(self.config)
        t2i.gen_image(prompt, img_path)

    def run_image_understanding(self, prompt: str, img_path: str):
        """
        图片理解
        :param prompt: 输入的提示文本
        :param img_path: 输入的图片文件路径
        :return: 图片理解结果
        """
        iu = ImageUnderstanding(self.config)
        return iu.understanding(prompt, img_path)

    def run_get_embedding(self, text: str):
        """
        获取文本的嵌入向量
        :param text: 输入的文本
        :return: 文本的嵌入向量
        """
        em = EmbeddingModel(self.config)
        return em.get_embedding(text)

    def save_prompts(self, ask_batch, answer_batch, batch_related_classes, language='en'):
        """
        将当前批次的输入数据和相关信息保存到文本文件
        :param ask_batch: 输入问题的批次
        :param answer_batch: 输出答案的批次
        :param batch_related_classes: 相关类别
        :param language: 处理的语言
        """
        file_name = f'prompts_{language}.txt'
        with open(file_name, 'a', encoding='utf-8') as f:
            for ask, answer, label in zip(ask_batch, answer_batch, batch_related_classes):
                f.write(f"{ask}\t{answer}\t{label}\n")

if __name__ == '__main__':
    # 初始化配置
    config = Config()
    # 创建应用
    app = SparkApp(config)
    # 启动聊天界面
    app.chat_interface()

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

# app.py

import os
import gradio as gr
import random
import time
from sparkai.core.messages import ChatMessage
from dwspark.config import Config
from dwspark.models import ChatModel, Text2Img, ImageUnderstanding, Text2Audio, Audio2Text, EmebddingModel
from loguru import logger

# 加载讯飞的api配置
SPARKAI_APP_ID = os.environ.get("SPARKAI_APP_ID", "your_app_id")
SPARKAI_API_SECRET = os.environ.get("SPARKAI_API_SECRET", "your_api_secret")
SPARKAI_API_KEY = os.environ.get("SPARKAI_API_KEY", "your_api_key")
config = Config(SPARKAI_APP_ID, SPARKAI_API_KEY, SPARKAI_API_SECRET)

# 初始化模型
stream_model = ChatModel(config, stream=True)

# 中译英提示语
zh2en_prompt = '你是中英文互译高手。给定一句中文文本，请你帮我翻译成英文。文本：{}'
# 英译中提示语
en2zh_prompt = '你是中英文互译高手。给定一句英文文本，请你帮我翻译成中文。文本：{}'

def chat(chat_query, chat_history, prompt_type):
    if prompt_type == '中译英':
        final_query = zh2en_prompt.format(chat_query)
    else:
        final_query = en2zh_prompt.format(chat_query)
    # 添加最新问题
    prompts = [ChatMessage(role='user', content=final_query)]

    # 将问题设为历史对话
    chat_history.append((chat_query, ''))
    # 对话同时流式返回
    for chunk_text in stream_model.generate_stream(prompts):
        # 总结答案
        answer = chat_history[-1][1] + chunk_text
        # 替换最新的对话内容
        chat_history[-1] = (chat_query, answer)
        # 返回
        yield '', chat_history

# 随机聊天函数
def random_chat(chat_query, chat_history):
    bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
    chat_history.append((chat_query, bot_message))
    return "", chat_history

with gr.Blocks() as demo:
    warning_html_code = """
        <div class="hint" style="text-align: center;background-color: rgba(255, 255, 0, 0.15); padding: 10px; margin: 10px; border-radius: 5px; border: 1px solid #ffcc00;">
            <p>中英翻译助手是Datawhale开源《讯飞2024星火杯》第一阶段的baseline。</p>
            <p>🐱 欢迎体验或交流【公众号】：Datawhale 【B站主页】https://space.bilibili.com/431850986</p>
            <p>相关地址: <a href="https://challenge.xfyun.cn/h5/xinghuo?ch=dwm618">比赛地址</a>、<a href="https://datawhaler.feishu.cn/wiki/Aee0wU4KlipwY9kHJyecQFT3nTg">学习手册</a></p>
        </div>
    """
    gr.HTML(warning_html_code)

    prompt_type = gr.Radio(choices=['中译英', '英译中'], value='中译英', label='翻译类型')
    chatbot = gr.Chatbot([], elem_id="chat-box", label="聊天历史")
    chat_query = gr.Textbox(label="输入问题", placeholder="输入需要咨询的问题")
    llm_submit_tab = gr.Button("发送", visible=True)
    gr.Examples([
        "Datawhale 是一个专注于数据科学与 AI 领域的开源组织...",
        "Python is a programming language that lets you work quickly and integrate systems more effectively."
    ], chat_query)

    # 按钮触发逻辑
    llm_submit_tab.click(fn=chat, inputs=[chat_query, chatbot, prompt_type], outputs=[chat_query, chatbot])
    chat_query.submit(fn=random_chat, inputs=[chat_query, chatbot], outputs=[chat_query, chatbot])

    # 添加 SDK 功能示例
    with gr.Accordion("SDK 功能示例", open=False):
        with gr.Column():
            text_to_audio = gr.Button("文字生成语音")
            audio_to_text = gr.Button("语音识别文字")
            generate_image = gr.Button("生成图片")
            understand_image = gr.Button("图片解释")
            get_embedding = gr.Button("获取文本向量")
            
            # SDK 示例功能
            def t2a_function():
                text = '2023年5月，讯飞星火大模型正式发布...'
                audio_path = './demo.mp3'
                t2a = Text2Audio(config)
                t2a.gen_audio(text, audio_path)
                return f"音频已生成: {audio_path}"

            def a2t_function():
                a2t = Audio2Text(config)
                audio_text = a2t.gen_text('./demo.mp3')
                return audio_text

            def t2i_function():
                prompt = '一只鲸鱼在快乐游泳的卡通头像'
                t2i = Text2Img(config)
                t2i.gen_image(prompt, './demo.jpg')
                return './demo.jpg'

            def iu_function():
                iu = ImageUnderstanding(config)
                understanding = iu.understanding('请理解一下图片', './demo.jpg')
                return understanding

            def em_function():
                em = EmebddingModel(config)
                vector = em.get_embedding("我们是datawhale")
                return str(vector)

            text_to_audio.click(fn=t2a_function, outputs=gr.Textbox())
            audio_to_text.click(fn=a2t_function, outputs=gr.Textbox())
            generate_image.click(fn=t2i_function, outputs=gr.Image())
            understand_image.click(fn=iu_function, outputs=gr.Textbox())
            get_embedding.click(fn=em_function, outputs=gr.Textbox())

if __name__ == "__main__":
    demo.queue().launch()
