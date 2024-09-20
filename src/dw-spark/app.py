#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: Soraichi Kaku
# @Date  : 2024/8/26
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
