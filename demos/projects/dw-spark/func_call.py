# func_call.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : func_call.py
# @Author: Richard Chiming Xu
# @Date  : 2024/6/24
# @Desc  : 讯飞API统一调用脚本

import os
from loguru import logger
from sparkai.core.messages import ChatMessage
from dwspark.config import Config
from dwspark.models import ChatModel, Text2Img, ImageUnderstanding, Text2Audio, Audio2Text, EmbeddingModel

class Config():
    def __init__(self, appid: str = None, apikey: str = None, apisecret: str = None):
        '''
        讯飞API统一的环境配置
        :param appid: appid
        :param apikey: api key
        :param apisecret: api secret

        调用方式：
        # 加载系统环境变量SPARKAI_APP_ID、SPARKAI_API_KEY、SPARKAI_API_SECRET
        # 系统环境变量需要进行赋值
        config = Config()
        # 自定义key写入
        config = Config('14****93', 'eb28b****b82', 'MWM1MzBkOD****Mzk0')
        '''
        if appid is None:
            self.XF_APPID = os.environ["SPARKAI_APP_ID"]
        else:
            self.XF_APPID = appid
        if apikey is None:
            self.XF_APIKEY = os.environ["SPARKAI_API_KEY"]
        else:
            self.XF_APIKEY = apikey
        if apisecret is None:
            self.XF_APISECRET = os.environ["SPARKAI_API_SECRET"]
        else:
            self.XF_APISECRET = apisecret

# 加载系统环境变量SPARKAI_APP_ID、SPARKAI_API_KEY、SPARKAI_API_SECRET
config = Config()

# 自定义key写入（如果需要的话）
# config = Config('14****93', 'eb28b****b82', 'MWM1MzBkOD****Mzk0')

# 模拟问题
question = input("输入你想问星火大模型的问题吧：\n")

'''
批式调用对话
'''
logger.info('----------批式调用对话----------')
model = ChatModel(config, stream=False)
logger.info(model.generate([ChatMessage(role="user", content=question)]))

'''
流式调用对话
'''
logger.info('----------流式调用对话----------')
model = ChatModel(config, stream=True)
for r in model.generate_stream(question):
    logger.info(r)
logger.info('done.')

'''
文字生成语音
'''
text = '2023年5月，讯飞星火大模型正式发布，迅速成为千万用户获取知识、学习知识的“超级助手”，成为解放生产力、释放想象力的“超级杠杆”。2024年4月，讯飞星火V3.5春季升级长文本、长图文、长语音三大能力。一年时间内，讯飞星火从1.0到3.5，每一次迭代都是里程碑式飞跃。'
audio_path = './demo.mp3'
t2a = Text2Audio(config)
t2a.gen_audio(text, audio_path)

'''
语音识别文字
'''
a2t = Audio2Text(config)
audio_text = a2t.gen_text(audio_path)
logger.info(audio_text)

'''
生成图片
'''
logger.info('----------生成图片----------')
prompt = '一只鲸鱼在快乐游泳的卡通头像'
t2i = Text2Img(config)
t2i.gen_image(prompt, './demo.jpg')

'''
图片解释
'''
logger.info('----------图片解释----------')
prompt = '请理解一下图片'
iu = ImageUnderstanding(config)
logger.info(iu.understanding(prompt, './demo.jpg'))

'''
获取文本向量
'''
logger.info('----------获取文本向量----------')
em = EmbeddingModel(config)
vector = em.get_embedding("我们是datawhale")
logger.info(vector)
