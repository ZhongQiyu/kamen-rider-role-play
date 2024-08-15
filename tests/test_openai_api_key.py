# test_openai_api_key.py

import os
import openai

# 获取 API 密钥
openai.api_key = os.environ.get('OPENAI_API_KEY')

# 发送请求
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Hello, world!",
  max_tokens=1024,
  n=1,
  stop=None,
  temperature=0.5,
)

print(response.choices[0].text.strip())