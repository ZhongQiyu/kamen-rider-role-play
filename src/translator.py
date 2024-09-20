# translator.py

import requests

class Translator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://translation.googleapis.com/language/translate/v2"

    def translate(self, text, target_lang):
        params = {
            'q': text,
            'target': target_lang,
            'key': self.api_key
        }
        response = requests.get(self.url, params=params)
        return response.json().get('data', {}).get('translations', [{}])[0].get('translatedText')

# 测试翻译
if __name__ == "__main__":
    translator = Translator('YOUR_API_KEY')  # 替换为你的API密钥
    translated_text = translator.translate("你好", "en")
    print("翻译结果:", translated_text)
