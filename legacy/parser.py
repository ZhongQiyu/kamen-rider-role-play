# parser.py

import json
from bs4 import BeautifulSoup

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    titles = soup.find_all('h1')
    return [title.text for title in titles]

def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 解析每一行的 JSON 对象
            obj = json.loads(line.strip())
            data.append(obj)
    return data

# 示例：解析 JSONL 文件
file_path = 'example.jsonl'
parsed_data = parse_jsonl(file_path)
print(parsed_data)
