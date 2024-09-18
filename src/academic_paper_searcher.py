# searcher.py

import time
import requests
import networkx as nx
from bs4 import BeautifulSoup
from selenium import webdriver
import matplotlib.pyplot as plt
from selenium.webdriver.common.keys import Keys

class AcademicPaperSearcher:
    def __init__(self, driver_path='path_to_your_chromedriver'):
        self.base_url = "https://scholar.google.com/scholar"
        self.driver = webdriver.Chrome(executable_path=driver_path)  # 使用合适的浏览器驱动程序

    def build_query_url(self, keywords, conference):
        """构建带有关键词和会议名称的查询URL。"""
        query_string = '+'.join(keywords.split()) + f"+AND+{conference}"
        return f"{self.base_url}?q={query_string}"

    def perform_search(self, keyword):
        """使用Selenium在Google Scholar上执行搜索。"""
        search_url = self.build_query_url(keyword, "IJCAI")  # 假设默认会议为IJCAI
        self.driver.get(search_url)
        time.sleep(2)  # 等待页面加载

        return self.driver.page_source

    def parse_results(self, html_content):
        """解析页面并打印论文标题和链接。"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = soup.find_all('h3', {'class': 'gs_rt'})
        for result in results:
            title = result.text
            link = result.a['href'] if result.a else 'No link available'
            print(f"Title: {title}\nLink: {link}\n")

    def search_papers(self, keywords, conference="IJCAI"):
        """搜索论文并打印结果。"""
        for keyword in keywords:
            html_content = self.perform_search(keyword + " AND " + conference)
            self.parse_results(html_content)

    def close(self):
        """关闭浏览器。"""
        self.driver.quit()

    def create_keyword_graph(self, keywords, edges):
        """创建关键字图并展示。"""
        G = nx.Graph()
        G.add_nodes_from(keywords)
        G.add_edges_from(edges)
        nx.draw(G, with_labels=True)
        plt.show()

# 使用示例
if __name__ == "__main__":
    searcher = AcademicPaperSearcher(driver_path='path_to_your_chromedriver')
    # 定义要搜索的关键词列表和会议名称
    keywords = ['multi-agent systems', 'agent-based modeling', 'role-playing', 'distributed AI', 'multimodal data mining', 'emotional intelligence']
    # 搜索IJCAI会议的论文
    searcher.search_papers(keywords, 'IJCAI')
    # 关闭浏览器
    searcher.close()
    # 定义可视化网络需要的边和节点
    edges = [('multi-agent systems', 'IJCAI'), ('role-playing', 'IJCAI'), ('multimodal data mining', 'IJCAI')]
    # 构建关键字网络
    searcher.create_keyword_graph(keywords, edges)



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

# spider.py

import requests

def fetch_url(url):
    response = requests.get(url)
    return response.text

# Jupyter Notebook - Cell 1
# 安装awscli和boto3
!pip install awscli boto3

# Jupyter Notebook - Cell 2
# 配置AWS CLI（需要手动输入密钥）
!aws configure

# Jupyter Notebook - Cell 3
# 创建一个S3存储桶
!aws s3 mb s3://your-bucket-name

# Jupyter Notebook - Cell 4
# 上传模型文件到S3存储桶
!aws s3 cp model_file s3://your-bucket-name/model_file

# Jupyter Notebook - Cell 5
# 导入boto3和定义辅助函数
import boto3
import json

def download_model_from_s3(bucket_name, model_key, download_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, model_key, download_path)

def generate_question_answer_pairs(model_file_path, input_text):
    # 这里你需要加载和运行你的模型
    with open(model_file_path, 'r') as model_file:
        model_data = model_file.read()
    return f'Question: {input_text}\nAnswer: {model_data}'  # 模拟生成的问答对

# Jupyter Notebook - Cell 6
# 使用函数从S3下载模型并生成问答对
bucket_name = 'your-bucket-name'
model_key = 'model_file'
download_path = 'local_model_file'

# 从S3下载模型
download_model_from_s3(bucket_name, model_key, download_path)

# 使用模型生成问答对
input_text = 'What is the capital of France?'
qa_pairs = generate_question_answer_pairs(download_path, input_text)

print(qa_pairs)
