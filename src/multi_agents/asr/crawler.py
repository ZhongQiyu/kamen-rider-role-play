# crawler.py

import itchat
import platform
import os
import csv
import json
from datetime import datetime

# 确定当前操作系统
def get_platform():
    current_platform = platform.system()
    if current_platform == "Windows":
        return "windows"
    elif current_platform == "Darwin":
        return "macos"
    elif current_platform == "Linux":
        return "linux"
    else:
        raise Exception("Unsupported platform")

# 根据平台选择文件保存路径
def get_save_path(filename):
    current_platform = get_platform()
    home_dir = os.path.expanduser("~")
    
    if current_platform == "windows":
        save_dir = os.path.join(home_dir, "Documents", "WeChatRecords")
    else:
        save_dir = os.path.join(home_dir, "WeChatRecords")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return os.path.join(save_dir, filename)

# 导出消息记录到 CSV 文件
def export_to_csv(messages, filename="messages.csv"):
    save_path = get_save_path(filename)
    
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FromUserName", "Content", "MsgType"])
        for msg in messages:
            writer.writerow([msg["time"], msg["FromUserName"], msg["Content"], msg["MsgType"]])
    
    print(f"消息已保存到: {save_path}")

# 导出消息记录到 JSON 文件
def export_to_json(messages, filename="messages.json"):
    save_path = get_save_path(filename)
    
    with open(save_path, mode='w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)
    
    print(f"消息已保存到: {save_path}")

# 消息处理和记录
messages = []

@itchat.msg_register(itchat.content.TEXT)
def handle_text(msg):
    msg_record = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "FromUserName": msg['FromUserName'],
        "Content": msg['Text'],
        "MsgType": msg['Type'],
    }
    messages.append(msg_record)
    print(f"收到消息: {msg_record['Content']}")

# 启动微信并登录
def start_wechat():
    itchat.auto_login(hotReload=True)
    itchat.run()

# 导出所有收到的消息
def export_messages():
    export_to_csv(messages)
    export_to_json(messages)

if __name__ == "__main__":
    try:
        start_wechat()
    except KeyboardInterrupt:
        # 程序结束时导出消息记录
        print("正在导出消息记录...")
        export_messages()



import itchat
import platform
import os
import csv
import json
from datetime import datetime

# 确定当前操作系统
def get_platform():
    current_platform = platform.system()
    if current_platform == "Windows":
        return "windows"
    elif current_platform == "Darwin":
        return "macos"
    elif current_platform == "Linux":
        return "linux"
    else:
        raise Exception("Unsupported platform")

# 根据平台选择文件保存路径
def get_save_path(filename, folder=None):
    current_platform = get_platform()
    home_dir = os.path.expanduser("~")
    
    if current_platform == "windows":
        base_dir = os.path.join(home_dir, "Documents", "WeChatRecords")
    else:
        base_dir = os.path.join(home_dir, "WeChatRecords")
    
    if folder:
        save_dir = os.path.join(base_dir, folder)
    else:
        save_dir = base_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return os.path.join(save_dir, filename)

# 导出消息记录到 CSV 文件
def export_to_csv(messages, filename="messages.csv"):
    save_path = get_save_path(filename)
    
    with open(save_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "FromUserName", "Content", "MsgType"])
        for msg in messages:
            writer.writerow([msg["time"], msg["FromUserName"], msg["Content"], msg["MsgType"]])
    
    print(f"消息已保存到: {save_path}")

# 导出消息记录到 JSON 文件
def export_to_json(messages, filename="messages.json"):
    save_path = get_save_path(filename)
    
    with open(save_path, mode='w', encoding='utf-8') as file:
        json.dump(messages, file, ensure_ascii=False, indent=4)
    
    print(f"消息已保存到: {save_path}")

# 消息处理和记录
messages = []

@itchat.msg_register(itchat.content.TEXT)
def handle_text(msg):
    msg_record = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "FromUserName": msg['FromUserName'],
        "Content": msg['Text'],
        "MsgType": msg['Type'],
    }
    messages.append(msg_record)
    print(f"收到文本消息: {msg_record['Content']}")

# 处理图片消息
@itchat.msg_register(itchat.content.PICTURE)
def handle_picture(msg):
    file_name = msg['FileName']
    file_path = get_save_path(file_name, folder="images")
    msg['Text'](file_path)  # 下载图片
    print(f"收到图片消息，保存到: {file_path}")
    
    msg_record = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "FromUserName": msg['FromUserName'],
        "Content": f"图片已保存: {file_path}",
        "MsgType": msg['Type'],
    }
    messages.append(msg_record)

# 处理视频消息
@itchat.msg_register(itchat.content.VIDEO)
def handle_video(msg):
    file_name = msg['FileName']
    file_path = get_save_path(file_name, folder="videos")
    msg['Text'](file_path)  # 下载视频
    print(f"收到视频消息，保存到: {file_path}")
    
    msg_record = {
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "FromUserName": msg['FromUserName'],
        "Content": f"视频已保存: {file_path}",
        "MsgType": msg['Type'],
    }
    messages.append(msg_record)

# 启动微信并登录
def start_wechat():
    itchat.auto_login(hotReload=True)
    itchat.run()

# 导出所有收到的消息
def export_messages():
    export_to_csv(messages)
    export_to_json(messages)

if __name__ == "__main__":
    try:
        start_wechat()
    except KeyboardInterrupt:
        # 程序结束时导出消息记录
        print("正在导出消息记录...")
        export_messages()



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



