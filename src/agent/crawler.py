#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : crawler.py
# @Author: Qiyu (Allen) Zhong
# @Date  : 2024/10/4
# @Desc  : 爬取论文以及微信聊天记录数据

import os
import csv
import json
import platform
import itchat
import time
import requests
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

class WeChatAcademicSearcher:
    def __init__(self, browser='chrome', driver_path='path_to_your_driver'):
        self.messages = []
        self.base_url = "https://scholar.google.com/scholar"
        self.driver_path = driver_path
        
        # 根据选择的浏览器初始化 WebDriver
        if browser.lower() == 'chrome':
            self.driver = webdriver.Chrome(executable_path=self.driver_path)
        elif browser.lower() == 'edge':
            self.driver = webdriver.Edge(executable_path=self.driver_path)
        else:
            raise ValueError("Unsupported browser. Please use 'chrome' or 'edge'.")

    def get_platform(self):
        current_platform = platform.system()
        if current_platform == "Windows":
            return "windows"
        elif current_platform == "Darwin":
            return "macos"
        elif current_platform == "Linux":
            return "linux"
        else:
            raise Exception("Unsupported platform")

    def get_save_path(self, filename, folder=None):
        current_platform = self.get_platform()
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

    def export_to_csv(self, filename="messages.csv"):
        save_path = self.get_save_path(filename)

        with open(save_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "FromUserName", "Content", "MsgType"])
            for msg in self.messages:
                writer.writerow([msg["time"], msg["FromUserName"], msg["Content"], msg["MsgType"]])

        print(f"消息已保存到: {save_path}")

    def export_to_json(self, filename="messages.json"):
        save_path = self.get_save_path(filename)

        with open(save_path, mode='w', encoding='utf-8') as file:
            json.dump(self.messages, file, ensure_ascii=False, indent=4)

        print(f"消息已保存到: {save_path}")

    @itchat.msg_register(itchat.content.TEXT)
    def handle_text(self, msg):
        msg_record = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "FromUserName": msg['FromUserName'],
            "Content": msg['Text'],
            "MsgType": msg['Type'],
        }
        self.messages.append(msg_record)
        print(f"收到文本消息: {msg_record['Content']}")

    @itchat.msg_register(itchat.content.PICTURE)
    def handle_picture(self, msg):
        file_name = msg['FileName']
        file_path = self.get_save_path(file_name, folder="images")
        msg['Text'](file_path)  # 下载图片
        print(f"收到图片消息，保存到: {file_path}")

        msg_record = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "FromUserName": msg['FromUserName'],
            "Content": f"图片已保存: {file_path}",
            "MsgType": msg['Type'],
        }
        self.messages.append(msg_record)

    @itchat.msg_register(itchat.content.VIDEO)
    def handle_video(self, msg):
        file_name = msg['FileName']
        file_path = self.get_save_path(file_name, folder="videos")
        msg['Text'](file_path)  # 下载视频
        print(f"收到视频消息，保存到: {file_path}")

        msg_record = {
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "FromUserName": msg['FromUserName'],
            "Content": f"视频已保存: {file_path}",
            "MsgType": msg['Type'],
        }
        self.messages.append(msg_record)

    def start_wechat(self):
        itchat.auto_login(hotReload=True)
        itchat.run()

    def export_messages(self):
        self.export_to_csv()
        self.export_to_json()

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
        if self.driver:
            self.driver.quit()

    def create_keyword_graph(self, keywords, edges):
        """创建关键字图并展示。"""
        G = nx.Graph()
        G.add_nodes_from(keywords)
        G.add_edges_from(edges)
        nx.draw(G, with_labels=True)
        plt.show()

if __name__ == "__main__":
    # 创建微信和学术搜索器实例，选择浏览器类型（'chrome' 或 'edge'）
    browser_choice = 'chrome'  # 可替换为 'edge'
    searcher = WeChatAcademicSearcher(browser=browser_choice, driver_path='path_to_your_driver')

    # 启动微信并记录消息
    try:
        searcher.start_wechat()
    except KeyboardInterrupt:
        print("正在导出消息记录...")
        searcher.export_messages()

    # 学术论文搜索示例
    try:
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
    except Exception as e:
        print(f"学术论文搜索过程中出错: {e}")
