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
    