import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 定义文本
text = '''
pip3 install, ERROR: Could not find a version, No matching distribution found, gnutools,
算法, 物理规律, 对算法添加物理规则, 降1倍, 降2倍,
项目结构, 模型比较, 性能评测, 部署方法, 量化处理, KV Cache,
株式会社, 财务报告, IJCAI, 游戏理论, 角色扮演
'''

# 创建词云对象
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

# 显示词云图像
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 不显示坐标轴
plt.show()
