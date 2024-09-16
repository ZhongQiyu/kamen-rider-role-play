import requests
from bs4 import BeautifulSoup
import os

url = 'https://wiki.tvnihon.com/wiki/Kamen_Rider_Blade_Cards'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
images = soup.find_all('img')

for image in images:
    # 获取图片的URL
    src = image.get('src')
    # 为图片构建完整的URL（如果需要）
    image_url = requests.compat.urljoin(url, src)
    # 获取图片的名字
    image_name = os.path.basename(src)
    # 下载图片
    with open(image_name, 'wb') as f:
        image_response = requests.get(image_url)
        f.write(image_response.content)
        print(f"已下载图片：{image_name}")
