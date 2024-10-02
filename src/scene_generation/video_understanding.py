# video_understanding.py

import os
import cv2
import requests
from bs4 import BeautifulSoup

def extract_frames_from_video(video_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        frame_path = os.path.join(save_dir, f"frame_{count:04}.png")
        cv2.imwrite(frame_path, frame)
        count += 1
    video.release()

def download_images_from_wiki(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')
    for image in images:
        src = image.get('src')
        image_url = requests.compat.urljoin(url, src)
        image_name = os.path.basename(src)
        with open(image_name, 'wb') as f:
            image_response = requests.get(image_url)
            f.write(image_response.content)
            print(f"已下载图片：{image_name}")

if __name__ == "__main__":
    # 视频帧提取
    video_path = 'C:\\Users\\MSI\\Desktop\\test_video.mp4'
    save_dir = 'C:\\Users\\MSI\\Desktop\\Frames'
    extract_frames_from_video(video_path, save_dir)

    # 下载图片
    wiki_url = 'https://wiki.tvnihon.com/wiki/Kamen_Rider_Blade_Cards'
    download_images_from_wiki(wiki_url)
