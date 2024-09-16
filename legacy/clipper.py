import os
import cv2

# 定义视频文件和保存帧的目录
video_path = 'C:\\Users\\MSI\\Desktop\\test_video.mp4'
save_dir = 'C:\\Users\\MSI\\Desktop\\Frames'

# 如果保存帧的目录不存在，则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)  # 添加exist_ok=True参数

# 打开视频文件
video = cv2.VideoCapture(video_path)

count = 0
while True:
    success, frame = video.read()
    if not success:
        break
    
    # 构建每一帧的完整路径
    frame_path = os.path.join(save_dir, f"frame_{count:04}.png")
    
    # 保存帧到指定路径
    cv2.imwrite(frame_path, frame)
    
    count += 1

# 释放视频对象
video.release()
