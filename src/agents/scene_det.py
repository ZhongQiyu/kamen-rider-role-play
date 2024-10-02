from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path, threshold=30.0):
    # 创建视频管理器对象
    video_manager = VideoManager([video_path])
    # 创建场景管理器对象
    scene_manager = SceneManager()
    # 添加基于内容变化的场景检测器
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # 启动视频管理器
    video_manager.start()

    # 处理视频并检测场景
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取检测到的场景列表
    scene_list = scene_manager.get_scene_list()

    print(f'检测到 {len(scene_list)} 个场景。')

    # 输出场景的起止时间
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_timecode()
        end_time = scene[1].get_timecode()
        print(f'场景 {i+1}: 从 {start_time} 到 {end_time}')

    # 关闭视频管理器
    video_manager.release()

# 使用示例
video_path = 'your_video.mkv'
detect_scenes(video_path, threshold=30.0)



import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

# 场景抽取函数
def extract_scenes(video_path, output_dir, threshold=30.0):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载视频
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    # 使用 ContentDetector 进行场景检测
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # 开始视频处理
    video_manager.start()

    # 检测场景
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取所有场景的帧号范围
    scene_list = scene_manager.get_scene_list()

    # 提取每个场景并保存为图片
    cap = cv2.VideoCapture(video_path)
    for i, scene in enumerate(scene_list):
        start_frame, end_frame = scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret:
            scene_img_path = os.path.join(output_dir, f"scene_{i:04d}.png")
            cv2.imwrite(scene_img_path, frame)
            print(f"Scene {i} saved to {scene_img_path}")
    
    # 释放资源
    cap.release()
    video_manager.release()

# 示例使用
video_path = 'kamen_rider_blade_episode.mp4'
output_dir = 'scenes_output'
extract_scenes(video_path, output_dir, threshold=30.0)