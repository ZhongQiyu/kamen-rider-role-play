# wechat_msg_crawler.py

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