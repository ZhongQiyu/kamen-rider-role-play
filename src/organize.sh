#!/bin/bash

# 检查是否存在某个文件
if [ -f "yourfile.txt" ]; then
    echo "找到文件 yourfile.txt"
    # 在这里执行相应的命令
else
    echo "未找到文件 yourfile.txt"
    # 在这里执行其他命令
fi

# 检查是否存在某个目录
if [ -d "yourdirectory" ]; then
    echo "找到目录 yourdirectory"
    # 在这里执行相应的命令
else
    echo "未找到目录 yourdirectory"
    # 在这里执行其他命令
fi

