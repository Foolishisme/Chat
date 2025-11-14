#!/bin/bash

echo "启动RAG对话系统..."
echo

# 检查.env文件是否存在
if [ ! -f .env ]; then
    echo "错误: 未找到.env文件"
    echo "请复制.env.example为.env并配置你的API密钥"
    exit 1
fi

# 运行应用
python main.py

