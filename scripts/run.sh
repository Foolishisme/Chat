#!/bin/bash
# RAG对话系统启动脚本

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "虚拟环境不存在，正在创建..."
    python -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "检查依赖..."
pip install -r requirements.txt

# 运行应用
echo "启动RAG对话系统..."
python -m app.main

