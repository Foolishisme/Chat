@echo off
REM RAG对话系统启动脚本 (Windows)

REM 切换到项目根目录
cd /d %~dp0\..

REM 检查虚拟环境
if not exist "venv\" (
    echo 虚拟环境不存在，正在创建...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 安装依赖
echo 检查依赖...
pip install -r requirements.txt

REM 运行应用
echo 启动RAG对话系统...
python -m app.main

pause

