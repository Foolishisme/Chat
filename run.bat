@echo off
echo 启动RAG对话系统...
echo.

REM 检查.env文件是否存在
if not exist .env (
    echo 错误: 未找到.env文件
    echo 请复制.env.example为.env并配置你的API密钥
    pause
    exit /b 1
)

REM 运行应用
python main.py

pause

