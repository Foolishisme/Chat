"""
允许使用 python -m app 直接运行应用
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",  # 使用字符串格式以支持 reload
        host="0.0.0.0",
        port=8100,
        reload=True,  # 开发模式下启用热重载
        log_level="info"
    )

