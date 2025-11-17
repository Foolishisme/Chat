"""
配置文件管理
使用 pydantic-settings 管理环境变量
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置"""
    
    # Google Gemini API Key
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    
    # 向量数据库持久化目录
    chroma_persist_directory: str = Field(
        default="./data/vector_db",
        alias="CHROMA_PERSIST_DIRECTORY"
    )
    
    # PDF文档路径
    pdf_document_path: str = Field(
        default="./data/documents/文档.pdf",
        alias="PDF_DOCUMENT_PATH"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()

