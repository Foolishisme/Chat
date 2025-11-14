"""
FastAPI主应用
提供RAG对话系统的RESTful API接口
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import os
import json

from rag_service import rag_service
from config import settings


# 创建FastAPI应用
app = FastAPI(
    title="RAG对话系统",
    description="基于LangChain和Gemini的RAG对话系统 - Hello World版本",
    version="0.1.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)

# 挂载静态文件目录
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求/响应模型
class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., description="用户的问题", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "文档的主要内容是什么？"
            }
        }


class SourceDocument(BaseModel):
    """来源文档模型"""
    page: str = Field(..., description="页码")
    content: str = Field(..., description="文档内容片段")


class QuestionResponse(BaseModel):
    """问题响应模型"""
    question: str = Field(..., description="原始问题")
    answer: str = Field(..., description="生成的答案")
    sources: List[SourceDocument] = Field(..., description="来源文档")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "文档的主要内容是什么？",
                "answer": "根据文档内容，主要讨论了...",
                "sources": [
                    {
                        "page": "1",
                        "content": "文档的第一页内容摘要..."
                    }
                ]
            }
        }


class StatusResponse(BaseModel):
    """状态响应模型"""
    status: str = Field(..., description="服务状态")
    message: str = Field(..., description="状态信息")
    initialized: bool = Field(..., description="是否已初始化")


# API端点
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化RAG服务"""
    try:
        print("正在初始化RAG服务...")
        rag_service.initialize()
        print("RAG服务初始化成功！")
    except Exception as e:
        print(f"RAG服务初始化失败: {str(e)}")
        # 不抛出异常，允许应用启动，但API调用时会返回错误


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回前端页面"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {
        "message": "欢迎使用RAG对话系统！",
        "docs": "/docs",
        "health": "/health",
        "version": "0.1.0",
        "frontend": "前端页面未找到，请访问 /docs 使用API"
    }


@app.get("/health", response_model=StatusResponse, tags=["系统"])
async def health_check():
    """健康检查端点"""
    return StatusResponse(
        status="healthy" if rag_service._initialized else "initializing",
        message="RAG服务运行正常" if rag_service._initialized else "RAG服务正在初始化",
        initialized=rag_service._initialized
    )


@app.post("/chat", response_model=QuestionResponse, tags=["对话"])
async def chat(request: QuestionRequest):
    """
    对话接口（非流式）
    
    基于上传的PDF文档回答用户问题
    
    - **question**: 用户的问题（必填）
    
    返回:
    - **question**: 原始问题
    - **answer**: 基于文档生成的答案
    - **sources**: 答案的来源文档片段
    """
    try:
        if not rag_service._initialized:
            raise HTTPException(
                status_code=503,
                detail="RAG服务尚未初始化完成，请稍后再试"
            )
        
        # 执行RAG查询
        result = rag_service.query(request.question)
        
        # 构建响应
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceDocument(page=str(s["page"]), content=s["content"])
                for s in result["sources"]
            ]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时发生错误: {str(e)}"
        )


@app.post("/chat/stream", tags=["对话"])
async def chat_stream(request: QuestionRequest):
    """
    流式对话接口
    
    基于上传的PDF文档回答用户问题，采用流式输出
    
    - **question**: 用户的问题（必填）
    
    返回: Server-Sent Events (SSE) 流
    """
    if not rag_service._initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG服务尚未初始化完成，请稍后再试"
        )
    
    async def generate():
        try:
            for chunk in rag_service.query_stream(request.question):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = {
                "type": "error",
                "content": f"处理问题时发生错误: {str(e)}"
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/reset", tags=["管理"])
async def reset_vectorstore():
    """
    重置向量数据库
    
    重新加载PDF文档并重建向量索引
    """
    try:
        rag_service.reset_vectorstore()
        return {
            "status": "success",
            "message": "向量数据库已重置并重新加载"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"重置向量数据库时发生错误: {str(e)}"
        )


# 主函数
if __name__ == "__main__":
    # 运行FastAPI应用
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式下启用热重载
        log_level="info"
    )

