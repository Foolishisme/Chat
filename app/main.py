"""
FastAPI主应用
提供RAG对话系统的RESTful API接口
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uvicorn
import os
import json
import uuid
from datetime import datetime

from app.services.rag_service import rag_service
from app.config import settings


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

# 挂载图片目录（用于显示提取的PDF图片）
if os.path.exists("data/images"):
    app.mount("/images", StaticFiles(directory="data/images"), name="images")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Session管理（内存存储，支持多用户会话）
sessions: Dict[str, List[Dict]] = {}  # {session_id: [{"role": "user/assistant", "content": "...", "timestamp": "..."}]}
MAX_HISTORY = 10  # 最多保留10轮对话（20条消息）


# 请求/响应模型
class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str = Field(..., description="用户的问题", min_length=1)
    session_id: Optional[str] = Field(None, description="会话ID，用于保持对话历史")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "文档的主要内容是什么？",
                "session_id": "550e8400-e29b-41d4-a716-446655440000"
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


# Session辅助函数
def get_or_create_session(session_id: Optional[str]) -> str:
    """获取或创建会话ID"""
    if not session_id:
        session_id = str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []
    return session_id


def get_session_history(session_id: str) -> List[Dict]:
    """获取会话历史（最近MAX_HISTORY轮对话）"""
    history = sessions.get(session_id, [])
    # 只保留最近的MAX_HISTORY轮对话（每轮2条消息：user + assistant）
    return history[-(MAX_HISTORY * 2):]


def add_to_session(session_id: str, role: str, content: str):
    """添加消息到会话历史"""
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    # 保持最大历史长度
    if len(sessions[session_id]) > MAX_HISTORY * 2:
        sessions[session_id] = sessions[session_id][-(MAX_HISTORY * 2):]


def format_history_for_prompt(history: List[Dict]) -> str:
    """格式化历史对话为提示词"""
    if not history:
        return ""
    
    formatted = "历史对话:\n"
    for msg in history:
        role_name = "用户" if msg["role"] == "user" else "AI助手"
        formatted += f"{role_name}: {msg['content']}\n"
    formatted += "\n"
    return formatted


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
    流式对话接口（支持对话记忆）
    
    基于上传的PDF文档回答用户问题，采用流式输出
    
    - **question**: 用户的问题（必填）
    - **session_id**: 会话ID（可选，不提供则自动创建新会话）
    
    返回: Server-Sent Events (SSE) 流，包含session_id
    """
    if not rag_service._initialized:
        raise HTTPException(
            status_code=503,
            detail="RAG服务尚未初始化完成，请稍后再试"
        )
    
    # 获取或创建会话ID
    session_id = get_or_create_session(request.session_id)
    
    # 获取历史对话
    history = get_session_history(session_id)
    history_text = format_history_for_prompt(history)
    
    # 添加用户问题到历史
    add_to_session(session_id, "user", request.question)
    
    async def generate():
        try:
            # 先发送session_id
            yield f"data: {json.dumps({'type': 'session_id', 'content': session_id}, ensure_ascii=False)}\n\n"
            
            # 收集完整答案用于保存到历史
            full_answer = ""
            
            # 流式生成答案（传递历史对话）
            for chunk in rag_service.query_stream(request.question, history_text):
                if chunk.get("type") == "token":
                    full_answer += chunk.get("content", "")
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            
            # 添加AI回答到历史
            add_to_session(session_id, "assistant", full_answer)
            
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


@app.post("/session/new", tags=["会话管理"])
async def new_session():
    """
    创建新会话
    
    返回新的session_id，用于开始全新的对话
    """
    session_id = str(uuid.uuid4())
    sessions[session_id] = []
    return {
        "status": "success",
        "session_id": session_id,
        "message": "新会话已创建"
    }


@app.delete("/session/{session_id}", tags=["会话管理"])
async def delete_session(session_id: str):
    """
    删除指定会话
    
    清空该会话的对话历史
    """
    if session_id in sessions:
        del sessions[session_id]
        return {
            "status": "success",
            "message": f"会话 {session_id} 已删除"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"会话 {session_id} 不存在"
        )


@app.get("/session/{session_id}/history", tags=["会话管理"])
async def get_session_history_endpoint(session_id: str):
    """
    获取指定会话的历史记录
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"会话 {session_id} 不存在"
        )
    return {
        "session_id": session_id,
        "history": sessions[session_id],
        "count": len(sessions[session_id])
    }


@app.post("/upload", tags=["文档管理"])
async def upload_document(file: UploadFile = File(...)):
    """
    上传PDF文档
    
    上传新的PDF文档并重建向量索引
    
    - **file**: PDF文件（必填）
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="只支持PDF文件格式"
        )
    
    try:
        # 保存文件
        file_path = os.path.join(os.path.dirname(settings.pdf_document_path), file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 重新加载并索引文档
        rag_service.load_document(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"文档 {file.filename} 已上传并索引完成",
            "size": len(content)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"上传文档时发生错误: {str(e)}"
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
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式下启用热重载
        log_level="info"
    )

