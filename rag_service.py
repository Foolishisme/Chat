"""
RAG服务实现
负责文档加载、向量化存储和检索增强生成
"""
import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

from config import settings


class RAGService:
    """RAG服务类"""
    
    def __init__(self):
        """初始化RAG服务"""
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self._initialized = False
    
    def initialize(self):
        """初始化服务组件"""
        if self._initialized:
            return
        
        # 初始化本地嵌入模型（支持中文，约420MB）
        print("正在初始化本地embedding模型（首次运行会自动下载）...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},  # 使用CPU，如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
        print("本地embedding模型初始化完成！")
        
        # 初始化LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        # 检查向量数据库是否已存在
        faiss_index_path = settings.chroma_persist_directory + "/index.faiss"
        if os.path.exists(faiss_index_path):
            print("加载已有的向量数据库...")
            self.vectorstore = FAISS.load_local(
                settings.chroma_persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("创建新的向量数据库...")
            self._load_and_index_documents()
        
        # 创建检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # 返回最相关的3个文档片段
            ),
            return_source_documents=True
        )
        
        self._initialized = True
        print("RAG服务初始化完成！")
    
    def _load_and_index_documents(self):
        """加载PDF文档并创建向量索引"""
        if not os.path.exists(settings.pdf_document_path):
            raise FileNotFoundError(
                f"PDF文档不存在: {settings.pdf_document_path}"
            )
        
        # 加载PDF文档
        print(f"正在加载PDF文档: {settings.pdf_document_path}")
        loader = PyPDFLoader(settings.pdf_document_path)
        documents = loader.load()
        print(f"成功加载 {len(documents)} 页文档")
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 每个文本块的大小
            chunk_overlap=200,  # 文本块之间的重叠
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档已分割为 {len(splits)} 个文本块")
        
        # 创建向量存储
        print("正在创建向量索引...")
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        # 保存向量数据库
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        self.vectorstore.save_local(settings.chroma_persist_directory)
        print("向量索引创建完成！")
    
    def query(self, question: str) -> dict:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            
        Returns:
            包含答案和来源文档的字典
        """
        if not self._initialized:
            self.initialize()
        
        # 执行查询
        result = self.qa_chain.invoke({"query": question})
        
        # 提取来源文档信息
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({
                    "page": doc.metadata.get("page", "未知"),
                    "content": doc.page_content[:200] + "..."  # 只显示前200字符
                })
        
        return {
            "answer": result["result"],
            "sources": sources,
            "question": question
        }
    
    def reset_vectorstore(self):
        """重置向量数据库（重新加载文档）"""
        import shutil
        if os.path.exists(settings.chroma_persist_directory):
            shutil.rmtree(settings.chroma_persist_directory)
        self._initialized = False
        self.initialize()


# 创建全局RAG服务实例
rag_service = RAGService()

