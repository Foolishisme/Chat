"""
RAG服务实现 - V2 多模态版本
负责文档加载、向量化存储和检索增强生成（支持文本+图片）
使用CLIP直接向量化图片，检索时返回原图
"""
import os
import base64
from typing import List, Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.embeddings import Embeddings
import numpy as np
import faiss

from app.config import settings
from app.services.image_processor import create_image_processor


class CLIPImageEmbeddings(Embeddings):
    """CLIP图片嵌入适配器，用于FAISS"""
    
    def __init__(self, image_processor):
        self.image_processor = image_processor
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """这个方法不会被调用，因为我们直接提供embedding"""
        return [[0.0] * 512] * len(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """将查询文本用CLIP向量化，实现跨模态检索"""
        embedding = self.image_processor.get_text_embedding(text)
        return embedding.tolist()


class RAGService:
    """RAG服务类 - 支持多模态（文本+图片）"""
    
    def __init__(self):
        """初始化RAG服务"""
        self.text_embeddings = None  # 文本嵌入模型
        self.image_processor = None  # 图片处理器（包含CLIP）
        self.text_vectorstore = None  # 文本向量库
        self.image_vectorstore = None  # 图片向量库
        self.llm = None
        self.qa_chain = None
        self._initialized = False
        self._reranker = None  # Cross-Encoder重排序模型
        self.use_reranking = True  # 是否启用重排序
    
    def initialize(self):
        """初始化服务组件"""
        if self._initialized:
            return
        
        # 初始化文本嵌入模型
        print("正在初始化文本embedding模型...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 文本embedding模型初始化完成！")
        
        # 初始化图片处理器（包含CLIP）
        self.image_processor = create_image_processor()
        
        # 初始化多模态LLM（支持图片输入）
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # 使用支持Vision的模型
            google_api_key=settings.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        print("✅ 多模态LLM初始化完成（Gemini 2.0 Flash Exp）")
        
        # 检查向量数据库是否已存在
        text_index_path = settings.chroma_persist_directory + "/text_index.faiss"
        image_index_path = settings.chroma_persist_directory + "/image_index.faiss"
        
        if os.path.exists(text_index_path):
            print("加载已有的文本向量数据库...")
            self.text_vectorstore = FAISS.load_local(
                settings.chroma_persist_directory,
                self.text_embeddings,
                allow_dangerous_deserialization=True,
                index_name="text_index"
            )
        
        if os.path.exists(image_index_path):
            print("加载已有的图片向量数据库...")
            clip_embeddings = CLIPImageEmbeddings(self.image_processor)
            self.image_vectorstore = FAISS.load_local(
                settings.chroma_persist_directory,
                clip_embeddings,
                allow_dangerous_deserialization=True,
                index_name="image_index"
            )
        
        if not os.path.exists(text_index_path):
            print("创建新的向量数据库...")
            self._load_and_index_documents()
        
        # 创建检索QA链（仅用于文本检索）
        if self.text_vectorstore:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.text_vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True
            )
        
        self._initialized = True
        print("✅ 多模态RAG服务初始化完成！")
    
    def _get_reranker(self):
        """获取或初始化重排序模型"""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Cross-Encoder重排序模型加载完成")
            except ImportError:
                print("⚠️  sentence-transformers未安装，将使用相似度重排序")
                self._reranker = "similarity"  # 使用相似度作为fallback
        return self._reranker
    
    def _rerank_documents(self, question: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        使用Cross-Encoder重排序文档
        
        Args:
            question: 用户问题
            documents: 待排序的文档列表
            top_k: 返回的文档数量
            
        Returns:
            重排序后的Top-K文档
        """
        if len(documents) <= top_k:
            return documents
        
        reranker = self._get_reranker()
        
        if reranker == "similarity":
            # 使用embedding相似度作为fallback
            query_embedding = self.text_embeddings.embed_query(question)
            scored_docs = []
            
            for doc in documents:
                doc_text = doc.page_content[:500]  # 取前500字符避免过长
                doc_embedding = self.text_embeddings.embed_query(doc_text)
                # 计算余弦相似度
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                scored_docs.append((doc, similarity))
            
            # 按相似度排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:top_k]]
        else:
            # 使用Cross-Encoder重排序
            pairs = [(question, doc.page_content[:500]) for doc in documents]
            scores = reranker.predict(pairs)
            
            # 排序并返回Top-K
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in ranked[:top_k]]
    
    def _load_and_index_documents(self):
        """加载PDF文档（文本+图片）并创建向量索引"""
        if not os.path.exists(settings.pdf_document_path):
            raise FileNotFoundError(
                f"PDF文档不存在: {settings.pdf_document_path}"
            )
        
        # 1. 加载PDF文本内容
        print(f"正在加载PDF文档: {settings.pdf_document_path}")
        loader = PyPDFLoader(settings.pdf_document_path)
        documents = loader.load()
        print(f"成功加载 {len(documents)} 页文档")
        
        # 2. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        text_splits = text_splitter.split_documents(documents)
        print(f"文本已分割为 {len(text_splits)} 个文本块")
        
        # 3. 提取和向量化图片（使用CLIP）
        print("\n开始处理PDF中的图片...")
        images_info = self.image_processor.process_pdf_images(settings.pdf_document_path)
        
        # 4. 创建文本向量存储
        print("\n正在创建文本向量索引...")
        self.text_vectorstore = FAISS.from_documents(
            documents=text_splits,
            embedding=self.text_embeddings
        )
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        self.text_vectorstore.save_local(
            settings.chroma_persist_directory,
            index_name="text_index"
        )
        print(f"✅ 文本向量索引已保存 ({len(text_splits)} 个文本块)")
        
        # 5. 创建图片向量存储（如果有图片）
        if images_info:
            print("\n正在创建图片向量索引（CLIP）...")
            
            # 准备图片文档
            image_documents = []
            image_embeddings_list = []
            
            for img_info in images_info:
                # 创建简短的文档对象
                doc = Document(
                    page_content=f"[图片 - 第{img_info['page']}页]",
                    metadata={
                        "page": img_info['page'],
                        "source": "image",
                        "image_path": img_info['image_path'],
                        "image_index": img_info['image_index'],
                        "type": "image",
                        "width": img_info['width'],
                        "height": img_info['height']
                    }
                )
                image_documents.append(doc)
                image_embeddings_list.append(img_info['embedding'])
            
            # 使用预计算的CLIP向量创建FAISS索引
            clip_embeddings = CLIPImageEmbeddings(self.image_processor)
            
            # 手动创建FAISS索引
            dimension = 512  # CLIP向量维度
            embeddings_array = np.array(image_embeddings_list, dtype=np.float32)
            
            # 创建FAISS索引
            index = faiss.IndexFlatIP(dimension)  # 内积相似度
            index.add(embeddings_array)
            
            # 创建文档存储
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(image_documents)})
            index_to_docstore_id = {i: str(i) for i in range(len(image_documents))}
            
            # 创建向量存储
            self.image_vectorstore = FAISS(
                embedding_function=clip_embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            # 保存图片向量存储
            self.image_vectorstore.save_local(
                settings.chroma_persist_directory,
                index_name="image_index"
            )
            print(f"✅ 图片向量索引已保存 ({len(image_documents)} 张图片)")
        else:
            print("PDF中没有找到图片")
        
        print("\n✅ 所有向量索引创建完成！")
    
    def query(self, question: str) -> dict:
        """
        执行RAG查询（非流式）
        
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
                    "content": doc.page_content[:200] + "..."
                })
        
        return {
            "answer": result["result"],
            "sources": sources,
            "question": question
        }
    
    def query_stream(self, question: str, history_text: str = ""):
        """
        执行流式RAG查询（支持多模态：文本+图片）
        
        Args:
            question: 用户问题
            history_text: 格式化的历史对话文本
            
        Yields:
            流式生成的文本块、来源文档和图片
        """
        if not self._initialized:
            self.initialize()
        
        # 1. 检索相关文本（Top 20，然后重排序到Top 5）
        text_docs = []
        if self.text_vectorstore:
            if self.use_reranking:
                # 优化方案：检索Top-20，然后重排序到Top-5
                text_retriever = self.text_vectorstore.as_retriever(search_kwargs={"k": 20})
                candidate_docs = text_retriever.invoke(question)
                text_docs = self._rerank_documents(question, candidate_docs, top_k=5)
            else:
                # 原始方案：直接检索Top-3
                text_retriever = self.text_vectorstore.as_retriever(search_kwargs={"k": 3})
                text_docs = text_retriever.invoke(question)
        
        # 2. 检索相关图片（Top 2）
        image_docs = []
        image_paths = []
        if self.image_vectorstore:
            try:
                image_retriever = self.image_vectorstore.as_retriever(search_kwargs={"k": 2})
                image_docs = image_retriever.invoke(question)
                image_paths = [doc.metadata.get("image_path") for doc in image_docs if doc.metadata.get("image_path")]
            except Exception as e:
                print(f"图片检索失败: {str(e)}")
        
        # 3. 构建文本上下文
        text_context = "\n\n".join([doc.page_content for doc in text_docs])
        
        # 4. 构建多模态prompt
        prompt_text = f"""基于以下文档内容、图片和历史对话回答问题。

{history_text}文档内容:
{text_context}

{'提供了 ' + str(len(image_paths)) + ' 张相关图片供参考。' if image_paths else ''}

当前问题: {question}

请结合文本和图片内容给出详细回答。"""
        
        # 5. 构建消息（包含文本和图片）
        message_content = [{"type": "text", "text": prompt_text}]
        
        # 6. 添加图片到prompt
        for img_path in image_paths[:2]:  # 最多2张图片
            try:
                with open(img_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                    message_content.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_data}"
                    })
            except Exception as e:
                print(f"读取图片失败 {img_path}: {str(e)}")
        
        message = HumanMessage(content=message_content)
        
        # 7. 流式生成答案
        for chunk in self.llm.stream([message]):
            if hasattr(chunk, 'content'):
                yield {
                    "type": "token",
                    "content": chunk.content
                }
        
        # 8. 发送来源信息（文本+图片）
        sources = []
        
        # 添加文本来源
        for doc in text_docs:
            sources.append({
                "page": doc.metadata.get("page", "未知"),
                "content": doc.page_content[:200] + "...",
                "type": "text"
            })
        
        # 添加图片来源
        for doc in image_docs:
            sources.append({
                "page": doc.metadata.get("page", "未知"),
                "content": f"[图片 - 第{doc.metadata.get('page')}页]",
                "type": "image",
                "image_path": doc.metadata.get("image_path", "")
            })
        
        yield {
            "type": "sources",
            "content": sources
        }
    
    def load_document(self, pdf_path: str):
        """
        加载新文档（文本+图片）并更新向量索引
        
        Args:
            pdf_path: PDF文档路径
        """
        print(f"正在加载新文档: {pdf_path}")
        
        # 1. 加载文本内容
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"成功加载 {len(documents)} 页文档")
        
        # 2. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        text_splits = text_splitter.split_documents(documents)
        print(f"文本已分割为 {len(text_splits)} 个文本块")
        
        # 3. 提取和向量化图片
        print("\n开始处理PDF中的图片...")
        images_info = self.image_processor.process_pdf_images(pdf_path)
        
        # 4. 重建文本向量存储
        print("\n正在重建文本向量索引...")
        self.text_vectorstore = FAISS.from_documents(
            documents=text_splits,
            embedding=self.text_embeddings
        )
        self.text_vectorstore.save_local(
            settings.chroma_persist_directory,
            index_name="text_index"
        )
        print(f"✅ 文本向量索引已保存")
        
        # 5. 重建图片向量存储
        if images_info:
            print("\n正在重建图片向量索引...")
            
            image_documents = []
            image_embeddings_list = []
            
            for img_info in images_info:
                doc = Document(
                    page_content=f"[图片 - 第{img_info['page']}页]",
                    metadata={
                        "page": img_info['page'],
                        "source": "image",
                        "image_path": img_info['image_path'],
                        "image_index": img_info['image_index'],
                        "type": "image",
                        "width": img_info['width'],
                        "height": img_info['height']
                    }
                )
                image_documents.append(doc)
                image_embeddings_list.append(img_info['embedding'])
            
            # 创建图片向量存储
            clip_embeddings = CLIPImageEmbeddings(self.image_processor)
            dimension = 512
            embeddings_array = np.array(image_embeddings_list, dtype=np.float32)
            
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings_array)
            
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(image_documents)})
            index_to_docstore_id = {i: str(i) for i in range(len(image_documents))}
            
            self.image_vectorstore = FAISS(
                embedding_function=clip_embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            self.image_vectorstore.save_local(
                settings.chroma_persist_directory,
                index_name="image_index"
            )
            print(f"✅ 图片向量索引已保存 ({len(image_documents)} 张图片)")
        else:
            print("PDF中没有找到图片")
            # 删除旧的图片索引
            self.image_vectorstore = None
        
        # 6. 重建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.text_vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
        
        print(f"\n✅ 文档 {pdf_path} 加载完成！")
    
    def reset_vectorstore(self):
        """重置向量数据库（重新加载文档）"""
        import shutil
        if os.path.exists(settings.chroma_persist_directory):
            shutil.rmtree(settings.chroma_persist_directory)
        self._initialized = False
        self.initialize()


# 创建全局RAG服务实例
rag_service = RAGService()

