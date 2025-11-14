# RAG对话系统 - Hello World版本

基于LangChain、Chroma和Google Gemini 2.5 Flash的简单RAG（检索增强生成）对话系统。

## 项目简介

这是一个简单的RAG对话系统，可以读取PDF文档并回答相关问题。系统使用向量数据库存储文档内容，通过语义检索找到相关内容，然后使用大语言模型生成答案。

## 技术栈

- **LangChain**: LLM应用开发框架
- **FAISS**: 向量数据库（替代Chroma，兼容性更好）
- **HuggingFace Sentence-Transformers**: 本地向量化模型（无需API，离线可用）
- **Google Gemini 2.5 Flash**: 大语言模型API（仅用于生成回答）
- **FastAPI**: Web框架
- **Poetry**: 依赖管理工具
- **Python**: 3.10+

## 项目结构

```
.
├── main.py                 # FastAPI应用主文件
├── rag_service.py          # RAG服务实现
├── config.py               # 配置管理
├── static/                 # 前端静态文件目录
│   └── index.html          # Web对话界面
├── pyproject.toml          # Poetry配置文件
├── .env.example            # 环境变量示例
├── .gitignore              # Git忽略文件
├── README.md               # 项目说明文档
├── 文档.pdf                # 知识库文档
└── chroma_db/              # 向量数据库（自动生成）
```

## 快速开始

### 1. 环境要求

- Python 3.10 或更高版本
- Poetry（推荐）或 pip

### 2. 安装依赖

#### 使用Poetry（推荐）

```bash
# 安装Poetry（如果尚未安装）
pip install poetry

# 安装项目依赖
poetry install

# 激活虚拟环境
poetry shell
```

#### 使用pip

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install fastapi uvicorn langchain langchain-google-genai langchain-community chromadb pypdf python-dotenv pydantic pydantic-settings
```

### 3. 配置环境变量

```bash
# 复制环境变量示例文件
# Windows PowerShell
Copy-Item env.example .env

# Linux/Mac
cp env.example .env

# 编辑.env文件，填入你的API密钥
# GOOGLE_API_KEY=你的真实Google_API密钥
```

**获取Google API Key:**
访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取免费的API密钥。

### 4. 准备文档

确保根目录下有 `文档.pdf` 文件，这是系统将要处理的知识库文档。

### 5. 运行应用

```bash
# 使用Poetry
poetry run python main.py

# 或直接运行
python main.py
```

应用将在 `http://localhost:8000` 启动。

### 6. 访问系统

**🎨 Web界面（推荐）**:
- 前端界面: http://localhost:8000

**📚 API文档**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API接口说明

### 1. 根路径

```
GET /
```

返回API基本信息。

### 2. 健康检查

```
GET /health
```

检查服务运行状态。

**响应示例:**
```json
{
  "status": "healthy",
  "message": "RAG服务运行正常",
  "initialized": true
}
```

### 3. 对话接口

```
POST /chat
```

基于PDF文档回答用户问题。

**请求体:**
```json
{
  "question": "文档的主要内容是什么？"
}
```

**响应示例:**
```json
{
  "question": "文档的主要内容是什么？",
  "answer": "根据文档内容，主要讨论了...",
  "sources": [
    {
      "page": "1",
      "content": "文档的第一页内容摘要..."
    }
  ]
}
```

### 4. 重置向量数据库

```
POST /reset
```

重新加载PDF文档并重建向量索引。

## 使用示例

### 使用Web界面（最简单）

直接在浏览器中打开 http://localhost:8000，您将看到一个美观的对话界面：

- ✨ 现代化的渐变UI设计
- 💬 实时对话交互
- 📚 显示答案来源
- ⚡ 系统状态实时监控
- 📱 响应式设计，支持移动端

在输入框中输入问题，点击"发送"即可开始对话！

### 使用curl

```bash
# 发送问题
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "文档中讲了什么内容？"}'
```

### 使用Python

```python
import requests

# 发送问题
response = requests.post(
    "http://localhost:8000/chat",
    json={"question": "文档中讲了什么内容？"}
)

result = response.json()
print(f"问题: {result['question']}")
print(f"答案: {result['answer']}")
print(f"来源: {len(result['sources'])} 个文档片段")
```

### 使用Swagger UI

访问 http://localhost:8000/docs，在交互式API文档中直接测试接口。

## 前端界面说明

系统提供了一个简洁美观的Web前端界面：

### 功能特性

- **🎨 现代化设计**: 使用渐变色和流畅动画
- **💬 实时对话**: 即时显示问题和答案
- **📚 来源追溯**: 显示答案来自文档的哪一页
- **⚡ 状态监控**: 实时显示系统初始化状态
- **📱 响应式布局**: 完美支持手机、平板和桌面
- **🚀 快速提问**: 预设示例问题，一键提问

### 文件位置

前端代码位于 `static/index.html`，您可以自由修改和定制界面样式。

## 工作原理

1. **文档加载**: 系统启动时，使用PyPDFLoader加载PDF文档
2. **文本分割**: 将文档分割成较小的文本块（chunk），便于检索
3. **向量化**: 使用本地HuggingFace模型将文本块转换为向量（**完全本地，无需网络**）
4. **存储**: 将向量存储在FAISS数据库中（`./chroma_db`目录）
5. **检索**: 用户提问时，将问题向量化并检索最相关的文档片段
6. **生成**: 将检索到的文档片段和问题一起发送给Gemini API生成答案（**仅此步骤需要网络**）

### 为什么采用本地向量化？

- ⚡ **速度快**: 本地计算比API调用快10-100倍
- 🔒 **隐私保护**: 文档内容不会上传到外部服务器
- 📡 **网络友好**: 向量化过程不依赖网络稳定性
- 💰 **零成本**: 无API调用费用
- 🔌 **离线可用**: 向量化完成后可完全离线使用（仅回答需要网络）

## 配置说明

### 环境变量

在 `.env` 文件中配置以下参数：

- `GOOGLE_API_KEY`: Google Gemini API密钥（必填）
- `CHROMA_PERSIST_DIRECTORY`: Chroma数据库存储目录（默认：./chroma_db）
- `PDF_DOCUMENT_PATH`: PDF文档路径（默认：./文档.pdf）

### RAG参数调整

在 `rag_service.py` 中可以调整以下参数：

- `model_name`: embedding模型名称（默认：`paraphrase-multilingual-MiniLM-L12-v2`，支持中文）
- `chunk_size`: 文本分割块大小（默认：1000）
- `chunk_overlap`: 文本块重叠大小（默认：200）
- `k`: 检索返回的文档数量（默认：3）
- `temperature`: LLM生成温度（默认：0.7）

### 本地Embedding模型说明

当前使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`：
- **大小**: 约420MB
- **语言**: 支持50+种语言（包括中文）
- **速度**: 快速（CPU可运行）
- **质量**: 良好

其他可选模型：
- `distiluse-base-multilingual-cased-v2`: 约500MB，质量更好
- `shibing624/text2vec-base-chinese`: 专门针对中文优化
- `BAAI/bge-small-zh-v1.5`: 中文专用，性能优秀

## 开发说明

### 项目特点

- ✅ 遵循OpenAPI规范
- ✅ 自动生成Swagger文档
- ✅ 使用Poetry进行依赖管理
- ✅ 配置与代码分离
- ✅ 类型提示和数据验证
- ✅ 完整的错误处理

### 后续扩展建议

1. 添加对话历史记录
2. 支持多种文档格式（Word, TXT, Markdown等）
3. 实现多文档管理
4. 添加用户认证
5. 实现流式响应
6. 添加对话上下文管理
7. 优化检索策略（如混合检索、重排序等）
8. 添加缓存机制

## 常见问题

### 1. API密钥错误

确保在 `.env` 文件中正确配置了 `GOOGLE_API_KEY`。

### 2. 找不到PDF文档

确保 `文档.pdf` 文件存在于项目根目录，或在 `.env` 中配置正确的路径。

### 3. 向量数据库初始化失败

删除 `chroma_db` 目录后重启应用，系统会重新创建向量索引。

### 4. 首次启动很慢

首次运行时会自动下载embedding模型（约420MB），这是正常的。下载完成后会缓存在本地，后续启动会很快。

### 5. 向量化进度

首次向量化大文档可能需要1-3分钟（取决于CPU性能）。完成后会在根目录生成 `chroma_db/` 文件夹。

### 4. 依赖安装失败

尝试更新pip和Poetry到最新版本：
```bash
pip install --upgrade pip poetry
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

---

**注意**: 这是一个学习和演示项目，不建议直接用于生产环境。生产环境需要考虑更多的安全性、性能优化和错误处理。

---

## 开发日志

### 2025-11-14: 项目精简优化
- **完成内容**: 清理项目多余文件，保留核心对话功能
- **删除内容**:
  - 性能测试工具（8个文件）：`compare_results.py`、`performance_comparison.py`、`performance_dashboard.py`、`quick_comparison.py`、`run_comparison_experiment.py`、`simple_comparison.py` 及相关输出文件
  - 多余文档（5个文件）：`COMPARISON_GUIDE.md`、`FRONTEND_GUIDE.md`、`PERFORMANCE_TOOLS.md`、`PROJECT_AUDIT.md`、`PROJECT_STATUS.md`
  - 备份文件（2个）：`rag_service.py.backup`、`rag_service.py.backup_20251113_165823`
  - result/ 文件夹
- **保留内容**:
  - 核心功能代码：`main.py`、`rag_service.py`、`config.py`
  - Web界面：`static/index.html`
  - 配置文件：`pyproject.toml`、`requirements.txt`、`env.example`
  - 启动脚本：`run.bat`、`run.sh`
  - 文档：简化后的 `README.md`、`需求说明`、`文档.pdf`
- **优化效果**: 项目结构更清晰，专注于核心RAG对话功能

### 2025-11-13: 项目初始化
- **完成内容**: 创建了一个基于RAG（检索增强生成）的简单对话系统
- **技术栈**:
  - LangChain: 用于构建LLM应用的框架
  - FAISS: 向量数据库（替代Chroma，兼容性更好）
  - HuggingFace Sentence-Transformers: 本地向量化模型
  - Google Gemini 2.5 Flash: 大语言模型API
  - FastAPI: Web框架，提供RESTful API
  - Poetry: Python依赖管理工具
  - Python 3.10+
- **主要功能**:
  1. PDF文档加载和本地向量化存储（**无需网络**）
  2. 基于语义检索的RAG问答系统
  3. FastAPI接口，提供Swagger自动文档
  4. 完整的配置管理和错误处理
  5. 支持向量数据库持久化和重置
- **文件结构**:
  - `main.py`: FastAPI应用主文件，提供API接口
  - `rag_service.py`: RAG服务核心实现（使用本地embedding模型）
  - `config.py`: 配置管理模块
  - `pyproject.toml`: Poetry依赖配置
  - `.env.example`: 环境变量模板
  - `requirements.txt`: pip依赖文件
  - `run.bat` / `run.sh`: 快速启动脚本
- **特点**:
  - ⚡ **本地向量化**: 速度快，不依赖网络，保护隐私
  - 📡 **网络友好**: 仅LLM回答需要API，向量化完全本地
  - 🔒 **数据安全**: 文档内容不上传外部服务器
  - 遵循OpenAPI规范
  - API密钥配置灵活（通过环境变量）
  - 完整的类型提示和数据验证
  - 自动生成Swagger文档

