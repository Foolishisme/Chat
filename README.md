# 多模态RAG智能对话系统 V2.1 🚀

> 基于 LangChain + CLIP + Gemini 的生产级多模态RAG系统  
> 支持PDF文档（文本+图片）智能问答、对话记忆、流式输出

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 项目简介

**多模态RAG智能对话系统**是一个生产级的文档问答助手，能够深度理解PDF文档中的**文本和图片**内容，提供准确、智能的问答服务。

### 🎯 核心能力

- **📝 双模态理解**：同时处理文本和图片，LLM可以直接"看"原图回答
- **🧠 对话记忆**：支持10轮对话历史，理解上下文和指代
- **⚡ 流式输出**：实时打字机效果，用户体验优秀
- **🎨 Markdown渲染**：支持代码高亮、表格、列表等格式化显示
- **🔍 语义检索**：基于FAISS向量数据库的高效语义搜索
- **📤 动态上传**：支持运行时上传新文档，自动重建索引

### 🌟 V2.1 版本亮点

相比传统RAG系统的重大突破：

```
传统方案：图片 → AI文字描述 → 向量化
问题：信息损失、速度慢、成本高

V2.1方案：图片 → CLIP直接向量化 → 原图传给LLM
优势：
  ✅ 信息无损（LLM看原图）
  ✅ 速度提升87%（1.3秒 vs 10秒）
  ✅ 成本降低50%（减少API调用）
  ✅ 检索准确性提升（基于视觉特征）
```

---

## 🏗️ 项目架构

### 目录结构

```
Chat/
├── 📁 app/                          # 应用核心代码
│   ├── __init__.py
│   ├── __main__.py                  # 模块入口 (python -m app)
│   ├── main.py                      # FastAPI应用主文件
│   ├── config.py                    # 配置管理
│   │
│   ├── 📁 services/                 # 业务服务层
│   │   ├── __init__.py
│   │   ├── rag_service.py           # RAG核心服务
│   │   └── image_processor.py       # 图片处理服务
│   │
│   ├── 📁 models/                   # 数据模型
│   │   └── __init__.py
│   │
│   └── 📁 utils/                    # 工具函数
│       └── __init__.py
│
├── 📁 static/                       # 前端静态资源
│   └── index.html                   # 单页应用界面
│
├── 📁 data/                         # 数据存储目录
│   ├── documents/                   # PDF文档存储
│   ├── images/                      # 提取的图片
│   └── vector_db/                   # 向量数据库
│       ├── text_index.faiss         # 文本向量索引
│       ├── text_index.pkl
│       ├── image_index.faiss        # 图片向量索引
│       └── image_index.pkl
│
├── 📁 docs/                         # 项目文档
│   ├── api/                         # API文档
│   ├── guides/                      # 用户指南
│   ├── technical/                   # 技术文档
│   └── development/                 # 开发文档
│
├── 📁 scripts/                      # 启动脚本
│   ├── run.sh                       # Linux/Mac启动脚本
│   └── run.bat                      # Windows启动脚本
│
├── 📁 tests/                        # 测试代码
│   └── __init__.py
│
├── 📁 configs/                      # 配置文件
│   └── .env.example                 # 环境变量示例
│
├── .gitignore                       # Git忽略文件
├── README.md                        # 项目说明
├── requirements.txt                 # Python依赖
└── pyproject.toml                   # Poetry配置
```

### 技术架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面 (Frontend)                    │
│  HTML/CSS/JavaScript + Marked.js + Highlight.js             │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP/SSE
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Web服务层 (FastAPI)                       │
│  - REST API (/chat, /upload, /session)                      │
│  - 静态文件服务                                               │
│  - CORS中间件                                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG服务层 (RAGService)                    │
│  - 文档加载与分割                                              │
│  - 双向量库管理                                                │
│  - 多模态检索                                                  │
│  - 对话历史管理                                                │
└─────────────────────────────────────────────────────────────┘
        │                                        │
        ▼                                        ▼
┌────────────────────┐              ┌────────────────────┐
│  文本处理模块       │              │  图片处理模块       │
│  - PyPDFLoader     │              │  - PyMuPDF提取     │
│  - 文本分割         │              │  - CLIP向量化      │
│  - Sentence-Trans  │              │  - 原图存储         │
│    (384维)         │              │    (512维)         │
└────────────────────┘              └────────────────────┘
        │                                        │
        ▼                                        ▼
┌────────────────────┐              ┌────────────────────┐
│  FAISS文本向量库    │              │  FAISS图片向量库    │
│  text_index.faiss  │              │  image_index.faiss │
└────────────────────┘              └────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI模型层 (Google Gemini)                     │
│  - Gemini 2.0 Flash Exp (多模态LLM)                          │
│  - 输入：文本上下文 + 原图base64                               │
│  - 输出：流式生成答案                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 技术栈

### 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| HTML5/CSS3/ES6 | - | 基础界面 |
| Marked.js | 12.0+ | Markdown解析 |
| Highlight.js | 11.0+ | 代码高亮 |
| Fetch API | - | 流式数据接收 |

### 后端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 编程语言 |
| FastAPI | 0.109+ | Web框架 |
| Uvicorn | 0.27+ | ASGI服务器 |
| LangChain | 0.1+ | LLM应用框架 |
| LangChain-Google-GenAI | 0.0.6+ | Gemini集成 |
| PyPDF | 4.0+ | PDF文本解析 |
| PyMuPDF (fitz) | 1.23+ | PDF图片提取 |

### AI模型

| 模型 | 用途 | 维度 |
|------|------|------|
| Gemini 2.0 Flash Exp | 多模态LLM（文本+图片理解） | - |
| CLIP ViT-B/32 | 图片向量化（跨模态检索） | 512维 |
| Sentence-Transformers | 文本向量化 | 384维 |

### 数据存储

| 技术 | 用途 |
|------|------|
| FAISS | 高效向量检索（双索引：文本+图片） |
| 内存存储 | 会话管理（支持多用户） |

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- RAM 8GB+（推荐 32GB 用于CLIP模型）
- Google Gemini API密钥

### 2. 安装依赖

#### 方式一：使用 pip

```bash
# 克隆项目
git clone <repository-url>
cd Chat

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 方式二：使用 Poetry

```bash
poetry install
poetry shell
```

### 3. 配置环境变量

```bash
# 复制环境变量示例文件
cp configs/.env.example .env

# 编辑 .env 文件，填入你的 Gemini API 密钥
# GOOGLE_API_KEY=your_api_key_here
```

**获取 Gemini API 密钥**：
1. 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
2. 创建或登录 Google 账号
3. 生成 API 密钥并复制

### 4. 准备文档

将您的 PDF 文档放入 `data/documents/` 目录：

```bash
cp your_document.pdf data/documents/
```

### 5. 启动应用

#### 方式一：使用启动脚本（推荐）

```bash
# Linux/Mac
bash scripts/run.sh

# Windows
scripts\run.bat
```

#### 方式二：直接运行

```bash
# 使用模块方式启动
python -m app.main

# 或使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8100 --reload
```

### 6. 访问应用

打开浏览器访问：

- **Web界面**：http://localhost:8100
- **API文档**：http://localhost:8100/docs
- **健康检查**：http://localhost:8100/health

---

## 📚 使用指南

### 基础对话

1. 打开 Web 界面
2. 在输入框输入问题
3. 点击"发送"或按 Enter
4. 查看实时流式输出的答案

### 上传新文档

1. 点击"上传文档"按钮
2. 选择 PDF 文件
3. 等待系统自动处理（提取文本+图片，建立索引）
4. 开始基于新文档的对话

### 新建对话

点击"新建对话"按钮，清空当前对话历史，开始全新的会话。

### 查看来源

每次回答后，系统会显示：
- 📄 **文本来源**：相关的文本片段
- 🖼️ **图片来源**：相关的图片缩略图（点击可查看大图）

---

## ⚙️ 配置说明

### 环境变量配置

编辑 `.env` 文件或 `configs/.env.example`：

```env
# API密钥
GOOGLE_API_KEY=your_google_api_key_here

# 向量数据库路径
CHROMA_PERSIST_DIRECTORY=./data/vector_db

# PDF文档路径
PDF_DOCUMENT_PATH=./data/documents/文档.pdf
```

### 应用配置

编辑 `app/config.py` 可以调整：

- 向量数据库路径
- 文档路径
- 模型参数

### 启动参数

在 `app/__main__.py` 或 `app/main.py` 中调整：

```python
uvicorn.run(
    app,
    host="0.0.0.0",     # 监听地址
    port=8100,          # 端口号
    reload=True,        # 热重载（开发模式）
    log_level="info"    # 日志级别
)
```

---

## 🔌 API 接口文档

### 核心接口

#### 1. 流式对话（推荐）

```http
POST /chat/stream
Content-Type: application/json

{
  "question": "文档的核心观点是什么？",
  "session_id": "optional-session-id"
}
```

**响应**：Server-Sent Events (SSE) 流

```
data: {"type":"session_id","content":"uuid"}
data: {"type":"token","content":"答"}
data: {"type":"token","content":"案"}
data: {"type":"sources","content":[...]}
data: [DONE]
```

#### 2. 上传文档

```http
POST /upload
Content-Type: multipart/form-data

file: your_document.pdf
```

#### 3. 会话管理

```http
# 创建新会话
POST /session/new

# 获取会话历史
GET /session/{session_id}/history

# 删除会话
DELETE /session/{session_id}
```

详细 API 文档访问：http://localhost:8100/docs

---

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_rag_service.py

# 查看覆盖率
pytest --cov=app tests/
```

### 手动测试

```bash
# 测试健康检查
curl http://localhost:8100/health

# 测试对话接口
curl -X POST http://localhost:8100/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"测试问题"}'
```

---

## 🛠️ 开发指南

### 添加新功能

1. **创建服务模块**：在 `app/services/` 下创建新文件
2. **定义数据模型**：在 `app/models/` 下定义 Pydantic 模型
3. **添加 API 端点**：在 `app/main.py` 中添加路由
4. **编写测试**：在 `tests/` 下创建测试文件

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 编写文档字符串
- 保持函数简洁（<50行）

### Git 工作流

```bash
# 创建功能分支
git checkout -b feature/new-feature

# 提交更改
git add .
git commit -m "Add: 新功能描述"

# 推送到远程
git push origin feature/new-feature

# 创建 Pull Request
```

---

## 📊 性能优化

### 当前性能指标

| 指标 | 数值 |
|------|------|
| 首Token延迟 | ~4.5秒 |
| 图片处理时间 | ~1.3秒（2张图） |
| 向量检索时间 | <0.1秒 |
| 内存占用 | ~2GB（CLIP模型） |

### 优化建议

1. **使用 GPU**：如有 NVIDIA GPU，设置 CUDA 加速
2. **调整分块策略**：修改 `chunk_size` 和 `chunk_overlap`
3. **缓存向量**：避免重复加载向量数据库
4. **批量处理**：上传多个文档时使用批量索引

---

## 🐛 常见问题

### 1. API密钥错误

**问题**：`401 Unauthorized` 或 `Invalid API key`

**解决**：
- 检查 `.env` 文件中的 `GOOGLE_API_KEY` 是否正确
- 确认 API 密钥已激活 Gemini API

### 2. 模型下载失败

**问题**：`Connection timeout` 下载 CLIP 或 Sentence-Transformers 模型

**解决**：
```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到 ~/.cache/huggingface/
```

### 3. 内存不足

**问题**：`OOM` 错误

**解决**：
- 关闭其他应用释放内存
- 减小 `chunk_size` 参数
- 使用更小的模型（如 `paraphrase-MiniLM-L6-v2`）

### 4. PDF 图片无法提取

**问题**：PDF 中有图片但系统未识别

**解决**：
- 确认 PDF 不是扫描件（使用 OCR 预处理）
- 检查图片尺寸是否 >100x100 像素
- 查看 `data/images/` 目录确认提取结果

---

## 📈 版本历史

### V2.1.3（当前版本）- 2025-11-17
- 🔧 端口迁移：从8000迁移到8100
  - 解决8000端口被占用问题
  - 统一所有配置为8100端口
  - 更新文档和前端配置

### V2.1.2 - 2025-11-17
- 🚀 前端优化：添加请求超时处理
  - ⏱️ 5秒超时机制，避免页面无限加载
  - 🔄 自动重试机制，提升用户体验
  - 💬 改进错误提示，区分超时和连接失败
  - ⏳ 延迟首次健康检查，避免阻塞页面加载

### V2.1.1 - 2025-11-17
- 🏗️ 项目重构：重组为分层架构
  - 代码模块化：`app/services/`、`app/models/`、`app/utils/`
  - 数据集中化：统一存放在 `data/` 目录
  - 文档分类化：按类型组织在 `docs/` 子目录
  - 配置独立化：配置文件集中到 `configs/`
  - 脚本独立化：启动脚本移至 `scripts/`
- 📚 文档完善：新增《项目结构说明》文档
- ✅ 路径优化：使用统一的绝对导入路径
- 🚀 启动优化：支持 `python -m app` 模块化启动

### V2.1 - 2025-11-17
- ✨ 多模态RAG：CLIP直接向量化图片
- ⚡ 性能提升：图片处理速度提升87%
- 💰 成本优化：API调用减少50%
- 🎯 准确性提升：基于视觉特征的跨模态检索

### V2.0 - 2025-11-16
- ✨ 对话记忆：支持10轮上下文
- 📤 动态上传：运行时上传新文档
- 🎨 Markdown渲染：代码高亮与格式化

### V1.0 - 2025-11-15
- 🎉 初始版本：基础RAG问答功能
- ⚡ 流式输出：实时打字机效果
- 🔍 向量检索：FAISS语义搜索

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [LangChain](https://www.langchain.com/) - LLM 应用开发框架
- [Google Gemini](https://deepmind.google/technologies/gemini/) - 强大的多模态大语言模型
- [OpenAI CLIP](https://openai.com/research/clip) - 视觉-语言预训练模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Python Web 框架
- [FAISS](https://github.com/facebookresearch/faiss) - 高效向量检索库

---

## 📞 联系方式

- 项目主页：[GitHub Repository](#)
- 问题反馈：[Issues](#)
- 技术交流：[Discussions](#)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star ⭐**

Made with ❤️ by [Your Name]

</div>

---

## 📝 开发日志

### 2025-11-17：项目结构重组（V2.1.1）

**完成内容**：
- 将混乱的根目录文件重组为清晰的分层架构
- 创建 `app/` 包，整合所有业务代码
- 创建 `data/` 目录，统一管理文档、图片、向量库
- 创建 `docs/` 子分类，按类型组织文档（guides/technical/development）
- 创建 `scripts/` 目录，独立管理启动脚本
- 创建 `configs/` 目录，集中管理配置文件
- 更新所有导入路径为绝对导入
- 创建 `app/__main__.py` 支持模块化启动
- 编写《项目结构说明》文档

**使用技术**：
- Python 包管理（`__init__.py`、`__main__.py`）
- 绝对导入路径（`from app.services.xxx import xxx`）
- 分层架构设计（Layered Architecture）
- 职责分离原则（Separation of Concerns）

**重组效果**：
- ✅ 代码模块化：业务逻辑清晰分层
- ✅ 文档分类化：按类型组织，易于查找
- ✅ 数据集中化：统一管理，避免混乱
- ✅ 配置独立化：环境变量集中管理
- ✅ 符合规范：遵循 Python 最佳实践

**测试结果**：
- ✅ 配置加载测试：通过
- ✅ 模块导入测试：通过
- ✅ 路径检查测试：通过
- ✅ FastAPI应用测试：通过（15个路由注册）
- ✅ RAG服务测试：通过（所有方法完整）
- ✅ 图片处理器测试：通过（所有方法完整）
- ✅ 启动流程测试：通过（模块化启动正常）
- ✅ 数据目录测试：通过（所有数据已迁移）

详细测试报告：`docs/development/项目重组测试报告.md`

**时间**：2025年11月17日

---

### 2025-11-17：前端超时处理优化

**问题描述**：
- 前端页面一直加载，无法进入
- 健康检查请求无超时，服务器未响应时页面卡住

**解决方案**：
- ✅ 添加 `AbortController` 超时处理（5秒超时）
- ✅ 改进错误处理，区分超时和连接失败
- ✅ 延迟首次健康检查（1秒），避免阻塞页面加载
- ✅ 添加自动重试机制（5秒后重试）

**技术实现**：
- 使用 `AbortController` API 实现请求超时
- 使用 `setTimeout` 延迟首次检查
- 改进错误提示，提供明确的用户反馈

**优化效果**：
- ✅ 页面不再无限加载
- ✅ 5秒超时，快速反馈
- ✅ 明确的错误提示
- ✅ 自动重试机制

**测试结果**：
- ✅ 超时处理测试：通过
- ✅ 正常连接测试：通过
- ✅ 页面加载测试：通过

详细文档：`docs/development/前端超时处理优化.md`

**时间**：2025年11月17日
