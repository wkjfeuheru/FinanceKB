# 金融 Agentic RAG 知识库平台

层级切块、稠密/稀疏混合检索和 BGE 重排序的金融问答服务。文档解析、元数据抽取、切块、向量化和写入可以走在线上传，也可以走离线脚本。

## 环境

- Python 3.11 及以上。入库和启动都用项目虚拟环境 `.venv`（Python 3.13）
- Docker
- DeepSeek API Key
- 通义 API Key（查询改写与图片描述）

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

`.env` 至少填写 `DEEPSEEK_API_KEY` 和 `JWT_ADMIN_PASSWORD`。查询改写模型由 `QUERY_REWRITE_MODEL` 指定。MinerU 模型从 ModelScope 拉取：`MINERU_MODEL_SOURCE=modelscope`。嵌入使用本地 ONNX INT8 的 `BAAI/bge-m3`（1024 维，CLS pooling）；模型、维度或 pooling 变更后需要重建集合并全量重新入库。

## 启动

本仓库 Compose 启动依赖服务：

```powershell
docker compose up -d etcd postgres redis milvus
.\.venv\Scripts\python.exe main.py
```

本机已有独立 Milvus 时，启动这三个容器即可，服务仍监听 `http://localhost:19530`：

```powershell
docker start milvus-etcd milvus-minio
docker start milvus-standalone
```

浏览器打开 http://localhost:8000 。

## 离线入库

离线脚本走完整生产路径：解析、元数据抽取、切块、向量化、写入。不启动 Web 服务，也不经过上传队列。在现有集合上增量写入；内容未变的文件按指纹跳过。

先确认 Milvus 已启动，再在项目目录执行。目录会递归扫描：

```powershell
cd "F:\python project\FinanceKB"
.\.venv\Scripts\python.exe scripts\ingest_offline.py "F:\新文件目录"
```

单个文件：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_offline.py "F:\新文件目录\某研报.pdf"
```

| 参数 | 作用 |
|------|------|
| `--category` | 文档分类，默认 `investment_research`。另有 `compliance_risk`、`business_operations`、`management` |
| `--only 片段` | 只处理文件名包含该片段的文件，可重复，多个为并集 |
| `--limit N` | 最多处理 N 个文件 |
| `--force` | 内容未变也重新入库 |
| `--collection` | 集合名，默认 `finance_kb` |

指定分类入库：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_offline.py "F:\新文件目录" --category compliance_risk
```

只重跑文件名里带某段文字、且内容未变也要覆盖的文件：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_offline.py "F:\新文件目录" --only AP20260926 --force
```

支持 `.pdf`、`.docx`、`.pptx`、`.xlsx`、`.xls`、`.md`、`.txt`。超过 `MAX_UPLOAD_SIZE_MB` 的文件会跳过（`.env.example` 默认为 20）。投研类文档必须抽出报告类型（个股 / 行业 / 宏观 / 策略）；正则抽不到时由查询改写模型补 `report_type` 和 `broker`。

`MINERU_BACKEND=pipeline` 对应 MinerU 4 的本地 `basic` 档。

## 在线上传

登录后在「文档管理」上传。默认走常驻流水线：入队即返回 `task_id`，用 `GET /api/tasks/{task_id}` 看进度。队列满时返回 503，退避重试即可；已入队且内容未变的文件会被指纹跳过。

分类写入 `category`。内置四类与自定义类别在「知识库管理」中维护，共用同一个 Milvus 集合。改名会迁移已有文档的分类；有文档的类别需先清空再删除。

| 分类值 | 标签 |
|--------|------|
| `investment_research` | 投研类 |
| `compliance_risk` | 合规风控类 |
| `business_operations` | 业务运营类 |
| `management` | 管理类 |

`ENABLE_VERSIONING=true` 时同一文档保留历史版本，检索只命中 `is_current`。旧集合需要先 `KnowledgeBase.rebuild_collection()`。按版本删除：`DELETE /api/documents/{source}?version=v2`。

聊天过滤示例：

```json
{"query": "...", "filters": {"category": ["compliance_risk"], "date": {"gte": "2024-01-01"}}}
```

`date` 是原文发布日期。无法确认时留空，不用文件修改时间或上传时间填补。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 登录，返回 JWT |
| POST | `/api/chat` | 问答。粗排默认 50，精排默认 5 |
| POST | `/api/chat/stream` | SSE 问答 |
| POST | `/api/documents/upload` | 批量上传，可选 `category` |
| GET | `/api/documents` | 文档列表。`?include_versions=true` 含历史版本 |
| PATCH | `/api/documents/{source}/metadata` | 修正标量元数据，不重新嵌入 |
| DELETE | `/api/documents/{source}` | 删除文档或指定版本 |
| GET | `/api/documents/{source}/page/{page}` | 原文页 PNG，页码从 1 起 |
| GET/POST/PATCH/DELETE | `/api/knowledge-bases` | 类别管理 |
| GET | `/api/tasks/{task_id}` | 入库进度 |
| GET | `/api/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

其余配置见 `.env.example`：切块、混合检索、重排序、拒答、对象存储和 JWT。

## 评测

评测口径见 [docs/evaluation-framework.md](docs/evaluation-framework.md)。常用入口：

```powershell
.\.venv\Scripts\python.exe scripts\ab_rag.py --list
.\.venv\Scripts\python.exe scripts\ab_rag.py --experiment hyde_on_off --fast --sample 5
.\.venv\Scripts\python.exe scripts\evaluate.py gate --report <report.json> --policy release
```

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | FastAPI |
| 向量库 | Milvus 2.6（稠密向量 + BM25 稀疏向量，RRF） |
| 关系数据库 | PostgreSQL（父块、整表） |
| LLM | DeepSeek；查询改写与图片描述走通义 |
| 嵌入 / 重排序 | BAAI/bge-m3 ONNX INT8，BAAI/bge-reranker-v2-m3 |
| 解析 | MinerU 4.1.2 |
| 前端 | Vue 3 + Element Plus + Vite |
| 对象存储 | 本地文件或阿里云 OSS |

## 开发

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
cd frontend
npm install
npm run dev
```

前端开发服务器端口 5173。导入统一使用 `src.*`。
