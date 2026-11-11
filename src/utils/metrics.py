"""Prometheus 指标收集模块。

基于 prometheus-fastapi-instrumentator 自动暴露 HTTP 指标，
并注册自定义业务指标：重排序分数分布、文档操作计数。
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# 自定义业务指标
# ---------------------------------------------------------------------------

# 检索结果质量（rerank_score 分布）
rerank_score = Histogram(
    "finance_rag_rerank_score",
    "Reranker score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

# 文档操作计数
document_ops = Counter(
    "finance_rag_document_ops_total",
    "Document operations",
    ["operation"],  # upload, delete, skip
)

# ---------------------------------------------------------------------------
# 入库流水线指标
# ---------------------------------------------------------------------------

# 各阶段入口队列深度（label 为该队列的消费阶段）。
# 持续上涨说明该阶段是瓶颈：pending→解析、chunk→切块、embed→嵌入、write→写库。
ingest_queue_depth = Gauge(
    "finance_rag_ingest_queue_depth",
    "Ingestion pipeline queue depth by consuming stage",
    ["stage"],  # pending, chunk, embed, write
)

# 阶段 worker 重启次数（>0 说明 worker 曾异常退出并被自愈重启）
ingest_worker_restarts = Counter(
    "finance_rag_ingest_worker_restarts_total",
    "Ingestion pipeline stage worker restarts",
    ["stage"],  # parse, chunk, embed, write
)

# 入库队列满导致的拒绝次数（背压触发）
ingest_queue_rejected = Counter(
    "finance_rag_ingest_queue_rejected_total",
    "Documents rejected because the ingestion queue was full",
)

# 各阶段处理耗时（逐文档），用于定位入库链路的真实瓶颈
ingest_stage_seconds = Histogram(
    "finance_rag_ingest_stage_seconds",
    "Per-document ingestion stage duration in seconds",
    ["stage"],  # parse, chunk, embed, write
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

