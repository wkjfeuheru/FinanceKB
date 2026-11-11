"""问答知识库管理模块（Agentic RAG）。

基于 Milvus 2.4+ 的 BM25 稀疏向量 + ONNX INT8 本地稠密向量实现 RRF 混合检索，
使用单一集合 ``finance_kb`` 管理问答知识库。

核心能力（委托给专门模块）：
* 层级父子结构感知切块 → :mod:`src.chunking.chunker`
* 稠密向量 + 稀疏向量 RRF 混合检索 + BGE 重排序 → :mod:`src.retrieve.hybrid_retriever`
* 文档增量入库 / 删除 / 列表
"""

from __future__ import annotations

import logging
import copy
import hashlib
from contextlib import nullcontext
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import (
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)


def get_milvus_client() -> MilvusClient:
    """创建配置好的 Milvus 客户端，供基础设施适配器及管理服务复用。"""
    kwargs: dict[str, Any] = {
        "uri": MILVUS_URI,
        "timeout": MILVUS_TIMEOUT_SECONDS,
    }
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)

from src.chunking.chunker import DoclingChunks
from src.chunking.context_injection import embed_text_for_chunk
from src.knowledge.document_version import extract_document_version
from src.chunking.metadata_extractor import (
    build_canonical_metadata,
    merge_metadata_patch,
    resolve_report_title,
)
from src.core.config import (
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    ENABLE_VERSIONING,
    KB_COLLECTION_NAME,
    MAX_VERSIONS_PER_DOC,
    MILVUS_NLIST,
    MILVUS_TIMEOUT_SECONDS,
    MILVUS_TOKEN,
    MILVUS_URI,
    RRF_K,
    REPORT_COLLECTION_NAME,
    ENABLE_CLIFF_DETECTION,
    RETRIEVAL_FUSION_MODE,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_SPARSE_WEIGHT,
    RETRIEVAL_CANDIDATE_SOURCE_LIMIT,
    RETRIEVAL_CANDIDATE_MAX_PER_SOURCE,
    normalize_fusion_weights,
    TENANT_ID,
)
from .onnx_embedder import OnnxEmbedder
from .fingerprint_store import FingerprintStore, fingerprint_path_for
from src.index.parent_store import ParentStore
from src.index.table_store import TableStore
from src.retrieve.hybrid_retriever import HybridRetriever
from src.core.exceptions import (
    EmbeddingError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)


# 研报维度与证据字段：``(字段名, 类型, VARCHAR 长度 或 None)``。
# 单独抽出来是为了让「集合该有哪些字段」这件事可以直接被单测锁住，
# 而不是只能靠连一个真实 Milvus 才能验证。
#
# 顺序与新建集合的字段顺序一致；变更时同步更新
# ``scripts/migrate_kb_schema.py`` 的 ``REPORT_FIELDS``。
# 证券名、行业、抽取来源和复核标记已从契约删除，不再进入新 schema。
_REPORT_FIELD_SPECS: tuple[tuple[str, Any, int | None], ...] = (
    ("report_type", DataType.VARCHAR, 16),
    ("broker", DataType.VARCHAR, 64),
    ("block_type", DataType.VARCHAR, 16),
    ("start_page", DataType.INT64, None),
    ("end_page", DataType.INT64, None),
    ("image_key", DataType.VARCHAR, 256),
)

# 展示标题的 VARCHAR 上限（UTF-8 字节）。新建集合用这个长度；
# 旧集合可能仍是 128，写入时按集合真实上限截断。
_TITLE_MAX_BYTES = 512

# 检索过滤会用到的标量。稠密索引仍是 IVF_FLAT + COSINE，这里只补倒排。
_SCALAR_INDEX_FIELDS: tuple[str, ...] = (
    "is_current",
    "category",
    "date",
    "report_type",
    "broker",
    "source",
)

# 旧集合可能还留着这些列。新代码不再取值，但插入时必须补空值，
# 否则固定 schema 会因缺字段整批失败。原地删列要等重建集合。
_RETIRED_SCALAR_FIELDS = frozenset({
    "security_name",
    "industry_l1",
    "industry_l2",
    "meta_source",
    "needs_review",
})

# 旧集合至少要有这个字段，才认为它属于「含研报字段」的一代 schema。
# 注意：不能再用它推断整组字段齐备——线上集合可能由更新版本创建，
# 多出本代码不认识的标量字段。实际写入时按真实字段集合裁剪/补默认值。
_REPORT_FIELD_SENTINEL = "block_type"

# 文档级研报字段：可在不重新嵌入的前提下原地更新。
# 其余研报字段（block_type / 页码 / image_key）是块级信息，文档级更新不得覆盖。
_DOCUMENT_LEVEL_REPORT_FIELDS = frozenset({"report_type", "broker"})


def _add_report_fields(schema: Any) -> None:
    """把研报维度与证据字段加入集合 schema。"""
    for name, dtype, max_length in _REPORT_FIELD_SPECS:
        if max_length is None:
            schema.add_field(name, dtype)
        else:
            schema.add_field(name, dtype, max_length=max_length)


def _clip(value: Any, limit: int) -> str:
    """VARCHAR 超长会让整篇文档入库失败，这里一律截断而不是抛错。"""
    return str(value or "").encode("utf-8")[:limit].decode("utf-8", errors="ignore")


# 集合里存在、但本代码没有专门取值逻辑的标量字段：用类型默认值补齐，
# 而不是让整批插入因「Insert missed an field」失败。
_UNSUPPORTED = object()


def _scalar_default(dtype: Any) -> Any:
    """给出某标量类型的非空默认值；向量/未知类型返回 ``_UNSUPPORTED``。"""
    # STRING 是较新版本 pymilvus 对 VARCHAR 的别名；旧版本没有该枚举成员
    if dtype == DataType.VARCHAR or dtype == getattr(DataType, "STRING", None):
        return ""
    if dtype in (DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64):
        return 0
    if dtype in (DataType.FLOAT, DataType.DOUBLE):
        return 0.0
    if dtype == DataType.BOOL:
        return False
    if dtype == DataType.JSON:
        return {}
    if dtype == DataType.ARRAY:
        return []
    return _UNSUPPORTED


def _fill_missing_scalars(
    rows: list[dict[str, Any]], defaults: dict[str, Any]
) -> list[str]:
    """给行集合补齐缺失的标量字段（原地修改），返回实际补齐的字段名。"""
    if not rows or not defaults:
        return []
    missing = {name: value for name, value in defaults.items() if name not in rows[0]}
    if not missing:
        return []
    for row in rows:
        for name, value in missing.items():
            row.setdefault(name, copy.deepcopy(value))
    return sorted(missing)


def _varchar_max_length(fields: list[dict[str, Any]], name: str) -> int | None:
    """从 describe_collection 的字段列表读取 VARCHAR 上限；读不到返回 None。"""
    for field in fields:
        if field.get("name") != name:
            continue
        params = field.get("params") or {}
        raw = params.get("max_length", field.get("max_length"))
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


def _add_scalar_indexes(index_params: Any) -> None:
    """给过滤字段加 INVERTED 索引。已有向量索引的参数对象也可以接着用。"""
    for field_name in _SCALAR_INDEX_FIELDS:
        index_params.add_index(field_name=field_name, index_type="INVERTED")


def _report_scalars(
    meta: dict[str, Any],
    chunk_meta: dict[str, Any],
    *,
    raw_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造研报标量字段值。

    文档级元数据（report_type / broker）来自入库时的抽取结果；
    块级字段（block_type / 页码 / image_key）来自切块结果。
    Milvus 标量不接受 ``None``，缺失一律给非空默认值。

    ``raw_meta`` 仍接受，便于旧调用方传入原始元数据；证券名、行业、
    抽取来源和复核标记已退役，不再从中回写。
    """
    del raw_meta
    canonical = build_canonical_metadata(meta)
    return {
        "report_type": _clip(canonical.get("report_type"), 16),
        "broker": _clip(canonical.get("broker"), 64),
        "block_type": _clip(chunk_meta.get("block_type") or "text", 16),
        "start_page": int(chunk_meta.get("start_page") or 0),
        "end_page": int(chunk_meta.get("end_page") or 0),
        "image_key": _clip(chunk_meta.get("image_key"), 256),
    }




@dataclass(frozen=True)
class ParsedDocument:
    """解析（及可选切块）结果。

    ``KnowledgeBase.parse_document`` 产出**仅解析**形态：``chunks`` 只携带
    解析产物（markdown / blocks / images / figure_captions），子块/父块/整表
    为空。切块与增强由共享准备流程
    （:func:`src.ingest.pipeline.prepare_chunked_document`）
    执行且仅执行一次，之后 ``chunks`` 才是完整的切块结果。
    """

    path: Path
    source: str
    title: str
    chunks: DoclingChunks
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# 集合级写锁
# ---------------------------------------------------------------------------
# 「删旧行 + 插新行」、文档间 SimHash 去重（先查再插）、版本切换与删除都必须
# **按 collection 串行**：并发写同一集合会让近似重复文档同时通过去重检查，
# 或让同 source 的覆盖互相删除。
#
# 锁刻意放在**资源侧**（而不是让各调用方自行持锁）：此前入库流水线用
# asyncio 写锁、document_service 用模块级 threading 锁，两把锁互不排斥——
# 对象存储事件回调与流水线写入并发时会同时进入 add_parsed_document。
# 放在这里以后，流水线、事件回调、管理脚本等所有调用方自动受保护。
_COLLECTION_WRITE_LOCKS: dict[str, threading.RLock] = {}
_COLLECTION_WRITE_LOCKS_GUARD = threading.Lock()


def _collection_write_lock(collection_name: str) -> threading.RLock:
    """取得（并缓存）某 collection 的写锁。

    使用 ``RLock``：同线程内的嵌套调用（例如某写方法复用另一写方法）
    不会自锁死，跨线程仍然互斥。
    """
    lock = _COLLECTION_WRITE_LOCKS.get(collection_name)
    if lock is not None:
        return lock
    with _COLLECTION_WRITE_LOCKS_GUARD:
        lock = _COLLECTION_WRITE_LOCKS.get(collection_name)
        if lock is None:
            lock = threading.RLock()
            _COLLECTION_WRITE_LOCKS[collection_name] = lock
    return lock


def _dedup_text_for_document(parsed: ParsedDocument) -> str:
    """文档判重输入：完整正文 + 完整表体（而不是检索用的表格索引摘要）。

    检索摘要（``_table_index_text``：表头 + 首行）刻意省略表体数值以提升信噪比，
    但正因如此**不能用于判重**：两张表头相同、行标签相同、金额不同的表，
    索引文本逐字相同，SimHash 海明距离为 0，会把合法新报告判成重复而拒绝入库。
    这里改用完整 markdown 与整表 markdown；两者都缺失时退化为文档身份。
    """
    chunks = parsed.chunks
    parts: list[str] = []
    markdown = getattr(chunks, "markdown", None)
    if markdown:
        parts.append(str(markdown))
    else:
        parents = getattr(chunks, "parents", None) or []
        parts.extend(str(getattr(parent, "content", "") or "") for parent in parents)
        if not parts:
            parts.extend(
                str(getattr(chunk, "page_content", "") or "")
                for chunk in (getattr(chunks, "chunks", None) or [])
            )
    # 整表 markdown 必须计入：PDF 路径的 markdown 可能已被切块器抽走表体
    parts.extend(
        str(getattr(table, "markdown", "") or "")
        for table in (getattr(chunks, "tables", None) or [])
    )
    text = "\n".join(part for part in parts if part.strip())
    if text.strip():
        return text
    return f"{parsed.source}\n{parsed.title}"


class KnowledgeBase:
    """问答知识库：Docling Hybrid 切块 + BM25 稀疏 + 稠密向量混合检索 + BGE 重排序。

    使用单一集合 ``finance_kb``，支持文档上传增量管理。
    切块逻辑委托给 :class:`DoclingHybridChunker`，
    检索逻辑委托给 :class:`HybridRetriever`。
    """

    def __init__(
        self,
        collection_name: str = KB_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client: MilvusClient | None = None
        self._embeddings: OnnxEmbedder | None = None
        self._parent_store: ParentStore | None = None
        self._table_store: TableStore | None = None
        self._retriever: HybridRetriever | None = None
        self._fingerprint_store: FingerprintStore | None = None
        self._schema_fields: set[str] | None = None
        # 集合标量字段的默认值（按实例缓存；用于 schema 漂移时补齐缺字段）
        self._scalar_defaults: dict[str, Any] | None = None
        # 已提示过的退役列，避免每次入库都重复打日志
        self._noted_retired_fields: set[str] = set()
        # 当前集合 title 的字节上限；None 表示尚未从 schema 读到
        self._title_max_bytes: int | None = None
        # 惰性初始化锁必须是**可重入**的：_get_retriever() 在持锁状态下会再调用
        # _get_client() / _get_parent_store()，而这两个方法同样 `with self._init_lock`。
        # 用普通 threading.Lock 会自死锁——只在「先入库、后检索」的顺序下侥幸不触发，
        # 新进程直接检索（CLI / 评测脚本）会永久挂起。
        self._init_lock = threading.RLock()
        # 集合级写锁（同 collection 的写操作串行；见 _collection_write_lock）
        self._write_lock = _collection_write_lock(collection_name)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """确保集合存在并已加载，不存在则创建空集合。"""
        client = self._get_client()
        try:
            if not client.has_collection(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            ):
                self._create_collection(client)
            else:
                self._warn_if_stale_schema(client)
                self._ensure_scalar_indexes(client)
                self._ensure_title_capacity(client)
            client.load_collection(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"集合 {self.collection_name} 检查/加载失败：{exc}"
            ) from exc

    def _collection_fields(self, client: MilvusClient) -> set[str]:
        """读取集合字段名（按实例缓存；读取失败返回空集合并允许下次重试）。"""
        if self._schema_fields is None:
            try:
                desc = client.describe_collection(self.collection_name)
                self._schema_fields = {
                    f.get("name") for f in desc.get("fields", [])
                }
            except Exception as exc:
                logger.warning("读取集合 schema 失败：%s", exc)
                return set()
        return self._schema_fields

    def _collection_has_field(self, client: MilvusClient, field: str) -> bool:
        """判断集合是否含指定字段：旧 schema 集合缺字段时跳过对应写入。"""
        return field in self._collection_fields(client)

    def _collection_scalar_defaults(self, client: MilvusClient) -> dict[str, Any]:
        """集合中所有可写标量字段的类型默认值（按实例缓存）。

        用于兼容「集合由更新版本代码创建、多出本代码未声明的标量字段」：
        Milvus 固定 schema 下缺一个必填字段就会整批插入失败，而字段是否
        可空/有默认值由建表时决定，不能指望服务端兜底。这里按类型补空值，
        并由调用方记录告警，让 schema 漂移可见而不是静默丢数据。
        """
        if self._scalar_defaults is None:
            try:
                desc = client.describe_collection(self.collection_name)
            except Exception as exc:  # noqa: BLE001 - 读取失败时保留原始插入错误
                logger.warning("读取集合 schema 失败（无法补齐缺省标量）：%s", exc)
                return {}
            # BM25 等 Function 的输出字段由服务端生成，不能手动写入
            function_outputs = {
                str(name)
                for function in desc.get("functions", []) or []
                for name in function.get("output_field_names", []) or []
            }
            defaults: dict[str, Any] = {}
            for field in desc.get("fields", []) or []:
                name = field.get("name")
                if not name or field.get("is_primary") or name in function_outputs:
                    continue
                value = _scalar_default(field.get("type"))
                if value is not _UNSUPPORTED:
                    defaults[str(name)] = value
            self._scalar_defaults = defaults
        return self._scalar_defaults

    def _complete_rows_with_schema_defaults(
        self, client: MilvusClient, rows: list[dict[str, Any]]
    ) -> None:
        """按真实集合 schema 补齐行里缺失的标量字段，并告警 schema 漂移。"""
        filled = _fill_missing_scalars(
            rows, self._collection_scalar_defaults(client)
        )
        if not filled:
            return
        retired = [name for name in filled if name in _RETIRED_SCALAR_FIELDS]
        unknown = [name for name in filled if name not in _RETIRED_SCALAR_FIELDS]
        if unknown:
            logger.warning(
                "集合 %s 含当前写入链路未提供的标量字段（%s），已按类型默认值补齐；"
                "这通常意味着集合 schema 与代码版本不一致，请检查是否需要重建或升级写入路径",
                self.collection_name,
                ", ".join(unknown),
            )
        newly_noted = [name for name in retired if name not in self._noted_retired_fields]
        if newly_noted:
            self._noted_retired_fields.update(newly_noted)
            logger.info(
                "集合 %s 仍含已退役标量（%s），已按空值补齐；这些列要等重建集合后才会消失",
                self.collection_name,
                ", ".join(newly_noted),
            )

    def _collection_has_version_fields(self, client: MilvusClient) -> bool:
        """判断集合是否包含版本相关字段（version/ingested_at/is_current）。"""
        return {"version", "ingested_at", "is_current"} <= self._collection_fields(client)

    def _warn_if_stale_schema(self, client: MilvusClient) -> None:
        """检测集合是否为旧 schema（缺少新字段），是则提示重建。"""
        try:
            desc = client.describe_collection(self.collection_name)
            fields = {f.get("name") for f in desc.get("fields", [])}
            missing = {
                "category", "date", "version", "ingested_at", "is_current",
                "heading_path", _REPORT_FIELD_SENTINEL,
            } - fields
            if missing:
                logger.warning(
                    "集合 %s 是旧 schema（缺少 %s 字段），请执行 rebuild_collection() 重建",
                    self.collection_name, ", ".join(sorted(missing)),
                )
        except Exception as exc:
            logger.warning("检测集合 schema 失败：%s", exc)

    def _describe_fields(self, client: MilvusClient) -> list[dict[str, Any]]:
        """读取集合字段描述。失败时抛给调用方，由调用方决定降级。"""
        desc = client.describe_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )
        return list(desc.get("fields") or [])

    def _field_has_index(self, client: MilvusClient, field: str) -> bool:
        """该字段是否已有索引。空索引名也算已有，避免反复补建。"""
        names = client.list_indexes(
            self.collection_name,
            field_name=field,
            timeout=MILVUS_TIMEOUT_SECONDS,
        )
        return bool(names)

    def _ensure_scalar_indexes(self, client: MilvusClient) -> None:
        """给已有集合补上缺失的过滤字段索引。失败不阻断加载。"""
        try:
            present = self._collection_fields(client)
            if not present:
                logger.warning(
                    "无法读取集合 %s 的字段，跳过标量索引补建", self.collection_name
                )
                return
            missing = [
                field for field in _SCALAR_INDEX_FIELDS
                if field in present and not self._field_has_index(client, field)
            ]
            if not missing:
                return
            created: list[str] = []
            for field in missing:
                params = client.prepare_index_params()
                params.add_index(field_name=field, index_type="INVERTED")
                try:
                    client.create_index(
                        self.collection_name,
                        params,
                        timeout=MILVUS_TIMEOUT_SECONDS,
                    )
                except Exception as exc:  # noqa: BLE001 - 单字段失败不影响其余字段
                    logger.warning(
                        "集合 %s 的 %s 标量索引补建失败：%s",
                        self.collection_name, field, exc,
                    )
                    continue
                created.append(field)
            if created:
                logger.info(
                    "已为集合 %s 补建标量索引：%s",
                    self.collection_name, ", ".join(created),
                )
        except Exception as exc:  # noqa: BLE001 - 索引探测失败不让服务起不来
            logger.warning(
                "集合 %s 补建标量索引失败，检索仍可用：%s",
                self.collection_name, exc,
            )

    def _ensure_title_capacity(self, client: MilvusClient) -> None:
        """把过短的 title 上限放宽到 ``_TITLE_MAX_BYTES``。只能加长，失败则沿用旧值。"""
        try:
            current = _varchar_max_length(self._describe_fields(client), "title")
        except Exception as exc:  # noqa: BLE001 - 读不到上限时按历史安全值截断
            logger.warning(
                "读取集合 %s 的 title 上限失败，写入仍按 128 字节截断：%s",
                self.collection_name, exc,
            )
            self._title_max_bytes = 128
            return
        if current is None:
            logger.warning(
                "集合 %s 的 title 字段没有可读的 max_length，写入按 128 字节截断",
                self.collection_name,
            )
            self._title_max_bytes = 128
            return
        if current >= _TITLE_MAX_BYTES:
            self._title_max_bytes = current
            return
        try:
            client.alter_collection_field(
                self.collection_name,
                "title",
                {"max_length": _TITLE_MAX_BYTES},
                timeout=MILVUS_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - 放宽失败时继续用旧上限
            logger.warning(
                "集合 %s 的 title 上限仍为 %d 字节，放宽失败：%s",
                self.collection_name, current, exc,
            )
            self._title_max_bytes = current
            return
        self._title_max_bytes = _TITLE_MAX_BYTES
        logger.info(
            "已将集合 %s 的 title 上限从 %d 放宽到 %d 字节",
            self.collection_name, current, _TITLE_MAX_BYTES,
        )

    def _title_byte_limit(self, client: MilvusClient) -> int:
        """写入 title 时使用的字节上限：优先用已确认的集合上限。"""
        if isinstance(self._title_max_bytes, int) and self._title_max_bytes > 0:
            return self._title_max_bytes
        try:
            current = _varchar_max_length(self._describe_fields(client), "title")
        except Exception as exc:  # noqa: BLE001 - 读失败时按历史安全值截断
            logger.warning(
                "读取集合 %s 的 title 上限失败，按 128 字节截断：%s",
                self.collection_name, exc,
            )
            return 128
        if current is None:
            return 128
        self._title_max_bytes = current
        return current

    def rebuild_collection(self) -> None:
        """删除并重建集合（用于 schema 变更后的迁移）。

        **重建会连带清空指纹库与 PG 侧派生数据**（``parent_chunks`` /
        ``table_chunks``）。不清指纹是一个静默陷阱：集合已经空了，但指纹仍记着
        「这篇已入库」，于是后续上传全部被判为「内容未变更」跳过——不报错、
        不告警，最后拿到一个永远是空的知识库。
        """
        client = self._get_client()
        if client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            client.drop_collection(self.collection_name)
            logger.info("已删除旧集合 %s", self.collection_name)
        self._schema_fields = None  # 清空 schema 缓存
        self._scalar_defaults = None
        self._title_max_bytes = None
        self._noted_retired_fields = set()
        self._create_collection(client)
        client.load_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )
        try:
            cleared = self._get_fingerprint_store().clear()
            logger.info("已清空指纹库（%d 条）：重建后旧指纹不再可信", cleared)
        except Exception as exc:  # noqa: BLE001 - 指纹清理失败不阻断重建
            logger.warning("清空指纹库失败，后续上传可能被误判为未变更：%s", exc)
        self._clear_derived_stores()
        logger.info("集合 %s 重建完成（含研报维度与证据字段）", self.collection_name)

    def _clear_derived_stores(self) -> None:
        """清空 PG 侧由向量库派生的数据（父块全文、整表）。

        向量库被重建后这些行已无对应块，留着只会让检索展开出孤儿内容。
        """
        from sqlalchemy import text as _sql_text

        for label, store in (
            ("parent_chunks", self._get_parent_store()),
            ("table_chunks", self._get_table_store()),
        ):
            try:
                engine = store._repo._get_engine()
                with engine.begin() as connection:
                    connection.execute(
                        _sql_text(f"DELETE FROM {label} WHERE collection = :c"),
                        {"c": self.collection_name},
                    )
                logger.info("已清空 %s（collection=%s）", label, self.collection_name)
            except Exception as exc:  # noqa: BLE001 - PG 不可用不阻断重建
                logger.warning("清空 %s 失败：%s", label, exc)

    def _create_collection(self, client: MilvusClient) -> None:
        """创建支持 BM25 稀疏向量的集合。"""
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=_TITLE_MAX_BYTES)
        schema.add_field("chunk", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_key", DataType.VARCHAR, max_length=64)
        schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
        # 顶层到本块的标题路径（层级切块骨架），无结构时为空串
        schema.add_field("heading_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("date", DataType.VARCHAR, max_length=32)
        schema.add_field("version", DataType.VARCHAR, max_length=32)
        schema.add_field("ingested_at", DataType.INT64)
        schema.add_field("is_current", DataType.BOOL)
        schema.add_field("is_deleted", DataType.BOOL)
        # 研报维度与证据字段（个股/行业/宏观/策略 + 块类型 + 页码）
        _add_report_fields(schema)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

        # Milvus BM25 Function：自动从 content 生成 sparse_vector
        schema.add_function(Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
        ))

        index_params = client.prepare_index_params()
        # 稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": MILVUS_NLIST},
        )
        # 稀疏向量索引（BM25）
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        _add_scalar_indexes(index_params)

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        # 新建集合的字段集合/默认值缓存需要失效
        self._schema_fields = None
        self._scalar_defaults = None
        self._title_max_bytes = _TITLE_MAX_BYTES
        logger.info("知识库集合 %s 创建成功（含 BM25 稀疏向量）", self.collection_name)

    # ------------------------------------------------------------------
    # 文档入库
    # ------------------------------------------------------------------

    def check_unchanged(self, source: str, file_path: str | Path) -> bool:
        """检查文件是否未变更，可用于增量构建跳过。

        Returns:
            True 表示文件哈希与上次入库时一致，可跳过解析+嵌入。
        """
        try:
            file_hash = FingerprintStore.hash_file(file_path)
            return self._get_fingerprint_store().is_unchanged(source, file_hash)
        except (OSError, FileNotFoundError):
            return False

    def check_unchanged_by_hash(self, source: str, content_hash: str) -> bool:
        """基于内容哈希判断文件是否未变更（避免读取本地文件）。

        Returns:
            True 表示未变更，可跳过解析；False 表示已变更或首次入库。
        """
        return not self._get_fingerprint_store().is_changed_by_hash(source, content_hash)

    def parse_document(
        self,
        file_path: str | Path,
        *,
        source: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """解析文档（**仅解析，不切块**）。

        返回的 ``ParsedDocument.chunks`` 只携带解析产物：markdown（PDF、TXT、
        Markdown 与受支持的 Office 格式都统一产出）、分页块、内嵌图片与原文图注；
        子块/父块/整表为空。切块由共享准备流程
        :func:`src.ingest.pipeline.prepare_chunked_document`
        执行且**仅执行一次**（流水线切块阶段、串行回退、对象存储事件共用）。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 未提取到有效文本（空文档不生成空索引）。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")

        source = source or path.name
        title = title or resolve_report_title(path, source)
        from src.ingest import get_parser

        result = get_parser().parse(path, source=source, title=title)
        markdown = str(result.markdown or "")
        if not markdown.strip():
            raise ValueError(f"文档未提取到有效文本：{source}")

        # 分类已废弃：metadata 原样保留，Milvus 的 category 字段统一写空串。
        return ParsedDocument(
            path=path,
            source=source,
            title=title,
            chunks=DoclingChunks(
                markdown=markdown,
                images=list(result.images or ()),
                blocks=list(result.blocks or ()),
                figure_captions=dict(result.figure_captions or {}),
            ),
            metadata=dict(metadata or {}),
        )

    def add_parsed_document(
        self,
        parsed: ParsedDocument,
        path: str | Path | None = None,
        content_hash: str | None = None,
        *,
        existing_vectors: dict[str, list[float]] | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """嵌入并存储已经完成解析和切块的文档（**同 collection 串行**）。

        本方法内部持有集合级写锁（:func:`_collection_write_lock`），
        因此**所有调用方都无需再自行加锁**：入库流水线的写入阶段、
        对象存储事件回调、管理脚本共用同一把锁。

        参数说明见 :meth:`_add_parsed_document_locked`。
        """
        with self._write_lock:
            return self._add_parsed_document_locked(
                parsed,
                path,
                content_hash,
                existing_vectors=existing_vectors,
                dense_vectors=dense_vectors,
            )

    def _add_parsed_document_locked(
        self,
        parsed: ParsedDocument,
        path: str | Path | None = None,
        content_hash: str | None = None,
        *,
        existing_vectors: dict[str, list[float]] | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """嵌入并存储已经完成解析和切块的文档（写锁内的实现）。

        Args:
            parsed: 解析 + 切块结果。
            path: 原始文件路径（未传 ``content_hash`` 时用于计算文件指纹）。
            content_hash: 上层已知的内容哈希，避免重复读取文件。
            existing_vectors: 集合内该 source 已有块的 ``{content_hash: 稠密向量}``。
                入库流水线在 Embedding **之前**查询并以它跳过重复嵌入；
                为 None 时本方法自行查询（保持旧调用方语义不变）。
            dense_vectors: 与 ``parsed.chunks.chunks`` 等长且一一对应的稠密向量
                （入库流水线在 Embedding 线程池中预先算好）。
                为 None 时本方法内联补齐缺失块。

        Note:
            本方法为**同步**写操作且**不自行加锁**，调用方必须通过
            :meth:`add_parsed_document` 进入（由它保证同一 collection 串行）。
        """
        canonical_meta = build_canonical_metadata(parsed.metadata or {})
        source = parsed.source
        title = parsed.title
        chunks = parsed.chunks

        # 嵌入输入带上下文注入（公司简称 + 章节路径）：短标题块靠块外元数据
        # 才能拿到查询词对齐信号；content 本体不变，content_hash / 块 id 不受影响。
        # 已有向量按 content_hash 复用不追溯，强制重嵌后才全量生效。
        doc_meta = canonical_meta
        embed_texts = [
            embed_text_for_chunk(chunk, doc_meta) for chunk in chunks.chunks
        ]

        # 文档间去重：判重输入是**完整正文 + 完整表体**（见 _dedup_text_for_document），
        # 不是检索用的表格索引摘要——摘要省略表体数值，会把金额不同的报告判成重复。
        dedup_text = _dedup_text_for_document(parsed)
        dedup = self._check_duplicate_document(source, dedup_text) or {}
        doc_simhash = dedup.get("simhash")
        content_sha256 = str(dedup.get("content_sha256") or "")
        duplicate_review = list(dedup.get("near_duplicates") or [])

        if dense_vectors is not None:
            if len(dense_vectors) != len(chunks.chunks):
                raise ValueError(
                    f"稠密向量数量与切块数量不一致（source={source}）："
                    f"{len(dense_vectors)} != {len(chunks.chunks)}"
                )
            resolved_vectors = list(dense_vectors)
        else:
            # 先读取同 source 的已有块向量；相同 content_hash 的块无需重复 embedding。
            if existing_vectors is None:
                existing_vectors = self.fetch_existing_vectors(source)
            missing_indexes = [
                index for index, chunk in enumerate(chunks.chunks)
                if str(chunk.metadata.get("content_hash", "")) not in existing_vectors
            ]
            new_vectors = self._embed_documents(
                [embed_texts[index] for index in missing_indexes]
            )
            resolved_vectors = [
                existing_vectors.get(str(chunk.metadata.get("content_hash", "")), [])
                for chunk in chunks.chunks
            ]
            for index, vector in zip(missing_indexes, new_vectors, strict=True):
                resolved_vectors[index] = vector
        dense_vectors = resolved_vectors

        # 从文档级元数据中拆出日期/版本，写入独立标量字段
        # category 列保留以兼容旧 schema 与快照，新写入一律为空
        meta = canonical_meta
        doc_category = ""
        doc_date = str(meta.get("date", "") or "")
        doc_version = extract_document_version(source, meta)
        ingested_at = int(time.time())

        client = self._get_client()
        self.ensure_collection()
        # 先收敛该 source 上一次未完成的发布（崩溃/失败遗留），再开始新的写入
        self.recover_pending_publishes(source=source)
        # 旧 schema 集合（未重建）无 version/ingested_at/is_current 字段
        has_version_fields = self._collection_has_version_fields(client)
        # 旧 schema 集合无 heading_path 字段：跳过该字段写入，避免插入报错
        has_heading_path = self._collection_has_field(client, "heading_path")
        # 按真实字段集合裁剪研报标量：线上集合可能仍留着已退役列，
        # 或多出本代码尚未声明的字段。缺的列由默认值补齐。
        available_fields = self._collection_fields(client)
        title_limit = self._title_byte_limit(client)

        keep_versions = ENABLE_VERSIONING and bool(doc_version)
        if keep_versions and not has_version_fields:
            raise RuntimeError(
                "已启用版本保留（ENABLE_VERSIONING=true）但集合缺少 version 字段，"
                "请调用 KnowledgeBase.rebuild_collection() 重建集合后重新上传文档"
            )

        # 构造插入行（sparse_vector 由 Milvus Function 自动生成，不传）
        rows = []
        for chunk, dense in zip(chunks.chunks, dense_vectors, strict=True):
            chunk_id = chunk.metadata["id"]
            parent_id = chunk.metadata.get("parent_id", "")
            row: dict[str, Any] = {
                "id": chunk_id,
                "content": chunk.page_content,
                "source": source,
                "title": _clip(title, title_limit),
                "chunk": int(chunk.metadata.get("chunk", 0)),
                "parent_id": parent_id,
                "chunk_key": str(chunk.metadata.get("chunk_key", chunk_id)),
                "content_hash": str(chunk.metadata.get("content_hash", "")),
                "tenant_id": TENANT_ID,
                "category": doc_category,
                "date": doc_date,
                "dense_vector": dense,
            }
            if has_version_fields:
                row.update({
                    "version": doc_version,
                    "ingested_at": ingested_at,
                    "is_current": True,
                    "is_deleted": False,
                })
            if has_heading_path:
                # 字段上限 1024：超长标题路径截断，避免整篇文档入库失败
                row["heading_path"] = str(
                    chunk.metadata.get("heading_path", "")
                )[:1024]
            report_scalars = _report_scalars(
                meta, chunk.metadata, raw_meta=parsed.metadata
            )
            row.update({
                key: value
                for key, value in report_scalars.items()
                if key in available_fields
            })
            rows.append(row)

        # 集合若由更新版本创建，可能还有本代码未取值的标量字段；Milvus 固定
        # schema 下缺一个必填字段就会整批失败，这里按类型默认值补齐并告警。
        self._complete_rows_with_schema_defaults(client, rows)

        # ------------------------------------------------------------------
        # 跨存储发布：暂存 → 激活 → 提交指纹
        # ------------------------------------------------------------------
        # 向量库 / 关系库 / 对象存储无法组成一个原子事务，因此这里不做
        # 「先删旧再写新」：旧版本在所有新数据校验通过、可见性切换完成之前
        # 始终保持可检索。中间态登记在指纹存储的发布日志里，进程在任意阶段
        # 退出后可由 :meth:`recover_pending_publishes` 收敛（未激活则回滚删掉
        # 暂存行，已激活则补完切换）。
        chunk_ids = [str(row["id"]) for row in rows]
        parent_ids = {str(p.id) for p in (chunks.parents or [])}
        table_ids = {str(t.id) for t in (getattr(chunks, "tables", None) or [])}
        if has_version_fields:
            previous_ids = self._current_ids(client, source)
        else:
            # 旧集合没有 is_current 字段，无法做可见性切换：只能保证「首次入库」安全。
            # 覆盖既有版本必须重建集合，否则写入失败时会丢失旧版本。
            if self._count_by_source(client, source):
                raise VectorStoreError(
                    f"集合 {self.collection_name} 缺少 is_current 字段，无法安全覆盖既有文档"
                    f"（source={source}）：请先调用 KnowledgeBase.rebuild_collection()"
                    "重建集合后重新上传"
                )
            previous_ids = []

        if content_hash:
            file_hash = content_hash
        elif path:
            file_hash = FingerprintStore.hash_file(path)
        else:
            file_hash = "unknown"
        file_path = str(path) if path else parsed.source

        # 派生数据（父块全文 / 整表）先落 PostgreSQL：按内容哈希幂等 upsert，
        # 不影响旧版本可见性；向量发布失败时这些行只是暂时无人引用。
        parent_store = self._get_parent_store()
        if chunks.parents:
            parent_store.store_batch([
                {
                    "id": p.id,
                    "content": p.content,
                    "heading": p.heading,
                    "heading_path": p.heading_path,
                    "source": source,
                }
                for p in chunks.parents
            ])

        # 整表落 PostgreSQL：向量库里的表格子块只有「表头 + 首行」，
        # 表体存这里，检索侧按 table_id 展开成完整表格
        if getattr(chunks, "tables", None):
            self._get_table_store().store_batch([
                {
                    "id": t.id,
                    "parent_id": t.parent_id,
                    "markdown": t.markdown,
                    "payload": t.payload,
                    "row_count": t.row_count,
                    "source": source,
                    "heading_path": t.heading_path,
                    "start_page": t.start_page,
                }
                for t in chunks.tables
            ])

        if has_version_fields:
            self._publish_with_visibility(
                client,
                source=source,
                rows=rows,
                chunk_ids=chunk_ids,
                previous_ids=previous_ids,
                keep_versions=keep_versions,
                version=doc_version,
                simhash=doc_simhash,
                content_sha256=content_sha256,
                file_hash=file_hash,
                file_path=file_path,
                metadata=canonical_meta,
            )
        else:
            self._insert_rows(client, rows)

        # 发布成功后回收不再被可见版本引用的 PG 派生行（可重试；失败不阻断发布）
        self._gc_derived_rows(source, parent_ids, table_ids)

        # 版本数超限时清理最旧版本
        if keep_versions and MAX_VERSIONS_PER_DOC > 0:
            self._prune_old_versions(client, source)

        logger.info(
            "文档入库成功：%s（%d 个 Hybrid 块）",
            Path(parsed.source).name, len(rows),
        )

        return {
            "source": source,
            "title": title,
            "chunk_count": len(rows),
            "parent_count": len(chunks.parents),
            # 近似文档只作为复核信号，不单独决定拒绝（RAG-05）
            "duplicate_review": duplicate_review,
        }

    # ------------------------------------------------------------------
    # 跨存储发布：暂存 / 激活 / 回滚 / 恢复
    # ------------------------------------------------------------------

    def _insert_rows(self, client: MilvusClient, rows: list[dict[str, Any]]) -> None:
        """插入不可变行并 flush；失败统一转成 VectorStoreError。"""
        try:
            client.insert(collection_name=self.collection_name, data=rows)
            client.flush(self.collection_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"向量库写入失败（{len(rows)} 行）：{exc}") from exc

    def _upsert_rows(self, client: MilvusClient, rows: list[dict[str, Any]]) -> None:
        """按主键覆盖写入（激活可见版本）；失败统一转成 VectorStoreError。"""
        try:
            client.upsert(collection_name=self.collection_name, data=rows)
            client.flush(self.collection_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"向量库激活失败（{len(rows)} 行）：{exc}") from exc

    def _query_rows_by_ids(self, client: MilvusClient, ids: list[str]) -> list[dict[str, Any]]:
        """按主键批量读取整行（分批，避免表达式过长）。"""
        rows: list[dict[str, Any]] = []
        for index in range(0, len(ids), 100):
            batch = ids[index : index + 100]
            quoted = ", ".join(f'"{self._escape_expr(item)}"' for item in batch)
            try:
                found = client.query(
                    collection_name=self.collection_name,
                    filter=f"id in [{quoted}]",
                    output_fields=["*"],
                    limit=len(batch) + 10,
                )
            except Exception as exc:  # noqa: BLE001 - 读取失败按缺失处理，由调用方决定
                logger.warning("按主键读取行失败（%d 个主键）：%s", len(batch), exc)
                continue
            rows.extend(found or [])
        return rows

    def _ids_present(self, client: MilvusClient, ids: list[str]) -> set[str]:
        """返回这批主键里已经存在的那些（不论可见性）。

        真实 Milvus 主键唯一，已有主键再 ``insert`` 会失败。读失败必须显式
        抛出：当成「不存在」再插入，会把一次发布变成主键冲突。
        """
        found: set[str] = set()
        for index in range(0, len(ids), 100):
            batch = ids[index : index + 100]
            quoted = ", ".join(f'"{self._escape_expr(item)}"' for item in batch)
            try:
                rows = client.query(
                    collection_name=self.collection_name,
                    filter=f"id in [{quoted}]",
                    output_fields=["id"],
                    limit=len(batch) + 10,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"读取主键是否存在失败（{len(batch)} 个主键）：{exc}"
                ) from exc
            found.update(str(row.get("id")) for row in rows or [] if row.get("id"))
        return found

    def _current_ids(self, client: MilvusClient, source: str) -> list[str]:
        """该 source 当前可见行（``is_current == true``）的主键。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}" and is_current == true'
        )
        try:
            rows = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["id"],
                limit=16384,
            )
        except Exception as exc:  # noqa: BLE001 - 读取失败视为无旧版本会丢数据，必须显式失败
            raise VectorStoreError(
                f"读取 source={source} 当前可见版本失败：{exc}"
            ) from exc
        return [str(row.get("id")) for row in rows or [] if row.get("id")]

    def _count_ids(
        self, client: MilvusClient, ids: list[str], *, current: bool | None = None
    ) -> int:
        """统计指定主键中满足可见性条件的行数（分批求和）。"""
        total = 0
        for index in range(0, len(ids), 100):
            batch = ids[index : index + 100]
            quoted = ", ".join(f'"{self._escape_expr(item)}"' for item in batch)
            expr = f"id in [{quoted}]"
            if current is True:
                expr += " and is_current == true"
            elif current is False:
                expr += " and is_current == false"
            total += self._count_matching(client, expr)
        return total

    def _delete_ids(
        self,
        client: MilvusClient,
        ids: list[str],
        *,
        current_only: bool | None = None,
    ) -> None:
        """按主键删除行；``current_only`` 可把删除限制在可见/暂存行上。

        主键唯一：同一 id 不会同时有暂存行和可见行。``current_only=False``
        仍用于只删不可见行，避免误删仍在服务的旧版本。
        """
        if not ids:
            return
        for index in range(0, len(ids), 100):
            batch = ids[index : index + 100]
            quoted = ", ".join(f'"{self._escape_expr(item)}"' for item in batch)
            expr = f"id in [{quoted}]"
            if current_only is True:
                expr += " and is_current == true"
            elif current_only is False:
                expr += " and is_current == false"
            client.delete(collection_name=self.collection_name, filter=expr)
        client.flush(self.collection_name)

    def _safe_delete_ids(
        self, client: MilvusClient, ids: list[str], *, current_only: bool | None = None
    ) -> None:
        """尽力删除（回收/回滚路径）：失败只告警，不掩盖原始异常。"""
        if not ids:
            return
        try:
            self._delete_ids(client, ids, current_only=current_only)
        except Exception as exc:  # noqa: BLE001 - 回收失败留待下次入库或恢复重试
            logger.warning("按主键清理向量行失败（%d 个主键）：%s", len(ids), exc)

    def _delete_invisible_rows(self, client: MilvusClient, source: str) -> None:
        """回收该 source 的全部不可见行（旧版本与历史暂存残留）。

        覆盖模式下这些行已无任何引用；按 source + ``is_current == false`` 整体
        删除，天然幂等，也能顺带清掉上一次发布崩溃遗留的暂存行。
        """
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}" and is_current == false'
        )
        client.delete(collection_name=self.collection_name, filter=expr)
        client.flush(self.collection_name)

    def _set_current(self, client: MilvusClient, ids: list[str], value: bool) -> None:
        """按主键切换可见性标志（读整行 → upsert；幂等）。"""
        if not ids:
            return
        rows = self._query_rows_by_ids(client, ids)
        if not rows:
            return
        for row in rows:
            row["is_current"] = value
        self._upsert_rows(client, rows)

    def _publish_with_visibility(
        self,
        client: MilvusClient,
        *,
        source: str,
        rows: list[dict[str, Any]],
        chunk_ids: list[str],
        previous_ids: list[str],
        keep_versions: bool,
        version: str,
        simhash: int | None,
        content_sha256: str,
        file_hash: str,
        file_path: str,
        metadata: dict[str, Any],
    ) -> None:
        """具备可见性字段时的发布：暂存 → 校验 → 登记 → 激活 → 停用旧版。

        真实 Milvus 主键唯一，不能为已有主键再插入一条暂存行。因此只对
        **尚不存在**的主键插入 ``is_current=false``；已有主键留到激活时
        用 ``upsert`` 覆盖。覆盖模式在激活成功后删掉「上一可见版本里、
        不在新版本主键集合中」的行。

        任何失败都不会让旧版本消失：激活前失败只清理暂存行；激活失败时回滚
        本次新增行并显式恢复旧版可见性，随后抛出原始失败。
        """
        store = self._get_fingerprint_store()
        present = self._ids_present(client, chunk_ids)
        staged_rows = [
            {**row, "is_current": False}
            for row in rows
            if str(row["id"]) not in present
        ]
        staged_ids = [str(row["id"]) for row in staged_rows]

        # 1) 暂存（不可见）：只插入新主键，避免撞上仍在服务的旧版本
        try:
            if staged_rows:
                self._insert_rows(client, staged_rows)
        except BaseException:
            self._safe_delete_ids(client, staged_ids, current_only=False)
            raise

        # 2) 校验暂存完整性：不完整就绝不进入可见性切换
        try:
            staged_count = (
                self._count_ids(client, staged_ids, current=False) if staged_ids else 0
            )
            if staged_count != len(staged_rows):
                raise VectorStoreError(
                    f"暂存校验失败（source={source}）：暂存行 {staged_count} != 期望 {len(staged_rows)}"
                )
        except BaseException:
            self._safe_delete_ids(client, staged_ids, current_only=False)
            raise

        # 3) 登记发布日志：此后崩溃可由 recover_pending_publishes 收敛
        store.begin_publish(
            source,
            sha256=file_hash,
            chunk_ids=chunk_ids,
            previous_ids=previous_ids,
            version=version,
            simhash=simhash,
            chunk_count=len(rows),
            file_path=file_path,
            metadata=metadata,
            record=store.get(source),
            content_sha256=content_sha256,
            staged_ids=staged_ids,
        )

        # 4) 激活：单次 upsert 覆盖同名主键，成为新的可见版本
        try:
            self._upsert_rows(client, rows)
            activated = self._count_ids(client, chunk_ids, current=True)
            if activated != len(rows):
                raise VectorStoreError(
                    f"激活校验失败（source={source}）：可见行 {activated} != 期望 {len(rows)}"
                )
            # 5) 清理暂存行。主键唯一时 upsert 已经把这些行改成可见，
            # 这里通常删不到东西；删完后「新主键不再有 is_current=false 行」
            # 就是恢复逻辑向前滚动的依据，因此必须排在提交指纹之前。
            if staged_ids:
                self._delete_ids(client, staged_ids, current_only=False)
            store.mark_publish_activated(source)
        except BaseException as exc:
            self._rollback_publish(client, source, chunk_ids, previous_ids)
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError(f"发布失败，已回滚到上一可见版本（source={source}）：{exc}") from exc

        # 6) 停用旧版残留（版本保留模式保留历史行；覆盖模式直接回收）
        obsolete = [item for item in previous_ids if item not in set(chunk_ids)]
        try:
            if keep_versions:
                self._set_current(client, obsolete, False)
            else:
                # 覆盖模式：正文变化会产生新主键，旧可见行不会被 upsert 覆盖，
                # 必须按主键删掉；再清掉该 source 残留的不可见行。
                self._delete_ids(client, obsolete)
                self._delete_invisible_rows(client, source)
        except Exception as exc:  # noqa: BLE001 - 新版本已发布，旧行残留不影响正确性
            logger.warning(
                "旧版本行回收失败（source=%s），下次入库会重试：%s", source, exc,
            )

        # 7) 提交指纹并清除发布登记（发布点：指纹只在此刻生效）
        store.commit_publish(
            source,
            sha256=file_hash,
            file_path=file_path,
            chunk_count=len(rows),
            simhash=simhash,
            version=version,
            metadata=metadata,
            content_sha256=content_sha256,
        )

    def _rollback_publish(
        self,
        client: MilvusClient,
        source: str,
        chunk_ids: list[str],
        previous_ids: list[str],
    ) -> None:
        """回滚未完成的发布：删除本次新增行，恢复旧版可见性，清除发布登记。"""
        previous = set(previous_ids)
        # 只删本次新增的主键。与旧版本重合的主键最多被 upsert 覆盖过内容，
        # 不能按主键删掉，否则旧版本一起消失。
        new_only = [item for item in chunk_ids if item not in previous]
        self._safe_delete_ids(client, new_only)
        # 不可见残留单独清理，不触碰仍在服务的旧版本
        self._safe_delete_ids(client, chunk_ids, current_only=False)
        if previous_ids:
            try:
                self._set_current(client, previous_ids, True)
            except Exception as exc:  # noqa: BLE001 - 旧行本就未被停用，失败不致命
                logger.warning("恢复旧版本可见性失败（source=%s）：%s", source, exc)
        store = self._get_fingerprint_store()
        pending = store.pending_publish(source) or {}
        store.abort_publish(source, restore_record=pending.get("previous_record"))

    def recover_pending_publishes(self, source: str | None = None) -> list[dict[str, Any]]:
        """收敛未完成的跨存储发布（进程崩溃恢复入口）。

        对每个存在发布日志的来源：新版本行已全部可见 → 补完切换与指纹提交
        （roll forward）；否则删除暂存/新增行并恢复旧版本（roll back）。
        幂等，可在启动时或下次入库前安全重复调用。
        """
        store = self._get_fingerprint_store()
        targets = [source] if source else store.pending_sources()
        outcomes: list[dict[str, Any]] = []
        for target in targets:
            pending = store.pending_publish(target)
            if not pending:
                continue
            outcome = self._recover_one(target, pending)
            outcomes.append(outcome)
            logger.info("未完成发布已收敛：source=%s 结果=%s", target, outcome["action"])
        return outcomes

    def _recover_one(self, source: str, pending: dict[str, Any]) -> dict[str, Any]:
        """单个来源的发布收敛（写锁内串行）。

        判定依据是**新插入的暂存行是否仍存在**，而不是「新主键是否可见」：
        与旧版本重合的主键在激活前就是可见的，按可见性计数会把未完成的发布
        误判为已激活。没有新主键时，改看发布日志里的 ``activated`` 标志。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            chunk_ids = [str(item) for item in pending.get("chunk_ids") or []]
            previous_ids = [str(item) for item in pending.get("previous_ids") or []]
            if "staged_ids" in pending:
                staged_ids = [str(item) for item in pending.get("staged_ids") or []]
            else:
                previous = set(previous_ids)
                staged_ids = [item for item in chunk_ids if item not in previous]
            staged = (
                self._count_ids(client, staged_ids, current=False) if staged_ids else 0
            )
            activated = bool(pending.get("activated"))

            if activated or (bool(staged_ids) and staged == 0):
                # 暂存行已不在，或激活标志已写下：新版本已可见，补删旧主键并提交指纹
                self._safe_delete_ids(
                    client,
                    [item for item in previous_ids if item not in set(chunk_ids)],
                )
                store = self._get_fingerprint_store()
                store.commit_publish(
                    source,
                    sha256=str(pending.get("sha256") or ""),
                    file_path=str(pending.get("file_path") or source),
                    chunk_count=int(pending.get("chunk_count") or len(chunk_ids)),
                    simhash=pending.get("simhash"),
                    version=str(pending.get("version") or ""),
                    metadata=pending.get("metadata"),
                    content_sha256=str(pending.get("content_sha256") or ""),
                )
                return {"source": source, "action": "rolled_forward", "chunks": len(chunk_ids)}

            # 暂存行仍在（或没有新主键）：发布未完成，回滚并保留旧版本
            self._rollback_publish(client, source, chunk_ids, previous_ids)
            return {"source": source, "action": "rolled_back", "chunks": len(chunk_ids)}

    def _gc_derived_rows(
        self, source: str, parent_ids: set[str], table_ids: set[str]
    ) -> None:
        """回收不再被可见版本引用的父块/整表行（发布后执行，失败可重试）。

        注意 keep 集合必须是**父块 / 整表各自的主键**（不是向量块主键）：
        用错主键会把刚写入的派生行当成过期数据删掉。
        """
        for label, store, keep_ids in (
            ("parent_chunks", self._get_parent_store(), parent_ids),
            ("table_chunks", self._get_table_store(), table_ids),
        ):
            try:
                store.delete_by_source_except(source, keep_ids)
            except Exception as exc:  # noqa: BLE001 - 回收失败不影响可见版本正确性
                logger.warning("回收 %s（source=%s）失败：%s", label, source, exc)

    def fetch_existing_vectors(self, source: str) -> dict[str, list[float]]:
        """读取指定 source 已有块的 ``{content_hash: 稠密向量}``。

        供入库流水线在 Embedding 之前调用：命中相同 content_hash 的块无需重新嵌入。
        查询失败时返回空字典（退化行为与旧实现一致：全部重新嵌入）。
        """
        try:
            client = self._get_client()
            self.ensure_collection()
            rows = client.query(
                collection_name=self.collection_name,
                filter=(
                    f'source == "{self._escape_expr(source)}" '
                    f'and tenant_id == "{TENANT_ID}"'
                ),
                output_fields=["content_hash", "dense_vector"],
                limit=16384,
            )
        except Exception as exc:
            logger.debug("读取旧块向量失败，全部重新嵌入：%s", exc)
            return {}
        return {
            str(row.get("content_hash")): row["dense_vector"]
            for row in rows
            if row.get("content_hash") and row.get("dense_vector")
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量计算文档稠密向量（公开接口，供 Embedding 线程池调用）。

        内部按 ``EMBED_BATCH_SIZE`` 分批，失败统一抛 :class:`EmbeddingError`。
        """
        return self._embed_documents(texts)

    def _check_duplicate_document(self, source: str, full_text: str) -> dict[str, Any]:
        """文档间去重检查（RAG-05）。

        判定依据分两级，避免「有损摘要」误杀合法文档：

        * **完全相同**（完整正文 + 表体的内容哈希一致，排除自身 source）→ 按明确的
          重复策略拒绝入库，并说明匹配对象；
        * **近似**（SimHash 海明距离 <= ``DUPLICATE_HAMMING_THRESHOLD``）→ 仅作为
          复核信号返回，**不单独决定拒绝**：期间不同、主体不同或关键事实不同的
          报告必须能入库。

        Returns:
            ``{"simhash": int|None, "content_sha256": str, "near_duplicates": [...]}``
        """
        from src.ingest.cleaner import (
            DUPLICATE_HAMMING_THRESHOLD,
            find_similar_documents,
            simhash,
        )

        text = full_text or ""
        if not text.strip():
            return {"simhash": None, "content_sha256": "", "near_duplicates": []}

        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        store = self._get_fingerprint_store()
        twin = store.find_exact_content(content_sha256, exclude_source=source)
        if twin:
            raise ValueError(
                f"完全相同的文档已入库：{twin}（内容哈希 {content_sha256[:12]}），"
                "已拒绝重复入库"
            )

        doc_simhash = simhash(text)
        similar = find_similar_documents(
            doc_simhash,
            store.get_all_simhashes(),
            DUPLICATE_HAMMING_THRESHOLD,
            exclude_source=source,
        )
        near_duplicates = [
            {"source": dup_source, "hamming_distance": int(dist)}
            for dup_source, dist in similar
        ]
        if near_duplicates:
            logger.warning(
                "近似文档（source=%s）：SimHash 海明距离 %s，仅作为复核信号，继续入库",
                source,
                ", ".join(
                    f"{item['source']}={item['hamming_distance']}"
                    for item in near_duplicates
                ),
            )
        return {
            "simhash": doc_simhash,
            "content_sha256": content_sha256,
            "near_duplicates": near_duplicates,
        }

    def soft_delete_document(self, source: str) -> dict[str, Any]:
        """Mark all source rows deleted and non-current, preserving audit data.

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            expr = f'source == "{self._escape_expr(source)}" and tenant_id == "{TENANT_ID}"'
            rows = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["*"],
                limit=16384,
            )
            for row in rows:
                row["is_current"] = False
                row["is_deleted"] = True
            if rows:
                client.upsert(collection_name=self.collection_name, data=rows)
                client.flush(self.collection_name)
            return {"source": source, "deleted_count": len(rows), "soft_deleted": True}

    def update_document_metadata(self, source: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """只更新文档级元数据（**不重新嵌入、不改向量、不动块级字段**）。

        标量可以原地更新：文本与向量都没变，因此没必要重跑嵌入，
        也不应触发指纹失效（否则下次上传会被判为「已变更」而全量重跑）。

        ``block_type`` / 页码 / ``image_key`` 是**块级**字段，文档级更新不能碰它们，
        否则一张表或一张图的定位信息会被整篇覆盖掉。

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            if not self._collection_has_field(client, _REPORT_FIELD_SENTINEL):
                raise RuntimeError(
                    "集合缺少研报元数据字段，请先执行 rebuild_collection() 重建集合"
                )

            expr = f'source == "{self._escape_expr(source)}" and tenant_id == "{TENANT_ID}"'
            rows = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["*"],
                limit=16384,
            )
            if not rows:
                return {"source": source, "updated_count": 0, "missing": True}

            if len(rows) >= 16384:
                raise ValueError("来源行数达到查询上限，不能确认已读取全部行")
            store = self._get_fingerprint_store()
            fingerprint_before = store.get(source)
            payloads = [merge_metadata_patch(dict(row), metadata) for row in rows]
            before = copy.deepcopy(rows)
            for row, payload in zip(rows, payloads, strict=True):
                row.update({key: payload[key] for key in _DOCUMENT_LEVEL_REPORT_FIELDS})
                for key in ("category", "date"):
                    if key in payload:
                        row[key] = payload[key]
            manifest = {"status": "not_applicable", "before": []}
            # documents 当前不包含 collection；仅默认集合拥有该来源映射。
            transaction = self._get_manifest_repository().metadata_date_transaction(
                TENANT_ID, source, payloads[0]["date"],
            ) if "date" in metadata and self.collection_name == REPORT_COLLECTION_NAME else nullcontext(manifest)
            attempted = False
            try:
                with transaction as manifest:
                    attempted = True
                    client.upsert(collection_name=self.collection_name, data=rows)
                    client.flush(self.collection_name)
                    store.update_metadata(source, payloads[0])
            except Exception as exc:
                compensation = {}
                if attempted:
                    for name, restore in (
                        ("milvus", lambda: (client.upsert(collection_name=self.collection_name, data=before), client.flush(self.collection_name))),
                        ("fingerprint", lambda: store.restore_record(source, fingerprint_before)),
                    ):
                        try:
                            restore()
                            compensation[name] = "restored"
                        except Exception as recovery_error:
                            compensation[name] = "failed"
                            compensation[name + "_error"] = str(recovery_error)
                    if "date" in metadata and self.collection_name == REPORT_COLLECTION_NAME:
                        try:
                            compensation["manifest"] = self._get_manifest_repository().restore_metadata_dates(
                                TENANT_ID, source, manifest["before"], payloads[0]["date"])
                        except Exception as recovery_error:
                            compensation["manifest"] = "failed"
                            compensation["manifest_error"] = str(recovery_error)
                error = RuntimeError(f"metadata update failed: {exc}")
                error.report = {"source": source, "status": "failed", "error": str(exc),
                                "compensation": compensation, "manifest": manifest,
                                "before_rows": before, "before_fingerprint": fingerprint_before}
                raise error from exc
            logger.info("研报元数据更新：%s（%d 行）", source, len(rows))
            return {"source": source, "updated_count": len(rows), "missing": False,
                    "metadata": payloads[0], "manifest": manifest}

    def _get_manifest_repository(self):
        from src.storage.relational_db.document_manifest import DocumentManifestRepository
        if not hasattr(self, "_metadata_manifest_repository"):
            self._metadata_manifest_repository = DocumentManifestRepository()
        return self._metadata_manifest_repository

    def remove_document(
        self, source: str, version: str | None = None
    ) -> dict[str, Any]:
        """按 source（可选指定版本）删除向量记录，保留本地原文件。

        * ``version=None``：删除该 source 的全部版本；
        * 指定版本：仅删除该版本的行，其余版本保留；
        * 仅当 source 已无任何记录时清理父块存储与指纹。

        内部持有集合级写锁（同 collection 串行）。
        """
        with self._write_lock:
            client = self._get_client()
            self.ensure_collection()
            deleted = self._delete_by_source(client, source, version=version)
            client.flush(self.collection_name)

            remaining = self._count_by_source(client, source)
            if remaining == 0:
                # 同步清理父块与整表存储及指纹（文档已完全删除）
                self._get_parent_store().delete_by_source(source)
                self._get_table_store().delete_by_source(source)
                self._get_fingerprint_store().remove(source)

            logger.info(
                "知识库向量删除：%s（version=%s，%d 条记录，本地文件保留）",
                source, version or "all", deleted,
            )
            return {"source": source, "version": version or "", "deleted_count": deleted}

    def _mark_old_versions_inactive(self, client: MilvusClient, source: str) -> None:
        """将该 source 现有的 is_current==true 行标记为 False。

        保留的历史辅助方法：当前**入库发布协议不再调用它**——它会连刚激活的新行
        一起停用，取而代之的是按主键差集调用的 :meth:`_set_current`。
        """
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}" and is_current == true'
        )
        rows: list[dict[str, Any]] = []
        iterator = client.query_iterator(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["*"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    row["is_current"] = False
                    rows.append(row)
        finally:
            iterator.close()

        if rows:
            client.upsert(collection_name=self.collection_name, data=rows)

    def _prune_old_versions(self, client: MilvusClient, source: str) -> None:
        """保留该 source 最新 ``MAX_VERSIONS_PER_DOC`` 个版本，删除更旧版本的行。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        rows: list[dict[str, Any]] = []
        iterator = client.query_iterator(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["id", "version", "ingested_at"],
            batch_size=1000,
        )
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()

        # 每个版本取最大 ingested_at 作为版本时间
        version_time: dict[str, int] = {}
        for row in rows:
            v = row.get("version", "") or ""
            ts = int(row.get("ingested_at", 0) or 0)
            if ts > version_time.get(v, 0):
                version_time[v] = ts

        kept_versions = {
            v for v, _ in sorted(
                version_time.items(), key=lambda item: item[1], reverse=True
            )[: MAX_VERSIONS_PER_DOC]
        }
        stale_ids = [
            row["id"] for row in rows if (row.get("version", "") or "") not in kept_versions
        ]
        if stale_ids:
            self._delete_by_ids(client, stale_ids)
            logger.info(
                "版本清理：%s 删除 %d 行（保留最新 %d 个版本）",
                source, len(stale_ids), MAX_VERSIONS_PER_DOC,
            )

    def list_documents(self, include_versions: bool = False) -> list[dict[str, Any]]:
        """列出知识库中的文档（按 source 聚合）。

        ``include_versions=True`` 时返回全部版本明细（含历史版本），
        否则仅统计当前版本（``is_current == true``）并附带当前版本号。
        """
        client = self._get_client()
        if not client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        ):
            return []
        self.ensure_collection()

        filter_expr = f'tenant_id == "{TENANT_ID}"'
        # 可见性门控按字段能力判断：未发布的暂存行与软删除行不得出现在列表里
        if not include_versions and (
            ENABLE_VERSIONING or self._collection_has_version_fields(client)
        ):
            filter_expr += " and is_current == true"

        # 研报元数据随列表返回：前端要靠它显示并支持人工修正。
        # 旧集合没有这些字段，逐字段过滤后再请求。
        report_fields = [
            name for name in _DOCUMENT_LEVEL_REPORT_FIELDS
            if self._collection_has_field(client, name)
        ]
        try:
            results = client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=[
                    "source", "title", "chunk", "category", "date",
                    "version", "ingested_at", "is_current",
                    *sorted(report_fields),
                ],
                limit=16384,
            )
        except Exception as exc:
            logger.warning("列出文档失败：%s", exc)
            return []

        agg: dict[str, dict[str, Any]] = {}
        for r in results:
            src = r.get("source", "")
            if src not in agg:
                agg[src] = {
                    "source": src,
                    "title": r.get("title", ""),
                    "category": r.get("category", ""),
                    "date": r.get("date", ""),
                    "chunk_count": 0,
                    "version": r.get("version", ""),
                    "versions": [],
                    **{name: r.get(name, "") for name in report_fields},
                }
            item = agg[src]
            item["chunk_count"] += 1
            if include_versions:
                item["versions"].append({
                    "version": r.get("version", "") or "",
                    "chunk_count": 0,  # 在下方汇总
                    "ingested_at": int(r.get("ingested_at", 0) or 0),
                    "is_current": bool(r.get("is_current", True)),
                })

        if include_versions:
            for item in agg.values():
                # 按版本汇总块数与最新入库时间
                merged: dict[str, dict[str, Any]] = {}
                for v in item["versions"]:
                    key = v["version"]
                    if key not in merged:
                        merged[key] = {"version": key, "chunk_count": 0,
                                       "ingested_at": 0, "is_current": False}
                    merged[key]["chunk_count"] += 1
                    merged[key]["ingested_at"] = max(
                        merged[key]["ingested_at"], v["ingested_at"]
                    )
                    merged[key]["is_current"] = (
                        merged[key]["is_current"] or v["is_current"]
                    )
                item["versions"] = sorted(
                    merged.values(), key=lambda v: v["ingested_at"], reverse=True
                )

        return list(agg.values())

    def has_rows(self) -> bool:
        """集合存在且至少有一行；不创建集合、不加载嵌入模型。

        供检索 API 在空集合（如尚未入库的 ``finance_policy``）上短路返回空结果，
        避免为了确认「没数据」而加载 ONNX/重排模型。
        """
        client = self._get_client()
        try:
            if not client.has_collection(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            ):
                return False
            stats = client.get_collection_stats(
                self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
            )
        except VectorStoreUnavailableError:
            raise
        except Exception as exc:
            raise VectorStoreUnavailableError(
                f"集合 {self.collection_name} 状态检查失败：{exc}"
            ) from exc
        return int(stats.get("row_count", 0) or 0) > 0

    def get_stats(self) -> dict[str, Any]:
        """知识库统计信息。"""
        client = self._get_client()
        has_collection = client.has_collection(
            self.collection_name, timeout=MILVUS_TIMEOUT_SECONDS
        )
        if not has_collection:
            return {
                "collection": self.collection_name,
                "exists": False,
                "document_count": 0,
                "chunk_count": 0,
            }
        self.ensure_collection()
        docs = self.list_documents()
        return {
            "collection": self.collection_name,
            "exists": True,
            "document_count": len(docs),
            "chunk_count": sum(d["chunk_count"] for d in docs),
        }

    # ------------------------------------------------------------------
    # 混合检索（委托给 HybridRetriever）
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        *,
        use_dense_only: bool = False,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
        filters: dict[str, Any] | None = None,
        rrf_k: int = RRF_K,
        keywords: list[str] | None = None,
        fusion_mode: str = RETRIEVAL_FUSION_MODE,
        dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
        sparse_weight: float = RETRIEVAL_SPARSE_WEIGHT,
        cliff_detection: bool = ENABLE_CLIFF_DETECTION,
        candidate_source_limit: int = RETRIEVAL_CANDIDATE_SOURCE_LIMIT,
        candidate_max_per_source: int = RETRIEVAL_CANDIDATE_MAX_PER_SOURCE,
    ) -> list[dict[str, Any]]:
        """稠密 + 稀疏（BM25）加权检索，可选 BGE 重排序 + 关键词增强。

        Args:
            query: 查询文本。
            k: 返回结果数（重排序关闭时的返回数量）。
            use_dense_only: 兼容开关，True 强制使用 dense 模式。
            expand_parents: 是否按 parent_id 去重并替换为父块全文。
            use_rerank: 是否启用 BGE 本地重排序。
            rerank_top_n: 启用重排序时的最终返回数量。
            filters: 元数据过滤条件（需 ENABLE_METADATA_FILTER=true）。
            rrf_k: 保留兼容参数，加权融合不使用它。
            keywords: 可选的关键词列表，用于过滤和重排序。
            fusion_mode: dense、weighted 或 auto。
            dense_weight: 稠密融合权重。
            sparse_weight: 稀疏融合权重。
            cliff_detection: 显式断崖开关，默认关闭。
            candidate_source_limit: 重排前候选文档数。
            candidate_max_per_source: 重排前每文档轮转配额。

        Returns:
            检索结果列表，每项含
            ``id / chunk_key / content / source / title / chunk / score / parent_id``
            （父子扩展时另含 ``child_id`` / ``parent_key``）。
            重排序启用时额外含 ``rerank_score`` 字段。

        Note:
            重排序失败时自动回退到 Milvus 原始排序，保证检索链路可用。
        """
        dense_weight, sparse_weight = normalize_fusion_weights(dense_weight, sparse_weight)
        self.ensure_collection()
        retriever = self._get_retriever()
        results = retriever.search(
            query,
            k=k,
            use_dense_only=use_dense_only,
            expand_parents=expand_parents,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
            filters=filters,
            rrf_k=rrf_k,
            keywords=keywords,
            fusion_mode=fusion_mode,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            cliff_detection=cliff_detection,
            candidate_source_limit=candidate_source_limit,
            candidate_max_per_source=candidate_max_per_source,
        )
        self.last_retrieval_cliff = retriever._last_cliff
        self.last_retrieval_diagnostics = dict(retriever._last_diagnostics)
        return results

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _get_retriever(self) -> HybridRetriever:
        """延迟创建混合检索器。"""
        if self._retriever is None:
            with self._init_lock:
                if self._retriever is None:
                    self._retriever = HybridRetriever(
                        client=self._get_client(),
                        collection_name=self.collection_name,
                        embed_query_fn=self._embed_query,
                        parent_store=self._get_parent_store(),
                    )
        return self._retriever

    @staticmethod
    def _escape_expr(value: str) -> str:
        """转义 Milvus 过滤表达式中的字符串字面量。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _delete_by_source(
        self, client: MilvusClient, source: str, version: str | None = None
    ) -> int:
        """删除指定 source（可选指定版本）的所有记录，返回删除行数。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        if version is not None:
            expr += f' and version == "{self._escape_expr(version)}"'
        try:
            # Milvus delete 不返回精确计数，删除前先统计匹配行数
            deleted = self._count_matching(client, expr)
            client.delete(
                collection_name=self.collection_name,
                filter=expr,
            )
            return deleted
        except Exception as exc:
            raise VectorStoreError(
                f"删除 source={source} version={version or 'all'} 失败：{exc}"
            ) from exc

    def _count_by_source(self, client: MilvusClient, source: str) -> int:
        """统计指定 source 的剩余记录行数（用于判断是否完全删除）。"""
        expr = (
            f'source == "{self._escape_expr(source)}" '
            f'and tenant_id == "{TENANT_ID}"'
        )
        return self._count_matching(client, expr)

    def _count_matching(self, client: MilvusClient, expr: str) -> int:
        """统计匹配过滤表达式的行数；失败抛 VectorStoreError。"""
        try:
            res = client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["count(*)"],
            )
            return int(res[0].get("count(*)", 0)) if res else 0
        except Exception as exc:
            raise VectorStoreError(f"统计行数失败（expr={expr}）：{exc}") from exc

    def _delete_by_ids(self, client: MilvusClient, ids: list[str]) -> None:
        """按主键批量删除记录。"""
        if not ids:
            return
        # 分批拼接 id in [...] 过滤表达式，避免表达式过长
        for i in range(0, len(ids), 100):
            batch = ids[i : i + 100]
            quoted = ", ".join(f'"{self._escape_expr(rid)}"' for rid in batch)
            client.delete(
                collection_name=self.collection_name,
                filter=f"id in [{quoted}]",
            )

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    try:
                        self._client = get_milvus_client()
                    except Exception as exc:
                        raise VectorStoreUnavailableError(
                            f"向量数据库客户端初始化失败（uri={MILVUS_URI}）：{exc}"
                        ) from exc
        return self._client

    def _get_embeddings(self) -> OnnxEmbedder:
        if self._embeddings is None:
            with self._init_lock:
                if self._embeddings is None:
                    self._embeddings = OnnxEmbedder(model_name=self.embedding_model)
        return self._embeddings

    def _get_parent_store(self) -> ParentStore:
        if self._parent_store is None:
            with self._init_lock:
                if self._parent_store is None:
                    # 父块全文持久化到 PostgreSQL，按集合名隔离
                    self._parent_store = ParentStore(collection=self.collection_name)
        return self._parent_store

    def _purge_derived_rows(self, source: str) -> None:
        """清掉某 source 在 PG 侧的派生行（父块全文、整表）。

        覆盖模式下重新入库前必须调用：父块 id 由内容哈希决定，重复入库会算出
        同一批 id，旧行还在就会撞 ``parent_chunks_pkey``，整篇文档入库失败。
        PG 不可用时只告警——真正的问题会在随后 insert 时以更明确的方式暴露。
        """
        for label, store in (
            ("parent_chunks", self._get_parent_store()),
            ("table_chunks", self._get_table_store()),
        ):
            try:
                store.delete_by_source(source)
            except Exception as exc:  # noqa: BLE001 - 清理由写锁保护，失败不吞掉写入
                logger.warning("清理 %s（source=%s）失败：%s", label, source, exc)

    def _get_table_store(self) -> TableStore:
        if self._table_store is None:
            with self._init_lock:
                if self._table_store is None:
                    # 整表持久化到 PostgreSQL（向量库只留「表头 + 首行」索引）
                    self._table_store = TableStore(collection=self.collection_name)
        return self._table_store

    def _get_fingerprint_store(self) -> FingerprintStore:
        if self._fingerprint_store is None:
            with self._init_lock:
                if self._fingerprint_store is None:
                    # 默认集合沿用基础指纹文件，其余集合按集合名隔离
                    self._fingerprint_store = FingerprintStore(
                        fingerprint_path_for(self.collection_name)
                    )
        return self._fingerprint_store

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        """分批嵌入文档"""
        embeddings: list[list[float]] = []
        try:
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i:i + EMBED_BATCH_SIZE]
                embeddings.extend(self._get_embeddings().embed_documents(batch))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"文档向量化失败：{exc}") from exc
        return embeddings

    def _embed_query(self, text: str) -> list[float]:
        try:
            return self._get_embeddings().embed_query(text)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"查询向量化失败：{exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量，供检索前的语义一致性校验使用。"""
        return self._embed_query(text)

    def get_embeddings(self) -> OnnxEmbedder:
        """公开获取嵌入器实例（供引用校验等跨模块使用）。"""
        return self._get_embeddings()

    def get_fingerprint_store(self) -> FingerprintStore:
        """公开获取指纹存储实例（供文档服务做增量检查）。"""
        return self._get_fingerprint_store()

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        """计算两个向量的余弦相似度；零向量返回 0。"""
        if len(left) != len(right) or not left:
            raise ValueError("向量维度不一致或为空")
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        denominator = left_norm * right_norm
        return dot_product / denominator if denominator else 0.0


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_KB_INSTANCES: dict[str, KnowledgeBase] = {}
_KB_LOCK = threading.Lock()


def get_knowledge_base(collection_name: str = KB_COLLECTION_NAME) -> KnowledgeBase:
    """获取（并缓存）指定集合的 KnowledgeBase 实例（线程安全）。

    默认返回全局配置集合的实例；传入其他集合名则返回对应实例，
    实例按集合名缓存，供多知识库管理与跨库检索使用。
    """
    kb = _KB_INSTANCES.get(collection_name)
    if kb is None:
        with _KB_LOCK:
            kb = _KB_INSTANCES.get(collection_name)
            if kb is None:
                kb = KnowledgeBase(collection_name=collection_name)
                _KB_INSTANCES[collection_name] = kb
    return kb
