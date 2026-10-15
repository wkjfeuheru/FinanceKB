"""混合检索与重排序模块。

提供：
* :class:`BGEReranker` — 本地 BGE 重排序器（延迟加载并缓存模型）
* :class:`HybridRetriever` — 稠密 + 稀疏（BM25）混合检索器

检索流程：
1. 稠密向量检索（ONNX INT8 本地嵌入 + COSINE）
2. 稀疏向量检索（Milvus 自动对 query 文本做 BM25 编码）
3. WeightedRanker 按显式权重融合（或 dense / auto 模式）
4. 可选父子扩展：按 parent_id 去重并替换为父块全文
5. 可选断崖检测：按粗排分数截断，再把留下的候选送入重排序
6. 可选 BGE 重排序（异常时回退到原始排序，不再二次截断）
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    MilvusClient,
    WeightedRanker,
)

from finance_rag.src.core.config import (
    APP_ENV,
    ENABLE_VERSIONING,
    ENABLE_CLIFF_DETECTION,
    MILVUS_NPROBE,
    MILVUS_TIMEOUT_SECONDS,
    MODEL_OFFLINE,
    RERANKER_CLIFF_ABS_THRESHOLD,
    RERANKER_CLIFF_MIN_TOP_K,
    RERANKER_CLIFF_REL_THRESHOLD,
    RERANKER_DEVICE,
    RERANKER_MODEL,
    RERANK_MAX_NON_TEXT_BLOCKS,
    RERANK_MAX_PER_SOURCE,
    RRF_K,
    RETRIEVAL_FUSION_MODE,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_SPARSE_WEIGHT,
    RETRIEVAL_CANDIDATE_SOURCE_LIMIT,
    RETRIEVAL_CANDIDATE_MAX_PER_SOURCE,
    normalize_fusion_weights,
)
from finance_rag.src.rag.context_injection import (
    is_non_text_block,
    rerank_text_for_match,
)
from finance_rag.src.core.exceptions import VectorStoreError
from finance_rag.src.core.logger import agent_logger
from finance_rag.src.rag.retrieval.parent_store import ParentStore

logger = logging.getLogger(__name__)

# 检索时过采样倍数（为重排序提供更多候选）
_OVERSAMPLE_FACTOR = 3


# ---------------------------------------------------------------------------
# 本地 BGE 重排序器
# ---------------------------------------------------------------------------

class BGEReranker:
    """本地 BGE 重排序器（延迟加载并缓存模型）。

    使用 ``sentence-transformers`` 的 ``CrossEncoder`` 加载
    ``BAAI/bge-reranker-v2-m3``（或兼容模型），完全本地推理，无需 API。

    模型在首次调用时加载，并以类变量缓存，避免重复分配约 2GB 内存。
    """

    _model_instance = None  # 类级缓存
    # 类级推理锁：模型实例为类变量单例，多线程（asyncio.to_thread 后并发请求）
    # 同时调用 predict 时串行化推理，避免 sentence-transformers 非线程安全问题
    _infer_lock = threading.Lock()
    # 模型加载锁：保护 _model_instance 的 check-then-act，避免并发重复加载约 2GB 权重
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_length: int = 512,
        device: str = RERANKER_DEVICE,
        use_fp16: bool = False,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._device = device
        self._use_fp16 = use_fp16

    def _get_model(self):
        """延迟加载并缓存 CrossEncoder 模型。"""
        if BGEReranker._model_instance is None:
            with BGEReranker._model_lock:
                if BGEReranker._model_instance is None:
                    from sentence_transformers import CrossEncoder

                    device = self._device
                    if device != "cpu" and APP_ENV != "production":
                        try:
                            import torch

                            if not torch.cuda.is_available():
                                logger.warning(
                                    "RERANKER_DEVICE=%s 但 CUDA 不可用，回退到 CPU", device
                                )
                                device = "cpu"
                        except ImportError:
                            device = "cpu"

                    model_kwargs = {"local_files_only": True} if MODEL_OFFLINE else {}
                    BGEReranker._model_instance = CrossEncoder(
                        self._model_name,
                        max_length=self._max_length,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                    if self._use_fp16 and device != "cpu":
                        BGEReranker._model_instance.model.half()
        return BGEReranker._model_instance

    def warmup(self) -> None:
        """启动预热：加载模型并执行一次小规模推理。

        将模型加载（可能含下载/解析权重）从「首个请求」挪到服务启动阶段，
        避免生产环境首请求冷启动超时。预热失败必须由调用方处理，防止
        服务在模型不可用时被误标记为就绪。
        """
        model = self._get_model()
        with BGEReranker._infer_lock:
            model.predict([["预热请求", "预热上下文"]])
        logger.info(
            "BGE 重排序模型预热完成：%s (device=%s)", self._model_name, self._device
        )
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 3,
        text_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> list[dict[str, Any]]:
        """对候选结果重排序，返回 top_n 条。

        Args:
            query: 查询文本。
            candidates: 候选结果列表（每项含 ``content`` 字段）。
            top_n: 返回的结果数量。
            text_fn: 可选的候选文本构造器。缺省直接用 ``content``；
                检索器会传入上下文注入版（公司简称 + 章节路径 + 正文），
                让短标题块在 cross-encoder 里拿到查询词对齐信号。

        Returns:
            重排序后的 top_n 结果，每项含 ``rerank_score`` 字段。

        Note:
            模型加载或推理失败时回退到原始排序，保证检索链路可用。
        """
        if not candidates:
            return []

        agent_logger.rag_rerank_start(candidate_count=len(candidates))
        t0 = time.perf_counter()

        try:
            model = self._get_model()
            if text_fn is None:
                pairs = [[query, c.get("content", "")[:2000]] for c in candidates]
            else:
                pairs = [[query, text_fn(c)] for c in candidates]
            with BGEReranker._infer_lock:
                scores = model.predict(pairs)
            for i, score in enumerate(scores):
                if i < len(candidates):
                    candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        except Exception as exc:
            logger.warning("BGE 重排序失败，回退到原始排序：%s", exc)
            for c in candidates:
                c.setdefault("rerank_score", c.get("score", 0.0))
            candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        result_count = min(top_n, len(candidates))
        agent_logger.rag_rerank_end(
            elapsed_ms=elapsed_ms,
            result_count=result_count,
        )
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# 断崖检测
# ---------------------------------------------------------------------------

def _apply_cliff_detection(
    candidates: list[dict[str, Any]],
    rel_threshold: float = RERANKER_CLIFF_REL_THRESHOLD,
    abs_threshold: float = RERANKER_CLIFF_ABS_THRESHOLD,
    min_top_k: int = RERANKER_CLIFF_MIN_TOP_K,
) -> list[dict[str, Any]]:
    """按粗排 ``score`` 检测断崖，截掉不必送进重排序的尾部。

    三条约束，满足任一即判定断崖（取**最早**一处截断）：

    1. **绝对差距**：``gap = s[i] - s[i+1] >= abs_threshold``（默认 0.8）；
    2. **相对跌幅**：``rel = gap / s[i] >= rel_threshold``（默认 0.40）；
    3. **兜底**：即使命中，也至少保留 ``min_top_k`` 条送进重排序，避免误判把
       上下文砍到只剩一两条（调用方会取 ``max(min_top_k, rerank_top_n)``，
       因此重排候选数不会低于客户端请求的最终条数）。

    两点分数尺度的硬事实（决定这两条规则各自的真实作用面）：

    * 分数 <= 1 时 ``gap >= 0.8`` 必然蕴含 ``rel >= 0.8 > 0.40``——**绝对规则
      被相对规则完全蕴含**；而 RRF 融合分上界约 ``2/61≈0.033``（``1/(60+rank)``
      两路求和），0.8 在这种模式下永不可达。保留绝对规则是为了显式表达
      「绝对断层」的意图，并在未来分数尺度 >1 时才有独立意义：它不改变任何
      当前结果。
    * 扫描**单次、首次命中即停**（最早断崖优先），不做「找最深落差」。
    * 非正 ``s[i]``（COSINE 分可为负）只让相对跌幅失效，绝对规则仍按差距判定。

    规则级禁用：对应阈值 ``<= 0`` 即关闭该规则；两条都 ``<= 0`` 则整体禁用
    （原样返回，不截断）。
    """
    if not candidates or len(candidates) <= 1:
        return candidates
    if rel_threshold <= 0 and abs_threshold <= 0:
        return candidates

    scores = [float(c.get("score", 0.0) or 0.0) for c in candidates]
    floor = max(1, int(min_top_k))
    cut = len(candidates)  # 默认不截断

    for i in range(len(scores) - 1):
        gap = scores[i] - scores[i + 1]
        # 绝对规则不涉及除法，因此对非正分数同样成立（COSINE 分可以为负）；
        # 相对跌幅只在 s[i] > 0 时有定义，非正时该规则自动失效。
        hit_abs = abs_threshold > 0 and gap >= abs_threshold
        rel = gap / scores[i] if scores[i] > 0 else 0.0
        hit_rel = rel_threshold > 0 and scores[i] > 0 and rel >= rel_threshold
        if not (hit_abs or hit_rel):
            continue
        cut = max(floor, i + 1)
        if hit_abs and hit_rel:
            rule = "绝对+相对"
        elif hit_abs:
            rule = "绝对"
        else:
            rule = "相对"
        logger.info(
            "断崖检测：score[%d]=%.4f → score[%d]=%.4f（差距 %.4f，跌幅 %.1f%%，"
            "命中 %s 规则），截断为 %d/%d 条",
            i, scores[i], i + 1, scores[i + 1], gap, rel * 100, rule,
            cut, len(candidates),
        )
        break

    return candidates[:cut]


def _apply_diversity(
    matches: list[dict[str, Any]],
    top_n: int,
    max_per_source: int = RERANK_MAX_PER_SOURCE,
    max_non_text: int = RERANK_MAX_NON_TEXT_BLOCKS,
) -> list[dict[str, Any]]:
    """重排后最终选取的多样性配额（MMR-lite：确定性规则，不引入新模型）。

    gold20 复盘的两个失败模式都由「重排分靠前的候选同源」直接造成：

    * 同一篇研报的表格块（表头摘要关键词密度高）占满 top-5，
      把问题真正指向的标题/正文块挤出槽位；
    * 跨文档（多跳）题的证据分布在不同文档，单文档霸榜让第二篇文档
      的证据一条都进不了 top-5。

    规则：按重排顺位扫描，每篇文档（``source``）最多 ``max_per_source`` 条、
    表格/图片块合计最多 ``max_non_text`` 条；配额把候选耗尽导致选不满时，
    按原顺位**回填**被跳过的候选（宁可有偏，不能缺条）。两个配额传 0 表示
    对应规则关闭。候选数本就 <= top_n 时无取舍空间，原样返回。
    """
    if top_n <= 0 or len(matches) <= top_n:
        return matches[:top_n]
    if max_per_source <= 0 and max_non_text <= 0:
        return matches[:top_n]

    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    non_text_count = 0

    for match in matches:
        source = str(match.get("source", "") or "")
        is_non_text = is_non_text_block(match)
        over_source = max_per_source > 0 and per_source.get(source, 0) >= max_per_source
        over_non_text = is_non_text and max_non_text > 0 and non_text_count >= max_non_text
        if over_source or over_non_text:
            skipped.append(match)
            continue
        per_source[source] = per_source.get(source, 0) + 1
        if is_non_text:
            non_text_count += 1
        selected.append(match)
        if len(selected) == top_n:
            break

    if len(selected) < top_n:
        taken = {id(m) for m in selected}
        for match in skipped:
            if id(match) not in taken:
                selected.append(match)
                if len(selected) == top_n:
                    break
    return selected


# ---------------------------------------------------------------------------
# 混合检索器
# ---------------------------------------------------------------------------

def _rebalance_source_candidates(
    matches: list[dict[str, Any]],
    target: int,
    source_limit: int,
    max_per_source: int,
    source_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    """按文档最高粗排分路由，轮转取块，最后按原顺位回填。"""
    unique: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for match in matches:
        key = match.get("id") or match.get("chunk_key") or id(match)
        if key not in seen:
            seen.add(key)
            unique.append(match)
    groups: dict[str, list[dict[str, Any]]] = {}
    for match in unique:
        groups.setdefault(str(match.get("source") or ""), []).append(match)
    sources = source_order if source_order is not None else sorted(
        groups, key=lambda s: max(float(m.get("score") or 0) for m in groups[s]), reverse=True
    )
    sources = sources[:max(1, source_limit)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for offset in range(max(1, max_per_source)):
        for source in sources:
            if offset < len(groups.get(source, [])):
                match = groups[source][offset]
                selected.append(match)
                selected_ids.add(id(match))
                if len(selected) >= target:
                    return selected[:target]
    for match in unique:
        if id(match) not in selected_ids:
            selected.append(match)
            if len(selected) >= target:
                break
    return selected[:target]


class HybridRetriever:
    """稠密 + 稀疏（BM25）混合检索器。

    封装 Milvus hybrid_search / search 调用，支持：
    * 稠密向量 + BM25 稀疏向量双路检索
    * WeightedRanker 融合（默认 dense/sparse = 0.8/0.2）
    * 纯稠密模式（use_dense_only=True 时切换）
    * 父子扩展：按 parent_id 去重并替换为父块全文
    * 可选 BGE 重排序（异常时回退到原始排序）

    表格子块在向量库里只有「表头 + 摘要」索引文本（整表存 PostgreSQL）。
    召回与重排全程基于这份紧凑文本——信噪比高、重排成本低；完整表格由
    聊天链路在**输出拼上下文前**按 ``parent_id``（= table_id）展开
    （见 ``chat_service.expand_table_docs``），本类不做整表展开。

    Args:
        client: MilvusClient 实例。
        collection_name: Milvus 集合名。
        embed_query_fn: 查询文本 -> 稠密向量的函数。
        reranker: 可选的 BGEReranker 实例（为 None 时按需创建）。
        parent_store: 可选的 ParentStore 实例，用于父子扩展。
    """

    #: 基础检索字段：任何集合都必须有
    BASE_OUTPUT_FIELDS: tuple[str, ...] = (
        "id", "content", "source", "title", "chunk", "parent_id",
        "chunk_key", "category", "date",
    )
    #: 研报维度与证据字段：仅在新 schema 集合上请求（旧集合请求会直接报错）。
    #: heading_path 供重排上下文注入（短标题块靠它拿到章节词对齐信号）。
    REPORT_OUTPUT_FIELDS: tuple[str, ...] = (
        "block_type", "start_page", "end_page", "image_key",
        "security_name", "industry_l1", "industry_l2",
        "report_type", "broker", "heading_path",
    )
    #: 走父块扩展的块类型：表格与图片的 parent_id 指向各自存储
    _NON_PARENT_BLOCK_TYPES = frozenset({"table", "image"})

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embed_query_fn,
        reranker: BGEReranker | None = None,
        parent_store: ParentStore | None = None,
    ):
        self._client = client
        self._collection_name = collection_name
        self._embed_query = embed_query_fn
        self._reranker = reranker
        self._parent_store = parent_store
        # 集合字段缓存：用于按需请求研报字段（None = 尚未探测）
        self._schema_fields: frozenset[str] | None = None
        # 最近一次重排前断崖：pre 是送入断崖的粗召回条数，post 是进入重排的条数
        self._last_cliff: dict[str, int] | None = None
        self._last_diagnostics: dict[str, Any] = {}

    def search(
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
        """混合检索 + 可选父子扩展 + 可选 BGE 重排序 + 关键词增强。

        流程：
        1. 稠密向量检索（ONNX INT8 本地嵌入 + COSINE）
        2. 稀疏向量检索（Milvus 自动对 query 文本做 BM25 编码）
        3. WeightedRanker 融合两路结果（或 dense / auto 模式）
        4. 可选父子扩展：按 parent_id 去重并替换为父块全文
        5. **关键词过滤重排序**：如果提供了关键词，优先保留包含关键词的文档
        6. 可选断崖检测：用粗排分数截断候选，再送入 BGE
        7. 可选 BGE 重排序（异常时回退到原始排序，不再按重排分数截断）

        Args:
            query: 查询文本。
            k: 返回结果数（重排序关闭时的返回数量）。
            use_dense_only: 兼容开关，True 强制映射到 dense。
            expand_parents: 是否按 parent_id 去重并替换为父块全文。
            use_rerank: 是否启用 BGE 本地重排序。
            rerank_top_n: 启用重排序时的最终返回数量。
            filters: 元数据过滤条件，如 {"source": "report.pdf"}。
                仅当 ENABLE_METADATA_FILTER=true 时生效。
            rrf_k: 保留兼容参数；WeightedRanker 不使用它。
            keywords: 可选的关键词列表，用于过滤和重排序。
            fusion_mode: dense、weighted 或 auto；auto 仅精确信号走 weighted。
            dense_weight: 稠密融合权重，默认 0.8。
            sparse_weight: 稀疏融合权重，默认 0.2。
            cliff_detection: 显式启用粗排断崖，默认关闭。
            candidate_source_limit: 按文档最高粗排分选择的文档数。
            candidate_max_per_source: 轮转阶段每篇文档的块数上限，不足按原序回填。

        Returns:
            检索结果列表，每项含
            ``id / chunk_key / content / source / title / chunk / score / parent_id``；
            启用父子扩展时额外含 ``child_id``（扩展前的子块主键）与 ``parent_key``；
            重排序启用时额外含 ``rerank_score``。

            ``id`` 为 Milvus 主键（**子块** id），父子扩展只替换 ``content``，
            不改 ``id``，因此评测侧可据此对齐标注的证据块。

        Note:
            重排序失败时自动回退到 Milvus 原始排序，保证检索链路可用。
        """
        t0 = time.perf_counter()

        dense_weight, sparse_weight = normalize_fusion_weights(dense_weight, sparse_weight)
        # 重排序开启时过采样候选，否则按 k 检索
        fetch_k = max(k, rerank_top_n * _OVERSAMPLE_FACTOR) if use_rerank else k
        limit = fetch_k * _OVERSAMPLE_FACTOR

        dense_vector = self._embed_query(query)
        filter_expr = self._build_filter(filters)
        # 版本保留模式：仅检索最新版本（is_current == true）
        if ENABLE_VERSIONING:
            filter_expr = self._combine_filter(filter_expr, "is_current == true")
        output_fields = self._output_fields()

        mode = "dense" if use_dense_only else fusion_mode
        if mode not in {"dense", "weighted", "auto"}:
            raise ValueError(f"未知融合模式：{mode}")
        if mode == "auto":
            exact_signal = re.search(r'\d|[A-Za-z]+|[“「『\"\x27][^”」』\"\x27]+[”」』\"\x27]', query)
            mode = "weighted" if exact_signal else "dense"
        self._last_diagnostics = {"fusion_mode": mode, "cliff_enabled": bool(cliff_detection and use_rerank)}
        self._last_parent_expansion = {"input": 0, "output": 0, "restored": 0, "failures": {}}
        matches = self._search_candidates(
            query, dense_vector, mode, dense_weight, sparse_weight,
            filter_expr, limit, output_fields,
        )

        source_count = len({m.get("source") or "" for m in matches})
        from copy import deepcopy
        self._last_diagnostics.update({
            "coarse_candidates": deepcopy(matches),
            "coarse_candidate_count": len(matches),
            "coarse_candidate_chunk_ids": [str(m.get("id") or "") for m in matches],
            "coarse_source_count": source_count,
            "coarse_duplicate_source_slots": len(matches) - source_count,
        })

        # 文档路由只使用第一阶段全局原始粗排分，不受父块去重/关键词顺位影响。
        source_scores: dict[str, float] = {}
        for match in matches:
            source = str(match.get("source") or "")
            if source:
                source_scores[source] = max(source_scores.get(source, float("-inf")), match["score"])
        routed_sources = sorted(source_scores, key=source_scores.get, reverse=True)[:max(1, candidate_source_limit)]

        # 重排开启时保留粗召回池，断崖才能看到尾部；关闭时仍按最终 k 截断
        pool = limit if use_rerank else k

        # 父子扩展（表格/图片块被跳过：其 parent_id 指向各自存储而非父块；
        # 表格的整表展开推迟到输出拼上下文时按 table_id 进行）
        if expand_parents and self._parent_store and matches:
            matches = self._expand_to_parents(matches, pool)

        # 关键词过滤与重排序（核心优化）
        if keywords and matches:
            matches = self._apply_keyword_boost(matches, keywords, pool)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        agent_logger.knowledge_retrieved(
            faq_hits=len(matches),
            rule_hits=0,
            product_hits=0,
            elapsed_ms=elapsed_ms,
            query_len=len(query),
        )

        # 断崖在重排之前，只决定送进 BGE 的候选；重排后按配额选最终 top_n
        self._last_cliff = None
        if use_rerank and matches:
            pre_cliff = len(matches)
            # 兜底下限同时不得低于最终需要的条数：客户端可把 rerank_top_n
            # 调到 10，若下限固定为 5，断崖会让重排拿不到足够的候选
            cliff_floor = max(RERANKER_CLIFF_MIN_TOP_K, rerank_top_n)
            if cliff_detection:
                matches = _apply_cliff_detection(matches, min_top_k=cliff_floor)
            self._last_cliff = {"pre": pre_cliff, "post": len(matches)}
            # 全局粗排用于选文档；文档内再次检索才能找回不在全局池里的证据块。
            document_matches: list[dict[str, Any]] = []
            document_failures = 0
            for source in routed_sources:
                scoped_filter = self._combine_filter(
                    filter_expr, self._build_scalar_condition("source", source)
                )
                try:
                    local = self._search_candidates(
                        query, dense_vector, mode, dense_weight, sparse_weight,
                        scoped_filter, max(1, candidate_max_per_source), output_fields,
                    )
                    # 防御旧客户端忽略过滤的返回；仅允许路由文档贡献第二阶段候选。
                    local = [match for match in local if match.get("source") == source]
                    if expand_parents and self._parent_store and local:
                        local = self._expand_to_parents(local, len(local))
                    if keywords and local:
                        local = self._apply_keyword_boost(local, keywords, len(local))
                    if cliff_detection and local:
                        local = _apply_cliff_detection(local, min_top_k=cliff_floor)
                except Exception as exc:
                    document_failures += 1
                    logger.warning("文档内检索失败，回退全局候选（%s）：%s", source, exc)
                    local = []
                if not local:
                    local = [match for match in matches if match.get("source") == source]
                document_matches.extend(local[:max(1, candidate_max_per_source)])
            self._last_diagnostics.update({
                "routed_sources": list(routed_sources),
                "document_search_count": len(routed_sources),
                "document_search_failures": document_failures,
                "document_candidate_chunk_ids": [str(m.get("id") or "") for m in document_matches],
                "document_candidate_count": len(document_matches),
                "document_candidates": deepcopy(document_matches),
            })
            matches = _rebalance_source_candidates(
                document_matches + matches, fetch_k, candidate_source_limit,
                candidate_max_per_source, source_order=routed_sources or None,
            )
            rerank_source_count = len({m.get("source") or "" for m in matches})
            self._last_diagnostics.update({
                "rerank_candidate_count": len(matches),
                "rerank_candidate_chunk_ids": [str(m.get("id") or "") for m in matches],
                "rerank_source_count": rerank_source_count,
                "rerank_duplicate_source_slots": len(matches) - rerank_source_count,
                "rerank_candidates": deepcopy(matches),
            })
            reranker = self._get_reranker()
            # 1) 对整个断崖后候选池打分（top_n 传全量，多样性配额才有补位空间）；
            # 2) 候选文本带上下文注入：短标题块在正文之外靠 security_name /
            #    heading_path 拿到查询词对齐信号（否则评级/标题类证据永远进不了 top-5）。
            matches = reranker.rerank(
                query, matches, top_n=len(matches), text_fn=rerank_text_for_match
            )
            # 3) 多样性配额：限制单文档霸榜与表格/图片挤占最终槽位；
            #    跨文档证据（多跳题）因此能进 top-5。
            matches = _apply_diversity(
                matches,
                top_n=rerank_top_n,
                max_per_source=RERANK_MAX_PER_SOURCE,
                max_non_text=RERANK_MAX_NON_TEXT_BLOCKS,
            )
            # 重排序后以 rerank_score 作为对外展示的相关度分数
            for m in matches:
                if m.get("rerank_score") is not None:
                    m["score"] = m["rerank_score"]
        else:
            matches = matches[:k]

        # 分数归一化：将粗排融合分数或 BGE 重排序分数
        # 映射到 0-1 区间，使首条结果分数为 1.0，其余按比例缩放，便于前端直观展示。
        if matches:
            scores = [m.get("score", 0) for m in matches if m.get("score") is not None]
            max_score = max(scores) if scores else 0
            if max_score > 0:
                for m in matches:
                    raw = m.get("score")
                    if raw is not None:
                        m["score"] = round(float(raw) / max_score, 4)

        return matches

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _search_candidates(
        self, query: str, dense_vector: list[float], mode: str,
        dense_weight: float, sparse_weight: float, filter_expr: str,
        limit: int, output_fields: list[str],
    ) -> list[dict[str, Any]]:
        """全局与文档内搜索共用参数与命中映射，避免重复嵌入/模型调用。"""
        kwargs = {
            "collection_name": self._collection_name, "limit": limit,
            "output_fields": output_fields, "timeout": MILVUS_TIMEOUT_SECONDS,
        }
        try:
            if mode == "dense":
                kwargs.update(data=[dense_vector], anns_field="dense_vector",
                              search_params={"metric_type": "COSINE", "params": {"nprobe": MILVUS_NPROBE}})
                if filter_expr:
                    kwargs["filter"] = filter_expr
                results = self._client.search(**kwargs)
            else:
                kwargs.update(reqs=[
                    AnnSearchRequest(data=[dense_vector], anns_field="dense_vector",
                                     param={"metric_type": "COSINE", "params": {"nprobe": MILVUS_NPROBE}},
                                     limit=limit, expr=filter_expr or None),
                    AnnSearchRequest(data=[query], anns_field="sparse_vector", param={"metric_type": "BM25"},
                                     limit=limit, expr=filter_expr or None),
                ], ranker=WeightedRanker(dense_weight, sparse_weight))
                results = self._client.hybrid_search(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"向量检索失败（{mode}）：{exc}") from exc
        matches: list[dict[str, Any]] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            chunk_id = entity.get("id") or hit.get("id") or ""
            match = {
                "id": chunk_id, "chunk_key": entity.get("chunk_key", "") or chunk_id,
                "content": entity.get("content"), "source": entity.get("source"),
                "title": entity.get("title") or "金融文档", "chunk": entity.get("chunk"),
                "score": float(hit.get("distance", hit.get("score", 0.0))),
                "block_type": entity.get("block_type", "") or "text",
                "start_page": int(entity.get("start_page") or 0),
                "end_page": int(entity.get("end_page") or 0),
            }
            for field in ("parent_id", "category", "date", "image_key", "security_name",
                          "industry_l1", "industry_l2", "report_type", "broker", "heading_path"):
                match[field] = entity.get(field, "") or ""
            matches.append(match)
        return matches

    def _collection_field_names(self) -> frozenset[str]:
        """探测集合适用的字段名（失败返回空集，调用方据此降级）。"""
        if self._schema_fields is None:
            try:
                desc = self._client.describe_collection(self._collection_name)
                self._schema_fields = frozenset(
                    str(field.get("name", "")) for field in desc.get("fields", [])
                )
            except Exception as exc:  # noqa: BLE001 - 旧集合/探测失败只降级
                logger.warning("探测集合 %s 字段失败，仅取基础字段：%s",
                               self._collection_name, exc)
                self._schema_fields = frozenset()
        return self._schema_fields

    def _output_fields(self) -> list[str]:
        """按集合实际 schema 组装 output_fields。

        旧集合并无研报字段，直接请求会让整个检索报错，因此逐字段过滤；
        宁可少取几个字段，也不能让检索整体不可用。
        """
        present = self._collection_field_names()
        fields = list(self.BASE_OUTPUT_FIELDS)
        # 探测失败（present 为空）时保守只取基础字段：请求不存在的字段会让
        # 整个检索报错，而少取几个字段只是少一些元数据。
        fields.extend(name for name in self.REPORT_OUTPUT_FIELDS if name in present)
        return fields

    def _get_reranker(self) -> BGEReranker:
        """延迟创建 BGE 重排序器实例。"""
        if self._reranker is None:
            self._reranker = BGEReranker()
        return self._reranker

    @staticmethod
    def _quote(value: Any) -> str:
        """Milvus 表达式值序列化：字符串加双引号，其余原样。"""
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _build_scalar_condition(field: str, value: Any) -> str:
        """构造标量字段过滤条件（== 或 in）。"""
        if isinstance(value, (list, tuple)):
            items = ", ".join(HybridRetriever._quote(v) for v in value)
            return f"{field} in [{items}]"
        return f"{field} == {HybridRetriever._quote(value)}"

    @staticmethod
    def _build_date_condition(value: Any) -> str | None:
        """构造 date 字段过滤条件（ISO 日期字符串的字典序即时间序）。

        value 为标量或列表时精确匹配；为 dict 时支持 gte/lte/gt/lt 范围比较。
        """
        op_map = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<"}
        if isinstance(value, dict):
            conditions = [
                f"date {op_map[op]} {HybridRetriever._quote(v)}"
                for op, v in value.items()
                if op in op_map
            ]
            return " and ".join(conditions) or None
        if isinstance(value, (list, tuple)):
            return HybridRetriever._build_scalar_condition("date", value)
        if value is None or value == "":
            return None
        return f"date == {HybridRetriever._quote(value)}"

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> str:
        """构造 Milvus 过滤表达式。

        支持的顶层键：
        - ``category``：分类，标量（==）或列表（in）
        - ``date``：日期，标量（==）、列表（in）或含 gte/lte/gt/lt 的 dict（范围比较）
        - 其他标量字段（如 source）：精确匹配 ``==`` 或 ``in``（向后兼容）

        示例::

            {"category": ["compliance_risk"], "date": {"gte": "2024-01-01"}}
            -> category in ["compliance_risk"] and date >= "2024-01-01"
        """
        if not filters:
            return ""
        report_types = filters.get("report_type")
        if isinstance(report_types, (list, tuple)) and len(set(report_types)) > 1:
            type_fields = {"security_name", "industry_l1", "industry_l2"}
            common = {field: value for field, value in filters.items()
                      if field not in type_fields and field != "report_type"}
            applicable = {"个股": ("security_name",), "行业": ("industry_l1", "industry_l2")}
            branches = []
            for report_type in dict.fromkeys(report_types):
                branch = [HybridRetriever._build_scalar_condition("report_type", report_type)]
                branch.extend(HybridRetriever._build_scalar_condition(field, filters[field])
                              for field in applicable.get(report_type, ()) if field in filters)
                branches.append("(" + " and ".join(branch) + ")")
            union = "(" + " or ".join(branches) + ")"
            common_expr = HybridRetriever._build_filter(common)
            return f"{common_expr} and {union}" if common_expr else union
        conditions = []
        for field, value in filters.items():
            if field == "date":
                cond = HybridRetriever._build_date_condition(value)
                if cond:
                    conditions.append(cond)
            else:
                conditions.append(HybridRetriever._build_scalar_condition(field, value))
        return " and ".join(conditions)

    @staticmethod
    def _combine_filter(existing: str, extra: str) -> str:
        """将附加过滤条件与已有表达式按 and 组合。"""
        if not existing:
            return extra
        if not extra:
            return existing
        return f"({existing}) and ({extra})"

    def _expand_to_parents(
        self, matches: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """先批量读取父块，再仅合并可证明完整覆盖的子块组。

        优化：
        - 父块过大（>3000字符）时，保留原始子块内容避免噪音扩散
        - 父块过小（<50字符）时，保留原始子块内容（可能只是标题）
        """
        PARENT_MAX_CHARS = 3000
        PARENT_MIN_CHARS = 50

        seen_ids: set[str] = set()
        unique: list[dict[str, Any]] = []
        for m in matches:
            identity = m.get('id') or m.get('chunk_key')
            if identity and identity in seen_ids:
                continue
            if identity:
                seen_ids.add(identity)
            unique.append(dict(m))
        # 父块扩展会替换 content，但子块身份必须保留，供评测侧做证据块对齐。
        # 放在查父块之前计算，保证「父块查询失败回退」路径也带上身份字段。
        for m in unique:
            m.setdefault("child_id", m.get("id", ""))
            m["parent_key"] = m.get("parent_id", "")

        groups: dict[str, list[dict[str, Any]]] = {}
        for m in unique:
            if m.get('block_type') not in self._NON_PARENT_BLOCK_TYPES and m.get('parent_id'):
                groups.setdefault(m['parent_id'], []).append(m)
        parent_ids = list(groups)
        diagnostics = getattr(self, '_last_parent_expansion', None)
        if diagnostics is None:
            diagnostics = {"input": 0, "output": 0, "restored": 0, "failures": {}}
            self._last_parent_expansion = diagnostics
        diagnostics['input'] += len(matches)
        parents_map = {}
        query_failed = False
        if parent_ids:
            try:
                parents_map = {
                    p["id"]: p for p in self._parent_store.get_batch(parent_ids)
                }
            except Exception as exc:
                logger.warning("父子扩展查询父块失败，回退为子块内容：%s", exc)
                query_failed = True
        valid: dict[str, dict[str, Any]] = {}
        for pid, children in groups.items():
            parent = parents_map.get(pid)
            content = parent.get('content') if parent else None
            reason = ''
            if query_failed:
                reason = 'read_failure'
            elif parent is None:
                reason = 'missing'
            elif not isinstance(content, str):
                reason = 'invalid_content'
            elif len(content) < PARENT_MIN_CHARS:
                reason = 'too_short'
            elif len(content) > PARENT_MAX_CHARS:
                reason = 'too_long'
            elif any(not child.get('content') or child['content'] not in content for child in children):
                reason = 'content_mismatch'
            if reason:
                failures = diagnostics['failures']
                failures[reason] = failures.get(reason, 0) + len(children)
            else:
                valid[pid] = parent
        output = []
        emitted = set()
        for m in unique:
            pid = m.get('parent_id')
            if m.get('block_type') not in self._NON_PARENT_BLOCK_TYPES and pid in valid:
                if pid in emitted:
                    continue
                emitted.add(pid)
                parent = valid[pid]
                m['content'] = parent['content']
                m['heading'] = parent.get('heading', '')
                m['heading_path'] = parent.get('heading_path', '')
                if parent.get('title'):
                    m['title'] = parent['title']
                m['covered_child_ids'] = list(dict.fromkeys(
                    identity
                    for child in groups[pid]
                    for identity in [child.get('child_id') or child.get('id', ''), *child.get('covered_child_ids', [])]
                    if identity
                ))
                diagnostics['restored'] += 1
            output.append(m)
        output = output[:k]
        diagnostics['output'] += len(output)
        if hasattr(self, '_last_diagnostics'):
            self._last_diagnostics['parent_expansion'] = diagnostics
        logger.info('父块扩展诊断：%s', diagnostics)
        return output

    def _apply_keyword_boost(
        self, matches: list[dict[str, Any]], keywords: list[str], k: int
    ) -> list[dict[str, Any]]:
        """根据关键词对检索结果进行过滤和重排序。

        策略：
        1. 遍历所有匹配，计算每个文档的关键词命中分数
        2. 支持精确匹配和子串匹配
        3. 将文档分为两组：包含关键词的（relevant）和不包含的（other）
        4. 相关文档按关键词命中数排序，非相关文档保留在后面
        5. 返回 top_k 个结果
        """
        if not keywords or not matches:
            return matches

        # 计算每个文档的关键词命中情况
        for m in matches:
            content = m.get("content", "") or ""
            hit_count = 0
            for kw in keywords:
                if not kw:
                    continue
                # 精确匹配（直接包含）
                if kw in content:
                    hit_count += 2
                    continue
                # 多字符关键词的子串匹配
                # 检查关键词中是否有足够比例的字符出现在内容中
                kw_chars = [c for c in kw if c.strip()]
                if len(kw_chars) >= 2:
                    matched_chars = sum(1 for c in kw_chars if c in content)
                    # 至少60%的字符匹配
                    if matched_chars >= max(1, int(len(kw_chars) * 0.6)):
                        hit_count += 1
            m["_keyword_hits"] = hit_count

        # 分组：相关 vs 非相关
        relevant = [m for m in matches if m.get("_keyword_hits", 0) > 0]
        other = [m for m in matches if m.get("_keyword_hits", 0) == 0]

        # 相关文档按关键词命中数降序，然后按原始分数降序
        relevant.sort(key=lambda m: (m["_keyword_hits"], m.get("score", 0.0)), reverse=True)

        # 合并：相关文档在前，非相关文档在后
        boosted = relevant + other

        # 清理临时字段
        for m in boosted:
            m.pop("_keyword_hits", None)

        logger.info(
            "关键词增强：%d/%d 个文档包含关键词，top 关键词=%s",
            len(relevant), len(matches), keywords[:3],
        )

        return boosted[:k]


