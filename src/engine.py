"""Agentic RAG 问答链（LangChain LCEL + 流式输出）。

基础 Agentic RAG 流程（单轮）：
1. 元数据过滤自动推断（Self-querying）：LLM 根据问题推断分类/日期过滤条件
2. 混合检索 + 可选 BGE 重排序：统一由 KnowledgeBase.hybrid_search 完成
3. 上下文组装：按来源拼接，标注引用编号
4. 流式生成：LangChain ``astream`` 逐 token 输出

注：查询改写（多路子查询检索）已并入 LangGraph Agentic 链路
（ENABLE_LANGGRAPH，见 src.agent.graph），标准链路直接使用原始查询。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator, Callable
from copy import deepcopy
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agent.prompts.chat import (
    HYDE_GENERATION_PROMPT,
    METADATA_FILTER_INFER_PROMPT,
)
from src.core.config import (
    CHAT_ENABLE_QUERY_REWRITE,
    CHAT_RERANK_TOP_K,
    CHAT_TOP_K,
    ENABLE_HYDE,
    ENABLE_LANGGRAPH,
    ENABLE_RERANKER,
    HYDE_WEIGHT,
    KB_COLLECTION_NAME,
    LLM_TIMEOUT_SECONDS,
    get_answer_model,
    get_model,
    get_rewrite_model,
)
from src.index.milvus_kb import get_knowledge_base
from src.chunking.metadata_extractor import metadata_field_validators
from src.retrieve.filter_policy import build_filter_plan, validate_explicit_filters
from src.index.table_store import TableStore

logger = logging.getLogger(__name__)

# 单块进入 LLM 上下文的字符上限：表格与普通文本区别对待
TEXT_CONTEXT_MAX_CHARS = 2000
# 单表进入 LLM 上下文的字符上限（输出期整表展开时按行截断，绝不静默丢数据）
TABLE_MAX_CHARS = 8000
TABLE_TRUNCATION_NOTE = "\n\n（表已截断，可点击查看完整表）"
TABLE_CONTEXT_MAX_CHARS = TABLE_MAX_CHARS


# ---------------------------------------------------------------------------
# 输出期整表展开
# ---------------------------------------------------------------------------

# 集合名 -> TableStore 缓存：引擎懒创建，但每个集合只建一份，避免逐请求重复建
_table_stores: dict[str, TableStore] = {}


def _table_store_for(collection: str) -> TableStore | None:
    """按集合名获取整表存储；不可用时返回 None（调用方退化为紧凑摘要）。"""
    key = collection or KB_COLLECTION_NAME
    store = _table_stores.get(key)
    if store is not None:
        return store
    try:
        store = TableStore(collection=key)
    except Exception as exc:  # noqa: BLE001 - 无 PG 时退化为不展开
        logger.warning("整表存储不可用（collection=%s）：%s", key, exc)
        return None
    _table_stores[key] = store
    return store


def _truncate_table(text: str) -> str:
    """按行截断整表并附带提示语（不切断 Markdown 行）。"""
    budget = TABLE_MAX_CHARS - len(TABLE_TRUNCATION_NOTE)
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept) + TABLE_TRUNCATION_NOTE


def expand_table_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """输出期整表展开：把表格索引块（表头+摘要）替换为完整表格。

    与检索链路的分工：召回与 BGE 重排全程基于紧凑的「表头+摘要」块——
    信噪比高、重排成本低；只有最终拼入 LLM 上下文前的这一步，才按
    ``parent_id``（= table_id）从整表存储取回完整表格，大表不受普通
    文本块的 ``TEXT_CONTEXT_MAX_CHARS`` 预算约束（单独放宽到
    ``TABLE_MAX_CHARS``）。

    多知识库场景按 doc 自带的 ``collection`` 解析各自的整表存储；
    展开失败（存储不可用 / PG 无该行）一律保留紧凑摘要，链路不断。
    """
    table_docs = [
        d for d in docs
        if str(d.get("block_type") or "") == "table" and d.get("parent_id")
    ]
    if not table_docs:
        return docs

    # 按 collection 分组批量取整表（每组一次主键查询）
    rows_by_collection: dict[str, dict[str, dict[str, Any]]] = {}
    for doc in table_docs:
        collection = str(doc.get("collection") or "")
        if collection in rows_by_collection:
            continue
        rows_by_collection[collection] = {}
        store = _table_store_for(collection)
        if store is None:
            continue
        ids = [
            str(d.get("parent_id"))
            for d in table_docs
            if str(d.get("collection") or "") == collection
        ]
        try:
            rows_by_collection[collection] = {
                str(row.get("id")): row for row in store.get_batch(ids)
            }
        except Exception as exc:  # noqa: BLE001 - 展开失败退化为紧凑摘要
            logger.warning("整表展开失败（collection=%s），保留索引摘要：%s", collection, exc)

    for doc in table_docs:
        collection = str(doc.get("collection") or "")
        row = rows_by_collection.get(collection, {}).get(str(doc.get("parent_id")))
        doc.setdefault("truncated", False)
        if not row:
            continue
        text = str(row.get("markdown") or "").strip()
        if not text:
            continue
        truncated = len(text) > TABLE_MAX_CHARS
        if truncated:
            text = _truncate_table(text)
        doc["content"] = text
        doc["truncated"] = truncated
        doc["table_page"] = int(row.get("start_page") or 0)
        doc["table_row_count"] = int(row.get("row_count") or 0)
        # 页码兜底：切块期页码归属失败（0）时用整表存储里的起始页补位，
        # 前端「跳原文页」才不会因为 0 而失效
        if not int(doc.get("start_page") or 0) and doc["table_page"]:
            doc["start_page"] = doc["table_page"]
    return docs


# ---------------------------------------------------------------------------
# 上下文组装
# ---------------------------------------------------------------------------

def _context_budget(block_type: str) -> int:
    """单块进入 LLM 上下文的字符上限。

    表格放宽到 8000（数据密度最高，截到 2000 会把表体和结论一起砍掉）；
    其余块维持 2000，避免个别长块挤占其它证据。
    """
    return TABLE_CONTEXT_MAX_CHARS if block_type == "table" else TEXT_CONTEXT_MAX_CHARS


def _page_label(doc: dict[str, Any]) -> str:
    """页码标签；页码缺失（0）时返回空串而不是编一个页码。"""
    start = int(doc.get("start_page") or 0)
    end = int(doc.get("end_page") or 0)
    if start <= 0:
        return ""
    if end <= 0 or end == start:
        return f"页码：{start}"
    return f"页码：{start}-{end}"


def _block_type_label(block_type: str) -> str:
    return {"table": "表格", "image": "图片"}.get(block_type, "")


def build_context(
    docs: list[dict[str, Any]], *, context_docs: list[dict[str, Any]] | None = None,
    budget_report: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """将检索结果组装为 LLM 上下文文本，并返回引用来源列表。

    块头带**报告日期 / 页码 / 块类型**：分析师要判断「这个数字出自哪一天的报告、
    在哪一页」，缺了这些信息结论无法被验证。缺失时留空，不猜。

    总预算（RAG-15）：单块字符上限之外还要受**模型窗口级预算**约束，
    否则多块证据叠加会超窗。裁剪顺序即相关度顺序；被丢弃的块不进入上下文，
    也不进入来源编号与事实核验输入，因此 `[n]` 始终连续、无悬空或错位。
    表格按行边界裁剪，不会从金额或标签中间切断。

    Returns:
        (context_text, sources) — context_text 含编号的文档片段，
        sources 为精简的来源元数据列表。
    """
    if budget_report is not None:
        budget_report.clear()
    if not docs:
        if budget_report is not None:
            budget_report.update({"input_blocks": 0, "kept": 0, "dropped": 0,
                                  "truncated_blocks": 0, "tokens": 0})
        return "（未检索到相关文档）", []

    # 单块上限换算成 token，再叠加总预算
    from src.core.config import CHAT_MAX_CONTEXT_TOKENS
    from src.core.token_budget import fit_evidence
    from src.chunking.chunker import normalize_html_tables

    fitted, report = fit_evidence(
        [
            {**doc, "content": normalize_html_tables(doc.get("content") or "")[:_context_budget(
                str(doc.get("block_type") or "text"))]}
            for doc in docs
        ],
        CHAT_MAX_CONTEXT_TOKENS,
    )
    if budget_report is not None:
        budget_report.update(report)
    if report["dropped"] or report["truncated_blocks"]:
        logger.info(
            "上下文预算裁剪：纳入 %d/%d 块、截断 %d 块（约 %d token）",
            report["kept"], report["input_blocks"], report["truncated_blocks"], report["tokens"],
        )
    docs = fitted
    if not docs:
        return "（未检索到相关文档）", []

    chunks: list[str] = []
    sources: list[dict[str, Any]] = []

    from src.answer.citations import CitationLedger

    for i, doc in CitationLedger(docs).entries:
        block_type = str(doc.get("block_type") or "text")
        content = doc.get("content") or ""
        if doc.get("truncated") and block_type == "table":
            content += TABLE_TRUNCATION_NOTE
        if context_docs is not None:
            context_docs.append({**doc, "content": content})
        title = doc.get("title", "金融文档")

        labels = [
            label
            for label in (
                f"报告日期：{doc['date']}" if doc.get("date") else "",
                _page_label(doc),
                f"类型：{_block_type_label(block_type)}" if _block_type_label(block_type) else "",
            )
            if label
        ]
        header = f"[{i}] 来源：{title}"
        if labels:
            header += f"（{'，'.join(labels)}）"
        chunks.append(f"{header}\n{content}")

        sources.append({
            "index": i,
            "title": title,
            "source": doc.get("source", ""),
            "category": "",
            "collection": doc.get("collection", ""),
            "chunk": doc.get("chunk"),
            "score": round(doc.get("score", 0.0), 4),
            "rerank_score": (
                round(doc["rerank_score"], 4)
                if isinstance(doc.get("rerank_score"), (int, float))
                else None
            ),
            "preview": (doc.get("content") or "")[:200],
            # 证据定位：前端据此做页码跳转与元数据 chip
            "date": doc.get("date", ""),
            "block_type": block_type,
            "start_page": int(doc.get("start_page") or 0),
            "end_page": int(doc.get("end_page") or 0),
            "image_key": doc.get("image_key", ""),
            "truncated": bool(doc.get("truncated")),
            "report_type": doc.get("report_type", ""),
            **{key: deepcopy(doc[key]) for key in (
                "id", "chunk_id", "chunk_key", "child_id", "parent_id", "parent_key", "table_id",
                "covered_child_ids", "document_sha256", "document_version", "broker", "heading_path",
            ) if key in doc},
        })

    context_text = "\n\n---\n\n".join(chunks)
    return context_text, sources


def _is_timeout_error(exc: Exception) -> bool:
    """判断异常是否为 LLM 请求超时/连接错误。"""
    from src.core.exceptions import classify_llm_error

    return classify_llm_error(exc) in ("timeout", "connection")


# ---------------------------------------------------------------------------
# 查询改写 + 多路子查询（标准链路可选增强，受 CHAT_ENABLE_QUERY_REWRITE 控制）
# ---------------------------------------------------------------------------

# 查询改写 Prompt：提取关键词 + 改写主查询 + 生成多路子查询
QUERY_REWRITE_PROMPT = """你是一个金融领域的查询分析和改写助手。

用户问题：{query}
历史对话：{history}

请按以下格式输出（不要加任何解释）：
1. 关键词列表：提取3-5个最核心的关键词，用逗号分隔
2. 改写查询：将问题改写为简洁的检索查询（保留所有核心关键词）
3. 子查询1：提取关键词"[时间]"相关的查询
4. 子查询2：提取关键词"[核心领域]"相关的查询
5. 子查询3：提取关键词"[动作/措施]"相关的查询

输出示例：
关键词：房地产,2023年,金融支持
改写查询：2023年金融支持房地产市场政策措施
子查询1：2023年金融政策
子查询2：房地产金融支持
子查询3：金融支持措施

请开始输出："""


def _format_history(history: list[dict[str, str]] | None) -> str:
    """格式化对话历史为文本。"""
    if not history:
        return "（无）"
    parts: list[str] = []
    for msg in history[-6:]:  # 最近 3 轮
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        label = "用户" if role == "user" else "助手"
        parts.append(f"{label}: {content}")
    return "\n".join(parts) if parts else "（无）"


def _extract_keywords_simple(query: str) -> list[str]:
    """简单关键词提取（无需 LLM，LLM 改写失败时的回退方案）。"""
    keywords: list[str] = []

    # 时间关键词
    for year in ['2021', '2022', '2023', '2024', '2025', '2026']:
        if year in query:
            keywords.append(year + '年')

    # 核心领域关键词
    domain_keywords = ['房地产', '金融', '银行', '货币政策', '财政政策',
                      '支持', '稳定', '风险', '监管', '市场']
    for kw in domain_keywords:
        if kw in query:
            keywords.append(kw)

    # 动作/措施关键词
    action_keywords = ['措施', '政策', '支持', '采取', '实施', '推动', '促进']
    for kw in action_keywords:
        if kw in query:
            keywords.append(kw)

    return keywords[:5]  # 最多5个


def _generate_sub_queries(main_query: str, keywords: list[str]) -> list[str]:
    """根据主查询和关键词生成子查询（关键词两两组合）。"""
    sub_queries: list[str] = []

    if len(keywords) >= 2:
        # 两两组合生成子查询
        for i in range(min(len(keywords), 3)):
            if i + 1 < len(keywords):
                sq = keywords[i] + keywords[i + 1]
                sub_queries.append(sq)
            elif len(keywords) >= 2:
                # 与最后一个关键词组合
                sq = keywords[i] + keywords[0]
                sub_queries.append(sq)

    if not sub_queries:
        sub_queries = [main_query]

    return sub_queries[:3]


def _parse_rewrite_output(raw: str, original_query: str) -> dict[str, Any]:
    """解析 LLM 返回的改写输出（关键词 / 改写查询 / 子查询）。"""
    rewritten = original_query
    keywords: list[str] = []
    sub_queries: list[str] = []

    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue

        # 关键词
        if line.startswith('关键词') or line.startswith('关键字'):
            kw_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            keywords = [k.strip() for k in kw_part.split(',') if k.strip()]

        # 改写查询
        elif '改写查询' in line or '改写' in line:
            rw_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            if rw_part:
                rewritten = rw_part

        # 子查询
        elif '子查询' in line:
            sq_part = line.split('：', 1)[-1].split(':', 1)[-1].strip()
            if sq_part:
                sub_queries.append(sq_part)

    # 如果解析失败，使用简单提取
    if not keywords:
        keywords = _extract_keywords_simple(original_query)

    if not sub_queries:
        # 生成简单子查询
        sub_queries = _generate_sub_queries(rewritten, keywords)

    return {
        'rewritten': rewritten,
        'keywords': keywords,
        'sub_queries': sub_queries,
    }


def rewrite_query(query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """分析查询，返回改写查询、关键词和子查询。

    优先使用轻量改写模型（``rewrite_model``，默认 DashScope qwen-turbo），
    未配置时回退主模型 ``model``，两者均不可用或调用失败时使用规则改写。

    Returns:
        dict with keys:
        - 'rewritten': 改写后的主查询
        - 'keywords': 关键词列表
        - 'sub_queries': 子查询列表
    """
    chosen = get_rewrite_model() or get_model()
    if chosen is None:
        return {
            'rewritten': query,
            'keywords': _extract_keywords_simple(query),
            'sub_queries': [query],
        }

    history_text = _format_history(history) if history else "（无）"
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", QUERY_REWRITE_PROMPT),
        ])
        chain = prompt | chosen | StrOutputParser()
        raw_output = chain.invoke({"query": query, "history": history_text})

        # 解析结构化输出
        result = _parse_rewrite_output(raw_output, query)
        logger.info(
            "查询改写：%s → 主查询=%s, 关键词=%s, 子查询=%s",
            query, result['rewritten'], result['keywords'], result['sub_queries']
        )
        return result
    except Exception as exc:
        if _is_timeout_error(exc):
            logger.warning("查询改写超时（%ss），回退到原查询", int(LLM_TIMEOUT_SECONDS))
        else:
            logger.warning("查询改写失败，使用原查询：%s", exc)
        return {
            'rewritten': query,
            'keywords': _extract_keywords_simple(query),
            'sub_queries': [query],
        }


# ---------------------------------------------------------------------------
# 元数据过滤自动推断（Self-querying Retrieval）
# ---------------------------------------------------------------------------


def _normalize_date_value(value: Any) -> str | None:
    """规范化日期过滤值：'2024' -> '2024-01-01'，非法格式返回 None。"""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-01-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    return None


def _parse_filter_candidate(raw: str) -> dict[str, Any]:
    """从 LLM 文本取原始 JSON 对象，供过滤计划逐项审计。"""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    # 兼容 LLM 用 metadata 包装输出
    if isinstance(payload.get("metadata"), dict):
        payload = payload["metadata"]
    return payload


def _parse_filters_output(raw: str) -> dict[str, Any]:
    """兼容旧调用方的宽容白名单读取接口。"""
    return normalize_metadata_filters(_parse_filter_candidate(raw))


def _collect_valid_values(value: Any, validator: Callable[[str], bool]) -> list[str]:
    """把标量/列表统一收敛成「去重后的合法值列表」。"""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    collected: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and validator(text) and text not in collected:
            collected.append(text)
    return collected


def normalize_metadata_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """兼容旧读取接口的宽容白名单归一化。

    生产 API 使用严格的 ``validate_explicit_filters``，自动候选先保留原始字段
    交由过滤计划审计。此函数仍供旧调用方宽容读取，不能用于新增显式条件校验。

    值域与跨字段业务规则与严格路径共用 ``filter_policy`` 的共享规则，
    仅失败语义不同：非法或冲突的项在此静默丢弃，而非抛错。

    所有值统一以列表形式返回（Milvus 的 ``in [...]`` 对单值字段同样成立）。
    """
    if not filters:
        return {}
    result: dict[str, Any] = {}

    # date：范围条件
    date_cond = filters.get("date")
    if isinstance(date_cond, dict):
        cond: dict[str, str] = {}
        for op in ("gte", "lte"):
            normalized = _normalize_date_value(date_cond.get(op))
            if normalized:
                cond[op] = normalized
        if cond:
            result["date"] = cond

    # 研报维度字段（校验器与入库/API 修正共用一套，避免三条路径规则漂移）
    validators: dict[str, Callable[[str], bool]] = {
        field: validator
        for field, validator in metadata_field_validators().items()
    }
    for field, validator in validators.items():
        if field in result:
            continue
        values = _collect_valid_values(filters.get(field), validator)
        if values:
            result[field] = values

    return result


def infer_metadata_filter_candidates(query: str) -> dict[str, Any]:
    """用 LLM 取得可审计原始候选；失败时返回空对象。"""
    if get_model() is None:
        return {}
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("human", METADATA_FILTER_INFER_PROMPT),
        ])
        chain = prompt | get_model() | StrOutputParser()
        raw = chain.invoke({"query": query})
        filters = _parse_filter_candidate(raw)
        if filters:
            logger.info("自动元数据过滤候选：%s -> %s", query, filters)
        return filters
    except Exception as exc:
        if _is_timeout_error(exc):
            logger.warning("元数据过滤推断超时，跳过过滤")
        else:
            logger.warning("元数据过滤推断失败，忽略：%s", exc)
        return {}


def infer_metadata_filters(query: str) -> dict[str, Any]:
    """保留旧接口的宽容白名单读取行为。"""
    return normalize_metadata_filters(infer_metadata_filter_candidates(query))


# ---------------------------------------------------------------------------
# 检索 + 重排
# ---------------------------------------------------------------------------

def reranking_enabled(requested: bool) -> bool:
    return ENABLE_RERANKER and requested


def iter_active_kbs() -> list[tuple[str, Any]]:
    """返回参与问答检索的 ``(集合名, KnowledgeBase)`` 列表。

    平台当前只有研报一个检索集合，因此恒为默认集合一项。
    """
    return [(KB_COLLECTION_NAME, get_knowledge_base())]


# ---------------------------------------------------------------------------
# 流式问答
# ---------------------------------------------------------------------------

# 秩融合常数（无统一评分时的兜底协议；与 Milvus/BM25 无关，仅用于稳定排序）
_MERGE_RRF_K = 60


def _doc_identity(doc: dict[str, Any]) -> tuple[Any, ...]:
    """块身份：有主键按（集合, 主键），否则退化到（集合, source, 内容）。"""
    chunk_id = doc.get("id") or doc.get("chunk_key")
    if chunk_id:
        return (doc.get("collection", ""), chunk_id)
    return (doc.get("collection", ""), doc.get("source", ""), doc.get("content", "") or "")


def merge_ranked_groups(
    groups: list[tuple[list[dict[str, Any]], float]],
    top_k: int,
    *,
    boost_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """按统一协议合并多组检索结果（多子查询 / HyDE / 稠密回退），去重后保留多样性配额。

    合并协议分三种，**绝不混用不同分数尺度**（RAG-10）：

    * 全部候选都带 ``merge_score``（检索层保留的归一化前原始分：重排分或同一
      融合公式的粗排分）→ 直接按原始分排序，``weight`` 作为分组阻尼（HyDE 用）；
    * 全都缺失 ``merge_score``（外部直接构造的候选）→ 沿用「分数为主键、日期为次键」
      的旧语义；
    * 只有部分候选带 ``merge_score``（例如某个集合重排失败）→ 退化为跨组秩融合
      （RRF），因为它不依赖任何分数尺度。

    展示分数在**最终选取之后**重新归一化，因此合并排序不受展示缩放影响。
    """
    from copy import deepcopy

    from src.retrieve.hybrid_retriever import _apply_diversity

    all_docs = [doc for docs, _ in groups for doc in docs]
    scored = [doc for doc in all_docs if isinstance(doc.get("merge_score"), (int, float))]
    if not all_docs:
        return []
    if len(scored) == len(all_docs):
        mode = "raw"
    elif not scored:
        mode = "legacy"
    else:
        mode = "rrf"

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    order: list[tuple[Any, ...]] = []
    for rank_group, (docs, weight) in enumerate(groups):
        group_weight = 1.0 if weight is None else float(weight)
        for rank, doc in enumerate(docs):
            key = _doc_identity(doc)
            if mode == "raw":
                rank_score = float(doc.get("merge_score") or 0.0) * group_weight
            elif mode == "rrf":
                rank_score = group_weight / (_MERGE_RRF_K + rank)
            else:
                rank_score = float(doc.get("score") or 0.0) * group_weight
            if key not in merged:
                copy = deepcopy(doc)
                copy["_merge_rank_score"] = rank_score
                copy["_merge_group_order"] = rank_group
                merged[key] = copy
                order.append(key)
            else:
                # 同一块可能被多个查询命中：取最高排名分，并保留各组的合并信息
                existing = merged[key]
                if rank_score > existing.get("_merge_rank_score", float("-inf")):
                    existing["_merge_rank_score"] = rank_score
                for field in ("rerank_score", "merge_score"):
                    if existing.get(field) is None and doc.get(field) is not None:
                        existing[field] = doc[field]

    candidates = [merged[key] for key in order]

    # 关键词加权：命中核心关键词的证据优先（乘性权重，不改变跨组可比性）
    if boost_keywords:
        for doc in candidates:
            content = doc.get("content", "")
            boost = 1.0
            for kw in boost_keywords:
                if kw and kw in content:
                    boost += 0.15
            doc["_merge_rank_score"] = doc.get("_merge_rank_score", 0.0) * boost

    candidates.sort(
        key=lambda d: (d.get("_merge_rank_score", 0.0), str(d.get("date") or "")),
        reverse=True,
    )
    # 两组及以上、且名额够每侧一条时，先留下各组当前最高的一条，再按分数填充。
    # 名额不够时仍按全局分数取舍，避免 top-1 被低分一侧占住。
    nonempty = [docs for docs, _weight in groups if docs]
    reserved: list[dict[str, Any]] = []
    if len(nonempty) >= 2:
        seen_reserved: set[tuple[Any, ...]] = set()
        for docs in nonempty:
            key = _doc_identity(docs[0])
            if key in seen_reserved or key not in merged:
                continue
            seen_reserved.add(key)
            reserved.append(merged[key])
    if reserved and top_k >= len(reserved):
        reserved.sort(
            key=lambda d: (d.get("_merge_rank_score", 0.0), str(d.get("date") or "")),
            reverse=True,
        )
        reserved_keys = {_doc_identity(doc) for doc in reserved}
        rest = [doc for doc in candidates if _doc_identity(doc) not in reserved_keys]
        selected = _apply_diversity([*reserved, *rest], top_k)
    else:
        selected = _apply_diversity(candidates, top_k)

    # 展示分数：最终选取后再归一化到 0-1（首条 1.0），不影响上面的排序
    best = max((d.get("_merge_rank_score", 0.0) for d in selected), default=0.0)
    for doc in selected:
        raw = doc.pop("_merge_rank_score", None)
        doc.pop("_merge_group_order", None)
        if raw is not None:
            doc["merge_score"] = round(float(raw), 6)
            if best > 0:
                doc["score"] = round(float(raw) / best, 4)
    return selected


def merge_docs(docs_a: list[dict[str, Any]], docs_b: list[dict[str, Any]],
                top_k: int, boost_keywords: list[str] | None = None) -> list[dict[str, Any]]:
    """合并两组检索结果（:func:`merge_ranked_groups` 的双组简写）。

    排序以分数为主键、**报告日期为次键**（同分时新报告优先）：研报的时效性
    直接影响结论是否成立，不能只靠召回顺序决定。
    """
    return merge_ranked_groups(
        [(docs_a, 1.0), (docs_b, 1.0)], top_k, boost_keywords=boost_keywords
    )


def _resolve_k_params(
    k: int | None,
    rerank_top_n: int | None,
) -> tuple[int, int]:
    """解析召回参数：用户显式传参优先，否则用统一默认值（粗排 50 / 精排 5）。

    召回深度是统一常量、与问题复杂度无关：行为可预测，也便于评测对齐。
    """
    effective_k = k if k is not None else CHAT_TOP_K
    effective_rtn = rerank_top_n if rerank_top_n is not None else CHAT_RERANK_TOP_K
    return effective_k, effective_rtn


def generate_hyde_answer(query: str) -> str:
    """让 LLM 生成假设性回答（HyDE 检索增强用），失败返回空串。"""
    if get_model() is None:
        return ""
    try:
        prompt = ChatPromptTemplate.from_messages([("human", HYDE_GENERATION_PROMPT)])
        chain = prompt | get_model() | StrOutputParser()
        return str(chain.invoke({"query": query}) or "")
    except Exception as exc:
        logger.warning("HyDE 假设回答生成失败：%s", exc)
        return ""


def hyde_retrieve(
    query: str,
    k: int,
    kbs: list[tuple[str, Any]],
    filters: dict[str, Any] | None,
    use_rerank: bool,
    rerank_top_n: int,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """HyDE 检索：用假设回答的稠密向量检索，分数按 HYDE_WEIGHT 加权。"""
    hyde_text = generate_hyde_answer(query)
    if not hyde_text or not hyde_text.strip():
        return []

    merged: list[dict[str, Any]] = []
    for coll_name, kb in kbs:
        try:
            docs = kb.hybrid_search(
                hyde_text,
                k=max(k, 5),
                use_dense_only=True,
                expand_parents=True,
                use_rerank=use_rerank,
                rerank_top_n=rerank_top_n,
                filters=filters,
            )
        except Exception as exc:
            logger.warning("HyDE 检索失败（%s）：%s", coll_name, exc)
            if diagnostics is not None:
                diagnostics.append({"collection": coll_name, "query": hyde_text,
                                    "filters": filters.copy() if filters else {}, "reason": "hyde",
                                    "k": max(k, 5), "rerank_top_n": rerank_top_n,
                                    "dense_only": True, "candidate_count": 0,
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue
        for d in docs:
            d["collection"] = coll_name
            d["score"] = float(d.get("score", 0.0) or 0.0) * HYDE_WEIGHT
        merged.extend(docs)
        if diagnostics is not None:
            from copy import deepcopy
            diagnostics.append({"collection": coll_name, "query": hyde_text,
                                **deepcopy(getattr(kb, "last_retrieval_diagnostics", None) or {}),
                                "filters": deepcopy(filters) if filters else {}, "reason": "hyde",
                                "k": max(k, 5), "rerank_top_n": rerank_top_n,
                                "dense_only": True, "candidate_count": len(docs), "error": None})
    return merged


async def _retrieve_batch(
    query: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rerank: bool = True,
    k: int | None = None,
    rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None,
    infer_filters: bool = True,
    stage_cb: Callable[[str], Any] | None = None,
    keywords: list[str] | None = None,
    queries_override: list[str] | None = None,
    filter_plan_override: Any = None,
    kbs_override: list[tuple[str, Any]] | None = None,
    rewrite_enabled: bool | None = None,
    dense_only: bool = False,
) -> dict[str, Any]:
    """完整检索管线（无流式副作用）。

    流程：查询改写 + 多路子查询（可选）→ 元数据过滤推断（可选）→
    跨库混合检索 → 去重/关键词加权合并 → 可选 HyDE 增强 → 空结果稠密回退。

    查询改写由 ``CHAT_ENABLE_QUERY_REWRITE`` 开关控制（默认关闭，单查询检索）；
    改写逻辑与 LangGraph 的 rewrite 节点共用 :func:`rewrite_query`。

    供 :func:`chat_stream` 与 A/B 评估器复用。``stage_cb`` 为可选阶段
    回调（同步），阶段按执行顺序依次上报。

    Returns:
        ``{"docs", "rewritten", "filters", "k", "rerank_top_n"}``
    """
    def _stage(name: str) -> None:
        if stage_cb is not None:
            stage_cb(name)

    filters = validate_explicit_filters(filters)
    # 解析召回参数：用户显式传参优先，否则统一默认值（粗排 50 / 精排 5）
    k, rerank_top_n = _resolve_k_params(k, rerank_top_n)

    # Step 1: 查询改写 + 多路子查询（可选）
    # 开启时改写主查询并拆分为多个子查询，多路检索后按关键词加权合并
    # 若调用方显式传入 keywords（如合规查询理解产出），则优先使用，不再走改写
    if keywords is None:
        keywords = []
    if queries_override is not None:
        all_queries = list(dict.fromkeys(queries_override))
        rewritten = all_queries[0] if all_queries else query
    elif (CHAT_ENABLE_QUERY_REWRITE if rewrite_enabled is None else rewrite_enabled) and not keywords:
        _stage("rewriting")
        try:
            query_info = rewrite_query(query, history)
        except Exception as exc:
            logger.warning("查询改写失败，回退原问题：%s", exc)
            query_info = {"rewritten": query, "keywords": [], "sub_queries": [query]}
        rewritten = query_info['rewritten']
        keywords = query_info['keywords']
        all_queries = [rewritten] + query_info['sub_queries']
        all_queries = list(dict.fromkeys(all_queries))  # 去重保序
    else:
        rewritten = query
        all_queries = [query]

    # 显式参数在任何模型或检索调用前严格校验。{} 表示调用方明确清空。
    candidate_filters: dict[str, Any] | None = None
    if infer_filters and filters is None:
        _stage("inferring_filters")
        candidate_filters = infer_metadata_filter_candidates(query)
    filter_plan = filter_plan_override or build_filter_plan(query, explicit_filters=filters, inferred_filters=candidate_filters)
    filters = filter_plan.hard_filters

    # Step 3: 混合检索 + 可选 BGE 重排序
    effective_use_rerank = reranking_enabled(use_rerank)
    _stage("retrieving")
    if effective_use_rerank:
        _stage("reranking")

    # 单物理集合检索
    kbs = iter_active_kbs() if kbs_override is None else kbs_override
    infra_failed_kbs: set[str] = set()
    retrieval_diagnostics: list[dict[str, Any]] = []
    filter_attempts: list[dict[str, Any]] = []
    all_kb_names = {name for name, _ in kbs}

    def _search_one_kb(
        search_query: str,
        coll_name: str,
        kb: Any,
        *,
        effective_filters: dict[str, Any],
        reason: str,
        dense_only: bool = False,
    ) -> list[dict[str, Any]]:
        """对单个知识库执行检索，结果打上 collection 标记。"""
        from src.core.exceptions import (
            EmbeddingError,
            VectorStoreError,
        )

        try:
            docs = kb.hybrid_search(
                search_query,
                k=k,
                use_dense_only=dense_only,
                expand_parents=True,
                use_rerank=effective_use_rerank,
                rerank_top_n=rerank_top_n,
                filters=effective_filters,
                keywords=keywords,
            )
        except (VectorStoreError, EmbeddingError) as exc:
            infra_failed_kbs.add(coll_name)
            logger.error("检索基础设施异常（%s）: %s", coll_name, exc)
            retrieval_diagnostics.append({"collection": coll_name, "query": search_query,
                                          "filters": effective_filters.copy(), "reason": reason,
                                          "k": k, "rerank_top_n": rerank_top_n,
                                          "dense_only": dense_only, "candidate_count": 0,
                                          "error": f"{type(exc).__name__}: {exc}"})
            return []
        except Exception as exc:
            infra_failed_kbs.add(coll_name)
            logger.warning("检索失败（%s）: %s", coll_name, exc)
            retrieval_diagnostics.append({"collection": coll_name, "query": search_query,
                                          "filters": effective_filters.copy(), "reason": reason,
                                          "k": k, "rerank_top_n": rerank_top_n,
                                          "dense_only": dense_only, "candidate_count": 0,
                                          "error": f"{type(exc).__name__}: {exc}"})
            return []
        for d in docs:
            d["collection"] = coll_name
        from copy import deepcopy
        retrieval_diagnostics.append({
            "collection": coll_name, "query": search_query,
            **deepcopy(getattr(kb, "last_retrieval_diagnostics", None) or {}),
            "filters": deepcopy(effective_filters), "reason": reason,
            "k": k, "rerank_top_n": rerank_top_n, "dense_only": dense_only,
            "candidate_count": len(docs), "error": None,
        })
        return docs

    def _do_search(search_query: str, effective_filters: dict[str, Any], reason: str, dense_only: bool = False) -> list[dict[str, Any]]:
        """跨所有知识库执行单次检索并合并。"""
        merged: list[dict[str, Any]] = []
        for coll_name, kb in kbs:
            merged.extend(_search_one_kb(search_query, coll_name, kb,
                                         effective_filters=effective_filters, reason=reason,
                                         dense_only=dense_only))
        return merged

    # 多查询跨库检索（改写开启时为「改写查询 + 子查询」，关闭时为单查询）
    # to_thread：Milvus 检索 + BGE 重排序均为同步阻塞调用，放线程池避免阻塞事件循环
    #
    # 每组 = 一次（过滤尝试 × 子查询）检索：组边界必须保留，否则跨查询合并会拿
    # 各自归一化后的 score 比较（每条查询的首条都是 1.0），弱证据会挤掉强证据。
    result_groups: list[tuple[list[dict[str, Any]], float]] = []
    for attempt in filter_plan.attempts():
        before = len(retrieval_diagnostics)
        found: list[dict[str, Any]] = []
        for search_query in all_queries:
            group = await asyncio.to_thread(_do_search, search_query, attempt["filters"], attempt["reason"], dense_only)
            result_groups.append((group, 1.0))
            found.extend(group)
        calls = retrieval_diagnostics[before:]
        filter_attempts.append({"reason": attempt["reason"], "filters": attempt["filters"],
                                "dense_only": False, "k": k, "rerank_top_n": rerank_top_n,
                                "candidate_count": len(found),
                                "error": "; ".join(str(call["error"]) for call in calls if call.get("error")) or None})
        if infra_failed_kbs:
            break

    # k 只表达「召回深度」（现为 50），最终进入 LLM 上下文的证据数由精排条数决定。
    # 合并上限若继续用 k，多知识库/多子查询（以及关闭重排）时会把 50 块证据塞进
    # 上下文；单查询单库下 hybrid_search 已按 rerank_top_n 返回，因此该上限对
    # 常态无影响，只兜住放大路径。
    final_k = rerank_top_n

    # 合并去重结果（统一评分协议 + 关键词加权）
    docs = merge_ranked_groups(result_groups, final_k, boost_keywords=keywords)

    # HyDE 检索增强（可选）：假设回答向量检索结果按 HYDE_WEIGHT 阻尼后合并
    if ENABLE_HYDE and get_model() is not None and docs and not filter_plan.tentative_filters:
        _stage("hyde")
        before = len(retrieval_diagnostics)
        hyde_docs = await asyncio.to_thread(
            hyde_retrieve,
            query,
            k,
            kbs,
            filters,
            effective_use_rerank,
            rerank_top_n,
            retrieval_diagnostics,
        )
        calls = retrieval_diagnostics[before:]
        if calls:
            filter_attempts.append({"reason": "hyde", "filters": filters.copy(),
                                    "dense_only": True, "k": max(k, 5),
                                    "rerank_top_n": rerank_top_n,
                                    "candidate_count": len(hyde_docs),
                                    "error": "; ".join(str(call["error"]) for call in calls if call.get("error")) or None})
        if hyde_docs:
            docs = merge_ranked_groups(
                [(docs, 1.0), (hyde_docs, HYDE_WEIGHT)], final_k, boost_keywords=keywords
            )

    # 如果结果为空，回退到简单稠密检索
    if not docs and not infra_failed_kbs and not dense_only:
        logger.info("混合检索无结果，尝试简单稠密检索")
        simple_docs: list[dict[str, Any]] = []
        before = len(retrieval_diagnostics)
        simple_docs.extend(await asyncio.to_thread(_do_search, query, filters, "hard_only_dense", True))
        calls = retrieval_diagnostics[before:]
        filter_attempts.append({"reason": "hard_only_dense", "filters": filters.copy(),
                                "dense_only": True, "k": k, "rerank_top_n": rerank_top_n,
                                "candidate_count": len(simple_docs),
                                "error": "; ".join(str(call["error"]) for call in calls if call.get("error")) or None})
        if simple_docs:
            docs = merge_ranked_groups([(simple_docs, 1.0)], final_k, boost_keywords=keywords)

    # 所有知识库均因基础设施异常而失败（而非真正的空结果）时给出错误说明
    retrieval_error: str | None = None
    if not docs and infra_failed_kbs and infra_failed_kbs == all_kb_names:
        retrieval_error = (
            "检索服务暂不可用（向量数据库/嵌入服务异常），请稍后重试"
        )

    return {
        "docs": docs,
        "rewritten": rewritten,
        "filters": filters,
        "k": k,
        "rerank_top_n": rerank_top_n,
        "retrieval_error": retrieval_error,
        "diagnostics": {"retrieval_calls": retrieval_diagnostics,
                        "filter_plan": filter_plan.as_dict(), "filter_attempts": filter_attempts},
    }


async def retrieve_pipeline(
    query: str, history: list[dict[str, str]] | None = None, *,
    use_rerank: bool = True, k: int | None = None, rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None, infer_filters: bool = True,
    stage_cb: Callable[[str], Any] | None = None, keywords: list[str] | None = None,
) -> dict[str, Any]:
    """标准检索兼容入口；执行逻辑与Agent/评测共享。"""
    from src.retrieve.pipeline import (
        RetrievalPolicy,
        RetrievalRequest,
        execute_retrieval,
    )
    if stage_cb:
        stage_cb("retrieving")
    return await execute_retrieval(
        RetrievalRequest(query, history, filters, infer_filters, k, rerank_top_n, use_rerank),
        RetrievalPolicy(rewrite=CHAT_ENABLE_QUERY_REWRITE and not keywords, grade=False),
    )


async def retrieve_documents(query: str, history: list[dict[str, str]] | None = None, **kwargs: Any) -> dict[str, Any]:
    """透明的生产检索入口；与线上共享同一条管线和排序。"""
    return await retrieve_pipeline(query, history, **kwargs)


async def chat_stream(
    query: str, history: list[dict[str, str]] | None = None, *,
    use_rerank: bool = True, k: int | None = None, rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None, infer_filters: bool = True,
) -> AsyncGenerator[dict[str, Any], None]:
    """标准和Agent均在覆盖及引用核验完成后发布事实答案。"""
    from src.agent.nodes.decision import decide_intent
    from src.answer.generator import answer_events, answer_pipeline
    from src.retrieve.pipeline import (
        RetrievalRequest,
        execute_retrieval,
        resolve_policy,
    )
    filters = validate_explicit_filters(filters)
    intent = decide_intent(query, history)
    if intent != "retrieve":
        answer = ("你好，我可以检索研报并提供带来源的金融事实回答，请说明公司、指标和期间。"
                  if intent == "direct" else "请明确你指的是哪家公司或哪份报告，以及所需指标和期间。")
        result = {"answer": answer, "sources": [], "rewritten_query": query,
                  "filters": filters or {}, "citation_validation": {"status": "not_required", "score": None, "issues": []},
                  "answer_rejected": False, "low_confidence": False,
                  "answer_status": "complete" if intent == "direct" else "clarification", "missing_items": []}
        async for event in answer_events(result):
            yield event
        return
    if ENABLE_LANGGRAPH:
        from src.agent.graph import run_agent_stream
        async for event in run_agent_stream(query, history, use_rerank=use_rerank, k=k,
                                            rerank_top_n=rerank_top_n, filters=filters, infer_filters=infer_filters):
            yield event
        return
    yield {"type": "status", "stage": "retrieving"}
    try:
        pipeline = await execute_retrieval(
            RetrievalRequest(query, history, filters, infer_filters, k, rerank_top_n, use_rerank),
            resolve_policy(mode="standard", rewrite=CHAT_ENABLE_QUERY_REWRITE),
        )
        yield {"type": "status", "stage": "verifying"}
        result = await answer_pipeline(query, pipeline, model=get_answer_model(), prefer_text=True)
        async for event in answer_events(result):
            yield event
    except Exception as exc:
        logger.exception("问答执行失败")
        yield {"type": "error", "message": f"问答执行失败：{exc}"}


async def chat(
    query: str, history: list[dict[str, str]] | None = None, *,
    use_rerank: bool = True, k: int | None = None, rerank_top_n: int | None = None,
    filters: dict[str, Any] | None = None, infer_filters: bool = True,
) -> dict[str, Any]:
    """消费同一出口，不重复生成、核验或计算拒答。"""
    answer, sources, done = [], [], None
    async for event in chat_stream(query, history, use_rerank=use_rerank, k=k,
                                   rerank_top_n=rerank_top_n, filters=filters, infer_filters=infer_filters):
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == "token":
            answer.append(event.get("content", ""))
        elif event["type"] == "sources":
            sources = event.get("sources", [])
        elif event["type"] == "done":
            done = event
    if done is None:
        raise RuntimeError("回答缺少完成事件")
    return {"answer": "".join(answer), "sources": sources,
            **{key: done.get(key) for key in ("rewritten_query", "filters", "citation_validation",
                 "answer_rejected", "low_confidence", "answer_status", "missing_items", "warnings")}}
