"""Agent 检索工具边界。"""

from __future__ import annotations

from typing import Any


class RetrievalTool:
    """提供统一的检索输入和规范化证据输出。"""

    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    def invoke(
        self,
        query: str,
        *,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        use_dense_only: bool = False,
        expand_parents: bool = True,
        use_rerank: bool = False,
        rerank_top_n: int = 3,
    ) -> dict[str, Any]:
        """执行检索并返回稳定的 evidence/metadata 结构。"""
        if not query or not query.strip():
            return {"query": query, "evidence": [], "metadata": {"count": 0}}

        retriever = self._retriever or _default_retriever()
        search = retriever.search if hasattr(retriever, "search") else retriever
        matches = search(
            query.strip(),
            k=k,
            filters=filters,
            keywords=keywords,
            use_dense_only=use_dense_only,
            expand_parents=expand_parents,
            use_rerank=use_rerank,
            rerank_top_n=rerank_top_n,
        )
        evidence = [_normalize_evidence(item, index) for index, item in enumerate(matches)]
        return {
            "query": query,
            "evidence": evidence,
            "metadata": {"count": len(evidence), "filters": filters or {}},
        }

    __call__ = invoke


class AgentRetrievalTool(RetrievalTool):
    """将 RAG 检索能力暴露给 Agent，隔离底层检索实现。"""

    name = "retrieve_finance_evidence"
    description = "检索金融知识库并返回规范化证据。"


def _normalize_evidence(item: dict[str, Any], index: int) -> dict[str, Any]:
    """统一检索结果为平铺字段，与 ``KnowledgeBase.hybrid_search`` 的返回契约一致。

    平铺字段保证下游 ``build_context`` / 来源展示能直接读取 source/title/
    category/chunk 等键，避免嵌套 ``metadata`` 造成字段丢失。

    ``block_type`` / 页码 / ``image_key`` 必须透传：输出期的整表展开
    （``expand_table_docs``）与上下文预算/标签都靠它们识别表格与图片块——
    丢失时表格块会被当成普通文本（预算砍半、标签消失）。
    """
    return {
        "id": item.get("id") or item.get("chunk_id") or f"evidence-{index}",
        "chunk_id": item.get("chunk_id"),
        "child_id": item.get("child_id") or item.get("id") or item.get("chunk_id"),
        "chunk_key": item.get("chunk_key") or item.get("id") or item.get("chunk_id"),
        "parent_key": item.get("parent_key") or item.get("parent_id"),
        "covered_child_ids": list(item.get("covered_child_ids") or []),
        "content": item.get("content") or "",
        "source": item.get("source", ""),
        "title": item.get("title", ""),
        "chunk": item.get("chunk"),
        "parent_id": item.get("parent_id", ""),
        "category": item.get("category", ""),
        "collection": item.get("collection", ""),
        "security_name": item.get("security_name", ""),
        "industry_l1": item.get("industry_l1", ""),
        "industry_l2": item.get("industry_l2", ""),
        "report_type": item.get("report_type", ""),
        "broker": item.get("broker", ""),
        "table_id": item.get("table_id"),
        "document_sha256": item.get("document_sha256") or None,
        "document_version": item.get("document_version") or None,
        "date": item.get("date", ""),
        "heading": item.get("heading", ""),
        "heading_path": item.get("heading_path", ""),
        "score": float(item.get("score") or 0.0),
        "rerank_score": item.get("rerank_score"),
        # 块类型与证据定位字段（表格/图片块的展开、标签与页码跳转都依赖它们）
        "block_type": item.get("block_type", "") or "text",
        "start_page": int(item.get("start_page") or 0),
        "end_page": int(item.get("end_page") or 0),
        "image_key": item.get("image_key", "") or "",
    }


def _default_retriever() -> Any:
    """延迟创建默认知识库检索器，导入工具时不连接外部服务。"""
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    return get_knowledge_base()


def get_retrieval_tool(retriever: Any | None = None) -> RetrievalTool:
    """创建标准 Retrieval Tool，可注入测试替身或自定义检索器。"""
    return RetrievalTool(retriever)


def get_agent_tools(retriever: Any | None = None) -> list[AgentRetrievalTool]:
    """创建当前 Agent 可用工具集合。"""
    return [AgentRetrievalTool(retriever)]


__all__ = ["AgentRetrievalTool", "RetrievalTool", "get_agent_tools", "get_retrieval_tool"]
