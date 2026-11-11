"""Agent 状态模型。"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class Evidence(TypedDict, total=False):
    """检索、工具、上下文及评测共用的证据身份。"""
    id: str | int
    chunk_key: str
    child_id: str | int
    parent_id: str
    parent_key: str
    covered_child_ids: list[str | int]
    collection: str
    source: str
    content: str
    report_type: str
    document_sha256: str | None
    document_version: str | None


class AgentState(TypedDict):
    """LangGraph Agent 在各节点之间传递的 canonical 状态。"""

    query: str
    history: list[dict[str, str]]
    rewritten_query: str
    keywords: list[str]
    sub_queries: list[str]
    docs: list[dict[str, Any]]
    reflection_round: int
    answer: str
    sources: list[dict[str, Any]]
    error: str
    request_filters: NotRequired[dict[str, Any] | None]
    infer_filters: NotRequired[bool]
    k: NotRequired[int | None]
    rerank_top_n: NotRequired[int | None]
    use_rerank: NotRequired[bool]
    filters: NotRequired[dict[str, Any]]
    filter_plan: NotRequired[dict[str, Any]]
    retrieval_error: NotRequired[str | None]
    retry_count: NotRequired[int]
    decision: NotRequired[str]
    stop_reason: NotRequired[str]
    attempted_queries: NotRequired[list[str]]
    pipeline: NotRequired[dict[str, Any]]
    policy: NotRequired[Any]
    dependencies: NotRequired[Any]
    intent: NotRequired[str]
    answer_result: NotRequired[dict[str, Any]]


__all__ = ["AgentState", "Evidence"]
