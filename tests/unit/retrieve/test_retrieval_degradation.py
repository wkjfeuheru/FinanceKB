"""检索降级语义回归（RAG-11）。

``docs/rag-review.md`` 的问题：``execute_retrieval`` 只要发现任一
``retrieval_calls`` 诊断含错误就整体失败，即使正常检索已有证据；部分子查询或
可选 HyDE 失败也会丢弃可用结果。

验收标准：
* 区分「主检索不可用」「部分失败」与「增强降级」，不把基础设施错误伪装成空结果；
* 成功证据仍交给覆盖评估决定完整回答 / 部分回答 / 拒答；
* 确有可用证据时不得因单条子查询或可选增强失败而整体报错。
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.agent.planning import QueryPlan, Subproblem
from src.retrieve.pipeline import (
    RetrievalDependencies,
    RetrievalPolicy,
    RetrievalRequest,
    execute_retrieval,
)


def _plan(query: str = "甲公司2025年营业收入") -> QueryPlan:
    return QueryPlan(needs_decomposition=False, subproblems=[Subproblem(id="q1", query=query)])


def _coverage(status: str = "supported") -> dict:
    supported = status == "supported"
    return {
        "items": [{"item_id": "q1", "status": status,
                   "evidence_ids": ["kb:c1"] if supported else [], "reason": ""}],
        "sufficient": supported,
        "supported_items": ["q1"] if supported else [],
        "missing_items": [] if supported else ["q1"],
        "unknown": False,
    }


def _deps(batch: dict, *, grader_status: str = "supported") -> RetrievalDependencies:
    async def search(request, queries, keywords, filter_plan):
        del request, keywords, filter_plan
        return deepcopy(batch)

    return RetrievalDependencies(
        search=search,
        planner=lambda *a: _plan(),
        grader=lambda *a: _coverage(grader_status),
        prepare_context=lambda docs: {
            "context_docs": deepcopy(docs), "sources": [], "context_text": "上下文",
        },
    )


def _call(collection: str, query: str, *, error: str | None = None, candidates: int = 0) -> dict:
    return {"collection": collection, "query": query, "filters": {}, "reason": "hard_only",
            "k": 50, "rerank_top_n": 5, "dense_only": False,
            "candidate_count": candidates, "error": error}


@pytest.mark.asyncio
async def test_partial_subquery_failure_keeps_available_evidence():
    """一条子查询失败但有可用证据：不得整体报错，证据继续进入覆盖评估。"""
    batch = {
        "docs": [{"id": "c1", "collection": "kb", "content": "可用证据", "score": 0.9}],
        "diagnostics": {"retrieval_calls": [
            _call("kb", "甲公司2025年营业收入", candidates=1),
            _call("kb", "甲公司2025年毛利率", error="VectorStoreError: timeout"),
        ]},
        "retrieval_error": None,
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch)
    )

    assert result["decision"] != "error"
    assert result["retrieval_error"] is None
    assert [doc["id"] for doc in result["context_docs"]] == ["c1"]
    assert result["warnings"] and "timeout" in result["warnings"][0]


@pytest.mark.asyncio
async def test_optional_hyde_failure_does_not_break_answer():
    """可选 HyDE 增强失败：正常检索有证据时只降级为告警。"""
    batch = {
        "docs": [{"id": "c1", "collection": "kb", "content": "原始检索证据", "score": 0.8}],
        "diagnostics": {"retrieval_calls": [
            _call("kb", "甲公司2025年营业收入", candidates=1),
            _call("kb", "hyde", error="HyDEGenerationError: 模型超时"),
        ]},
        "retrieval_error": None,
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch)
    )

    assert result["decision"] == "generate"
    assert result["coverage"]["sufficient"] is True
    assert any("HyDE" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_alert_surfaces_every_failed_call_not_just_the_first():
    """告警必须覆盖全部失败调用（排障时要能看到是哪几条子查询降级）。"""
    failures = [f"错误{i}" for i in range(3)]
    batch = {
        "docs": [{"id": "c1", "collection": "kb", "content": "可用证据", "score": 0.9}],
        "diagnostics": {"retrieval_calls": [
            _call("kb", f"查询{i}", error=message) for i, message in enumerate(failures)
        ]},
        "retrieval_error": None,
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch)
    )

    assert len(result["warnings"]) == 3
    for message in failures:
        assert any(message in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_primary_search_unavailable_is_still_an_error():
    """主检索不可用（全部调用失败且没有证据）必须显式报错，不能伪装成空结果。"""
    batch = {
        "docs": [],
        "diagnostics": {"retrieval_calls": [
            _call("kb", "甲公司2025年营业收入", error="VectorStoreError: down"),
        ]},
        "retrieval_error": None,
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch)
    )

    assert result["decision"] == "error"
    assert result["stop_reason"] == "retrieval_error"
    assert result["retrieval_error"]


@pytest.mark.asyncio
async def test_aggregate_retrieval_error_is_still_an_error():
    """聚合错误标记（全部知识库基础设施失败）优先于一切降级判断。"""
    batch = {
        "docs": [],
        "diagnostics": {"retrieval_calls": []},
        "retrieval_error": "检索服务暂不可用（向量数据库/嵌入服务异常），请稍后重试",
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch)
    )

    assert result["decision"] == "error"
    assert "检索服务暂不可用" in result["retrieval_error"]


@pytest.mark.asyncio
async def test_empty_result_without_errors_is_not_an_infrastructure_error():
    """真正的空结果不是错误：必须交给覆盖评估决定拒答，而不是报基础设施故障。"""
    batch = {
        "docs": [],
        "diagnostics": {"retrieval_calls": [_call("kb", "甲公司2025年营业收入", candidates=0)]},
        "retrieval_error": None,
    }
    result = await execute_retrieval(
        RetrievalRequest("甲公司2025年营业收入", filters={}), RetrievalPolicy(), _deps(batch, grader_status="missing")
    )

    assert result["decision"] != "error"
    assert result["retrieval_error"] is None
    assert result["decision"] == "refuse"
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_coverage_uses_main_model_not_rewrite_model(monkeypatch):
    """覆盖评估直接走主模型，不把整份证据打到改写模型上。"""
    from src import engine as chat_service
    from src.core import config

    class RewriteModel:
        def invoke(self, prompt):
            raise AssertionError("覆盖评估不应调用改写模型")

    class MainModel:
        def invoke(self, prompt):
            assert "rerank_score" not in prompt
            return '{"items":[{"item_id":"q1","status":"supported","evidence_ids":["c1"],"reason":"证据给出营收"}]}'

    monkeypatch.setattr(chat_service, "get_rewrite_model", lambda: RewriteModel())
    monkeypatch.setattr(chat_service, "get_model", lambda: MainModel())
    monkeypatch.setattr(config, "get_grading_model", lambda: None)

    async def search(request, queries, keywords, filter_plan):
        del request, queries, keywords, filter_plan
        return {"docs": [{"id": "c1", "collection": "kb", "content": "甲公司营业收入10亿元", "block_type": "text", "score": 0.9, "rerank_score": 0.99}],
                "diagnostics": {"retrieval_calls": []}}

    result = await execute_retrieval(
        RetrievalRequest("甲公司营业收入是多少？", infer_filters=False),
        RetrievalPolicy(),
        RetrievalDependencies(search=search, planner=lambda *a: _plan("甲公司营业收入是多少？")),
    )

    assert result["decision"] == "generate"
    assert result["coverage"]["sufficient"]
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_coverage_rate_limit_stays_unverified(monkeypatch):
    """主模型限流时只调用一次，并保持未验证，不写成证据不足。"""
    from src import engine as chat_service
    from src.core import config

    class RateLimitError(Exception):
        status_code = 429

    calls = {"n": 0}

    class MainModel:
        def invoke(self, prompt):
            calls["n"] += 1
            raise RateLimitError("rate limit")

    monkeypatch.setattr(chat_service, "get_rewrite_model", lambda: None)
    monkeypatch.setattr(chat_service, "get_model", lambda: MainModel())
    monkeypatch.setattr(config, "get_grading_model", lambda: MainModel())

    async def search(request, queries, keywords, filter_plan):
        del request, queries, keywords, filter_plan
        return {"docs": [{"id": "c1", "collection": "kb", "content": "甲公司营业收入10亿元", "block_type": "text", "score": 0.9}],
                "diagnostics": {"retrieval_calls": []}}

    result = await execute_retrieval(
        RetrievalRequest("甲公司营业收入是多少？", infer_filters=False),
        RetrievalPolicy(),
        RetrievalDependencies(search=search, planner=lambda *a: _plan("甲公司营业收入是多少？")),
    )

    assert calls["n"] == 1
    assert result["decision"] == "unverified"
    assert result["stop_reason"] == "coverage_unknown"
    assert any("没有改判成证据不足" in warning for warning in result["warnings"])
