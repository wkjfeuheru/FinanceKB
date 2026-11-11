"""粗排深度（k）与精排条数（rerank_top_n）的解耦。

背景：``k`` 曾同时充当 ``merge_docs`` 的全局上限。把粗排统一为 50 之后，若
合并上限仍用 ``k``，多知识库 / 多子查询（改写）/ 关闭重排时会一次性把 50 块
证据塞进 LLM 上下文（50 × 2000 字符）。这里锁住新契约：

* ``k`` 只表达**召回深度**（50）；
* ``rerank_top_n`` 决定最终进入上下文的**证据条数**（5），且不受重排开关影响。

假知识库按「每路返回大量候选」构造，从而逼出合并上限这条放大路径。
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.config import CHAT_RERANK_TOP_K, CHAT_TOP_K
from src import engine as chat_service


class _FakeKB:
    """记录 ``hybrid_search`` 入参，并返回指定数量的候选。"""

    def __init__(self, hits: int):
        self._hits = hits
        self.calls: list[dict] = []

    def hybrid_search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return [
            {
                "id": f"c{i}",
                "chunk_key": f"c{i}",
                # 内容必须互不相同：merge_docs 按 content[:200] 去重
                "content": f"候选 {i} 的内容",
                "source": "s.pdf",
                "title": "研报",
                "score": 1.0 - i * 0.001,
                "block_type": "text",
                "parent_id": "",
            }
            for i in range(self._hits)
        ]


def _use_kb(monkeypatch, kb) -> None:
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("finance_kb", kb)])


def test_unified_k_defaults():
    """统一口径：粗排 50 / 精排 5（代码默认与 .env 均已同步）。"""
    assert (CHAT_TOP_K, CHAT_RERANK_TOP_K) == (50, 5)


def test_pipeline_requests_coarse_k_and_returns_only_final_k(monkeypatch):
    kb = _FakeKB(hits=200)
    _use_kb(monkeypatch, kb)

    pipeline = asyncio.run(chat_service.retrieve_pipeline("q", infer_filters=False))

    assert kb.calls[0]["k"] == CHAT_TOP_K == 50
    assert kb.calls[0]["rerank_top_n"] == CHAT_RERANK_TOP_K == 5
    assert len(pipeline["docs"]) == 5


def test_pipeline_caps_evidence_when_rerank_disabled(monkeypatch):
    """关闭重排时 k 条候选会被完整返回，合并上限必须把它压到精排条数。"""
    kb = _FakeKB(hits=200)
    _use_kb(monkeypatch, kb)

    pipeline = asyncio.run(
        chat_service.retrieve_pipeline("q", use_rerank=False, infer_filters=False)
    )

    assert kb.calls[0]["k"] == 50
    assert len(pipeline["docs"]) == 5


def test_pipeline_caps_evidence_across_sub_queries(monkeypatch):
    """改写开启时多路结果合并后同样收敛到精排条数。"""
    kb = _FakeKB(hits=50)
    _use_kb(monkeypatch, kb)
    monkeypatch.setattr(chat_service, "CHAT_ENABLE_QUERY_REWRITE", True)
    from src.retrieve import pipeline as retrieval_service
    from src.agent.planning import QueryPlan, Subproblem
    monkeypatch.setattr(retrieval_service, "plan_query", lambda *a, **kw: QueryPlan(
        needs_decomposition=True, subproblems=[Subproblem(id=f"q{i}", query=f"子问题{i}") for i in range(3)],
    ))
    pipeline = asyncio.run(chat_service.retrieve_pipeline("q", infer_filters=False))

    # 必要事项最多 3 个；初检另保留原问句，证据条数仍独立受限。
    assert [call["query"] for call in kb.calls] == ["q", "子问题0", "子问题1", "子问题2"]
    assert all(call["k"] == 50 and call["rerank_top_n"] == 5 for call in kb.calls)
    assert len(pipeline["docs"]) == 5


@pytest.mark.asyncio
async def test_langgraph_retrieve_node_keeps_final_count(monkeypatch):
    """LangGraph 检索节点同口径（共享执行器路径）：召回 k=50，合并后证据 <=5。"""
    from src.agent import graph
    from src.retrieve.pipeline import RetrievalPolicy

    kb = _FakeKB(hits=50)
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("finance_kb", kb)])
    # 离线策略：不规划、不核验、不补检，直接走单批检索（同 filter_contract 契约定式）
    monkeypatch.setattr(graph, "resolve_policy", lambda **kw: RetrievalPolicy(
        mode="agentic", rewrite=False, reflect=False, grade=False, use_tools=False,
    ))

    state = {
        "query": "q",
        "rewritten_query": "q",
        "sub_queries": ["子问题一", "子问题二"],
        "keywords": [],
        "docs": [],
        "reflection_round": 0,
        "infer_filters": False,
    }

    result = await graph._retrieve_node_async(state)

    assert kb.calls[0]["k"] == 50
    assert len(result["docs"]) <= 5
