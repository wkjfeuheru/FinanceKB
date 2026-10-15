"""从公开检索入口到实际知识库调用的过滤契约。"""

import pytest
from fastapi import FastAPI

from finance_rag.src.services import chat_service


@pytest.fixture(autouse=True)
def _offline_graph_scope(monkeypatch):
    """本文件仅验证范围传递；工具协议及核验由v2独立用例覆盖。"""
    from finance_rag.src.orchestration import graph
    from finance_rag.src.services.retrieval_service import RetrievalPolicy
    monkeypatch.setattr(graph, "resolve_policy", lambda **kw: RetrievalPolicy(
        mode="agentic", rewrite=False, reflect=False, grade=False, use_tools=False,
    ))


class RecordingKB:
    def __init__(self, answers=None, error=None):
        self.calls = []
        self.answers = answers or []
        self.error = error
        self.last_retrieval_diagnostics = {}

    def hybrid_search(self, query, **kwargs):
        self.calls.append((query, kwargs.copy()))
        if self.error:
            raise self.error
        index = len(self.calls) - 1
        return self.answers[index] if index < len(self.answers) else []


@pytest.mark.asyncio
async def test_empty_explicit_filters_and_disabled_inference_skip_model(monkeypatch):
    kb = RecordingKB()
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: pytest.fail("inference called"))
    for explicit, enabled in [({}, True), (None, False)]:
        kb.calls.clear()
        result = await chat_service.retrieve_documents("普通问题", filters=explicit, infer_filters=enabled)
        assert result["filters"] == {}
        assert len(kb.calls) == 2  # hybrid 空召回后 dense，同一硬范围


@pytest.mark.asyncio
async def test_tentative_priority_cannot_hide_hard_only_hits(monkeypatch):
    kb = RecordingKB(answers=[
        [{"id": "priority", "content": "优先"}],
        [{"id": "hard-only", "content": "必需"}],
    ])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: {"security_name": "贵州茅台"})
    result = await chat_service.retrieve_documents("比较贵州茅台与五粮液", k=10, rerank_top_n=5)
    assert {doc["id"] for doc in result["docs"]} == {"priority", "hard-only"}
    assert result["filters"] == {}
    assert [call[1]["filters"] for call in kb.calls] == [{"security_name": ["贵州茅台"]}, {}]
    assert [a["candidate_count"] for a in result["diagnostics"]["filter_attempts"]] == [1, 1]
    assert [a["k"] for a in result["diagnostics"]["filter_attempts"]] == [10, 10]
    assert [a["rerank_top_n"] for a in result["diagnostics"]["retrieval_calls"]] == [5, 5]


@pytest.mark.asyncio
async def test_two_publication_days_keep_exact_scope_in_query_and_diagnostics(monkeypatch):
    kb = RecordingKB(answers=[
        [{"id": "first", "content": "第一篇", "score": 0.9}],
        [{"id": "second", "content": "第二篇", "score": 0.8}],
    ])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: {
        "date": {"gte": "2026-09-27", "lte": "2026-09-27"}, "broker": "东吴证券",
    })
    result = await chat_service.retrieve_documents(
        "比较东吴证券2026年9月27日点评与太平洋证券2026年9月24日点评"
    )
    hard = {"date": ["2026-09-24", "2026-09-27"], "broker": ["东吴证券", "太平洋证券"]}
    assert result["filters"] == hard
    assert kb.calls[-1][1]["filters"] == hard
    assert result["diagnostics"]["filter_plan"]["hard"] == hard
    assert result["diagnostics"]["filter_attempts"][-1]["filters"] == hard
    assert result["diagnostics"]["retrieval_calls"][-1]["filters"] == hard
    assert {doc["id"] for doc in result["docs"]} == {"first", "second"}


@pytest.mark.asyncio
async def test_infrastructure_failure_never_starts_dense_fallback(monkeypatch):
    from finance_rag.src.core.exceptions import VectorStoreError

    kb = RecordingKB(error=VectorStoreError("down"))
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    result = await chat_service.retrieve_documents("问题", filters={"report_type": "个股"}, infer_filters=False)
    assert len(kb.calls) == 1
    assert kb.calls[0][1]["filters"] == {"report_type": ["个股"]}
    assert result["retrieval_error"]
    assert result["diagnostics"]["filter_attempts"][0]["error"]


@pytest.mark.asyncio
async def test_tentative_branch_failure_does_not_expand_to_hard_only(monkeypatch):
    from finance_rag.src.core.exceptions import VectorStoreError

    kb = RecordingKB(error=VectorStoreError("down"))
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: {"security_name": "贵州茅台"})
    result = await chat_service.retrieve_documents("比较贵州茅台与五粮液")
    assert len(kb.calls) == 1
    assert kb.calls[0][1]["filters"] == {"security_name": ["贵州茅台"]}
    assert result["retrieval_error"]


@pytest.mark.asyncio
async def test_hyde_attempt_uses_hard_filters_and_has_diagnostics(monkeypatch):
    kb = RecordingKB(answers=[
        [{"id": "hybrid", "content": "原始证据", "score": 0.9}],
        [{"id": "hyde", "content": "扩展证据", "score": 0.8}],
    ])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", True)
    monkeypatch.setattr(chat_service, "get_model", lambda: object())
    monkeypatch.setattr(chat_service, "generate_hyde_answer", lambda _q: "假设答案")
    result = await chat_service.retrieve_documents("问题", filters={"report_type": "个股"}, infer_filters=False)
    assert [call[1]["filters"] for call in kb.calls] == [{"report_type": ["个股"]}] * 2
    assert [attempt["reason"] for attempt in result["diagnostics"]["filter_attempts"]] == ["hard_only", "hyde"]
    assert result["diagnostics"]["retrieval_calls"][1]["reason"] == "hyde"


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
@pytest.mark.filterwarnings("ignore:Using .* with .* is deprecated:Warning")
def test_api_rejects_invalid_explicit_filters_before_external_call(monkeypatch, path):
    from fastapi.testclient import TestClient
    from finance_rag.src.api.dependencies import get_current_user
    from finance_rag.src.api.routes import chat as route

    app = FastAPI()
    app.include_router(route.router, prefix="/chat")
    app.dependency_overrides[get_current_user] = lambda: "test-user"
    monkeypatch.setattr(route, "_chat_fns", lambda: pytest.fail("external chat called"))
    response = TestClient(app).post(path, json={"query": "问题", "filters": {"date": {"gte": "2026-02-31"}}})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_direct_chat_stream_forwards_disabled_inference_and_empty_explicit(monkeypatch):
    from finance_rag.src.services import retrieval_service
    seen = []

    async def retrieve(request, policy):
        seen.append({"filters": request.filters, "infer_filters": request.infer_filters})
        return {"docs": [], "rewritten": request.query, "filters": {}, "retrieval_error": None}

    monkeypatch.setattr(chat_service, "get_model", lambda: object())
    monkeypatch.setattr(chat_service, "ENABLE_LANGGRAPH", False)
    monkeypatch.setattr(retrieval_service, "execute_retrieval", retrieve)
    async for _ in chat_service.chat_stream("问题", filters={}, infer_filters=False):
        pass
    assert seen[0]["filters"] == {}
    assert seen[0]["infer_filters"] is False


def test_langgraph_retrieval_preserves_request_scope(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB()
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    state = {"query": "问题", "rewritten_query": "问题", "sub_queries": [], "keywords": [],
             "filters": {"report_type": ["个股"]}, "infer_filters": False,
             "k": 17, "rerank_top_n": 3, "use_rerank": False}
    graph._retrieve_node(state)
    assert kb.calls[0][1]["filters"] == {"report_type": ["个股"]}
    assert kb.calls[0][1]["k"] == 17
    assert kb.calls[0][1]["rerank_top_n"] == 3
    assert kb.calls[0][1]["use_rerank"] is False


def test_langgraph_tentative_branch_also_searches_hard_only(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(answers=[
        [{"id": "priority", "source": "a.pdf", "content": "优先", "score": 0.9}],
        [{"id": "hard-only", "source": "b.pdf", "content": "必要", "score": 0.8}],
    ])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: {"security_name": "贵州茅台"})
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    state = {"query": "比较贵州茅台与五粮液", "rewritten_query": "比较贵州茅台与五粮液",
             "sub_queries": [], "keywords": [], "request_filters": None, "infer_filters": True}
    result = graph._retrieve_node(state)
    assert {doc["id"] for doc in result["docs"]} == {"priority", "hard-only"}
    assert [call[1]["filters"] for call in kb.calls] == [{"security_name": ["贵州茅台"]}, {}]
    assert result["filters"] == {}


def test_langgraph_reflection_reuses_confirmed_filter_plan(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB()
    inferred_calls = []

    def infer(_query):
        inferred_calls.append(1)
        return {"security_name": "贵州茅台"} if len(inferred_calls) == 1 else {}

    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", infer)
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    state = {"query": "贵州茅台的评级", "rewritten_query": "贵州茅台的评级",
             "sub_queries": [], "keywords": [], "request_filters": None, "infer_filters": True}
    first = graph._retrieve_node(state)
    graph._retrieve_node(first)
    assert len(inferred_calls) == 1
    assert [call[1]["filters"] for call in kb.calls] == [{"security_name": ["贵州茅台"]}] * 4
    assert [call[1]["use_dense_only"] for call in kb.calls] == [False, True, False, True]


@pytest.mark.asyncio
async def test_compiled_graph_stream_preserves_explicit_scope_and_request_options(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(answers=[[{"id": "one", "source": "a.pdf", "content": "证据", "score": 0.9}]])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "rewrite_query", lambda query, _history: {
        "rewritten": query, "keywords": [], "sub_queries": [query],
    })
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: pytest.fail("inference called"))
    monkeypatch.setattr(graph, "get_model", lambda: None)
    monkeypatch.setattr(graph, "get_langgraph_app", graph.build_langgraph_chain)
    events = [event async for event in graph.run_agent_stream(
        "问题", filters={"report_type": "个股"}, infer_filters=False,
        k=17, rerank_top_n=3, use_rerank=False,
    )]
    assert [event for event in events if event["type"] == "done"][0]["filters"] == {"report_type": ["个股"]}
    assert kb.calls[0][1]["filters"] == {"report_type": ["个股"]}
    assert kb.calls[0][1]["k"] == 17
    assert kb.calls[0][1]["rerank_top_n"] == 3
    assert kb.calls[0][1]["use_rerank"] is False


@pytest.mark.asyncio
async def test_compiled_graph_empty_hybrid_uses_same_hard_scope_for_dense(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(answers=[[], [{"id": "dense", "source": "a.pdf", "content": "证据", "score": 0.9}]])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "rewrite_query", lambda query, _history: {
        "rewritten": query, "keywords": [], "sub_queries": [query],
    })
    monkeypatch.setattr(graph, "get_model", lambda: None)
    monkeypatch.setattr(graph, "get_langgraph_app", graph.build_langgraph_chain)
    events = [event async for event in graph.run_agent_stream(
        "问题", filters={"report_type": "个股"}, infer_filters=False,
    )]
    assert [call[1]["use_dense_only"] for call in kb.calls] == [False, True]
    assert all(call[1]["filters"] == {"report_type": ["个股"]} for call in kb.calls)
    assert [event for event in events if event["type"] == "done"][0]["filters"] == {"report_type": ["个股"]}


@pytest.mark.asyncio
async def test_compiled_graph_reflection_keeps_first_confirmed_plan(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB()
    inference_count = []
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "rewrite_query", lambda query, _history: {
        "rewritten": query, "keywords": [], "sub_queries": [query],
    })

    def infer(_query):
        inference_count.append(1)
        return {"security_name": "贵州茅台"} if len(inference_count) == 1 else {}

    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", infer)
    monkeypatch.setattr(graph, "get_model", lambda: None)
    monkeypatch.setattr(graph, "get_langgraph_app", graph.build_langgraph_chain)
    events = [event async for event in graph.run_agent_stream("贵州茅台评级", infer_filters=True)]
    assert len(inference_count) == 1
    assert all(call[1]["filters"] == {"security_name": ["贵州茅台"]} for call in kb.calls)
    assert [event for event in events if event["type"] == "done"][0]["filters"] == {"security_name": ["贵州茅台"]}


@pytest.mark.asyncio
async def test_compiled_graph_reports_exact_publication_days_used_by_search(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(answers=[[{"id": "one", "source": "a.pdf", "content": "证据", "score": 0.9}]])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "rewrite_query", lambda query, _history: {
        "rewritten": query, "keywords": [], "sub_queries": [query],
    })
    monkeypatch.setattr(chat_service, "infer_metadata_filter_candidates", lambda _q: {})
    monkeypatch.setattr(graph, "get_model", lambda: None)
    monkeypatch.setattr(graph, "get_langgraph_app", graph.build_langgraph_chain)
    events = [event async for event in graph.run_agent_stream(
        "比较东吴证券2026年9月27日点评与太平洋证券2026年9月24日点评"
    )]
    hard = {"date": ["2026-09-24", "2026-09-27"], "broker": ["东吴证券", "太平洋证券"]}
    assert kb.calls[0][1]["filters"] == hard
    assert [event for event in events if event["type"] == "done"][0]["filters"] == hard


@pytest.mark.asyncio
async def test_compiled_graph_infrastructure_failure_skips_dense(monkeypatch):
    from finance_rag.src.core.exceptions import VectorStoreError
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(error=VectorStoreError("down"))
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "rewrite_query", lambda query, _history: {
        "rewritten": query, "keywords": [], "sub_queries": [query],
    })
    monkeypatch.setattr(graph, "get_model", lambda: None)
    monkeypatch.setattr(graph, "get_langgraph_app", graph.build_langgraph_chain)
    events = [event async for event in graph.run_agent_stream(
        "问题", filters={"report_type": "个股"}, infer_filters=False,
    )]
    assert len(kb.calls) == 1
    assert kb.calls[0][1]["use_dense_only"] is False
    assert kb.calls[0][1]["filters"] == {"report_type": ["个股"]}
    assert [event for event in events if event["type"] == "error"]


def test_langgraph_dense_fallback_does_not_trigger_hyde(monkeypatch):
    from finance_rag.src.orchestration import graph

    kb = RecordingKB(answers=[[], [{"id": "dense", "source": "a.pdf", "content": "证据", "score": 0.9}]])
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("kb", kb)])
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", True)
    monkeypatch.setattr(graph, "get_model", lambda: object())
    monkeypatch.setattr(chat_service, "hyde_retrieve", lambda *_args: pytest.fail("HyDE should not run"))
    result = graph._retrieve_node({
        "query": "问题", "rewritten_query": "问题", "sub_queries": [], "keywords": [],
        "request_filters": {"report_type": ["个股"]}, "infer_filters": False,
    })
    assert [call[1]["use_dense_only"] for call in kb.calls] == [False, True]
    assert result["docs"][0]["id"] == "dense"
