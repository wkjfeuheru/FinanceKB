"""Agentic v2：业务决策、共享检索闭环及验证后生成。"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph import END, StateGraph

from finance_rag.src.agent.state.models import AgentState
from finance_rag.src.core.config import LANGGRAPH_MAX_REFLECT_ROUNDS, get_model
from finance_rag.src.orchestration.nodes.decision import decide_intent
from finance_rag.src.rag.retrieval.filter_policy import validate_explicit_filters
from finance_rag.src.services.retrieval_service import (
    RetrievalRequest,
    execute_retrieval,
    resolve_policy,
)

MAX_REFLECTION_ROUNDS = LANGGRAPH_MAX_REFLECT_ROUNDS


def _request(state):
    return RetrievalRequest(state["query"], state.get("history"),
        state.get("request_filters", state.get("filters")), state.get("infer_filters", True),
        state.get("k"), state.get("rerank_top_n"), state.get("use_rerank", True))


def _rewrite_node(state: AgentState) -> dict:
    return {"intent": decide_intent(state["query"], state.get("history"))}


def _retrieve_node(state: AgentState) -> dict:
    """旧同步适配，共用批检索实现，不维护独立循环。"""
    from finance_rag.src.eval.retrieval import _run_async
    from finance_rag.src.rag.retrieval.filter_policy import build_filter_plan
    from finance_rag.src.services import chat_service
    explicit = validate_explicit_filters(state.get("request_filters", state.get("filters")))
    previous = state.get("filter_plan")
    inferred = None
    if explicit is None and state.get("infer_filters", True):
        inferred = previous.get("inferred") if previous else chat_service.infer_metadata_filter_candidates(state["query"])
    plan = build_filter_plan(state["query"], explicit_filters=explicit, inferred_filters=inferred)
    queries = list(dict.fromkeys([state.get("rewritten_query") or state["query"], *state.get("sub_queries", [])]))
    result = _run_async(lambda: chat_service._retrieve_batch(
        state["query"], state.get("history"), k=state.get("k"), rerank_top_n=state.get("rerank_top_n"),
        use_rerank=state.get("use_rerank", True), filters=explicit, infer_filters=False,
        keywords=state.get("keywords", []), queries_override=queries,
        filter_plan_override=plan, rewrite_enabled=False))
    return {**state, "docs": result["docs"], "filters": plan.hard_filters,
            "filter_plan": plan.as_dict(), "retrieval_error": result.get("retrieval_error")}


async def _retrieve_node_async(state: AgentState) -> dict:
    policy = state.get("policy") or resolve_policy(mode="agentic", max_retries=MAX_REFLECTION_ROUNDS)
    result = await execute_retrieval(_request(state), policy, state.get("dependencies"))
    return {"pipeline": result, "docs": result["docs"], "filters": result["filters"],
        "retry_count": result["retry_count"], "decision": result["decision"],
        "stop_reason": result["stop_reason"], "attempted_queries": result["attempted_queries"],
        "rewritten_query": result["rewritten"], "retrieval_error": result.get("retrieval_error")}


def _reflect_node(state: AgentState) -> dict:
    pipeline = state.get("pipeline") or {}
    return {"decision": pipeline.get("decision", state.get("decision", "unverified")),
            "retry_count": pipeline.get("retry_count", state.get("retry_count", 0))}


def _should_retrieve(state: AgentState) -> str:
    return "rewrite" if state.get("decision") == "rewrite" else "generate"


async def _generate_node(state: AgentState) -> dict:
    from finance_rag.src.services.answer_service import answer_pipeline
    if state.get("intent") in {"direct", "clarify"}:
        clarify = state["intent"] == "clarify"
        result = {"answer": "请明确公司、报告、指标和期间。" if clarify else "你好，我可以检索研报并提供带来源的金融事实回答。",
            "sources": [], "rewritten_query": state["query"], "filters": {},
            "citation_validation": {"status": "not_required", "score": None, "issues": []},
            "answer_rejected": False, "low_confidence": False,
            "answer_status": "clarification" if clarify else "complete", "missing_items": []}
    else:
        result = await answer_pipeline(state["query"], state["pipeline"], model=get_model())
    return {"answer_result": result, "answer": result["answer"], "sources": result["sources"]}


def build_langgraph_chain():
    graph = StateGraph(AgentState)
    graph.add_node("agent", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node_async)
    graph.add_node("reflect", _reflect_node)
    graph.add_node("generate", _generate_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", lambda state: "retrieve" if state["intent"] == "retrieve" else "generate")
    graph.add_edge("retrieve", "reflect")
    graph.add_edge("reflect", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_langgraph_app: Any = None


def get_langgraph_app():
    global _langgraph_app
    if _langgraph_app is None:
        _langgraph_app = build_langgraph_chain()
    return _langgraph_app


def initial_state(query, history=None, **kwargs):
    return {"query": query, "history": history or [], "rewritten_query": query,
        "keywords": [], "sub_queries": [], "docs": [], "reflection_round": 0,
        "retry_count": 0, "answer": "", "sources": [], "error": "", **kwargs}


async def run_agent_stream(query: str, history=None, *, use_rerank=True, k=None,
                          rerank_top_n=None, filters=None, infer_filters=True) -> AsyncGenerator[dict, None]:
    from finance_rag.src.services.answer_service import answer_events
    filters = validate_explicit_filters(filters)
    yield {"type": "status", "stage": "retrieving"}
    try:
        result = await get_langgraph_app().ainvoke(initial_state(
            query, history, use_rerank=use_rerank, k=k, rerank_top_n=rerank_top_n,
            request_filters=filters, infer_filters=infer_filters))
        yield {"type": "status", "stage": "verifying"}
        async for event in answer_events(result["answer_result"]):
            yield event
    except Exception as exc:
        yield {"type": "error", "message": f"Agent执行失败：{exc}"}


def build_graph():
    return build_langgraph_chain()


def run_agent(query: str, history=None):
    from finance_rag.src.eval.retrieval import _run_async
    return _run_async(lambda: get_langgraph_app().ainvoke(initial_state(query, history)))


def run_langgraph_retrieval(query: str, history=None) -> list[dict]:
    from finance_rag.src.eval.retrieval import _run_async
    result = _run_async(lambda: execute_retrieval(RetrievalRequest(query, history), resolve_policy(mode="agentic")))
    if result.get("retrieval_error"):
        raise RuntimeError(result["retrieval_error"])
    return result["docs"]
