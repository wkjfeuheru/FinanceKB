"""兼容旧评测函数名；所有检索轮次由共享执行器控制。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .retrieval import _run_async

SearchFn = Callable[..., list[dict[str, Any]]]
RewriteFn = Callable[[str], dict[str, Any]]
FilterFn = Callable[[str], dict[str, Any]]
ReflectFn = Callable[[dict[str, Any]], dict[str, Any]]


def retrieve_agentic(
    query: str, *, rewrite: bool = False, route: bool = False, reflect: bool = False,
    kb: Any | None = None, search: SearchFn | None = None,
    rewrite_query_fn: RewriteFn | None = None, infer_filters_fn: FilterFn | None = None,
    normalize_fn: Callable[[dict[str, Any] | None], dict[str, Any]] | None = None,
    reflect_fn: ReflectFn | None = None, max_reflect_rounds: int | None = None,
) -> list[dict[str, Any]]:
    """旧注入函数适配为依赖；reflection_round=99 仅解释旧测试钩子。"""
    from src.agent.planning import QueryPlan, Subproblem
    from src.agent.nodes.grading import evidence_key
    from src.retrieve import pipeline as service

    deps = service.RetrievalDependencies()
    state: dict[str, Any] = {"query": query, "history": [], "rewritten_query": query,
                             "keywords": [], "sub_queries": [], "reflection_round": 0,
                             "answer": "", "sources": [], "error": ""}
    legacy_filters: dict[str, Any] | None = None
    if rewrite_query_fn is not None or (search is not None and not rewrite):
        def planner(original, history, limit):
            if rewrite and rewrite_query_fn:
                info = rewrite_query_fn(original)
                state.update(rewritten_query=str(info.get("rewritten") or original),
                             keywords=list(info.get("keywords") or []), sub_queries=list(info.get("sub_queries") or []))
            queries = list(dict.fromkeys([state["rewritten_query"], *state["sub_queries"]]))
            if len(queries) > limit:
                return QueryPlan(needs_decomposition=True, subproblems=[],
                                 clarification=f"问题需要 {len(queries)} 个子查询，超过上限 {limit}；请明确或缩小范围。")
            return QueryPlan(needs_decomposition=len(queries) > 1,
                             subproblems=[Subproblem(id=f"q{i}", query=q) for i, q in enumerate(queries, 1)])
        deps.planner = planner
    if infer_filters_fn is not None:
        def infer(original):
            nonlocal legacy_filters
            value = infer_filters_fn(original)
            legacy_filters = normalize_fn(value) if normalize_fn else value
            return legacy_filters
        deps.infer_filters = infer
    if search is not None:
        def search_adapter(request, queries, keywords, filter_plan):
            filters = legacy_filters if normalize_fn and infer_filters_fn else filter_plan.hard_filters
            docs = search(queries, filters=filters or None, keywords=state["keywords"] or keywords)
            state["docs"] = docs
            return {"docs": docs, "diagnostics": {}}
        deps.search = search_adapter
    if reflect_fn is not None and reflect:
        # 整数哨兵是旧反思钩子的兼容语义，不进入共享生产状态。
        deps.model = object()
        def grader(plan, context_docs):
            state["docs"] = state.get("docs", context_docs)
            state.update(reflect_fn(dict(state)))
            sufficient = int(state.get("reflection_round") or 0) >= 99
            supported = [item.id for item in plan.subproblems] if sufficient else []
            return {"sufficient": sufficient, "unknown": False, "supported_items": supported,
                    "missing_items": [] if sufficient else [item.id for item in plan.subproblems],
                    "items": [{"id": item.id, "evidence_ids": [evidence_key(doc) for doc in context_docs]} for item in plan.subproblems]}
        deps.grader = grader
        deps.rewriter = lambda item, coverage, attempted: str(state.get("rewritten_query") or item.query)
    settings = dict(mode="agentic", rewrite=rewrite, route=route, reflect=reflect,
                    grade=reflect, use_tools=False)
    if max_reflect_rounds is not None:
        settings["max_retries"] = max_reflect_rounds
    policy = service.resolve_policy(**settings)
    request = service.RetrievalRequest(query, infer_filters=route)
    pipeline = _run_async(lambda: service.execute_retrieval(request, policy, deps, kb=kb))
    if pipeline.get("retrieval_error"):
        raise RuntimeError(pipeline["retrieval_error"])
    # 旧列表接口无法携带澄清状态；诊断证据保留在共享执行器的结构化结果中。
    if pipeline.get("decision") == "clarification":
        return []
    return pipeline["docs"]
