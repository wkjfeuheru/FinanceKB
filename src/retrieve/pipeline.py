"""线上和评测共用的有界检索执行器。"""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from src.agent.planning import QueryPlan, Subproblem, plan_query, rewrite_missing
from src.agent.nodes.grading import decide_from_coverage, evidence_key, grade_coverage
from src.retrieve.filter_policy import build_filter_plan, validate_explicit_filters

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "agentic-v2"


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    history: list[dict] | None = None
    filters: dict | None = None
    infer_filters: bool = True
    k: int | None = None
    rerank_top_n: int | None = None
    use_rerank: bool = True


@dataclass(frozen=True)
class RetrievalPolicy:
    mode: Literal["standard", "agentic"] = "standard"
    rewrite: bool = False
    route: bool = True
    reflect: bool = False
    max_subproblems: int = 3
    max_retries: int = 2
    grade: bool = True
    use_tools: bool = False
    dense_only: bool = False

    def __post_init__(self):
        if not 0 <= self.max_retries <= 2:
            raise ValueError("最多允许初检后2次补检")
        if not 1 <= self.max_subproblems <= 3:
            raise ValueError("复杂问题最多3个必要事项")


@dataclass
class RetrievalDependencies:
    search: Callable | None = None
    planner: Callable | None = None
    grader: Callable | None = None
    rewriter: Callable | None = None
    prepare_context: Callable | None = None
    infer_filters: Callable | None = None
    tool_decider: Callable | None = None
    model: Any = None


async def call(fn, *args):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    value = await asyncio.to_thread(fn, *args)
    return await value if inspect.isawaitable(value) else value


def query_key(query: str) -> str:
    return re.sub(r"\s+", "", query).casefold()


def _coverage_notes(coverage: dict | None) -> list[str]:
    """给界面看的覆盖评估说明。限流不能写成证据不足。"""
    if not coverage:
        return []
    if coverage.get("rate_limited"):
        return ["覆盖评估请求过于频繁，这次没有改判成证据不足。请稍后再问。"]
    if coverage.get("call_failed"):
        return ["覆盖评估暂时不可用，这次没有改判成证据不足。"]
    notes: list[str] = []
    for item in coverage.get("items") or []:
        reason = str(item.get("reason") or "").strip()
        if item.get("status") in {"missing", "conflicting"} and reason and reason not in notes:
            notes.append(f"覆盖评估：{reason}")
    return notes


def public_coverage_warnings(warnings: list[str] | None) -> list[str]:
    """只把覆盖评估说明送到对话里，检索诊断留在服务端日志。"""
    return [item for item in warnings or [] if str(item).startswith("覆盖评估")]


def prepare_context(docs: list[dict]) -> dict:
    from src.engine import build_context, expand_table_docs
    context_docs = []
    budget_report: dict = {}
    text, sources = build_context(
        expand_table_docs(deepcopy(docs)),
        context_docs=context_docs,
        budget_report=budget_report,
    )
    return {
        "context_text": text,
        "context_docs": context_docs,
        "sources": sources,
        # 上下文预算报告：纳入/丢弃/截断的块数与估算 token，供诊断与排障
        "context_budget": budget_report,
    }


def resolve_policy(*, mode: str | None = None, grade: bool = True, **overrides) -> RetrievalPolicy:
    from src.core import config
    agentic = mode == "agentic" if mode else config.ENABLE_LANGGRAPH
    settings = {"mode": "agentic" if agentic else "standard",
                "rewrite": agentic or config.CHAT_ENABLE_QUERY_REWRITE,
                "route": True, "reflect": agentic, "grade": grade,
                "use_tools": agentic,
                "max_subproblems": config.AGENT_MAX_SUBPROBLEMS,
                "max_retries": config.LANGGRAPH_MAX_REFLECT_ROUNDS}
    settings.update(overrides)
    return RetrievalPolicy(**settings)


async def default_search(request, queries, keywords, filter_plan, policy, kb=None):
    from src import engine as chat_service
    return await chat_service._retrieve_batch(
        request.query, request.history, use_rerank=request.use_rerank,
        k=request.k, rerank_top_n=request.rerank_top_n,
        filters=request.filters, infer_filters=False,
        queries_override=queries, filter_plan_override=filter_plan,
        kbs_override=[(getattr(kb, "collection_name", "kb"), kb)] if kb is not None else None,
        rewrite_enabled=False, keywords=keywords, dense_only=policy.dense_only,
    )


def select_docs(pool: list[dict], coverage: dict | None, top_n: int, fresh: list[dict] | None = None) -> list[dict]:
    selected, identities = [], set()
    by_id = {evidence_key(doc): doc for doc in pool}
    supported_ids = {key for item in (coverage or {}).get("items", [])
                     if item.get("status") == "supported" for key in item.get("evidence_ids", [])}
    fresh_candidates = sorted(
        [doc for doc in fresh or [] if evidence_key(doc) not in supported_ids],
        key=lambda d: float(d.get("rerank_score") if d.get("rerank_score") is not None else d.get("score") or 0), reverse=True,
    )
    reserved = min(len((coverage or {}).get("missing_items", [])), len(fresh_candidates), top_n)
    if coverage:
        for item in coverage.get("items", []):
            if item.get("status") != "supported":
                continue
            for key in item.get("evidence_ids", []):
                if key in by_id and key not in identities and len(selected) < top_n - reserved:
                    selected.append(deepcopy(by_id[key]))
                    identities.add(key)
                    break
    # 已支撑事项保留；缺失事项优先给最新补检证据留槽，避免旧高分模板霸占上下文。
    if coverage and coverage.get("missing_items"):
        fresh_slots = len(coverage["missing_items"])
        for doc in fresh_candidates:
            key = evidence_key(doc)
            if key not in identities and len(selected) < top_n and fresh_slots:
                selected.append(deepcopy(doc))
                identities.add(key)
                fresh_slots -= 1
    for doc in sorted(pool, key=lambda d: float(d.get("rerank_score") if d.get("rerank_score") is not None else d.get("score") or 0), reverse=True):
        key = evidence_key(doc)
        if key not in identities and len(selected) < top_n:
            selected.append(deepcopy(doc))
            identities.add(key)
    return selected


async def execute_retrieval(request: RetrievalRequest, policy: RetrievalPolicy,
                            dependencies: RetrievalDependencies | None = None, *, kb=None) -> dict:
    """初检不计补检；只对缺失事项补检，保留跨轮证据与原请求硬范围。"""
    from src.core import config
    from src import engine as chat_service
    deps = dependencies or RetrievalDependencies()
    explicit = validate_explicit_filters(request.filters)
    model = deps.model
    if model is None and (policy.grade or policy.rewrite):
        model = chat_service.get_rewrite_model() or chat_service.get_model()
    async def planning():
        if deps.planner:
            return await call(deps.planner, request.query, request.history, policy.max_subproblems)
        if policy.grade or policy.rewrite:
            return await asyncio.to_thread(plan_query, request.query, request.history,
                                           model=model, max_subproblems=policy.max_subproblems)
        return QueryPlan(needs_decomposition=False, subproblems=[Subproblem(id="q1", query=request.query)])

    async def inference():
        if explicit is None and request.infer_filters and policy.route:
            return await call(deps.infer_filters or chat_service.infer_metadata_filter_candidates, request.query)
        return None

    async def timed(stage, operation):
        started = time.perf_counter()
        try:
            return await operation()
        finally:
            logger.info("检索阶段耗时：stage=%s seconds=%.3f", stage, time.perf_counter() - started)

    tasks = [asyncio.create_task(timed("planning", planning)),
             asyncio.create_task(timed("filter_inference", inference))]
    try:
        plan, inferred = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if isinstance(plan, dict):
        plan = QueryPlan.model_validate(plan)
    top_n = request.rerank_top_n or config.CHAT_RERANK_TOP_K
    result = {"docs": [], "evidence_pool": [], "rewritten": request.query,
              "filters": explicit or {}, "k": request.k or config.CHAT_TOP_K,
              "rerank_top_n": top_n, "retry_count": 0, "decision": "retrieve",
              "stop_reason": "", "plan": plan.model_dump(), "coverage": None,
              "retrieval_error": None, "attempted_queries": [],
              # 部分失败 / 可选增强降级的告警：有证据就继续，但把降级原因如实带回
              "warnings": [],
              "diagnostics": {"protocol_version": PROTOCOL_VERSION, "retrieval_calls": [], "filter_attempts": [], "batches": []}}
    scope_warning = str(plan.clarification or "")
    search_plan = plan
    if scope_warning:
        # 规划不完整时仍检索原问句。澄清文案留给回答层，避免把缺项说成已经覆盖。
        result["warnings"].append(scope_warning)
        result["clarification"] = scope_warning
        search_plan = QueryPlan(
            needs_decomposition=False,
            subproblems=[Subproblem(id="q1", query=request.query)],
        )

    def finish(decision: str, stop_reason: str | None = None) -> dict:
        for note in _coverage_notes(result.get("coverage")):
            if note not in result["warnings"]:
                result["warnings"].append(note)
        if scope_warning:
            result.update(decision="clarification", stop_reason="invalid_plan", clarification=scope_warning)
        else:
            result["decision"] = decision
            if stop_reason is not None:
                result["stop_reason"] = stop_reason
        return result

    filter_plan = build_filter_plan(request.query, explicit_filters=explicit, inferred_filters=inferred)
    result["filters"] = deepcopy(filter_plan.hard_filters)
    result["diagnostics"]["filter_plan"] = filter_plan.as_dict()
    items = list(search_plan.subproblems)
    planned = ([item.query for item in items] if policy.rewrite or policy.use_tools
               else [items[0].query] if len(items) == 1 else [request.query])
    # 工具路径要和子问题一一对应；原问句在工具改写之后再追加。其余路径一开始就带上原问句。
    queries = list(planned)
    if not policy.use_tools:
        queries = list(dict.fromkeys([request.query, *queries]))
    pool, seen_queries = {}, set()
    last_coverage = None
    best_context = None
    for batch_no in range(policy.max_retries + 1):
        if policy.use_tools:
            pending = [(item, q) for item, q in zip(items, queries, strict=True) if query_key(q) not in seen_queries]
            items = [item for item, _ in pending]
            queries = [q for _, q in pending]
        else:
            queries = [q for q in dict.fromkeys(queries) if query_key(q) not in seen_queries]
        if not queries:
            result["stop_reason"] = "repeated_query"
            break
        if policy.use_tools:
            from src.agent.nodes.decision import request_tool_calls
            tool_calls = await call(deps.tool_decider or request_tool_calls, items, queries, chat_service.get_model())
            if tool_calls.get("error"):
                result.update(decision="error", retrieval_error=tool_calls["error"], stop_reason="tool_error")
                return result
            result["diagnostics"].setdefault("tool_calls", []).extend(tool_calls["calls"])
            queries = [item["args"]["query"] for item in tool_calls["calls"]]
            result["diagnostics"].setdefault("policy_forced", []).append(tool_calls["policy_forced"])
        if batch_no == 0:
            queries = list(dict.fromkeys([request.query, *queries]))
        if batch_no:
            result["retry_count"] += 1
        seen_queries.update(query_key(q) for q in queries)
        result["attempted_queries"].extend(queries)
        result["rewritten"] = queries[0]
        search_queries = list(dict.fromkeys(queries))
        stage_started = time.perf_counter()
        batch = (await call(deps.search, request, search_queries, [], filter_plan) if deps.search
                 else await default_search(request, search_queries, [], filter_plan, policy, kb))
        logger.info("检索阶段耗时：stage=search batch=%d seconds=%.3f", batch_no, time.perf_counter() - stage_started)
        diagnostic = deepcopy(batch.get("diagnostics") or {})
        result["diagnostics"]["batches"].append({"queries": queries, **diagnostic})
        result["diagnostics"]["retrieval_calls"].extend(diagnostic.get("retrieval_calls") or [])
        result["diagnostics"]["filter_attempts"].extend(diagnostic.get("filter_attempts") or [])
        # 区分「主检索不可用」与「部分失败 / 可选增强降级」（RAG-11）：
        # 只要还有可用证据，就不该把某条子查询或 HyDE 的失败升级为整体报错——
        # 那会丢弃已经拿到的证据，把可用结果变成基础设施错误。
        batch_docs = batch.get("docs") or []
        calls = diagnostic.get("retrieval_calls") or []
        call_errors = [
            f"{call.get('collection', '')}/{call.get('query', '')}: {call['error']}"
            for call in calls if call.get("error")
        ]
        primary_unavailable = bool(batch.get("retrieval_error")) or (
            not batch_docs and calls and len(call_errors) == len(calls)
        )
        if primary_unavailable:
            result.update(
                decision="error",
                retrieval_error=batch.get("retrieval_error") or "检索服务暂不可用",
                stop_reason="retrieval_error",
            )
            return result
        if call_errors:
            warnings = result.setdefault("warnings", [])
            warnings.extend(call_errors)
            logger.warning(
                "部分检索调用失败，按可用证据继续（%d 条）：%s",
                len(call_errors), "; ".join(call_errors[:3]),
            )
        if policy.use_tools:
            result["diagnostics"].setdefault("tool_results", []).extend(
                {"tool_call_id": c["id"], "name": c["name"], "evidence_ids": [evidence_key(d) for d in batch.get("docs", [])]}
                for c in tool_calls["calls"])
        for doc in batch.get("docs", []):
            key = evidence_key(doc)
            if key in pool:
                covered = set(pool[key].get("covered_child_ids") or []) | set(doc.get("covered_child_ids") or [])
                pool[key]["covered_child_ids"] = sorted(covered)
            else:
                pool[key] = deepcopy(doc)
        result["evidence_pool"] = deepcopy(list(pool.values()))
        result["docs"] = select_docs(list(pool.values()), last_coverage, top_n, batch.get("docs", []))
        stage_started = time.perf_counter()
        context = await call(deps.prepare_context or prepare_context, result["docs"])
        logger.info("检索阶段耗时：stage=context batch=%d seconds=%.3f", batch_no, time.perf_counter() - stage_started)
        result.update(context)
        if context.get("context_budget"):
            result["diagnostics"]["context_budget"] = context["context_budget"]
        if policy.grade:
            stage_started = time.perf_counter()
            if deps.grader:
                last_coverage = await call(deps.grader, search_plan, context["context_docs"])
            else:
                # 覆盖评估走主模型。改写模型只负责短调用，避免把整份证据打到低配额接口上。
                grading_model = config.get_grading_model() or chat_service.get_model() or model
                last_coverage = await asyncio.to_thread(
                    grade_coverage, search_plan, context["context_docs"], model=grading_model)
            logger.info("检索阶段耗时：stage=coverage batch=%d seconds=%.3f", batch_no, time.perf_counter() - stage_started)
            result["coverage"] = last_coverage
            if last_coverage.get("unknown"):
                return finish("unverified", "coverage_unknown")
            if best_context is None or len(last_coverage["supported_items"]) > len(best_context["coverage"]["supported_items"]):
                best_context = deepcopy({key: result[key] for key in ("docs", "coverage", "context_docs", "context_text", "sources")})
            if last_coverage["sufficient"]:
                return finish("generate", "sufficient")
        else:
            return finish("generate" if result["docs"] else "refuse", "retrieval_only")
        if not policy.reflect or batch_no == policy.max_retries:
            result["stop_reason"] = "retry_limit" if policy.reflect else "no_reflection"
            break
        missing_ids = set(last_coverage["missing_items"])
        items = [item for item in search_plan.subproblems if item.id in missing_ids]
        queries = []
        for item in items:
            if deps.rewriter:
                rewritten = await call(deps.rewriter, item, last_coverage, result["attempted_queries"])
            else:
                rewritten = await asyncio.to_thread(rewrite_missing, item, result["attempted_queries"], model=model)
            queries.append(rewritten)
    if best_context and last_coverage and len(best_context["coverage"]["supported_items"]) > len(last_coverage["supported_items"]):
        result.update(best_context)
        last_coverage = best_context["coverage"]
    return finish(decide_from_coverage(last_coverage))
