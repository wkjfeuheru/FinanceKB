"""共享答案出口：事实完成覆盖检查和逐声明核验后才允许发布。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from finance_rag.src.agent.prompts.chat import ANSWER_PROMPT
from finance_rag.src.services.citation_validator import REFUSAL_ANSWER, validate_claims
from finance_rag.src.services.retrieval_service import call, prepare_context

UNVERIFIED_ANSWER = "暂时无法完成事实核验，未发布事实答案。请查看原文来源，或稍后重试。"


async def invoke_text(model: Any, prompt: str) -> str:
    if model is None:
        raise RuntimeError("LLM未配置")
    response = (await model.ainvoke(prompt) if callable(getattr(model, "ainvoke", None))
                else await asyncio.to_thread(model.invoke, prompt))
    text = getattr(response, "content", response)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("生成模型未返回有效答案")
    return text.strip()


def missing_labels(pipeline: dict) -> list[str]:
    items = {i["id"]: i["query"] for i in (pipeline.get("plan") or {}).get("subproblems", [])}
    return [items.get(key, key) for key in (pipeline.get("coverage") or {}).get("missing_items", [])]


async def answer_pipeline(query: str, pipeline: dict, *, model: Any = None,
                          validation_model: Any = None, validator: Callable | None = None,
                          enabled: bool | None = None) -> dict:
    if pipeline.get("retrieval_error") or pipeline.get("decision") == "error":
        raise RuntimeError(pipeline.get("retrieval_error") or "检索执行失败")
    if "context_text" not in pipeline:
        pipeline = {**pipeline, **await call(prepare_context, pipeline.get("docs", []))}
    result = {"answer": "", "sources": deepcopy(pipeline.get("sources") or []),
              "rewritten_query": pipeline.get("rewritten", query), "filters": pipeline.get("filters", {}),
              "citation_validation": None, "answer_rejected": False, "low_confidence": False,
              "answer_status": "unverified", "missing_items": missing_labels(pipeline),
              "context_docs": deepcopy(pipeline.get("context_docs") or []),
              "context_text": pipeline.get("context_text", "")}
    decision = pipeline.get("decision", "generate")
    if decision == "clarification":
        result.update(answer=pipeline.get("clarification", "请明确公司、指标和期间。"), answer_status="clarification")
        return result
    if decision == "unverified" or (pipeline.get("coverage") or {}).get("unknown"):
        result.update(answer=UNVERIFIED_ANSWER, low_confidence=True,
                      citation_validation={"status":"unknown", "valid":False, "score":None,
                                           "issues":["证据覆盖无法验证"], "checks":[]})
        return result
    if decision == "refuse" or not result["context_docs"]:
        result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
        return result
    if model is None:
        from finance_rag.src.core.config import get_model
        model = get_model()
    if model is None:
        result.update(answer=UNVERIFIED_ANSWER, low_confidence=True,
                      citation_validation={"status":"unknown", "valid":False, "score":None,
                                           "issues":["核验模型不可用"], "checks":[]})
        return result
    if validation_model is None:
        validation_model = model
    if validator is None:
        from finance_rag.src.services import citation_validator
        effective_enabled = citation_validator.ENABLE_CITATION_VALIDATION if enabled is None else enabled
        if not effective_enabled:
            result.update(answer=UNVERIFIED_ANSWER, low_confidence=True,
                          citation_validation={"status":"unknown", "valid":False, "score":None,
                                               "issues":["事实核验未启用"], "checks":[]})
            return result
    if pipeline.get("coverage") is None:
        from finance_rag.src.agent.planning import QueryPlan, plan_query
        from finance_rag.src.orchestration.nodes.grading import grade_coverage
        plan = (QueryPlan.model_validate(pipeline["plan"]) if pipeline.get("plan")
                else await asyncio.to_thread(plan_query, query, model=model))
        if plan.clarification:
            result.update(answer=plan.clarification, answer_status="clarification")
            return result
        coverage = await asyncio.to_thread(grade_coverage, plan, result["context_docs"], model=validation_model)
        pipeline = {**pipeline, "coverage": coverage, "plan": plan.model_dump()}
        result["missing_items"] = missing_labels(pipeline)
        if coverage["unknown"]:
            result.update(answer=UNVERIFIED_ANSWER, low_confidence=True,
                          citation_validation={"status":"unknown", "valid":False, "score":None,
                                               "issues":["证据覆盖无法验证"], "checks":[]})
            return result
        decision = "generate" if coverage["sufficient"] else "partial" if coverage["supported_items"] else "refuse"
        if decision == "refuse":
            result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
            return result
    validate = validator or validate_claims
    supported = set((pipeline.get("coverage") or {}).get("supported_items", []))
    supported_items = [i["query"] for i in (pipeline.get("plan") or {}).get("subproblems", []) if i["id"] in supported]
    prompt = ANSWER_PROMPT.format(context=result["context_text"], query=query)
    if decision == "partial":
        prompt += "\n只回答已支撑事项，不讨论缺失事实，不生成解释性前言。已支撑事项：" + json.dumps(supported_items, ensure_ascii=False)
    for attempt in range(2):
        draft = await invoke_text(model, prompt)
        validation_options = {"model": validation_model, "enabled": enabled}
        if decision == "partial":
            all_items = (pipeline.get("plan") or {}).get("subproblems", [])
            validation_options.update(
                allowed_items=[item for item in all_items if item["id"] in supported],
                blocked_items=[item for item in all_items if item["id"] not in supported],
            )
        validation = await validate(draft, result["context_docs"], **validation_options)
        result["citation_validation"] = validation
        if validation.get("status") == "verified":
            result.update(answer=draft, answer_status="partial" if decision == "partial" else "complete")
            if result["missing_items"]:
                result["answer"] += "\n\n未覆盖事项：" + "；".join(result["missing_items"]) + "。以上为有据的部分答案。"
                result["answer_status"] = "partial"
            return result
        if validation.get("status") == "unknown":
            result.update(answer=UNVERIFIED_ANSWER, answer_status="unverified", low_confidence=True)
            return result
        if not attempt:
            supported_claims = [c["claim"] for c in validation.get("checks", []) if c.get("status") == "supported"]
            prompt += ("\n上一稿校验失败。严格依据来源重新生成一次，每句事实均需引用。"
                       "禁止保留未支撑声明；不得增加新事实。仅允许的事项："
                       + json.dumps(supported_claims or supported_items, ensure_ascii=False))
    result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
    return result


async def answer_events(result: dict):
    """这里只发布已经完成核验的结果，内部证据全文不进入公共事件。"""
    yield {"type": "sources", "sources": result["sources"]}
    if result["answer"]:
        yield {"type": "token", "content": result["answer"]}
    public = {key: result.get(key) for key in (
        "rewritten_query", "filters", "citation_validation", "answer_rejected", "low_confidence",
        "answer_status", "missing_items",
    )}
    if public["citation_validation"]:
        validation = deepcopy(public["citation_validation"])
        validation["checks"] = [{key: value for key, value in check.items()
                                 if key not in {"supporting_quote", "supporting_quotes"}}
                                for check in validation.get("checks", [])]
        public["citation_validation"] = validation
    yield {"type": "done", **public, "source_count": len(result["sources"])}
