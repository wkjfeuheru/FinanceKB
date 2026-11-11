"""共享答案出口：事实完成覆盖检查和逐声明核验后才允许发布。"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from src.agent.prompts.chat import ANSWER_PROMPT
from src.answer.citations import (
    REFUSAL_ANSWER,
    CitationLedger,
    normalize_citations,
    validate_claims,
)
from src.answer.llm import invoke_text
from src.retrieve.pipeline import call, prepare_context

logger = logging.getLogger(__name__)

UNVERIFIED_ANSWER = "暂时无法完成事实核验，未发布事实答案。请查看原文来源，或稍后重试。"

_IMAGE_EXTS = {"png", "jpeg", "gif", "webp", "bmp"}


def _image_ext(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    if ext == "jpg":
        ext = "jpeg"
    return ext if ext in _IMAGE_EXTS else "png"


def collect_context_images(context_docs: list[dict]) -> list[dict]:
    """下载已进入上下文的图片块原图。同一键只下载一次，失败则跳过该图。"""
    from src.core.config import ANSWER_MAX_IMAGES
    from src.storage.objects import get_storage

    limit = ANSWER_MAX_IMAGES
    cache: dict[str, str | None] = {}
    attached: list[str] = []
    images: list[dict] = []
    storage = None
    for index, doc in enumerate(context_docs, start=1):
        if str(doc.get("block_type") or "") != "image":
            continue
        key = str(doc.get("image_key") or "").strip()
        if not key:
            continue
        if key not in cache:
            if len(attached) >= limit:
                continue
            if storage is None:
                storage = get_storage()
            try:
                data = storage.download_sync(key)
            except Exception as exc:  # noqa: BLE001 - 单张失败不阻断回答
                logger.warning("原图读取失败（%s）：%s", key, exc)
                cache[key] = None
                continue
            if not data:
                cache[key] = None
                continue
            encoded = base64.b64encode(data).decode("ascii")
            cache[key] = f"data:image/{_image_ext(key)};base64,{encoded}"
            attached.append(key)
        if cache[key]:
            images.append({"index": index, "image_key": key, "data_url": cache[key]})
    return images


def missing_labels(pipeline: dict) -> list[str]:
    items = {i["id"]: i["query"] for i in (pipeline.get("plan") or {}).get("subproblems", [])}
    return [
        items.get(key, key) for key in (pipeline.get("coverage") or {}).get("missing_items", [])
    ]


async def answer_pipeline(
    query: str,
    pipeline: dict,
    *,
    model: Any = None,
    validation_model: Any = None,
    validator: Callable | None = None,
    enabled: bool | None = None,
    prefer_text: bool = False,
) -> dict:
    """生成、核验及重试共用一个截止时间，超时不发布草稿。"""
    from src.core.config import ANSWER_TIMEOUT_SECONDS

    try:
        async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
            return await _answer_pipeline(
                query,
                pipeline,
                model=model,
                validation_model=validation_model,
                validator=validator,
                enabled=enabled,
                prefer_text=prefer_text,
            )
    except TimeoutError:
        logger.warning("答案生成与核验超时：上限 %.1f 秒", ANSWER_TIMEOUT_SECONDS)
        return {
            "answer": UNVERIFIED_ANSWER,
            "sources": deepcopy(pipeline.get("sources") or []),
            "rewritten_query": pipeline.get("rewritten", query),
            "filters": pipeline.get("filters", {}),
            "citation_validation": {
                "status": "unknown",
                "valid": False,
                "score": None,
                "issues": ["答案生成与事实核验超时，请稍后重试"],
                "checks": [],
            },
            "answer_rejected": False,
            "low_confidence": True,
            "answer_status": "unverified",
            "missing_items": missing_labels(pipeline),
            "warnings": [],
        }


async def _answer_pipeline(
    query: str,
    pipeline: dict,
    *,
    model: Any = None,
    validation_model: Any = None,
    validator: Callable | None = None,
    enabled: bool | None = None,
    prefer_text: bool = False,
) -> dict:
    if pipeline.get("retrieval_error") or pipeline.get("decision") == "error":
        raise RuntimeError(pipeline.get("retrieval_error") or "检索执行失败")
    if "context_text" not in pipeline:
        pipeline = {**pipeline, **await call(prepare_context, pipeline.get("docs", []))}
    from src.retrieve.pipeline import public_coverage_warnings

    result = {
        "answer": "",
        "sources": deepcopy(pipeline.get("sources") or []),
        "rewritten_query": pipeline.get("rewritten", query),
        "filters": pipeline.get("filters", {}),
        "citation_validation": None,
        "answer_rejected": False,
        "low_confidence": False,
        "answer_status": "unverified",
        "missing_items": missing_labels(pipeline),
        "warnings": public_coverage_warnings(pipeline.get("warnings")),
        "context_docs": deepcopy(pipeline.get("context_docs") or []),
        "context_text": pipeline.get("context_text", ""),
    }
    decision = pipeline.get("decision", "generate")
    incoming = pipeline.get("coverage") or {}
    if decision == "refuse" and incoming.get("visual_pending_items"):
        decision = "partial" if incoming.get("missing_items") else "generate"
    if decision == "clarification":
        result.update(
            answer=pipeline.get("clarification", "请明确公司、指标和期间。"),
            answer_status="clarification",
        )
        return result
    if decision == "unverified" or (pipeline.get("coverage") or {}).get("unknown"):
        result.update(
            answer=UNVERIFIED_ANSWER,
            low_confidence=True,
            citation_validation={
                "status": "unknown",
                "valid": False,
                "score": None,
                "issues": ["证据覆盖无法验证"],
                "checks": [],
            },
        )
        return result
    if decision == "refuse" or not result["context_docs"]:
        result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
        return result
    if model is None:
        from src.core.config import get_answer_model

        model = get_answer_model()
    if model is None:
        result.update(
            answer=UNVERIFIED_ANSWER,
            low_confidence=True,
            citation_validation={
                "status": "unknown",
                "valid": False,
                "score": None,
                "issues": ["核验模型不可用"],
                "checks": [],
            },
        )
        return result
    if validator is None:
        from src.answer import citations as citation_validator

        effective_enabled = (
            citation_validator.ENABLE_CITATION_VALIDATION if enabled is None else enabled
        )
        if not effective_enabled:
            result.update(
                answer=UNVERIFIED_ANSWER,
                low_confidence=True,
                citation_validation={
                    "status": "unknown",
                    "valid": False,
                    "score": None,
                    "issues": ["事实核验未启用"],
                    "checks": [],
                },
            )
            return result
    if pipeline.get("coverage") is None:
        from src.agent.nodes.grading import grade_coverage
        from src.agent.planning import QueryPlan, plan_query
        from src.core.config import get_model

        plan = (
            QueryPlan.model_validate(pipeline["plan"])
            if pipeline.get("plan")
            else await asyncio.to_thread(plan_query, query, model=get_model())
        )
        if plan.clarification:
            result.update(answer=plan.clarification, answer_status="clarification")
            return result
        from src.agent.nodes.grading import decide_from_coverage

        coverage = await asyncio.to_thread(
            grade_coverage, plan, result["context_docs"], model=get_model()
        )
        pipeline = {**pipeline, "coverage": coverage, "plan": plan.model_dump()}
        result["missing_items"] = missing_labels(pipeline)
        if coverage["unknown"]:
            result.update(
                answer=UNVERIFIED_ANSWER,
                low_confidence=True,
                citation_validation={
                    "status": "unknown",
                    "valid": False,
                    "score": None,
                    "issues": ["证据覆盖无法验证"],
                    "checks": [],
                },
            )
            return result
        decision = decide_from_coverage(coverage)
        if decision == "refuse":
            result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
            return result
    validate = validator or validate_claims
    ledger = CitationLedger(result["context_docs"])
    coverage_now = pipeline.get("coverage") or {}
    answerable = set(coverage_now.get("supported_items") or []) | set(
        coverage_now.get("visual_pending_items") or []
    )
    supported_items = [
        i["query"]
        for i in (pipeline.get("plan") or {}).get("subproblems", [])
        if i["id"] in answerable
    ]
    prompt = ANSWER_PROMPT.format(context=result["context_text"], query=query)
    images = collect_context_images(result["context_docs"])
    if prefer_text and not images:
        from src.core.config import get_grading_model

        model = get_grading_model() or model
    if validation_model is None:
        # 只有实际附图的核验才需要视觉模型；纯文本复用已有的结构化覆盖核验模型。
        from src.core.config import get_grading_model

        validation_model = model if images else (get_grading_model() or model)
    if decision == "partial":
        prompt += "\n只回答已支撑事项，不讨论缺失事实，不生成解释性前言。已支撑事项：" + json.dumps(
            supported_items, ensure_ascii=False
        )
    # 窗口级校验（RAG-15）：输入 + 预留输出必须落在模型窗口内。
    # 超限时给出可理解的限制反馈，而不是让 provider 抛超时/截断错误。
    from src.core.token_budget import estimate_tokens, require_within_window

    try:
        require_within_window(estimate_tokens(prompt), kind="生成")
    except ValueError as exc:
        logger.warning("生成输入超出模型窗口：%s", exc)
        result.update(
            answer=UNVERIFIED_ANSWER,
            low_confidence=True,
            citation_validation={
                "status": "unknown",
                "valid": False,
                "score": None,
                "issues": [str(exc)],
                "checks": [],
            },
        )
        return result
    if not ledger.matches_sources(result["sources"]):
        result.update(
            answer=UNVERIFIED_ANSWER,
            low_confidence=True,
            citation_validation={
                "status": "unknown",
                "valid": False,
                "score": None,
                "issues": ["引用来源与上下文编号不一致"],
                "checks": [],
            },
        )
        return result
    for attempt in range(2):
        started = time.perf_counter()
        draft = normalize_citations(await invoke_text(model, prompt, images))
        logger.info(
            "答案生成模型耗时：attempt=%d seconds=%.3f output_chars=%d",
            attempt + 1,
            time.perf_counter() - started,
            len(draft),
        )
        validation_options = {"model": validation_model, "enabled": enabled, "images": images}
        if decision == "partial":
            all_items = (pipeline.get("plan") or {}).get("subproblems", [])
            validation_options.update(
                allowed_items=[item for item in all_items if item["id"] in answerable],
                blocked_items=[item for item in all_items if item["id"] not in answerable],
            )
        structure = ledger.validate(draft)
        if structure["status"] == "failed":
            validation = {
                "status": "failed",
                "valid": False,
                "score": 0,
                "issues": ["声明缺少引用或引用编号越界"],
                "checks": [],
                "citations": structure,
            }
        else:
            validation = await validate(draft, ledger.documents, **validation_options)
        result["citation_validation"] = validation
        if validation.get("status") != "verified":
            logger.info(
                "答案核验未通过：attempt=%d status=%s issues=%s",
                attempt + 1,
                validation.get("status"),
                validation.get("issues", []),
            )
        if (
            validation.get("status") == "verified"
            and validation.get("numbers", {}).get("status", "verified") == "verified"
        ):
            result.update(
                answer=draft, answer_status="partial" if decision == "partial" else "complete"
            )
            if result["missing_items"]:
                result["answer"] += (
                    "\n\n未覆盖事项："
                    + "；".join(result["missing_items"])
                    + "。以上为有据的部分答案。"
                )
                result["answer_status"] = "partial"
            return result
        if validation.get("status") == "unknown":
            result.update(answer=UNVERIFIED_ANSWER, answer_status="unverified", low_confidence=True)
            return result
        if not attempt:
            supported_claims = [
                {"claim": c["claim"], "citation_indices": c.get("citation_indices", [])}
                for c in validation.get("checks", [])
                if c.get("status") == "supported"
            ]
            prompt += (
                "\n上一稿校验失败，具体问题："
                + json.dumps(validation.get("issues", []), ensure_ascii=False)
                + "。严格依据来源重新生成一次，每句事实均需引用。"
                "禁止保留未支撑声明；不得增加新事实。以下已核验声明必须保留对应citation_indices的全部编号，"
                "不能删减、替换或重新分配引用。只组织这些已核验事实，禁止扩写应用、产品或数字。仅允许的事项："
                + json.dumps(supported_claims or supported_items, ensure_ascii=False)
            )
    issues = (result.get("citation_validation") or {}).get("issues") or []
    logger.info("事实核验未通过，改为拒答：%s", issues)
    result.update(answer=REFUSAL_ANSWER, answer_status="refused", answer_rejected=True)
    return result


async def answer_events(result: dict):
    """这里只发布已经完成核验的结果，内部证据全文不进入公共事件。"""
    yield {"type": "sources", "sources": result["sources"]}
    if result["answer"]:
        yield {"type": "token", "content": result["answer"]}
    public = {
        key: result.get(key)
        for key in (
            "rewritten_query",
            "filters",
            "citation_validation",
            "answer_rejected",
            "low_confidence",
            "answer_status",
            "missing_items",
        )
    }
    public["warnings"] = list(result.get("warnings") or [])
    if public["citation_validation"]:
        validation = deepcopy(public["citation_validation"])
        validation["checks"] = [
            {
                key: value
                for key, value in check.items()
                if key not in {"supporting_quote", "supporting_quotes"}
            }
            for check in validation.get("checks", [])
        ]
        public["citation_validation"] = validation
    yield {"type": "done", **public, "source_count": len(result["sources"])}
