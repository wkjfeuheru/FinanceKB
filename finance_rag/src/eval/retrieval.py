"""生产检索策略到统一评测运行器的适配层。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

from finance_rag.src.core.config import (
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_FUSION_MODE,
    RETRIEVAL_SPARSE_WEIGHT,
)

from .runner import EvaluationStrategy, RetrievalOutput

LAYER_SPECS: dict[str, dict[str, Any]] = {
    "L0": {
        "name": "纯稠密检索", "k": 50, "top_k": 5, "use_dense_only": True,
        "use_rerank": False, "cliff_detection": False,
    },
    "L1": {
        "name": "混合检索(Weighted)", "k": 50, "top_k": 5, "use_dense_only": False,
        "use_rerank": False, "cliff_detection": False,
    },
    "L2": {
        "name": "混合检索，未重排序", "k": 5, "top_k": 5, "use_dense_only": False,
        "use_rerank": False, "cliff_detection": False,
    },
    "L3": {
        "name": "混合检索 + BGE", "k": 50, "top_k": 5, "use_dense_only": False,
        "use_rerank": True, "cliff_detection": False,
    },
    "L5": {
        "name": "混合检索 + BGE + 断崖检测", "k": 50, "top_k": 5,
        "use_dense_only": False, "use_rerank": True, "cliff_detection": True,
    },
}


def build_retrieval_strategy(
    kb: Any,
    layer: str,
    *,
    candidate_k: int = 50,
    top_k: int = 5,
    fusion_mode: str = RETRIEVAL_FUSION_MODE,
    dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
    sparse_weight: float = RETRIEVAL_SPARSE_WEIGHT,
) -> EvaluationStrategy:
    """构造 L0/L1/L2/L3/L5 策略；L4 永久不在评测矩阵中。

    ``candidate_k`` 是重排层的粗排深度（现与生产一致为 50）；L2 无重排层，
    其 ``k`` 就是最终深度（= ``top_k``），否则「同深度比重排」的对照失去意义。
    """
    key = layer.upper()
    if key == "PRODUCTION":
        return _production_strategy(candidate_k, top_k, kb)
    if key not in LAYER_SPECS:
        raise ValueError(f"未知检索层：{layer}；可选 {', '.join(LAYER_SPECS)}")
    spec = dict(LAYER_SPECS[key])
    spec["top_k"] = top_k
    spec["k"] = top_k if key == "L2" else candidate_k
    spec.update(fusion_mode=fusion_mode, dense_weight=dense_weight, sparse_weight=sparse_weight)

    def retrieve(query: str) -> RetrievalOutput:
        results = kb.hybrid_search(
            query,
            k=spec["k"],
            use_rerank=spec["use_rerank"],
            rerank_top_n=top_k,
            # 块级指标需要原始 chunk id；父块扩展既改变上下文粒度，又会额外依赖
            # PostgreSQL，不属于本次检索层优化变量。
            expand_parents=False,
            use_dense_only=spec["use_dense_only"],
            cliff_detection=bool(spec["cliff_detection"]),
            fusion_mode=fusion_mode,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        final = list(results[:top_k])
        cliff = getattr(kb, "last_retrieval_cliff", None) or {}
        pre = cliff.get("pre")
        post = cliff.get("post")
        if spec["use_rerank"] and isinstance(pre, int) and isinstance(post, int):
            pre_cliff_depth = pre
            post_cliff_depth = post
            cliff_triggered = bool(spec["cliff_detection"] and post < pre)
        else:
            pre_cliff_depth = len(results)
            post_cliff_depth = len(final)
            cliff_triggered = False
        return RetrievalOutput(
            candidates=list(results),
            results=final,
            cliff_triggered=cliff_triggered,
            pre_cliff_depth=pre_cliff_depth,
            post_cliff_depth=post_cliff_depth,
            diagnostics=dict(getattr(kb, "last_retrieval_diagnostics", None) or {}),
        )

    return EvaluationStrategy(
        name=key,
        retrieve=retrieve,
        config={**spec, "embedding_model": getattr(kb, "embedding_model", None)},
        resolve_parent=_parent_resolver(kb),
    )


def _run_async(factory: Callable[[], Any]) -> Any:
    """同步评测调用异步生产入口；已有 loop 时在独立线程执行。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()


def _production_strategy(candidate_k: int, top_k: int, kb: Any) -> EvaluationStrategy:
    from finance_rag.src.services import answer_service, retrieval_service

    policy = retrieval_service.resolve_policy()

    def pipeline(query: str) -> dict[str, Any]:
        request = retrieval_service.RetrievalRequest(query, k=candidate_k, rerank_top_n=top_k)
        return _run_async(lambda: retrieval_service.execute_retrieval(request, policy, kb=kb))

    def retrieve(query: str) -> RetrievalOutput:
        return adapt_production_output(pipeline(query))

    def negative(case: Any) -> dict[str, Any]:
        retrieved = pipeline(case.query)
        answered = _run_async(lambda: answer_service.answer_pipeline(case.query, retrieved))
        if not isinstance(answered, dict) or "answer" not in answered or not answered.get("answer_status"):
            raise RuntimeError("生产回答缺少完成结果")
        if answered["answer_status"] == "unverified":
            answered = {**answered, "answer_rejected": None, "evaluation_status": "unverified",
                        "rejection_verification": "coverage_or_claim_verification_unknown",
                        "evaluation_issue": "model_verification_unavailable"}
        return {**answered, "production_done": answered,
                "retrieval_diagnostics": retrieved.get("diagnostics") or {}}

    return EvaluationStrategy(
        name="PRODUCTION", retrieve=retrieve, evaluate_negative=negative,
        config={"k": candidate_k, "top_k": top_k, "use_rerank": True, "production_path": "services.retrieval_service.execute_retrieval", "protocol_version": "agentic-v2", "embedding_model": getattr(kb, "embedding_model", None)},
    )


def adapt_production_output(pipeline: dict[str, Any]) -> RetrievalOutput:
    """共享生产证据、整表展开和实际上下文预算；不重新检索或评分。"""
    if pipeline.get("retrieval_error"):
        raise RuntimeError(pipeline["retrieval_error"])
    results = deepcopy(pipeline["docs"])
    if "context_docs" in pipeline and "context_text" in pipeline:
        context_results = deepcopy(pipeline["context_docs"])
        context_text, sources = pipeline["context_text"], deepcopy(pipeline.get("sources") or [])
    else:
        # 仅兼容旧包装返回值；共享执行器已经准备好的上下文不得再次展开。
        from finance_rag.src.services.retrieval_service import prepare_context
        context = prepare_context(results)
        context_results = context["context_docs"]
        context_text, sources = context["context_text"], context["sources"]
    diagnostic = deepcopy(pipeline.get("diagnostics") or {})
    calls = diagnostic.get("retrieval_calls") or []
    if len(calls) == 1:
        diagnostic.update(calls[0])
    elif len(calls) > 1:
        # 各过滤分支只各自有粗排顺序；不能拼出不存在的统一 Recall@50 排名。
        diagnostic["unified_coarse_stage"] = "unavailable"
        diagnostic.pop("coarse_candidate_chunk_ids", None)
        diagnostic.pop("coarse_candidates", None)
    for pipeline_key, diagnostic_key in (("filters", "filters"), ("rewritten", "rewritten_query"), ("k", "k"), ("rerank_top_n", "rerank_top_n")):
        if pipeline_key in pipeline:
            diagnostic[diagnostic_key] = deepcopy(pipeline[pipeline_key])
    diagnostic.update(context_text=context_text, context_sources=sources)
    for key in ("plan", "coverage", "decision", "stop_reason", "retry_count", "attempted_queries"):
        if key in pipeline:
            diagnostic[key] = deepcopy(pipeline[key])
    return RetrievalOutput(
        candidates=([] if len(calls) > 1 else deepcopy(diagnostic.get("coarse_candidates") or results)),
        results=results, diagnostics=diagnostic, context_results=context_results,
    )


def knowledge_content_fingerprints(kb: Any) -> dict[str, Any]:
    """只读确认实际源集合和完整 SHA；任何身份缺失均不能生成可冻结指纹。"""
    import re

    result: dict[str, Any] = {"pdf_set_fingerprint": "unavailable", "chunk_content_fingerprint": "unavailable"}
    sources: set[str] = set()
    source_identity_missing = False
    store = None
    try:
        getter = getattr(kb, "get_fingerprint_store", None)
        if getter is not None:
            store = getter()
        list_documents = getattr(kb, "list_documents", None)
        if list_documents is not None:
            try:
                for row in list_documents():
                    source = str(row.get("source") or "").strip()
                    if source:
                        sources.add(source)
                    else:
                        source_identity_missing = True
            except Exception as exc:
                result["document_listing_error"] = f"{type(exc).__name__}: {exc}"
        get_client = getattr(kb, "_get_client", None)
        if get_client is not None:
            client = get_client()
            iterator_fn = getattr(client, "query_iterator", None)
            if iterator_fn is not None:
                iterator = iterator_fn(collection_name=kb.collection_name, filter="", output_fields=["id", "source", "content"], batch_size=1000)
                chunks = []
                try:
                    while True:
                        batch = iterator.next()
                        if not batch:
                            break
                        for item in batch:
                            source = str(item.get("source") or "").strip()
                            if source:
                                sources.add(source)
                            else:
                                source_identity_missing = True
                            chunks.append((str(item.get("id") or ""), source, hashlib.sha256(str(item.get("content") or "").encode()).hexdigest()))
                finally:
                    iterator.close()
                result["chunk_content_fingerprint"] = hashlib.sha256(json.dumps(sorted(chunks), ensure_ascii=False).encode()).hexdigest()
                result["chunk_content_count"] = len(chunks)
        result["pdf_source_count"] = len(sources)
        if store is not None and sources and not source_identity_missing:
            payload, missing = [], []
            for source in sorted(sources):
                sha = (store.get(source) or {}).get("sha256")
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
                    missing.append(source)
                else:
                    payload.append((source, sha.lower()))
            if missing:
                result["pdf_identity_missing_sources"] = missing
            else:
                result["pdf_set_fingerprint"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
        elif source_identity_missing:
            result["pdf_identity_missing_source"] = True
    except Exception as exc:
        result["content_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_online_evidence_resolver(kb: Any) -> Callable[[Any], dict[str, Any] | None]:
    """按真实容器解析 schema2 证据；PDF 引文另外核验整页位置和文件 SHA。"""
    from finance_rag.src.core.config import UPLOAD_DIR
    root = Path(__file__).resolve().parents[3]
    fingerprints = kb.get_fingerprint_store()
    pdf_cache: dict[str, tuple[str, list[str]]] = {}

    def resolve(ref: Any) -> dict[str, Any] | None:
        kind = ref.location.get("kind", "chunk")
        source = ref.source
        sha = str((fingerprints.get(source) or {}).get("sha256") or "")
        anchor = str(ref.location.get("anchor_id") or ref.chunk_id)
        if kind == "parent":
            actual = kb._get_parent_store().get(anchor)
        elif kind == "table":
            actual = kb._get_table_store().get(anchor)
            if actual:
                actual = {**actual, "content": str(actual.get("markdown") or "")}
        elif kind == "pdf":
            if source not in pdf_cache:
                # basename 必须来自已标注 source，不接受路径穿越。
                if Path(source).name != source:
                    return None
                path = next((directory / source for directory in (root / "assets/reports", Path(UPLOAD_DIR)) if (directory / source).is_file()), None)
                if path is None:
                    return None
                import pymupdf
                with pymupdf.open(path) as document:
                    pages = [page.get_text() for page in document]
                pdf_cache[source] = (hashlib.sha256(path.read_bytes()).hexdigest(), pages)
            sha, pages = pdf_cache[source]
            page_number = ref.location.get("pdf_page", ref.location.get("start_page"))
            start, end = ref.location.get("pdf_start"), ref.location.get("pdf_end")
            if not isinstance(page_number, int) or not (1 <= page_number <= len(pages)) or not isinstance(start, int) or not isinstance(end, int):
                return None
            page = pages[page_number - 1]
            original = str(ref.location.get("original_quote") or "")
            if not (0 <= start < end <= len(page)) or page[start:end] != original or original != ref.quote:
                return None
            actual = {"source": source, "content": page[start:end]}
        elif kind == "chunk":
            escaped = ref.chunk_id.replace("\\", "\\\\").replace('"', '\\"')
            rows = kb._get_client().query(collection_name=kb.collection_name,
                                          filter=f'id == "{escaped}"', output_fields=["id", "source", "content", "version"], limit=1)
            actual = rows[0] if rows else None
        else:
            return None
        return {**actual, "document_sha256": sha} if actual else None

    return resolve


def _parent_resolver(kb: Any) -> Callable[[str], str | None] | None:
    """有父块存储时才挂上还原查询。假知识库不提供该接口，评测照常跑块级指标。"""
    getter = getattr(kb, "_get_parent_store", None)
    if getter is None:
        return None

    def resolve(parent_id: str) -> str | None:
        if not parent_id:
            return None
        row = getter().get(parent_id)
        if not row:
            return None
        content = row.get("content")
        return None if content is None else str(content)

    return resolve
