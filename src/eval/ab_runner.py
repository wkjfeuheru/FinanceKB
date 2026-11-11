"""A/B 实验臂执行器与编排（子进程隔离 + Ragas 批量评估）。

臂执行模型：
* 每臂在独立子进程中运行（``python -m src.eval.ab_runner``），
  臂的环境变量覆盖（``ABArm.env_overrides``）随子进程 env 传入，
  优先级高于 ``.env``，保证配置开关在进程内真实生效；
* 入库臂先按臂开关把 ``files/`` 索引进独立集合 ``finance_kb_ab_*``；
* 逐题执行 检索 → 生成 →（可选拒答策略），随后 Ragas 批量评估；
* 父进程编排双臂（顺序按 seed 随机）、聚合对比并输出报告。

用法（子进程入口）::

    python -m src.eval.ab_runner \
        --experiment hyde_on_off --arm hyde_on --testset <md> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT

ROOT = PROJECT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# 负样本使用共享答案出口的显式拒答标记；服务错误不计正确拒答。
from src.eval.metrics import GENERATION_METRICS  # noqa: E402

#: 检索侧客观指标（报告分组用，不参与计算）——按标注证据的 chunk id 精确匹配，
#: 是检索能力的地面真值，永远与 RAGAS judge 指标分开读。
RETRIEVAL_OBJECTIVE_METRICS: tuple[str, ...] = (
    "recall_at_5",
    "precision_at_5",
    "chunk_mrr",
    "hit_rate",
    "mrr",
    "ndcg",
    "doc_coverage",
    "doc_recall",
    "evidence_hit",
    "evidence_coverage",
    "evidence_all_in_topk",
    "multi_hop_hit",
    "context_evidence_recall_at_5",
    "context_evidence_slot_precision_at_5",
    "context_evidence_mrr",
)


def _stratified_sample(entries: list[Any], sample: int, seed: int) -> list[Any]:
    """按 ``question_type`` 分层抽样，保证小样本下每类问题都有代表。

    原先直接用 ``random.sample`` 全局抽样：``--sample 10`` 时 multi_hop 只有 2 题，
    很容易抽到 0 题或抽到同一个问题的重复条目，导致该类型指标（``multi_hop_hit``）
    失去意义——实际报告里 n=2 就是同一题在两臂各出现一次。

    策略：先按类型均分名额，再在类内随机；名额不足时把余量补给题量最多的类型。
    类型顺序与类内顺序都受 ``seed`` 控制，保证可复现。
    """
    rng = random.Random(seed)
    buckets: dict[str, list[Any]] = {}
    for entry in entries:
        buckets.setdefault(getattr(entry, "question_type", "") or "single_hop", []).append(entry)

    types = sorted(buckets)
    quota = {qtype: min(sample // len(types), len(buckets[qtype])) for qtype in types}
    remaining = sample - sum(quota.values())
    # 余量按「类内剩余题量」从多到少补齐，同类内保持稳定顺序
    while remaining > 0:
        candidates = [
            qtype for qtype in types if quota[qtype] < len(buckets[qtype])
        ]
        if not candidates:
            break
        candidates.sort(key=lambda qtype: (-len(buckets[qtype]), qtype))
        quota[candidates[0]] += 1
        remaining -= 1

    picked: list[Any] = []
    for qtype in types:
        pool = list(buckets[qtype])
        rng.shuffle(pool)
        picked.extend(pool[: quota[qtype]])
    # 输出顺序按类型稳定排列，便于逐题明细比对
    picked.sort(key=lambda entry: getattr(entry, "question_type", "") or "")
    return picked


def _flag_weak_multi_hop(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """筛出「标注的相关文档一篇都没被召回」的多跳条目。

    这类条目的 ``multi_hop_hit=0`` 说明不了检索能力：自动生成的跨文档多跳题
    是把两篇随机抽到的文档硬配在一起的（见 ``testset_generator._generate_multi_hop``），
    第二篇常常与问题无关，检不到是正常的。把这类条目单独列出来，才能把
    「检索/重排序失败」与「评测集标注不可信」分开归因。

    仅报警、不自动剔除，避免悄悄改写评测基准。
    """
    weak: list[dict[str, Any]] = []
    for item in diagnostics:
        expected = {
            Path(name).stem.lower()
            for name in re.split(r"[,，;；|]", item.get("expected_docs") or "")
            if name.strip()
        }
        retrieved = {
            Path(stem).stem.lower()
            for stem in (item.get("retrieved_docs") or "").split(";")
            if stem.strip()
        }
        if item.get("doc_coverage") == 0.0:
            missing = sorted(name for name in (expected - retrieved) if name)
            weak.append({
                "query": item.get("query", ""),
                "expected_docs": item.get("expected_docs", ""),
                "missing_docs": ", ".join(missing),
                "retrieved_docs": item.get("retrieved_docs", ""),
                "note": "标注文档一篇都未召回：疑似该文档与问题无关（自动配对产物）",
            })
    return weak


# ---------------------------------------------------------------------------
# 入库臂（ingest）：把 files/ 文档按臂开关索引进独立集合
# ---------------------------------------------------------------------------

# 入库臂文档源目录（约定为项目根 files/）
_ARM_DOCS_DIR = ROOT / "files"

# 入库臂支持的可解析文件扩展名
_SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def collect_kb_files(docs_dir: Path) -> list[Path]:
    """递归收集文档源目录下的可解析文件（md/txt/pdf）。"""
    if not docs_dir.exists():
        return []
    return [
        p for p in sorted(docs_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


def _arm_collection_name(experiment: Any, arm: Any) -> str:
    """入库臂独立集合名：``finance_kb_ab_<实验名>_<臂名>``。"""
    return f"finance_kb_ab_{experiment.name}_{arm.name}"


def _index_arm_collection(collection: str, docs_dir: Path) -> Any:
    """把 docs_dir 文档按当前进程开关（如 ENABLE_SEMANTIC_CHUNKER）索引进独立集合。

    切块器选择由当前进程的 ``ENABLE_SEMANTIC_CHUNKER`` 决定（臂 env_overrides
    随子进程 env 传入），走与生产入库一致的共享准备流程，实现
    「语义+层级 vs 仅层级」的索引差异。

    Returns:
        索引完成的 KnowledgeBase 实例。
    """
    from src.index.milvus_kb import KnowledgeBase
    from src.ingest.pipeline import prepare_chunked_document_sync

    kb = KnowledgeBase(collection_name=collection)
    kb.ensure_collection()
    files = collect_kb_files(docs_dir)
    for path in files:
        try:
            parsed = kb.parse_document(
                path, source=path.name, title=path.stem
            )
            # 解析只产出文本产物；切块 + 元数据/图片增强在此执行一次。
            parsed = prepare_chunked_document_sync(parsed)
            kb.add_parsed_document(parsed)
        except Exception as exc:
            logger.warning("入库臂索引失败（%s）：%s", path.name, exc)
    doc_count = len(kb.list_documents())
    logger.info("入库臂集合 %s 索引完成：%d 文档", collection, doc_count)
    return kb


# ---------------------------------------------------------------------------
# 检索 / 生成（臂开关在进程内已生效）
# ---------------------------------------------------------------------------

def _pipeline_for_arm(experiment: Any, arm: Any, kb: Any, query: str) -> dict[str, Any]:
    """单臂只选择共享策略；轮次、上下文和证据累计由生产执行器处理。"""
    from src.core import config
    from src.retrieve import pipeline as service

    from .retrieval import _run_async

    strategy = arm.strategy_overrides
    retrieval_only = bool(strategy.get("retrieval_only"))
    component = any(key in strategy for key in ("rewrite", "route", "reflect"))
    overrides = {key: bool(strategy[key]) for key in ("rewrite", "route", "reflect") if key in strategy}
    if component:
        overrides["mode"] = "agentic" if any(overrides.values()) else "standard"
        overrides["use_tools"] = False
    elif arm.retrieval == "langgraph":
        overrides["mode"] = "agentic"
    if retrieval_only:
        overrides["use_tools"] = False
        if not component:
            overrides["rewrite"] = bool(config.CHAT_ENABLE_QUERY_REWRITE) if experiment.name in {"rewrite_on_off", "query_rewrite_on_off"} else False
            overrides["reflect"] = False
    overrides["dense_only"] = bool(strategy.get("use_dense_only", False))
    # 补检实验需要覆盖评分作为补检触发，其余纯检索不评分。
    policy = service.resolve_policy(grade=not retrieval_only or bool(overrides.get("reflect")), **overrides)
    request = service.RetrievalRequest(
        query, infer_filters=policy.route, k=int(strategy.get("k", config.CHAT_TOP_K)),
        rerank_top_n=int(strategy.get("rerank_top_n", config.CHAT_RERANK_TOP_K)),
        use_rerank=bool(strategy.get("use_rerank", True)),
    )
    result = _run_async(lambda: service.execute_retrieval(request, policy, kb=kb))
    if result.get("retrieval_error"):
        raise RuntimeError(result["retrieval_error"])
    return result


def _retrieve_for_arm(experiment: Any, arm: Any, kb: Any, query: str) -> list[dict[str, Any]]:
    """保留旧列表返回包装名。"""
    return _pipeline_for_arm(experiment, arm, kb, query)["docs"]


def _langgraph_retrieve(query: str) -> list[dict[str, Any]]:
    """保留旧函数名，复用共享 Agentic 策略。"""
    from src.index.milvus_kb import get_knowledge_base
    from src.retrieve import pipeline as service

    from .retrieval import _run_async
    pipeline = _run_async(lambda: service.execute_retrieval(service.RetrievalRequest(query), service.resolve_policy(mode="agentic"), kb=get_knowledge_base()))
    if pipeline.get("retrieval_error"):
        raise RuntimeError(pipeline["retrieval_error"])
    return pipeline["docs"]


def _answer_for_arm(query: str, pipeline: dict[str, Any]) -> dict[str, Any]:
    from src.answer.generator import answer_pipeline

    from .retrieval import _run_async
    return _run_async(lambda: answer_pipeline(query, pipeline))


def _generate_answer(query: str, contexts: dict[str, Any] | list[str]) -> str:
    """旧包装名保留；生成和逐声明核验只走共享答案出口。"""
    if isinstance(contexts, list):
        from src.retrieve.pipeline import prepare_context
        docs = [{"id": str(i), "content": text} for i, text in enumerate(contexts, 1)]
        contexts = {"docs": docs, "decision": "generate", **prepare_context(docs)}
    return str(_answer_for_arm(query, contexts)["answer"])


def _evidence_slot_metrics(docs: list[dict[str, Any]], refs: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """实际上下文窗口的同源完整 Gold 引文覆盖；父块成员标签不能证明可见。"""
    from .metrics import compute_text_coverage

    required = {str(ref.get("chunk_id") or ref.get("evidence_id") or ""): ref for ref in refs}
    required.pop("", None)
    if not required or any(not ref.get("quote") or not ref.get("source") for ref in required.values()):
        return {"recall_at_5": None, "precision_at_5": None, "chunk_mrr": None,
                "evidence_hit": None, "evidence_coverage": None, "evidence_rank": None,
                "evidence_all_in_topk": None, "covered_gold_ids": [], "coverage_status": "annotation_unavailable"}
    ranks = {}
    for rank, doc in enumerate(docs[:k], 1):
        for key, ref in required.items():
            if key not in ranks and str(doc.get("source") or "") == ref["source"] and compute_text_coverage([str(doc.get("content") or "")], [ref["quote"]]) == 1.0:
                ranks[key] = rank
    recall = round(len(ranks) / len(required), 4)
    return {"recall_at_5": recall, "precision_at_5": round(len(set(ranks.values())) / k, 4),
            "chunk_mrr": round(1 / min(ranks.values()), 4) if ranks else 0.0,
            "evidence_hit": float(bool(ranks)), "evidence_coverage": recall,
            "evidence_rank": min(ranks.values()) if ranks else k + 1,
            "evidence_all_in_topk": float(len(ranks) == len(required)),
            "covered_gold_ids": sorted(ranks), "coverage_status": "visible_same_source_quotes"}


# ---------------------------------------------------------------------------
# 臂执行（子进程内）
# ---------------------------------------------------------------------------

def _pipeline_state(pipeline: dict[str, Any]) -> dict[str, Any]:
    """共享执行器的顶层状态（规划/覆盖/停止原因/轮次）。"""
    keys = ("plan", "coverage", "decision", "stop_reason", "retry_count",
            "attempted_queries", "clarification", "warnings")
    return {key: pipeline[key] for key in keys if pipeline.get(key) not in (None, "", [], {})}


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """检索/生成延迟的 p50、p95。

    第一个样本通常包含 BGE/ONNX 冷启动，单独记 ``cold_start_ms`` 并从分位统计里
    剔除——否则 p95 反映的是模型加载而不是链路成本。
    """
    summary: dict[str, Any] = {}
    for key, name in (("retrieval_latency_ms", "retrieval"), ("generation_latency_ms", "generation")):
        values = [
            float(item[key])
            for item in per_query
            if isinstance(item.get(key), (int, float)) and float(item[key]) > 0
        ]
        if not values:
            continue
        cold, warm = values[0], (values[1:] if len(values) > 3 else values)
        summary[name] = {
            "n": len(values),
            "mean_ms": round(sum(warm) / len(warm), 1),
            "p50_ms": round(_percentile(warm, 0.5) or 0.0, 1),
            "p95_ms": round(_percentile(warm, 0.95) or 0.0, 1),
            "cold_start_ms": round(cold, 1),
        }
    return summary


def _arm_view(item: dict[str, Any]) -> dict[str, Any]:
    """把 per_query 记录翻成归因函数的「臂视图」。"""
    diagnostics = item.get("diagnostics") or {}
    retrieved = diagnostics.get("retrieved_chunk_ids") or ""
    if isinstance(retrieved, str):
        retrieved = [part for part in retrieved.split(";") if part]
    return {
        "metrics": item.get("metrics") or {},
        "retrieved_chunk_ids": list(retrieved),
        "coarse_top50": [diagnostics.get("coarse_candidate_chunk_ids") or []],
        "rerank_candidate_chunk_ids": diagnostics.get("rerank_candidate_chunk_ids") or [],
        "hard_filters": (diagnostics.get("filter_plan") or {}).get("hard")
        or diagnostics.get("filters")
        or {},
        "attempted_queries": diagnostics.get("attempted_queries") or [],
        "clarification": diagnostics.get("clarification") or "",
        "stop_reason": diagnostics.get("stop_reason") or "",
        "error": "; ".join(str(err.get("message")) for err in item.get("errors") or []),
    }


def _attribution(experiment: Any, results: dict[str, Any]) -> dict[str, Any]:
    """候选臂相对基线臂的失败阶段计数与样例（复用 eval200 的唯一定义）。"""
    from src.eval.eval200 import classify_failure, stage_counts

    arms = (experiment.arm_a, experiment.arm_b)
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        for item in (results.get(arm.name) or {}).get("per_query") or []:
            case_id = str(item.get("case_id") or "")
            if case_id:
                indexed.setdefault(case_id, {})[arm.name] = item
    stages: dict[str, str] = {}
    examples: dict[str, list[str]] = {}
    for case_id, pair in indexed.items():
        baseline = pair.get(arms[0].name)
        candidate = pair.get(arms[1].name)
        if not baseline or not candidate:
            continue
        record = {
            "query": candidate.get("user_input", ""),
            "stratum": candidate.get("stratum", ""),
            "question_type": candidate.get("question_type", ""),
            "gold_chunk_ids": list(candidate.get("chunk_ids") or []),
            "gold_sources": list(candidate.get("gold_sources") or []),
            "evidence_meta": list(candidate.get("evidence_meta") or []),
            "entity_hints": dict(candidate.get("entity_hints") or {}),
        }
        stage = classify_failure(record, _arm_view(candidate), _arm_view(baseline))
        if not stage:
            continue
        stages[case_id] = stage
        bucket = examples.setdefault(stage, [])
        if len(bucket) < 3:
            bucket.append(f"{case_id} [{record['stratum']}] {str(record['query'])[:60]}")
    return {
        "n": len(stages),
        "stage_counts": stage_counts(stages.values()),
        "examples": examples,
    }


def execute_arm(
    experiment: Any,
    arm: Any,
    *,
    testset_path: Path,
    fast: bool,
    sample: int,
    seed: int,
    allow_missing: bool,
    include_silver: bool = False,
    cross_check: int = 0,
) -> dict[str, Any]:
    """执行单臂：逐题检索/生成 → Ragas 批量评估 →（可选）第二裁判交叉校验 → 聚合。

    Args:
        cross_check: >0 时在**同一批 case_id** 上再用第二裁判族重跑一遍四指标，
            供父进程做「自我强化偏差」的跨模型校验；0 表示关闭。
    """
    from src.eval.metrics import compute_chunk_retrieval_metrics
    from src.eval.ragas_eval import (
        RagasBatchEvaluator,
        _compute_doc_coverage,
        _compute_evidence_metrics,
        _compute_local_retrieval_metrics,
    )
    from src.eval.test_set import TestSetLoader
    from src.eval.testset_generator import filter_entries_by_kb
    from src.index.milvus_kb import get_knowledge_base

    if experiment.kind == "ingest":
        # 入库臂：按臂开关把 files/ 文档索引进独立集合，评估在该集合上进行
        collection = _arm_collection_name(experiment, arm)
        kb = _index_arm_collection(collection, _ARM_DOCS_DIR)
    else:
        kb = get_knowledge_base()
        collection = kb.collection_name

    entries = TestSetLoader(Path(testset_path)).load_test_set()
    if testset_path.suffix.lower() == ".jsonl" and not include_silver:
        entries = [entry for entry in entries if entry.tier == "gold"]
    entries, removed = filter_entries_by_kb(entries, kb, allow_missing=allow_missing)
    if sample and 0 < sample < len(entries):
        entries = _stratified_sample(entries, sample, seed)
    if not entries:
        raise RuntimeError("测试集为空（相关文档均不在知识库）")

    from .manifest import PROTOCOL_FINGERPRINT, PROTOCOL_VERSION, group_rows
    from .retrieval import knowledge_content_fingerprints
    from .testset_generator import knowledge_base_fingerprint

    annotation_rows = []
    if testset_path.suffix.lower() == ".jsonl":
        source_rows = {str(row["id"]): row for row in (json.loads(line) for line in testset_path.read_text(encoding="utf-8").splitlines() if line.strip())}
        annotation_rows = [{**source_rows[entry.unique_id], "case_id": entry.unique_id, "pool": "evaluation"} for entry in entries]
    else:
        # 旧 Markdown 默认 Gold/approved 是旧加载器假设，新协议不能据此升级。
        for entry in entries:
            entry.tier, entry.review_status = "silver", "pending"
            annotation_rows.append({"case_id": entry.unique_id or str(entry.question_id), "query": entry.query,
                                    "required_facts": list(entry.required_facts) or [part.strip() for part in re.split(r"[。！？；\n]", entry.ground_truth) if part.strip()][:5], "question_type": entry.question_type,
                                    "expected_behavior": entry.expected_behavior, "paired_case_id": entry.paired_case_id,
                                    "tier": "silver", "review_status": "pending", "pool": "evaluation"})
    group_rows(annotation_rows)
    group_by_id = {row["case_id"]: row["sample_group_id"] for row in annotation_rows}
    annotations_by_id = {row["case_id"]: row for row in annotation_rows}
    content_fingerprints = knowledge_content_fingerprints(kb)
    try:
        kb_identity = knowledge_base_fingerprint(kb)
        kb_payload = {"metadata": kb_identity,
                      "pdf_set": content_fingerprints.get("pdf_set_fingerprint"),
                      "chunk_content": content_fingerprints.get("chunk_content_fingerprint")}
        kb_fingerprint = hashlib.sha256(json.dumps(kb_payload, sort_keys=True).encode()).hexdigest()
    except Exception as exc:
        kb_fingerprint = None
        content_fingerprints["kb_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    dataset_payload = json.dumps({"source_sha256": hashlib.sha256(testset_path.read_bytes()).hexdigest(), "selected_annotations": annotation_rows}, ensure_ascii=False, sort_keys=True)
    run_fingerprints = {"protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": PROTOCOL_FINGERPRINT,
                        "dataset_fingerprint": hashlib.sha256(dataset_payload.encode()).hexdigest(),
                        "kb_fingerprint": kb_fingerprint, **content_fingerprints}
    if any(row.get("needs_review") for row in annotation_rows):
        run_fingerprints["comparable"] = False
        run_fingerprints["annotation_review_required"] = True
    if any(row.get("tier") == "gold" and (row.get("review_status") != "approved" or not row.get("reviewer") or (row.get("provenance") or {}).get("human_approved") is False) for row in annotation_rows):
        run_fingerprints["comparable"] = False
        run_fingerprints["gold_review_unverified"] = True
    if any(content_fingerprints.get(key) == "unavailable" for key in ("pdf_set_fingerprint", "chunk_content_fingerprint")):
        run_fingerprints["comparable"] = False

    retrieval_only = bool(arm.strategy_overrides.get("retrieval_only"))
    evaluator = None if retrieval_only else RagasBatchEvaluator(kb)
    ragas_rows: list[dict[str, Any]] = []
    for entry in entries:
        errors: list[dict[str, str]] = []
        t0 = time.perf_counter()
        try:
            pipeline = _pipeline_for_arm(experiment, arm, kb, entry.query)
            docs = pipeline["docs"]
        except Exception as exc:
            docs = []
            pipeline = {"docs": [], "context_docs": [], "context_text": "", "sources": [], "diagnostics": {}}
            errors.append({"stage": "retrieval", "message": str(exc)})
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 4)

        # RAGAS 消费实际生产预算快照；块级指标仍消费原始召回身份。
        contexts = [str(d.get("content") or "") for d in pipeline.get("context_docs", []) if d.get("content")]
        sources = [
            {
                # id / child_id 供评测侧做证据块对齐；父子扩展后 id 仍是子块主键
                "id": d.get("id", "") or d.get("child_id", ""),
                "child_id": d.get("child_id", ""),
                "covered_child_ids": list(d.get("covered_child_ids") or []),
                "document_sha256": d.get("document_sha256", ""),
                "document_version": d.get("document_version", ""),
                "chunk_key": d.get("chunk_key", ""),
                "parent_id": d.get("parent_id", ""),
                "title": d.get("title", ""),
                "source": d.get("source", ""),
                "score": round(float(d.get("score", 0.0) or 0.0), 4),
                "rerank_score": (
                    round(float(d["rerank_score"]), 4)
                    if isinstance(d.get("rerank_score"), (int, float)) else None
                ),
                "content": (d.get("content") or "")[:500],
            }
            for d in docs
        ]

        answered: dict[str, Any] = {}
        if retrieval_only or errors:
            response = ""
            generation_ms = 0.0
        else:
            t0 = time.perf_counter()
            try:
                answered = _answer_for_arm(entry.query, pipeline)
                response = str(answered.get("answer") or "")
            except Exception as exc:
                response = ""
                errors.append({"stage": "generation", "message": str(exc)})
            generation_ms = round((time.perf_counter() - t0) * 1000, 4)

        refs = annotations_by_id[entry.unique_id or str(entry.question_id)].get("evidence")
        if refs is None:
            names = entry.related_doc_names or (entry.related_docs,)
            refs = [{"chunk_id": chunk_id,
                     "quote": entry.evidence_snippets[index] if index < len(entry.evidence_snippets) else "",
                     "source": names[0] if len(names) == 1 else ""}
                    for index, chunk_id in enumerate(entry.chunk_ids)]
        context_evidence = _evidence_slot_metrics(pipeline.get("context_docs") or [], refs)
        ragas_rows.append({
            "case_id": entry.unique_id or str(entry.question_id),
            "tier": entry.tier,
            "sample_group_id": group_by_id[entry.unique_id or str(entry.question_id)],
            "review_status": entry.review_status,
            "user_input": entry.query,
            "retrieved_contexts": contexts,
            "response": response,
            "reference": entry.ground_truth,
            "related_docs": entry.related_docs,
            "test_type": entry.test_type,
            "difficulty": entry.difficulty,
            "question_type": entry.question_type,
            "stratum": entry.stratum,
            "paired_case_id": entry.paired_case_id,
            "chunk_ids": list(entry.chunk_ids),
            "gold_sources": [str(ref.get("source") or "") for ref in refs or []],
            "evidence_meta": list(
                (annotations_by_id.get(entry.unique_id or str(entry.question_id), {}).get("provenance") or {}).get("evidence_meta")
                or []
            ),
            "entity_hints": dict(
                (annotations_by_id.get(entry.unique_id or str(entry.question_id), {}).get("provenance") or {}).get("entity_hints")
                or {}
            ),
            "sources": sources,
            "context_text": pipeline.get("context_text", ""),
            "context_sources": pipeline.get("sources", []),
            "context_evidence": context_evidence,
            "diagnostics": {
                **(pipeline.get("diagnostics") or {}),
                # 规划/覆盖/停止原因在共享执行器的顶层；不搬进诊断，
                # 归因与「分解/Agentic 有没有真的生效」都无从判断。
                **_pipeline_state(pipeline),
            },
            "answer_status": answered.get("answer_status", "not_run"),
            "answer_rejected": answered.get("answer_rejected"),
            "citation_validation": answered.get("citation_validation"),
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "errors": errors,
        })

    t0 = time.perf_counter()
    positive_rows = [
        row for row in ragas_rows if row.get("question_type") != "negative"
    ]
    if retrieval_only or evaluator is None or not positive_rows:
        evaluation = {
            "per_query": [{"metrics": {}} for _ in positive_rows],
            "metric_errors": {},
            "fallback_metrics": [],
        }
    else:
        evaluation = evaluator.evaluate_batch(positive_rows, fast=fast)
    ragas_ms = round((time.perf_counter() - t0) * 1000, 4)

    # 逐题：本地检索指标 + 证据块指标 + 综合分 + 分类型本地指标
    evaluated = iter(evaluation["per_query"])
    per_query = [
        ({**row, "metrics": {}} if row.get("question_type") == "negative" else {**row, **next(evaluated)})
        for row in ragas_rows
    ]
    multi_hop_diag: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if i >= len(per_query):
            break
        retrieved_sources = [s["source"] for s in per_query[i]["sources"]]
        expected_docs = entry.related_doc_names or entry.related_docs
        local = _compute_local_retrieval_metrics(
            retrieved_sources, expected_docs, k=5
        )
        local.update(_compute_doc_coverage(retrieved_sources, expected_docs, k=5))

        # 证据块级指标：检索结果里的 chunk id（父子扩展后仍保留子块身份）
        retrieved_chunk_ids = [
            str(s.get("child_id") or s.get("id") or "")
            for s in per_query[i]["sources"]
        ]
        evidence = _compute_evidence_metrics(
            retrieved_chunk_ids, entry.chunk_ids, k=5
        )
        local.update({k: v for k, v in evidence.items() if k != "evidence_rank"})
        chunk_metrics = compute_chunk_retrieval_metrics(
            retrieved_chunk_ids, entry.chunk_ids, k=5
        )
        local["recall_at_5"] = chunk_metrics["recall_at_k"]
        local["precision_at_5"] = chunk_metrics["precision_at_k"]
        local["chunk_mrr"] = chunk_metrics["mrr"]
        # 原始 child_id 精确指标与父窗口覆盖指标严格分名；排名按实际窗口槽位。
        context_evidence = per_query[i]["context_evidence"]
        local["context_evidence_recall_at_5"] = context_evidence["recall_at_5"]
        local["context_evidence_slot_precision_at_5"] = context_evidence["precision_at_5"]
        local["context_evidence_mrr"] = context_evidence["chunk_mrr"]
        per_query[i]["metrics"].update(local)
        # 长度遥测：长度偏差的原始数据（不放进 metrics，避免拉高均值误读）
        per_query[i]["answer_chars"] = len(per_query[i].get("response") or "")
        # evidence_rank 未命中记 k+1，单独放在诊断里，避免拉高均值造成误读
        per_query[i]["diagnostics"] = {
            **(per_query[i].get("diagnostics") or {}),
            "evidence_rank": evidence["evidence_rank"],
            "retrieved_doc_stems": ";".join(
                Path(source).stem for source in retrieved_sources[:5] if source
            ),
            "retrieved_chunk_ids": ";".join(cid for cid in retrieved_chunk_ids[:5] if cid),
            "exact_chunk_metric_scope": "raw_selected_child_ids_at_5",
            "context_evidence_scope": "actual_budgeted_context_slots_at_5_same_source_full_gold_quote",
            "context_covered_gold_ids": context_evidence["covered_gold_ids"],
            "context_evidence_status": context_evidence["coverage_status"],
        }

        # 多跳：相关文档全部被召回才算命中（跨文档证据覆盖）
        if entry.question_type == "multi_hop":
            per_query[i]["metrics"]["multi_hop_hit"] = float(
                local["doc_recall"] if local["doc_recall"] is not None else 0.0
            )
            multi_hop_diag.append({
                "query": entry.query,
                "expected_docs": entry.related_docs,
                "retrieved_docs": per_query[i]["diagnostics"]["retrieved_doc_stems"],
                "doc_coverage": local["doc_coverage"],
                "evidence_coverage": local["evidence_coverage"],
                "evidence_rank": evidence["evidence_rank"],
            })
        # 服务错误不按零命中或空答案拒答记分；发布出口的显式拒答标记优先。
        if any(error.get("stage") == "retrieval" for error in per_query[i].get("errors", [])):
            per_query[i]["metrics"] = {name: None for name in per_query[i]["metrics"]}
        if any(error.get("stage") == "generation" for error in per_query[i].get("errors", [])):
            per_query[i]["metrics"].update({name: None for name in GENERATION_METRICS})
        if entry.question_type == "negative" and not retrieval_only:
            rejected = per_query[i].get("answer_rejected")
            verified = per_query[i].get("answer_status") != "unverified"
            per_query[i]["metrics"]["negative_rejection"] = float(rejected) if verified and isinstance(rejected, bool) and not per_query[i].get("errors") else None
            if not verified:
                per_query[i]["evaluation_issue"] = "model_verification_unavailable"

    # 弱标注检测：多跳条目标注的相关文档从未被召回 → 疑似评测集标注不可信
    suspect_annotations = _flag_weak_multi_hop(multi_hop_diag)

    # 第二裁判族交叉校验：在同一批 case_id 上用另一个模型族的裁判重跑四指标，
    # 供父进程判断「结论是否依赖自评模型」（自我强化偏差）。默认关闭。
    crosscheck: dict[str, Any] = {
        "status": "disabled", "sample": cross_check, "n": 0,
        "metrics": {}, "metric_errors": {},
    }
    if cross_check > 0 and not retrieval_only and evaluator is not None:
        from src.core.config import get_secondary_judge_model

        secondary = get_secondary_judge_model()
        if secondary is None:
            crosscheck = {
                "status": "unavailable", "sample": cross_check, "n": 0,
                "metrics": {}, "metric_errors": {},
                "note": "未配置第二裁判模型（EVAL_JUDGE_SECONDARY_MODEL/DASHSCOPE_API_KEY）"
                        "或与主裁判同族",
            }
        else:
            # 抽样只按 case_id 排序取前 N 条：两臂条目顺序一致，因此抽到同一批题
            positive_rows = [
                row for row in ragas_rows if row.get("question_type") != "negative"
            ]
            selected = sorted(
                positive_rows, key=lambda row: str(row.get("case_id"))
            )[:cross_check]
            try:
                secondary_eval = evaluator.evaluate_batch(
                    selected,
                    fast=fast,
                    judge_model=secondary,
                    judge_label="secondary",
                )
                crosscheck = {
                    "status": "ok",
                    "sample": cross_check,
                    "n": len(selected),
                    "metrics": secondary_eval["metrics"],
                    "metric_errors": secondary_eval["metric_errors"],
                    "judge_model_name": secondary_eval.get("judge_model_name", ""),
                    "per_query": secondary_eval["per_query"],
                }
            except Exception as exc:  # 第二裁判失败不得影响主臂结果
                logger.warning("第二裁判交叉校验失败（%s）：%s", arm.name, exc)
                crosscheck = {
                    "status": "failed", "sample": cross_check, "n": 0,
                    "metrics": {}, "metric_errors": {}, "error": str(exc),
                }

    from src.core import config

    from .metrics import METRICS_VERSION
    judge_name = "objective-only" if retrieval_only else evaluation.get("judge_model_name")
    judge_payload = {"metrics_version": METRICS_VERSION, "fast": fast,
                     "judge_model": judge_name,
                     "judge_endpoint": "" if retrieval_only else config.EVAL_JUDGE_BASE_URL}
    if judge_name in (None, "", "unavailable", "unknown"):
        run_fingerprints["judge_fingerprint"] = None
        run_fingerprints["comparable"] = False
        run_fingerprints["judge_unavailable"] = True
    else:
        run_fingerprints["judge_fingerprint"] = hashlib.sha256(json.dumps(judge_payload, sort_keys=True).encode()).hexdigest()
    run_fingerprints["metrics_version"] = METRICS_VERSION
    if evaluation.get("fallback_metrics"):
        run_fingerprints["comparable"] = False
        run_fingerprints["judge_fallback_used"] = True
    if any(row.get("answer_status") == "unverified" for row in per_query):
        run_fingerprints["comparable"] = False
        run_fingerprints["answer_verification_unknown"] = True
    for row in per_query:
        row["run_fingerprints"] = dict(run_fingerprints)

    # 聚合：各指标均值（仅有限值）
    metric_names: set[str] = set()
    for item in per_query:
        metric_names.update(item["metrics"].keys())
    all_aggregates: dict[str, float | None] = {}
    for name in sorted(metric_names):
        finite = [
            item["metrics"][name]
            for item in per_query
            if isinstance(item["metrics"].get(name), (int, float))
        ]
        all_aggregates[name] = round(sum(finite) / len(finite), 4) if finite else None
    # 生成侧主指标 = RAGAS 四指标 + 负样本拒答（拒答是独立的对抗层口径，
    # 不属于 RAGAS 四指标但必须留在主表里，否则负样本失败会被埋进诊断指标）。
    primary_names = {*GENERATION_METRICS, "negative_rejection"}
    aggregates = {name: all_aggregates.get(name) for name in sorted(primary_names) if name in all_aggregates}
    diagnostic_aggregates = {
        name: value for name, value in all_aggregates.items() if name not in primary_names
    }
    if retrieval_only:
        for name in ("recall_at_5", "precision_at_5", "chunk_mrr"):
            if name in all_aggregates:
                aggregates[name] = all_aggregates[name]
                diagnostic_aggregates.pop(name, None)

    # 两条链路的只读分组（仅供报告并排展示；检索侧客观指标算法与主指标口径不变）
    generation_metrics = {
        name: all_aggregates.get(name)
        for name in GENERATION_METRICS
        if name in all_aggregates
    }
    retrieval_objective_metrics = {
        name: all_aggregates.get(name)
        for name in RETRIEVAL_OBJECTIVE_METRICS
        if name in all_aggregates
    }

    metrics_by_type, counts_by_type = _mean_by(per_query, "question_type")
    metrics_by_stratum, counts_by_stratum = _mean_by(per_query, "stratum")

    return {
        "experiment": experiment.name,
        "protocol_version": PROTOCOL_VERSION,
        "run_fingerprints": run_fingerprints,
        "independent_sample_count": len(set(group_by_id.values())),
        "arm": arm.to_dict(),
        "collection": collection,
        "testset": str(testset_path),
        "fast": fast,
        "sample": sample,
        "seed": seed,
        "cross_check": cross_check,
        "entry_count": len(entries),
        "removed_entries": [
            {"query": e.query, "missing": missing} for e, missing in removed
        ],
        "metrics": aggregates,
        "diagnostic_metrics": diagnostic_aggregates,
        "generation_metrics": generation_metrics,
        "retrieval_metrics": retrieval_objective_metrics,
        "crosscheck": crosscheck,
        "metrics_by_type": metrics_by_type,
        "metrics_by_type_counts": counts_by_type,
        "metrics_by_stratum": metrics_by_stratum,
        "metrics_by_stratum_counts": counts_by_stratum,
        "suspect_annotations": suspect_annotations,
        "multi_hop_diagnostics": multi_hop_diag,
        "per_query": per_query,
        "metric_errors": evaluation["metric_errors"],
        "fallback_metrics": evaluation["fallback_metrics"],
        "ragas_latency_ms": ragas_ms,
        "latency": _latency_summary(per_query),
        "success": True,
        "created_at": datetime.now(UTC).astimezone().isoformat(),
    }


# ---------------------------------------------------------------------------
# 对比与报告（父进程）
# ---------------------------------------------------------------------------

def _mean_by(
    per_query: list[dict[str, Any]], key: str
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, int]]]:
    """按字段分层求均值。缺失值不进均值，但样本数仍记下。"""
    values: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for item in per_query:
        name = str(item.get(key) or "")
        if not name:
            continue
        bucket = values.setdefault(name, {})
        count_bucket = counts.setdefault(name, {})
        for metric, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                bucket.setdefault(metric, []).append(float(value))
                count_bucket[metric] = count_bucket.get(metric, 0) + 1
    means = {
        name: {
            metric: round(sum(vals) / len(vals), 4)
            for metric, vals in bucket.items()
            if vals
        }
        for name, bucket in values.items()
    }
    return means, counts


def _public_metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
    """主指标加上诊断指标，报告里才能看到 context_precision 和 Recall@5。"""
    if not result:
        return {}
    merged = dict(result.get("diagnostic_metrics") or {})
    merged.update(result.get("metrics") or {})
    return merged


def _with_public_metrics(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    copied = dict(result)
    copied["metrics"] = _public_metrics(result)
    return copied


def compare_arms(
    arm_a_result: dict[str, Any] | None,
    arm_b_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """聚合双臂指标，计算 Δ / 相对变化 / 胜出。"""
    def _metrics(result: dict[str, Any] | None) -> dict[str, float | None]:
        return (result or {}).get("metrics") or {}

    a, b = _metrics(arm_a_result), _metrics(arm_b_result)
    rows: dict[str, dict[str, Any]] = {}
    for name in sorted(set(a) | set(b)):
        av, bv = a.get(name), b.get(name)
        delta = (
            round(bv - av, 4)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float))
            else None
        )
        relative_pct = (
            round(delta / av * 100, 2)
            if delta is not None and isinstance(av, (int, float)) and av
            else None
        )
        winner = None
        if delta is not None:
            if delta > 0:
                winner = "B"
            elif delta < 0:
                winner = "A"
            else:
                winner = "tie"
        rows[name] = {
            "arm_a": av, "arm_b": bv,
            "delta": delta, "relative_pct": relative_pct, "winner": winner,
        }
    return {"metrics": rows,
            "arm_a_error": (arm_a_result or {}).get("error"),
            "arm_b_error": (arm_b_result or {}).get("error")}


def run_experiment(
    experiment: Any,
    *,
    testset_path: Path,
    fast: bool,
    sample: int,
    seed: int,
    allow_missing: bool,
    outdir: Path | None,
    include_silver: bool = False,
    pairwise: int = 0,
    cross_check: int = 0,
    anchors: bool = True,
    anchor_strict: bool | None = None,
    export_human_sample: int = 0,
    bootstrap: int = 2000,
    noise_floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    """编排双臂子进程执行、偏差防护与校准，并输出报告。

    Args:
        pairwise: >0 时抽 N 题做顺序随机 + 换序复评的成对裁判（位置偏差防护）。
        cross_check: >0 时用第二裁判族在同一批题上重跑四指标（自我强化偏差防护）。
        anchors: 是否跑锚点样本（默认跑，文件不存在则自动跳过）。
        anchor_strict: 锚点越界是否直接判失败；``None`` 时读 ``EVAL_ANCHOR_STRICT``。
        export_human_sample: >0 时导出 N 条待人工标注样本。
        bootstrap: 配对统计的 Bootstrap 次数。
    """
    outdir = Path(outdir) if outdir else ROOT / "scripts" / "results" / "ab"
    run_dir = outdir / experiment.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    arms = [experiment.arm_a, experiment.arm_b]
    rng = random.Random(seed)
    ordered = list(arms) if rng.random() < 0.5 else list(reversed(arms))
    logger.info("臂执行顺序：%s", " → ".join(a.name for a in ordered))

    results: dict[str, Any] = {}
    for arm in ordered:
        arm_out = run_dir / f"{arm.name}.json"
        cmd = [
            sys.executable, "-m", "src.eval.ab_runner",
            "--experiment", experiment.name,
            "--arm", arm.name,
            "--testset", str(testset_path),
            "--out", str(arm_out),
            "--sample", str(sample),
            "--seed", str(seed),
        ]
        if fast:
            cmd.append("--fast")
        if allow_missing:
            cmd.append("--allow-missing")
        if include_silver:
            cmd.append("--include-silver")
        if cross_check > 0:
            cmd.extend(["--cross-check", str(cross_check)])
        env = {**os.environ, **arm.env_overrides}
        logger.info("执行臂 %s（env=%s）", arm.name, arm.env_overrides or "{}")
        proc = subprocess.run(
            cmd, env=env, cwd=str(ROOT),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0 or not arm_out.exists():
            results[arm.name] = {
                "error": f"returncode={proc.returncode}",
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
            continue
        try:
            results[arm.name] = json.loads(arm_out.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            results[arm.name] = {"error": f"结果解析失败：{exc}"}

    comparison = compare_arms(
        _with_public_metrics(results.get(experiment.arm_a.name)),
        _with_public_metrics(results.get(experiment.arm_b.name)),
    )

    retrieval_only = _is_retrieval_only(results, arms)
    statistics_report = _run_statistics(
        experiment, results, arms, seed=seed, bootstrap=bootstrap,
        retrieval_only=retrieval_only, noise_floor=noise_floor,
    )
    judge_bias = _run_judge_bias(results, arms)
    pairwise_report, pairwise_records = _run_pairwise(results, arms, sample=pairwise, seed=seed)
    if pairwise_records:
        _write_jsonl(run_dir / "pairwise.jsonl", pairwise_records)
    crosscheck_report = _run_crosscheck(results, arms)
    anchor_report = _run_anchors(fast=fast, strict=anchor_strict)
    human_sample_path = _run_human_sample_export(
        results, arms, run_dir, sample=export_human_sample, seed=seed
    )

    report = {
        "experiment": experiment.to_dict(),
        "testset": str(testset_path),
        "fast": fast,
        "sample": sample,
        "seed": seed,
        "arm_order": [a.name for a in ordered],
        "retrieval_only": retrieval_only,
        "arms": {
            arm.name: {
                "label": arm.label,
                "metrics": _public_metrics(results.get(arm.name)),
                "generation_metrics": (results.get(arm.name) or {}).get("generation_metrics"),
                "retrieval_metrics": (results.get(arm.name) or {}).get("retrieval_metrics"),
                "metrics_by_type": (results.get(arm.name) or {}).get("metrics_by_type"),
                "metrics_by_type_counts": (
                    results.get(arm.name) or {}
                ).get("metrics_by_type_counts"),
                "metrics_by_stratum": (results.get(arm.name) or {}).get("metrics_by_stratum"),
                "metrics_by_stratum_counts": (
                    results.get(arm.name) or {}
                ).get("metrics_by_stratum_counts"),
                # 疑似标注不可信的条目：把「检索失败」与「评测集缺陷」分开归因
                "suspect_annotations": (
                    results.get(arm.name) or {}
                ).get("suspect_annotations"),
                "entry_count": (results.get(arm.name) or {}).get("entry_count"),
                "collection": (results.get(arm.name) or {}).get("collection"),
                "error": (results.get(arm.name) or {}).get("error"),
            }
            for arm in arms
        },
        "comparison": comparison,
        "statistics": statistics_report,
        "latency": {arm.name: (results.get(arm.name) or {}).get("latency") for arm in arms},
        "attribution": _attribution(experiment, results),
        "judge_bias": {**judge_bias, "position": pairwise_report},
        "crosscheck": crosscheck_report,
        "anchors": anchor_report,
        "human_sample": (
            str(human_sample_path) if human_sample_path else None
        ),
        "created_at": datetime.now(UTC).astimezone().isoformat(),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_json(run_dir / "judge_bias.json", report["judge_bias"])
    if crosscheck_report.get("status") != "disabled":
        _write_json(run_dir / "judge_crosscheck.json", crosscheck_report)
    if anchor_report.get("status") == "ok":
        _write_json(run_dir / "anchor_report.json", anchor_report)
    (run_dir / "report.md").write_text(render_report_md(report), encoding="utf-8")
    _write_per_query_csv(run_dir, experiment, results)
    print(render_report_text(report))
    return report


# ---------------------------------------------------------------------------
# 偏差防护与校准（父进程）
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _is_retrieval_only(results: dict[str, Any], arms: list[Any]) -> bool:
    """任一拳声明了 ``retrieval_only`` 即视为只比检索（不生成答案）。"""
    for arm in arms:
        overrides = ((results.get(arm.name) or {}).get("arm") or {}).get(
            "strategy_overrides"
        ) or {}
        if overrides.get("retrieval_only"):
            return True
    return False


def _paired_records(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """把 arm 的 per_query 转成 ``statistics.compare_runs`` 需要的记录形状。"""
    rows: list[dict[str, Any]] = []
    for row in (result or {}).get("per_query") or []:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        rows.append({
            "case_id": case_id,
            "tier": str(row.get("tier") or "gold"),
            "stratum": str(row.get("stratum") or ""),
            "metrics": row.get("metrics") or {},
            "sample_group_id": row.get("sample_group_id"),
            "paired_case_id": row.get("paired_case_id"),
            "run_fingerprints": row.get("run_fingerprints") or (result or {}).get("run_fingerprints"),
        })
    return rows


def _run_statistics(
    experiment: Any,
    results: dict[str, Any],
    arms: list[Any],
    *,
    seed: int,
    bootstrap: int,
    retrieval_only: bool,
    noise_floor: dict[str, float] | None = None,
) -> dict[str, Any]:
    """配对 Bootstrap 判定：主指标 95% CI + 护栏回退检查。

    agentic-v2 要求 95% 覆盖率并校验全部指纹；旧结果保留 50% 的历史口径。
    """
    from src.eval.statistics import compare_runs, select_experiment_spec

    spec = select_experiment_spec(experiment.name, retrieval_only=retrieval_only)
    try:
        report = compare_runs(
            _paired_records(results.get(arms[0].name)),
            _paired_records(results.get(arms[1].name)),
            spec,
            bootstrap_samples=bootstrap,
            seed=seed,
            minimum_gold_coverage=0.95 if any((result or {}).get("protocol_version") == "agentic-v2" for result in results.values()) else 0.5,
            baseline_config=(results.get(arms[0].name) or {}).get("run_fingerprints"),
            candidate_config=(results.get(arms[1].name) or {}).get("run_fingerprints"),
            noise_floor=noise_floor,
        )
    except Exception as exc:  # 重复 case_id 等结构问题：报告里可见，不打断整轮
        logger.warning("配对统计失败：%s", exc)
        return {"status": "failed", "error": str(exc), "experiment": spec.name}
    return {
        "status": "ok",
        "paired_with": arms[1].name,
        "baseline_arm": arms[0].name,
        **report,
    }


def _run_judge_bias(results: dict[str, Any], arms: list[Any]) -> dict[str, Any]:
    """长度偏差（无条件、无额外 LLM 调用）。"""
    from src.core.config import (
        EVAL_LENGTH_CORR_THRESHOLD,
        EVAL_LENGTH_MATCH_RATIO,
    )
    from src.eval.judge_bias import (
        length_matched_deltas,
        length_report,
    )

    rows_by_arm = {
        arm.name: list((results.get(arm.name) or {}).get("per_query") or [])
        for arm in arms
    }
    metrics = [*GENERATION_METRICS]
    length = length_report(
        rows_by_arm, metrics, corr_threshold=EVAL_LENGTH_CORR_THRESHOLD
    )
    matched = length_matched_deltas(
        rows_by_arm[arms[0].name],
        rows_by_arm[arms[1].name],
        metrics,
        max_ratio=EVAL_LENGTH_MATCH_RATIO,
    )
    return {"length": length, "length_matched": matched}


def _pairwise_pairs(
    results: dict[str, Any],
    arms: list[Any],
) -> list[dict[str, Any]]:
    """构造两臂可配对的题（正例、双方都有非空回答）。"""
    by_arm = {
        arm.name: {
            str(row.get("case_id")): row
            for row in (results.get(arm.name) or {}).get("per_query") or []
        }
        for arm in arms
    }
    left, right = arms[0].name, arms[1].name
    pairs: list[dict[str, Any]] = []
    for case_id in sorted(set(by_arm[left]) & set(by_arm[right])):
        row_a, row_b = by_arm[left][case_id], by_arm[right][case_id]
        if str(row_a.get("question_type")) == "negative":
            continue
        if not str(row_a.get("response") or "") or not str(row_b.get("response") or ""):
            continue
        pairs.append({
            "case_id": case_id,
            "question_type": str(row_a.get("question_type") or ""),
            "question": str(row_a.get("user_input") or ""),
            "row_a": row_a,
            "row_b": row_b,
        })
    return pairs


def _run_pairwise(
    results: dict[str, Any],
    arms: list[Any],
    *,
    sample: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """成对裁判（位置偏差防护）。``sample<=0`` 时关闭。"""
    if sample <= 0:
        return {"status": "disabled", "sample": 0}, []
    from src.core.config import (
        EVAL_POSITION_FLIP_THRESHOLD,
        get_judge_model,
        primary_judge_spec,
    )
    from src.eval.human_calibration import stratified_pick
    from src.eval.judge_bias import position_report
    from src.eval.pairwise_judge import run_pairwise

    judge = get_judge_model()
    if judge is None:
        return {"status": "disabled", "sample": sample, "reason": "LLM 未配置"}, []
    pairs = _pairwise_pairs(results, arms)
    if not pairs:
        return {"status": "no_pairs", "sample": sample}, []
    selected = stratified_pick(pairs, sample, seed)
    records = run_pairwise(selected, judge, seed=seed)
    report = position_report(records, flip_threshold=EVAL_POSITION_FLIP_THRESHOLD)
    return {
        "status": "ok",
        "sample": sample,
        "n_pairs": len(selected),
        "judge_model": primary_judge_spec(),
        "arm_a": arms[0].name,
        "arm_b": arms[1].name,
        "flip_threshold": EVAL_POSITION_FLIP_THRESHOLD,
        **report,
    }, records


def _run_crosscheck(results: dict[str, Any], arms: list[Any]) -> dict[str, Any]:
    """第二裁判族交叉校验（自我强化偏差防护）。"""
    from src.core.config import (
        EVAL_JUDGE_BIAS_THRESHOLD,
        generator_spec,
        model_family,
        primary_judge_spec,
    )
    from src.eval.judge_bias import self_preference_report

    arm_status = {
        arm.name: {
            "status": (results.get(arm.name) or {}).get("crosscheck", {}).get("status", "disabled"),
            "n": (results.get(arm.name) or {}).get("crosscheck", {}).get("n", 0),
            "judge_model_name": (
                (results.get(arm.name) or {}).get("crosscheck") or {}
            ).get("judge_model_name", ""),
            "error": ((results.get(arm.name) or {}).get("crosscheck") or {}).get("error"),
        }
        for arm in arms
    }
    primary_by_arm = {
        arm.name: list((results.get(arm.name) or {}).get("per_query") or [])
        for arm in arms
    }
    secondary_by_arm = {
        arm.name: list(
            ((results.get(arm.name) or {}).get("crosscheck") or {}).get("per_query") or []
        )
        for arm in arms
    }
    if not any(secondary_by_arm.values()):
        statuses = {item["status"] for item in arm_status.values()}
        status = "unavailable" if "unavailable" in statuses else (
            "failed" if "failed" in statuses else "disabled"
        )
        return {"status": status, "arms": arm_status}

    report = self_preference_report(
        primary_by_arm,
        secondary_by_arm,
        GENERATION_METRICS,
        generator_family=model_family(generator_spec()),
        primary_judge_family=model_family(primary_judge_spec()),
        bias_threshold=EVAL_JUDGE_BIAS_THRESHOLD,
    )
    status = (
        "ok"
        if all(item["status"] == "ok" for item in arm_status.values())
        else "partial"
    )
    return {"status": status, "arms": arm_status, **report}


def _run_anchors(*, fast: bool, strict: bool | None) -> dict[str, Any]:
    """锚点样本（judge 校准）：默认只报告，``strict`` 时越界直接失败。"""
    from pathlib import Path as _Path

    from src.core.config import (
        EVAL_ANCHOR_BAND_SLACK,
        EVAL_ANCHOR_STRICT,
        EVAL_ANCHORS_PATH,
        get_judge_model,
        get_model,
    )
    from src.eval.anchors import (
        AnchorDataError,
        evaluate_anchors,
        load_anchors,
        summarize_anchors,
    )
    from src.eval.ragas_eval import RagasBatchEvaluator

    strict = EVAL_ANCHOR_STRICT if strict is None else strict
    anchor_path = _Path(EVAL_ANCHORS_PATH)
    if not anchor_path.exists():
        return {"status": "skipped", "reason": f"锚点文件不存在：{anchor_path}"}
    if get_model() is None:
        return {"status": "disabled", "reason": "LLM 未配置"}
    try:
        anchors = load_anchors(anchor_path)
    except AnchorDataError as exc:
        return {"status": "failed", "error": str(exc), "path": str(anchor_path)}

    try:
        evaluator = RagasBatchEvaluator()
        evaluated = evaluate_anchors(
            evaluator, anchors, fast=fast, judge_model=get_judge_model()
        )
    except Exception as exc:  # 锚点失败只报告，不影响主结果（strict 时由下方门禁决定）
        logger.warning("锚点评估失败：%s", exc)
        return {"status": "failed", "error": str(exc), "path": str(anchor_path)}

    report = summarize_anchors(
        anchors, evaluated["observed"], slack=EVAL_ANCHOR_BAND_SLACK
    )
    report.update({
        "status": "ok",
        "path": str(anchor_path),
        "strict": bool(strict),
        "metric_errors": evaluated.get("metric_errors") or {},
        "fallback_metrics": evaluated.get("fallback_metrics") or [],
    })
    if strict and report.get("verdict") != "ok":
        raise RuntimeError(
            f"锚点漂移导致评估失败（verdict={report.get('verdict')}）："
            f"{report.get('violations')}"
        )
    return report


def _run_human_sample_export(
    results: dict[str, Any],
    arms: list[Any],
    run_dir: Path,
    *,
    sample: int,
    seed: int,
) -> Path | None:
    """导出待人工标注样本（每周 50–100 条的对齐机制）。"""
    if sample <= 0:
        return None
    from src.eval.human_calibration import export_human_sample

    per_query_by_arm = {
        arm.name: list((results.get(arm.name) or {}).get("per_query") or [])
        for arm in arms
    }
    return export_human_sample(
        per_query_by_arm,
        out_csv=run_dir / "human_sample.csv",
        n=sample,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "faithfulness": "忠实度 faithfulness",
    "answer_relevancy": "回答相关性 answer_relevancy",
    "context_precision": "上下文精确率 context_precision",
    "context_recall": "上下文召回率 context_recall",
    "answer_correctness": "答案正确性 answer_correctness",
    "context_entity_recall": "实体召回 context_entity_recall",
    "hit_rate": "命中率 hit_rate@5",
    "mrr": "MRR@5",
    "recall_at_5": "证据 Recall@5",
    "precision_at_5": "证据 Precision@5",
    "chunk_mrr": "证据 MRR@5",
    "ndcg": "NDCG@5",
    "doc_coverage": "文档覆盖 doc_coverage",
    "doc_recall": "文档全召回 doc_recall",
    "evidence_hit": "证据块命中 evidence_hit",
    "evidence_coverage": "证据块覆盖 evidence_coverage",
    "evidence_all_in_topk": "证据块全在 top5",
    "multi_hop_hit": "多跳全召回 multi_hop_hit",
    "negative_rejection": "负样本拒绝 negative_rejection",
    "composite_score": "综合分 composite",
}


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _table_rows(report: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(report["comparison"]["metrics"].items())


def render_report_text(report: dict[str, Any]) -> str:
    arms = report["experiment"]
    lines = [
        "=" * 78,
        f"  A/B 实验：{report['experiment']['label']}（{report['experiment']['name']}）",
        f"  臂 A [{arms['arm_a']['name']}]：{arms['arm_a']['label']}",
        f"  臂 B [{arms['arm_b']['name']}]：{arms['arm_b']['label']}",
        f"  测试集：{Path(report['testset']).name} | 样本 {report['sample'] or '全量'} | "
        f"模式：{'fast' if report['fast'] else 'full'} | seed={report['seed']}",
        "=" * 78,
        f"{'指标':<40} {'臂A':>10} {'臂B':>10} {'Δ(B-A)':>10} {'相对%':>10}",
        "-" * 78,
    ]
    for name, row in _table_rows(report):
        label = _METRIC_LABELS.get(name, name)
        rel = "N/A" if row["relative_pct"] is None else f"{row['relative_pct']:.2f}%"
        lines.append(
            f"{label:<40} {_fmt(row['arm_a']):>10} {_fmt(row['arm_b']):>10} "
            f"{_fmt(row['delta']):>10} {rel:>10}  ({row['winner'] or '-'})"
        )
    errors = []
    if report["comparison"].get("arm_a_error"):
        errors.append(f"臂A 错误：{report['comparison']['arm_a_error']}")
    if report["comparison"].get("arm_b_error"):
        errors.append(f"臂B 错误：{report['comparison']['arm_b_error']}")
    statistics = report.get("statistics") or {}
    if statistics.get("status") == "ok":
        lines.append("-" * 78)
        lines.append(
            f"配对统计判定：{statistics.get('decision')} "
            f"（主指标 {statistics.get('primary_metric')}；"
            f"护栏 {', '.join(statistics.get('guardrails') or []) or '无'}）"
        )
        if statistics.get("guardrail_failures"):
            lines.append(f"护栏回退：{', '.join(statistics['guardrail_failures'])}")
    judge_bias = report.get("judge_bias") or {}
    position = judge_bias.get("position") or {}
    if position.get("status") == "ok":
        lines.append(
            f"位置偏差：换序翻转率 {position.get('order_flip_rate')}"
            f"（{'疑似' if position.get('position_bias_suspect') else '正常'}）"
        )
    length = judge_bias.get("length") or {}
    if length.get("length_bias_suspect"):
        suspects = ", ".join(
            f"{item['arm']}/{item['metric']}" for item in length.get("suspects") or []
        )
        lines.append(f"长度偏差疑似：{suspects}")
    anchors = report.get("anchors") or {}
    if anchors.get("status") == "ok":
        lines.append(
            f"锚点漂移：{anchors.get('verdict')}"
            f"（通过率 {anchors.get('pass_rate')}，检查 {anchors.get('n_checked')} 项）"
        )
    if errors:
        lines.append("-" * 78)
        lines.extend(errors)
    return "\n".join(lines) + "\n"


def render_report_md(report: dict[str, Any]) -> str:
    arms = report["experiment"]
    lines = [
        f"# A/B 实验报告：{report['experiment']['label']}",
        "",
        f"- 实验：`{report['experiment']['name']}`",
        f"- 臂 A：**{arms['arm_a']['name']}**（{arms['arm_a']['label']}）",
        f"- 臂 B：**{arms['arm_b']['name']}**（{arms['arm_b']['label']}）",
        f"- 测试集：`{Path(report['testset']).name}`；样本数：{report['sample'] or '全量'}；"
        f"生成侧指标：RAGAS 四指标（faithfulness / answer_relevancy / "
        f"context_precision / context_recall）；seed：{report['seed']}",
        f"- 臂执行顺序：{' → '.join(report['arm_order'])}",
        "",
        "| 指标 | 臂 A | 臂 B | Δ (B-A) | 相对变化 | 胜出 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in _table_rows(report):
        label = _METRIC_LABELS.get(name, name)
        rel = "N/A" if row["relative_pct"] is None else f"{row['relative_pct']:.2f}%"
        lines.append(
            f"| {label} | {_fmt(row['arm_a'])} | {_fmt(row['arm_b'])} | "
            f"{_fmt(row['delta'])} | {rel} | {row['winner'] or '—'} |"
        )
    lines.extend(_render_chain_sections(report))
    lines.extend(_render_statistics_section(report))
    lines.extend(_render_latency_section(report))
    lines.extend(_render_attribution_section(report))
    lines.extend(_render_type_breakdown(report))
    lines.extend(_render_stratum_breakdown(report))
    lines.extend(_render_suspect_annotations(report))
    lines.extend(_render_bias_sections(report))
    lines.extend(_render_anchor_section(report))
    lines.extend(_render_crosscheck_section(report))
    return "\n".join(lines) + "\n"


def _metric_row(report: dict[str, Any], name: str) -> str:
    row = (report.get("comparison", {}).get("metrics") or {}).get(name)
    if not row:
        return f"| {_METRIC_LABELS.get(name, name)} | N/A | N/A | N/A |"
    return (
        f"| {_METRIC_LABELS.get(name, name)} | {_fmt(row.get('arm_a'))} | "
        f"{_fmt(row.get('arm_b'))} | {_fmt(row.get('delta'))} |"
    )


def _render_chain_sections(report: dict[str, Any]) -> list[str]:
    """两条链路的并排视图：生成侧 RAGAS 四指标 / 检索侧客观指标。

    只做分组展示，不改任何指标算法；`recall@5`、`MRR` 等客观指标是检索能力的
    地面真值，与 judge 指标分开读。
    """
    comparison_metrics = (report.get("comparison") or {}).get("metrics") or {}
    generation = [name for name in GENERATION_METRICS if name in comparison_metrics]
    retrieval = [
        name for name in RETRIEVAL_OBJECTIVE_METRICS if name in comparison_metrics
    ]
    lines: list[str] = []
    if generation:
        lines.extend(
            ["", "## 生成侧 RAGAS 四指标", "",
             "| 指标 | 臂 A | 臂 B | Δ (B-A) |", "|---|---:|---:|---:|"]
        )
        lines.extend(_metric_row(report, name) for name in generation)
    if retrieval:
        lines.extend([
            "",
            "## 检索侧客观指标（Recall@5 / MRR 等）",
            "",
            "按标注证据的 chunk id 精确匹配计算：命中即 `id` 字符串相等。"
            "这些是**检索能力的地面真值**，与生成侧 judge 指标分开看；"
            "生成侧实验也把它们当护栏（回退即判 fail）。",
            "",
            "| 指标 | 臂 A | 臂 B | Δ (B-A) |",
            "|---|---:|---:|---:|",
        ])
        lines.extend(_metric_row(report, name) for name in retrieval)
    return lines


def _render_statistics_section(report: dict[str, Any]) -> list[str]:
    """配对 Bootstrap 判定小节。"""
    statistics = report.get("statistics") or {}
    lines = ["", "## 配对统计判定（配对 Bootstrap）", ""]
    if not statistics or statistics.get("status") == "failed":
        lines.append(
            f"未完成：{statistics.get('error') or '未运行'}"
        )
        return lines
    lines.append(
        f"- 主指标：`{statistics.get('primary_metric')}`；"
        f"护栏：{', '.join(statistics.get('guardrails') or []) or '无'}；"
        f"Bootstrap {statistics.get('bootstrap_samples')} 次（seed={statistics.get('seed')}）"
    )
    tier = statistics.get("decision_tier") or "gold"
    grade = statistics.get("evidence_grade") or tier
    lines.append(f"- 判定：**{statistics.get('decision')}**；证据等级：**{grade}**（{tier} 样本）")
    if statistics.get("noise_suppressed"):
        lines.append(
            f"- ⚠ 提升未超过噪声下限（{statistics.get('noise_floor')}），按 inconclusive 处理"
        )
    section = ((statistics.get("tiers") or {}).get(tier) or {})
    lines.append("")
    lines.append("| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |")
    lines.append("|---|---:|---:|---:|---|---|---:|")
    for name, metric in (section.get("metrics") or {}).items():
        if name not in set(GENERATION_METRICS) | set(RETRIEVAL_OBJECTIVE_METRICS):
            continue
        ci = metric.get("ci95")
        ci_text = "N/A" if not ci else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(
            f"| {name} | {_fmt(metric.get('baseline'))} | {_fmt(metric.get('candidate'))} | "
            f"{_fmt(metric.get('delta'))} | {ci_text} | {metric.get('verdict')} | "
            f"{float(metric.get('coverage') or 0.0):.1%} |"
        )
    if statistics.get("guardrail_failures"):
        lines.append("")
        lines.append(f"- ⚠ 护栏回退：{', '.join(statistics['guardrail_failures'])}")
    if statistics.get("insufficient_coverage"):
        threshold = float(statistics.get("minimum_gold_coverage") or 0.95)
        lines.append(
            f"- ⚠ 覆盖率不足（<{threshold:.0%}）：{', '.join(statistics['insufficient_coverage'])}"
        )
    return lines


def _render_latency_section(report: dict[str, Any]) -> list[str]:
    """检索/生成延迟的 p50、p95；冷启动样本已剔除并单列。"""
    latency = report.get("latency") or {}
    if not any(latency.values()):
        return []
    arm_a = report["experiment"]["arm_a"]["name"]
    arm_b = report["experiment"]["arm_b"]["name"]

    def cell(stats: dict[str, Any] | None) -> str:
        if not stats:
            return "n/a"
        return f"{stats.get('p50_ms')} / {stats.get('p95_ms')} ms（n={stats.get('n')}）"

    lines = [
        "", "## 延迟（ms，p50 / p95）", "",
        "| 阶段 | 臂 A | 臂 B |", "|---|---:|---:|",
    ]
    for stage, label in (("retrieval", "检索链路"), ("generation", "答案生成")):
        lines.append(
            f"| {label} | {cell((latency.get(arm_a) or {}).get(stage))} | "
            f"{cell((latency.get(arm_b) or {}).get(stage))} |"
        )
    lines.extend(["", "首个样本含模型冷启动，已从 p50/p95 剔除（见各臂 cold_start_ms）。"])
    return lines


def _render_attribution_section(report: dict[str, Any]) -> list[str]:
    """候选臂相对基线臂的失败阶段计数与样例。"""
    attribution = report.get("attribution") or {}
    counts = attribution.get("stage_counts") or {}
    if not counts:
        return []
    lines = ["", "## 失败阶段归因（候选臂相对基线）", "", "| 阶段 | 条数 |", "|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name, count in counts.items())
    examples = attribution.get("examples") or {}
    if examples:
        lines.extend(["", "样例：", ""])
        for stage, items in examples.items():
            lines.append(f"- **{stage}**")
            lines.extend(f"  - {item}" for item in items)
    lines.extend([
        "",
        "归因只统计「基线答对、候选答错」的题；两臂同错归 `shared_failure`，"
        "不算到优化头上。",
    ])
    return lines


def _render_bias_sections(report: dict[str, Any]) -> list[str]:
    from src.eval.judge_bias import render_bias_markdown

    judge_bias = report.get("judge_bias") or {}
    return render_bias_markdown(judge_bias)


def _render_anchor_section(report: dict[str, Any]) -> list[str]:
    from src.eval.anchors import render_anchor_markdown

    anchors = report.get("anchors") or {}
    if anchors.get("status") in {"disabled", "skipped"}:
        return render_anchor_markdown(None)
    return render_anchor_markdown(anchors)


def _render_crosscheck_section(report: dict[str, Any]) -> list[str]:
    """跨模型校验的状态（具体一致性表已由 judge_bias 小节渲染）。"""
    crosscheck = report.get("crosscheck") or {}
    lines = ["", "### 跨模型校验状态", ""]
    if crosscheck.get("status") in {None, "disabled"}:
        lines.append("未运行（用 `--cross-check N` 开启，需要第二裁判模型）。")
        return lines
    for arm, status in (crosscheck.get("arms") or {}).items():
        lines.append(
            f"- 臂 `{arm}`：{status.get('status')}（n={status.get('n', 0)}，"
            f"裁判={status.get('judge_model_name') or 'N/A'}）"
            + (f"；错误：{status.get('error')}" if status.get("error") else "")
        )
    if crosscheck.get("status") == "unavailable":
        lines.append("- ⚠ 第二裁判不可用：自我强化偏差未校验。")
    return lines


def _render_type_breakdown(report: dict[str, Any]) -> list[str]:
    """按问题类型分层表：带上样本数，n/a 指标一眼可见。

    ``multi_hop_hit`` 这类指标在单跳/负样本条目上不存在，且小样本下
    某类型可能只有 1～2 题，不写清 n 很容易把噪声当成结论。
    """
    arms = report.get("arms") or {}
    arm_a_name = report["experiment"]["arm_a"]["name"]
    arm_b_name = report["experiment"]["arm_b"]["name"]
    arm_a = arms.get(arm_a_name) or {}
    arm_b = arms.get(arm_b_name) or {}
    by_type_a = arm_a.get("metrics_by_type") or {}
    by_type_b = arm_b.get("metrics_by_type") or {}
    counts_a = arm_a.get("metrics_by_type_counts") or {}
    counts_b = arm_b.get("metrics_by_type_counts") or {}
    if not by_type_a and not by_type_b:
        return []

    lines = ["", "### 分问题类型指标", "", "| 类型 | 指标 | 臂 A | 臂 B |", "|---|---|---:|---:|"]
    for qtype in sorted(set(by_type_a) | set(by_type_b)):
        type_a = by_type_a.get(qtype) or {}
        type_b = by_type_b.get(qtype) or {}
        for metric in sorted(set(type_a) | set(type_b)):
            label = _METRIC_LABELS.get(metric, metric)
            # 两个臂的样本数应一致；取任一非空值展示
            n = (counts_a.get(qtype) or {}).get(metric) or (
                counts_b.get(qtype) or {}
            ).get(metric)
            suffix = f"（n={n}）" if n else ""
            lines.append(
                f"| {qtype} | {label}{suffix} | {_fmt(type_a.get(metric))} | "
                f"{_fmt(type_b.get(metric))} |"
            )
    return lines


def _render_stratum_breakdown(report: dict[str, Any]) -> list[str]:
    """按 explicit / cross_doc / vague / negative 拆开，只保留面试要讲的指标。"""
    keep = {"context_precision", "recall_at_5", "chunk_mrr", "mrr"}
    arms = report.get("arms") or {}
    arm_a = arms.get(report["experiment"]["arm_a"]["name"]) or {}
    arm_b = arms.get(report["experiment"]["arm_b"]["name"]) or {}
    by_a = arm_a.get("metrics_by_stratum") or {}
    by_b = arm_b.get("metrics_by_stratum") or {}
    counts_a = arm_a.get("metrics_by_stratum_counts") or {}
    counts_b = arm_b.get("metrics_by_stratum_counts") or {}
    if not by_a and not by_b:
        return []
    lines = ["", "### 分层（stratum）", "", "| 层 | 指标 | 臂 A | 臂 B |", "|---|---|---:|---:|"]
    for name in ("explicit", "cross_doc", "vague", "negative"):
        if name not in by_a and name not in by_b:
            continue
        left = by_a.get(name) or {}
        right = by_b.get(name) or {}
        for metric in sorted((set(left) | set(right)) & keep):
            label = _METRIC_LABELS.get(metric, metric)
            n = (counts_a.get(name) or {}).get(metric) or (counts_b.get(name) or {}).get(metric)
            suffix = f"（n={n}）" if n else ""
            lines.append(
                f"| {name} | {label}{suffix} | {_fmt(left.get(metric))} | {_fmt(right.get(metric))} |"
            )
    return lines if len(lines) > 5 else []


def _render_suspect_annotations(report: dict[str, Any]) -> list[str]:
    """疑似评测集标注不可信的条目（多跳标注文档一篇都没召回）。"""
    suspects: dict[str, list[dict[str, Any]]] = {}
    for arm, result in (report.get("arms") or {}).items():
        flagged = (result or {}).get("suspect_annotations") or []
        if flagged:
            suspects[arm] = flagged
    if not suspects:
        return []
    lines = [
        "",
        "### [注意] 疑似标注不可信的条目",
        "",
        "以下多跳条目标注的相关文档**一篇都没有被召回**。自动生成的跨文档多跳题是",
        "把随机抽到的两篇文档硬配成对（见 `testset_generator._generate_multi_hop`），",
        "第二篇常常与问题无关，因此这类 `multi_hop_hit=0` **不能归因为检索失败**。",
        "",
    ]
    for arm, flagged in sorted(suspects.items()):
        lines.append(f"- 臂 `{arm}`：{len(flagged)} 条")
        for item in flagged[:10]:
            lines.append(f"  - 期望文档：{item.get('expected_docs', '')}")
            lines.append(f"    - 未召回：{item.get('missing_docs', '')}")
            lines.append(f"    - 实际召回：{item.get('retrieved_docs', '') or '（空）'}")
    return lines


def _write_per_query_csv(
    run_dir: Path, experiment: Any, results: dict[str, Any]
) -> None:
    """合并双臂逐题明细为 CSV。"""
    import csv

    columns = [
        "arm", "query", "question_type", "stratum", "chunk_ids", "related_docs",
        "difficulty", "test_type", "response", "errors",
        # 生成侧 RAGAS 四指标（answer_correctness / context_entity_recall 已下线）
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        # 检索侧客观指标：文档级 + 块级（Recall@5 / Precision@5 / MRR）
        "hit_rate", "mrr", "ndcg", "doc_coverage", "doc_recall",
        "recall_at_5", "precision_at_5", "chunk_mrr",
        # 块级证据指标（测试集未标注证据块时为空 = n/a）
        "evidence_hit", "evidence_coverage", "evidence_all_in_topk",
        # 证据块排名（未命中记 k+1=6），便于区分「未召回」与「被 rerank 截断」
        "evidence_rank",
        "multi_hop_hit", "negative_rejection",
        # 长度遥测（长度偏差排查）
        "answer_chars",
        # 实测检索结果（对齐排查用）
        "retrieved_doc_stems", "retrieved_chunk_ids",
        "retrieval_latency_ms", "generation_latency_ms",
        # 归因用：规划/停止原因/补检轮次（来自共享执行器顶层状态）
        "decision", "stop_reason", "retry_count",
    ]
    with (run_dir / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for arm in (experiment.arm_a, experiment.arm_b):
            result = results.get(arm.name) or {}
            for item in result.get("per_query", []):
                row = {
                    "arm": arm.name,
                    "query": item.get("user_input", ""),
                    "question_type": item.get("question_type", ""),
                    "stratum": item.get("stratum", ""),
                    "chunk_ids": ";".join(item.get("chunk_ids", [])),
                    "related_docs": item.get("related_docs", ""),
                    "difficulty": item.get("difficulty", ""),
                    "test_type": item.get("test_type", ""),
                    "response": item.get("response", ""),
                    "errors": json.dumps(item.get("errors", []), ensure_ascii=False),
                    "retrieval_latency_ms": item.get("retrieval_latency_ms"),
                    "generation_latency_ms": item.get("generation_latency_ms"),
                }
                row.update(item.get("metrics", {}))
                row["answer_chars"] = item.get("answer_chars")
                # 诊断字段单独取，避免污染 metrics 聚合
                diagnostics = item.get("diagnostics", {}) or {}
                row["evidence_rank"] = diagnostics.get("evidence_rank")
                row["retrieved_doc_stems"] = diagnostics.get("retrieved_doc_stems", "")
                row["retrieved_chunk_ids"] = diagnostics.get("retrieved_chunk_ids", "")
                row["decision"] = diagnostics.get("decision", "")
                row["stop_reason"] = diagnostics.get("stop_reason", "")
                row["retry_count"] = diagnostics.get("retry_count", "")
                writer.writerow(row)


# ---------------------------------------------------------------------------
# 子进程入口
# ---------------------------------------------------------------------------

def _parse_arm_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B 实验臂执行器（子进程入口）")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--testset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--include-silver", action="store_true")
    parser.add_argument(
        "--cross-check", type=int, default=0,
        help="用第二裁判族在同一批题上重跑四指标（自我强化偏差校验），0=关闭",
    )
    return parser.parse_args(argv)


def arm_main(argv: list[str] | None = None) -> int:
    """子进程入口：执行单臂并写结果 JSON。"""
    from src.eval.ab_catalog import select_experiments

    args = _parse_arm_args(argv)
    experiment = select_experiments([args.experiment])[0]
    arm = next(
        a for a in (experiment.arm_a, experiment.arm_b) if a.name == args.arm
    )
    result = execute_arm(
        experiment,
        arm,
        testset_path=Path(args.testset),
        fast=args.fast,
        sample=args.sample,
        seed=args.seed,
        allow_missing=args.allow_missing,
        include_silver=args.include_silver,
        cross_check=args.cross_check,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("臂 %s 结果已写入 %s", arm.name, out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    sys.exit(arm_main())
