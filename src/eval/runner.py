"""统一评测运行器：逐题执行、断点恢复并生成标准产物。"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from .dataset import EvalCase, EvaluationDataset, case_stratum
from .manifest import PROTOCOL_FINGERPRINT, PROTOCOL_VERSION, sample_group_ids
from .metrics import (
    GENERATION_METRICS,
    METRICS_VERSION,
    assess_parent_restore,
    compute_chunk_retrieval_metrics,
    compute_document_recall,
    compute_structured_fact_metrics,
    compute_text_coverage,
    fact_completion_ranks,
    summarize_parent_restore,
)

Mode = Literal["retrieval", "generation"]


def code_worktree_fingerprint(root: Path) -> str | None:
    """实际代码变更摘要，含未追踪 Python；排除结果、测试、缓存及敏感配置。"""
    try:
        paths = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
        ).decode("utf-8").split("\0")
        selected = []
        for raw in paths:
            path = Path(raw)
            if path.suffix != ".py" or any(part.lower() in {"tests", "results", "cache", "__pycache__", "secrets", ".venv"} for part in path.parts):
                continue
            if (root / path).is_file():
                selected.append((raw, hashlib.sha256((root / path).read_bytes()).hexdigest()))
        # Include deleted tracked paths: absence is a meaningful implementation change.
        payload = json.dumps(sorted(selected), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_diff_fingerprint(root: Path) -> str | None:
    """HEAD 到实际工作树的代码 diff，另纳入相关未跟踪源码。"""
    try:
        exclusions = [f":(exclude){name}/**" for name in ("tests", "scripts/results", ".venv", "secrets", "cache")]
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", "*.py", *exclusions], cwd=root, stderr=subprocess.DEVNULL)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root).decode("utf-8").split("\0")
        digest = hashlib.sha256(diff)
        for raw in sorted(untracked):
            path = Path(raw)
            if path.suffix != ".py" or any(part.lower() in {"tests", "results", "cache", "__pycache__", "secrets", ".venv"} for part in path.parts):
                continue
            digest.update(raw.encode("utf-8"))
            digest.update((root / path).read_bytes())
        return digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def parser_rules_fingerprint(root: Path) -> str:
    paths = ("src/ingest/mineru_parser.py", "src/ingest/cleaner.py",
             "src/chunking/chunker.py", "src/chunking/html_table.py",
             "src/chunking/recursive_splitter.py")
    digest = hashlib.sha256()
    for raw in paths:
        path = root / raw
        digest.update(raw.encode("utf-8"))
        digest.update(path.read_bytes() if path.is_file() else b"unavailable")
    return digest.hexdigest()


@dataclass(frozen=True)
class RetrievalOutput:
    """一次检索的完整输出；候选集用于诊断，results 用于正式计分。"""

    candidates: list[dict[str, Any]]
    results: list[dict[str, Any]]
    cliff_triggered: bool = False
    pre_cliff_depth: int | None = None
    post_cliff_depth: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    context_results: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class EvaluationStrategy:
    """把具体检索/生成实现注入统一运行器。"""

    name: str
    retrieve: Callable[[str], RetrievalOutput]
    generate: Callable[[EvalCase, list[str]], str] | None = None
    evaluate_answer: Callable[[EvalCase, list[str], str], dict[str, float | None]] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    resolve_parent: Callable[[str], str | None] | None = None
    evaluate_negative: Callable[[EvalCase], dict[str, Any]] | None = None


@dataclass(frozen=True)
class EvaluationRun:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    output_dir: Path


def _chunk_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("chunk_id") or item.get("pk") or "")


def _content(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("text") or "")


#: 拒答识别关键词。真实的"拒答"远不止「无法回答」这一种说法，实测中模型
#: 更常说「检索到的内容**未提供**…」「**未提及**…」——漏词会把正确拒答判成
#: 幻觉作答，直接把负样本拒答正确率打对折（18 篇语料基线上真实发生过：
#: 「检索到的内容未提供宁德时代…具体数据」被误判为未拒答）。
REJECTION_KEYWORDS = (
    "无法回答", "无法给出可靠回答", "无法给出", "拒绝回答", "不能回答",
    "未检索到", "没有检索到", "暂未检索", "未能检索", "未找到",
    "暂无相关", "无相关信息", "无相关数据", "没有相关",
    "未提供", "未提及", "未包含", "未涉及", "不含",
    "知识库中未", "现有知识库中", "语料中没有", "不足以支撑", "无法支撑",
)


def negative_rejection_score(answer: str) -> float:
    """负样本是否正确拒绝（含拒绝表述或空回答）返回 1.0，否则 0.0。

    这是**关键词代理指标**，不是语义判定：答案里出现"…未提供…"即视为拒答，
    因此必须同时保留原始答案供人工复核（见 scripts/baseline_e2e.py 的 JSONL）。
    """
    if not answer or not answer.strip():
        return 1.0
    return 1.0 if any(kw in answer for kw in REJECTION_KEYWORDS) else 0.0


def _negative_rejection_score(answer: str) -> float:
    """向后兼容的私有别名。"""
    return negative_rejection_score(answer)


class EvaluationRunner:
    """对同一数据集运行检索或生成评测，并在每题后落盘。"""

    def run(
        self,
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        checkpoint_dir: str | Path,
        *,
        include_silver: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationRun:
        if mode not in {"retrieval", "generation"}:
            raise ValueError(f"不支持的评测模式：{mode}")
        if mode == "generation" and (strategy.generate is None or strategy.evaluate_answer is None):
            raise ValueError("生成评测必须提供 generate 和 evaluate_answer")

        dataset.validate()
        output_dir = Path(checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / "records.jsonl"
        selected = self._select_cases(dataset, mode, include_silver)
        if mode == "retrieval" and strategy.evaluate_negative is not None:
            selected = list(dataset.cases)
        metadata = dict(metadata or {})
        grouping = sample_group_ids(list(dataset.cases))
        grouping.update(metadata.get("sample_group_ids") or {})
        provenance = self._run_provenance(selected, strategy, metadata)
        self._validate_checkpoint(output_dir, dataset, strategy, mode, selected, metadata, provenance)

        existing = self._read_records(records_path)
        completed = {record["case_id"] for record in existing if record.get("status") == "completed"}
        for case in selected:
            if case.id in completed:
                continue
            record = self._run_case(case, strategy, mode)
            record["sample_group_id"] = grouping[case.id]
            record["run_fingerprints"] = {
                key: provenance[key] for key in (
                    "dataset_fingerprint", "kb_fingerprint", "metrics_version", "annotation_schema_versions",
                    "code_worktree_sha256", "code_diff_sha256", "pdf_set_fingerprint",
                    "chunk_content_fingerprint", "parser_clean_chunk_rules",
                    "protocol_version", "protocol_fingerprint", "judge_fingerprint",
                )
            }
            with records_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        by_id = {record["case_id"]: record for record in self._read_records(records_path)}
        records = [by_id[case.id] for case in selected if case.id in by_id]
        summary = self._summarize(records, dataset, strategy, mode, include_silver, metadata or {})
        summary.update(provenance)
        self._write_outputs(output_dir, records, summary)
        return EvaluationRun(records, summary, output_dir)

    @staticmethod
    def _select_cases(dataset: EvaluationDataset, mode: Mode, include_silver: bool) -> list[EvalCase]:
        if mode == "retrieval":
            return [case for case in dataset.cases if case.question_type != "negative"]
        tiers = {"gold", "silver"} if include_silver else {"gold"}
        return [case for case in dataset.cases if case.tier in tiers]

    @staticmethod
    def _dataset_fingerprint(cases: list[EvalCase]) -> str:
        payload = "\n".join(
            json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) for case in cases
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_checkpoint(
        self,
        output_dir: Path,
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        selected: list[EvalCase],
        metadata: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        config_path = output_dir / "run_config.json"
        value = {
            "mode": mode,
            "strategy": strategy.name,
            "strategy_config": strategy.config,
            "dataset_fingerprint": self._dataset_fingerprint(selected),
            "metadata": metadata,
            **provenance,
        }
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != value:
                raise ValueError("检查点目录已属于不同的数据集、策略或运行配置")
            return
        config_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def _run_provenance(
        cls, cases: list[EvalCase], strategy: EvaluationStrategy, metadata: dict[str, Any],
    ) -> dict[str, Any]:
        from src.core import config

        def setting(name: str, fallback: Any) -> Any:
            value = metadata.get(name, strategy.config.get(name))
            return fallback if value is None else value

        kb_fingerprints = {str(case.provenance.get("kb_fingerprint")) for case in cases if case.provenance.get("kb_fingerprint")}
        root = config.PROJECT_ROOT
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8", errors="replace").strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, encoding="utf-8", errors="replace").strip())
        except (OSError, subprocess.CalledProcessError):
            head, dirty = None, None
        return {
            "dataset_fingerprint": cls._dataset_fingerprint(cases),
            "protocol_version": setting("protocol_version", None),
            "protocol_fingerprint": setting("protocol_fingerprint", PROTOCOL_FINGERPRINT if setting("protocol_version", None) == PROTOCOL_VERSION else None),
            "judge_fingerprint": setting("judge_fingerprint", None),
            "kb_fingerprint": metadata.get("kb_fingerprint") or strategy.config.get("kb_fingerprint") or (next(iter(kb_fingerprints)) if len(kb_fingerprints) == 1 else None),
            "embedding_model": setting("embedding_model", config.EMBEDDING_MODEL),
            "embedding_version": setting("embedding_version", config.EMBEDDING_VERSION),
            "embedding_dim": setting("embedding_dim", config.EMBEDDING_DIM),
            "embedding_pooling": setting("embedding_pooling", config.EMBEDDING_POOLING),
            "chunk_max_tokens": setting("chunk_max_tokens", config.DOCLING_CHUNK_MAX_TOKENS),
            "chunk_overlap_tokens": metadata.get("chunk_overlap_tokens", strategy.config.get("chunk_overlap_tokens", config.CHILD_OVERLAP_TOKENS)),
            "fusion_mode": "dense" if strategy.config.get("use_dense_only") else setting("fusion_mode", config.RETRIEVAL_FUSION_MODE),
            "dense_weight": setting("dense_weight", config.RETRIEVAL_DENSE_WEIGHT),
            "sparse_weight": setting("sparse_weight", config.RETRIEVAL_SPARSE_WEIGHT),
            "git_head": head,
            "git_dirty": dirty,
            "code_worktree_sha256": code_worktree_fingerprint(root),
            "code_diff_sha256": code_diff_fingerprint(root),
            "annotation_schema_versions": sorted({case.schema_version for case in cases}),
            "metrics_version": METRICS_VERSION,
            "pdf_set_fingerprint": setting("pdf_set_fingerprint", None),
            "chunk_content_fingerprint": setting("chunk_content_fingerprint", "unavailable"),
            "parser_clean_chunk_rules": setting("parser_clean_chunk_rules", parser_rules_fingerprint(root)),
        }

    @staticmethod
    def _read_records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _run_case(
        self, case: EvalCase, strategy: EvaluationStrategy, mode: Mode
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if case.question_type == "negative" and strategy.evaluate_negative is not None:
                outcome = strategy.evaluate_negative(case)
                answer = str(outcome.get("answer") or "")
                rejected = outcome.get("answer_rejected")
                if outcome.get("evaluation_status") == "unverified" or outcome.get("answer_status") == "unverified":
                    rejection = None
                    outcome["rejection_verification"] = "coverage_or_claim_verification_unknown"
                elif isinstance(rejected, bool):
                    rejection = float(rejected)
                    outcome["rejection_verification"] = "explicit_production_flag"
                elif answer.strip():
                    rejection = negative_rejection_score(answer)
                    outcome["rejection_verification"] = "answer_keyword_proxy"
                else:
                    rejection = None
                    outcome["rejection_verification"] = "unverified_empty_answer"
                return self._record(
                    case, status="completed", metrics={"negative_rejection": rejection},
                    diagnostics=outcome, answer=answer,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3), error=None,
                )
            retrieval = strategy.retrieve(case.query)
            result_ids = [_chunk_id(item) for item in retrieval.results if _chunk_id(item)]
            diagnostics: dict[str, Any] = {
                "candidate_chunk_ids": [
                    _chunk_id(item) for item in retrieval.candidates if _chunk_id(item)
                ],
                "top5_chunk_ids": result_ids[:5],
                "evidence_ranks": self._evidence_ranks(case, result_ids),
                "cliff_triggered": retrieval.cliff_triggered,
                "pre_cliff_depth": retrieval.pre_cliff_depth,
                "post_cliff_depth": retrieval.post_cliff_depth,
                **retrieval.diagnostics,
            }
            stage = "dense" if diagnostics.get("fusion_mode", strategy.config.get("fusion_mode")) == "dense" or strategy.config.get("use_dense_only") else "fusion"
            diagnostics["candidate_stage"] = stage
            diagnostics["resolved_fusion_mode"] = diagnostics.get("fusion_mode") or ("dense" if stage == "dense" else strategy.config.get("fusion_mode"))
            diagnostics.setdefault("cliff_enabled", bool(strategy.config.get("cliff_detection")))
            final_sources = {item.get("source") for item in retrieval.results if item.get("source")}
            diagnostics.update(final_candidate_count=len(retrieval.results), final_source_count=len(final_sources), final_duplicate_source_slots=len(retrieval.results) - len(final_sources))
            coarse_ids = diagnostics.get("coarse_candidate_chunk_ids")
            # 只有旧适配器返回的 candidates 能证明未截断时，才将其当作粗排池。
            if coarse_ids is None and not strategy.config.get("use_rerank"):
                coarse_ids = diagnostics["candidate_chunk_ids"]
                diagnostics["coarse_candidate_chunk_ids"] = coarse_ids
            gold_ids = list(dict.fromkeys(item.chunk_id for item in case.evidence))
            rerank_ids = diagnostics.get("rerank_candidate_chunk_ids")
            if not strategy.config.get("use_rerank"):
                rerank_ids = None
            diagnostics["evidence_survival"] = {
                stage: [cid for cid in gold_ids if cid in coarse_ids] if coarse_ids is not None else "unavailable",
                "rerank": [cid for cid in gold_ids if cid in rerank_ids] if rerank_ids is not None else "unavailable",
                "final_top5": [cid for cid in gold_ids if cid in result_ids[:5]],
                "sparse": "unavailable",
            }
            diagnostics["gold_evidence_count"] = len(gold_ids)
            answer: str | None = None
            if mode == "retrieval":
                raw = compute_chunk_retrieval_metrics(
                    result_ids, [item.chunk_id for item in case.evidence], k=5
                )
                metrics = {
                    "recall_at_5": raw["recall_at_k"],
                    "precision_at_5": raw["precision_at_k"],
                    "mrr": raw["mrr"],
                    "document_recall_at_5": compute_document_recall([str(item.get("source") or "") for item in retrieval.results], [item.source for item in case.evidence]),
                    "exact_candidate_recall_at_20": compute_chunk_retrieval_metrics(coarse_ids, gold_ids, k=20)["recall_at_k"] if coarse_ids is not None and len(coarse_ids) >= 20 else None,
                    "exact_candidate_recall_at_50": compute_chunk_retrieval_metrics(coarse_ids, gold_ids, k=50)["recall_at_k"] if coarse_ids is not None and len(coarse_ids) >= 50 else None,
                    "exact_candidate_recall_at_actual_depth": compute_chunk_retrieval_metrics(coarse_ids, gold_ids, k=max(1, len(coarse_ids)))["recall_at_k"] if coarse_ids is not None else None,
                    "quote_coverage_at_5": compute_text_coverage([_content(item) for item in retrieval.results[:5]], [item.quote for item in case.evidence]),
                    "required_fact_coverage_at_5": compute_text_coverage([_content(item) for item in retrieval.results[:5]], list(case.required_facts)),
                }
                metrics.update(compute_structured_fact_metrics(case, retrieval.results))
                if retrieval.context_results is not None:
                    metrics["context_required_fact_coverage"] = compute_structured_fact_metrics(case, retrieval.context_results, k=len(retrieval.context_results))["structured_required_fact_coverage_at_5"] if case.facts else compute_text_coverage([_content(item) for item in retrieval.context_results], list(case.required_facts))
                    diagnostics["context_results"] = retrieval.context_results
                diagnostics["coarse_actual_depth"] = len(coarse_ids) if coarse_ids is not None else None
                diagnostics["fact_group_survival"] = {
                    "final": {"actual_depth": len(retrieval.results), "completion_ranks": fact_completion_ranks(case, retrieval.results)},
                }
                for stage_name, prefix in (("global", "coarse"), ("document", "document"), ("rerank", "rerank")):
                    stage_docs = diagnostics.get(f"{prefix}_candidates")
                    stage_ids = diagnostics.get(f"{prefix}_candidate_chunk_ids")
                    if stage_docs is None and stage_ids is not None:
                        # 老适配器仅有ID，仍可证明映射ID命中，不能证明引文恢复。
                        source_by_id = {ref.chunk_id: ref.source for ref in case.evidence if ref.chunk_id}
                        stage_docs = [{"id": cid, "source": source_by_id.get(cid, "")} for cid in stage_ids]
                    diagnostics["fact_group_survival"][stage_name] = {
                        "actual_depth": len(stage_docs) if stage_docs is not None else None,
                        "completion_ranks": fact_completion_ranks(case, stage_docs) if stage_docs is not None else "unavailable",
                        "observation": "actual_content" if diagnostics.get(f"{prefix}_candidates") is not None else "mapped_ids_only" if stage_ids is not None else "unavailable",
                    }
                if strategy.resolve_parent is not None:
                    diagnostics["parent_restore_statuses"] = self._parent_restore_statuses(
                        case, retrieval.results, strategy.resolve_parent
                    )
            else:
                # 输出期整表展开：与线上 chat 链路同一口径（检索/重排只吃
                # 紧凑摘要块，拼上下文前才按 table_id 取回整表）
                if retrieval.context_results is not None:
                    expansions = deepcopy(retrieval.context_results)
                else:
                    from src.retrieve.pipeline import prepare_context
                    expansions = prepare_context(deepcopy(retrieval.results))["context_docs"]
                contexts = [_content(item) for item in expansions if _content(item)]
                assert strategy.generate is not None
                answer = strategy.generate(case, contexts)
                if case.question_type == "negative":
                    metrics = {"negative_rejection": _negative_rejection_score(answer)}
                else:
                    assert strategy.evaluate_answer is not None
                    evaluated = strategy.evaluate_answer(case, contexts, answer)
                    # 生成侧统一 RAGAS 四指标（与 ragas_eval.GENERATION_METRICS 同口径）
                    metrics = {
                        key: evaluated.get(key)
                        for key in GENERATION_METRICS
                    }
                    diagnostics["generation_metrics"] = {
                        key: value for key, value in evaluated.items() if key not in metrics
                    }
                diagnostics["answer_chars"] = len(answer or "")
            return self._record(
                case, status="completed", metrics=metrics, diagnostics=diagnostics,
                answer=answer, latency_ms=round((time.perf_counter() - started) * 1000, 3),
                error=None,
            )
        except Exception as exc:  # 单题失败也必须写检查点，避免整批静默丢失
            return self._record(
                case, status="failed", metrics={}, diagnostics={}, answer=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _record(
        case: EvalCase,
        *,
        status: str,
        metrics: dict[str, Any],
        diagnostics: dict[str, Any],
        answer: str | None,
        latency_ms: float,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "case_id": case.id,
            "tier": case.tier,
            "question_type": case.question_type,
            "stratum": case_stratum(case),
            "paired_case_id": str((case.provenance or {}).get("paired_case_id") or ""),
            "query": case.query,
            "expected_behavior": case.expected_behavior,
            "status": status,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "answer": answer,
            "latency_ms": latency_ms,
            "error": error,
        }

    @staticmethod
    def _parent_restore_statuses(
        case: EvalCase,
        results: list[dict[str, Any]],
        resolve_parent: Callable[[str], str | None],
    ) -> list[str]:
        """只统计落在 top-5 里的证据块。未命中的块不进入还原率分母。"""
        by_id = {_chunk_id(item): item for item in results if _chunk_id(item)}
        top_ids = [_chunk_id(item) for item in results[:5] if _chunk_id(item)]
        statuses: list[str] = []
        for evidence in case.evidence:
            if evidence.chunk_id not in top_ids:
                continue
            parent_id = str(by_id.get(evidence.chunk_id, {}).get("parent_id") or "")
            if not parent_id:
                statuses.append("no_parent")
                continue
            try:
                content = resolve_parent(parent_id)
            except Exception:
                statuses.append("lookup_failed")
                continue
            statuses.append(assess_parent_restore(evidence.quote, content))
        return statuses

    @staticmethod
    def _evidence_ranks(case: EvalCase, result_ids: list[str]) -> dict[str, int | None]:
        ranks = {chunk_id: rank for rank, chunk_id in enumerate(result_ids, 1)}
        return {evidence.chunk_id: ranks.get(evidence.chunk_id) for evidence in case.evidence}

    @staticmethod
    def _summarize(
        records: list[dict[str, Any]],
        dataset: EvaluationDataset,
        strategy: EvaluationStrategy,
        mode: Mode,
        include_silver: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        metric_names = sorted({key for row in records for key in row.get("metrics", {})})
        metrics: dict[str, float | None] = {}
        coverage: dict[str, float] = {}
        for name in metric_names:
            values = [
                row["metrics"][name]
                for row in records
                if row.get("metrics", {}).get(name) is not None
            ]
            metrics[name] = round(mean(values), 6) if values else None
            coverage[name] = round(len(values) / len(records), 6) if records else 0.0
        tiers: dict[str, Any] = {}
        for tier in ("gold", "silver"):
            subset = [row for row in records if row["tier"] == tier]
            tiers[tier] = {
                "n": len(subset),
                "metrics": {
                    name: round(mean(values), 6) if values else None
                    for name in metric_names
                    if (values := [
                        row["metrics"][name]
                        for row in subset
                        if row.get("metrics", {}).get(name) is not None
                    ])
                },
            }
        independent = [
            row for row in records if row.get("question_type") != "negative" and row.get("stratum") in {"explicit", "cross_doc"}
        ]
        # 两条链路的只读分组：生成侧（RAGAS 四指标）与检索侧客观层（Recall@5/MRR 等）。
        # 分组只用于报告展示，不改变任何指标算法与主指标口径。
        generation_names = [name for name in metric_names if name in GENERATION_METRICS]
        generation_metrics = _mean_metrics(records, generation_names)
        retrieval_metrics = _mean_metrics(
            records, [name for name in metric_names if name not in GENERATION_METRICS]
        )
        return {
            "mode": mode,
            "strategy": strategy.name,
            "strategy_config": strategy.config,
            "dataset_source": str(dataset.source_path) if dataset.source_path else None,
            "include_silver": include_silver,
            "n": len(records),
            "completed": sum(row["status"] == "completed" for row in records),
            "failed": sum(row["status"] == "failed" for row in records),
            "metrics": metrics,
            "metric_coverage": coverage,
            "generation_metrics": generation_metrics,
            "retrieval_metrics": retrieval_metrics,
            "tiers": tiers,
            "independent_positive": {
                "n": len(independent),
                "note": "不含 vague 配对题，也不含负样本",
                "metrics": _mean_metrics(independent, metric_names),
                "metric_coverage": _metric_coverage(independent, metric_names),
                "parent_restore": _parent_restore_summary(independent),
                "retrieval_diagnostics": _diagnostics_summary(independent),
            },
            "strata": _strata_summary(records, metric_names),
            "vague_pairs": _vague_pair_deltas(records),
            "parent_restore": _parent_restore_summary(records),
            "metadata": metadata,
            "retrieval_diagnostics": _diagnostics_summary(records),
        }

    def _write_outputs(
        self, output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        fields = [
            "case_id", "tier", "question_type", "stratum", "paired_case_id", "status", "query",
            "recall_at_5", "precision_at_5", "mrr",
            "document_recall_at_5", "exact_candidate_recall_at_20", "exact_candidate_recall_at_50",
            "exact_candidate_recall_at_actual_depth", "coarse_actual_depth",
            "evidence_group_recall_at_5", "fact_mrr", "structured_required_fact_coverage_at_5",
            "all_required_documents_at_5", "context_required_fact_coverage", "fact_group_survival",
            "quote_coverage_at_5", "required_fact_coverage_at_5",
            "candidate_stage", "resolved_fusion_mode", "cliff_enabled",
            "coarse_candidate_count", "coarse_source_count", "coarse_duplicate_source_slots",
            "rerank_candidate_count", "rerank_source_count", "rerank_duplicate_source_slots",
            "final_candidate_count", "final_source_count", "final_duplicate_source_slots",
            "coarse_candidate_chunk_ids", "rerank_candidate_chunk_ids", "evidence_survival",
            "faithfulness", "answer_relevancy", "context_precision", "context_recall",
            "negative_rejection", "answer_chars",
            "latency_ms", "error",
        ]
        with (output_dir / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    **{key: record.get(key) for key in fields},
                    **{key: value for key, value in record.get("metrics", {}).items() if key in fields and key not in record},
                    "answer_chars": (record.get("diagnostics") or {}).get("answer_chars"),
                    **{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in (record.get("diagnostics") or {}).items() if key in fields and key not in record},
                })
        lines = _render_retrieval_report(summary)
        (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_metrics(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name in metric_names:
        values = [
            row["metrics"][name]
            for row in rows
            if row.get("metrics", {}).get(name) is not None
        ]
        result[name] = round(mean(values), 6) if values else None
    return result


def _metric_coverage(rows: list[dict[str, Any]], names: list[str]) -> dict[str, float]:
    return {name: round(sum(row.get("metrics", {}).get(name) is not None for row in rows) / len(rows), 6) if rows else 0.0 for name in names}


def _diagnostics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [f"{stage}_{suffix}" for stage in ("coarse", "rerank", "final") for suffix in ("candidate_count", "source_count", "duplicate_source_slots")]
    diagnostics = [row.get("diagnostics") or {} for row in rows]
    modes: dict[str, int] = {}
    for item in diagnostics:
        mode = item.get("resolved_fusion_mode")
        if mode:
            modes[mode] = modes.get(mode, 0) + 1
    survival_recall: dict[str, float | None] = {}
    survival_coverage: dict[str, float] = {}
    for stage in ("dense", "fusion", "rerank", "final_top5"):
        values = [len(survived) / item["gold_evidence_count"] for item in diagnostics
                  if item.get("gold_evidence_count") and isinstance(survived := (item.get("evidence_survival") or {}).get(stage), list)]
        survival_recall[stage] = round(mean(values), 6) if values else None
        survival_coverage[stage] = round(len(values) / len(rows), 6) if rows else 0.0
    return {
        "means": {name: round(mean(values), 6) if (values := [item[name] for item in diagnostics if isinstance(item.get(name), (int, float))]) else None for name in names},
        "sparse_survival": "unavailable",
        "coverage": {name: round(sum(isinstance(item.get(name), (int, float)) for item in diagnostics) / len(rows), 6) if rows else 0.0 for name in names},
        "resolved_fusion_modes": modes,
        "cliff_enabled_count": sum(item.get("cliff_enabled") is True for item in diagnostics),
        "cliff_enabled_coverage": round(sum(isinstance(item.get("cliff_enabled"), bool) for item in diagnostics) / len(rows), 6) if rows else 0.0,
        "survival_recall": survival_recall,
        "survival_coverage": survival_coverage,
    }


def _parent_restore_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: list[str] = []
    for row in rows:
        statuses.extend((row.get("diagnostics") or {}).get("parent_restore_statuses") or [])
    return summarize_parent_restore(statuses)


def _strata_summary(records: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    names = sorted({row.get("stratum") or "" for row in records if row.get("stratum")})
    summary: dict[str, Any] = {}
    for name in names:
        subset = [row for row in records if row.get("stratum") == name]
        summary[name] = {
            "n": len(subset),
            "metrics": _mean_metrics(subset, metric_names),
            "metric_coverage": _metric_coverage(subset, metric_names),
            "parent_restore": _parent_restore_summary(subset),
            "retrieval_diagnostics": _diagnostics_summary(subset),
        }
    return summary


def _vague_pair_deltas(records: list[dict[str, Any]]) -> dict[str, Any]:
    """模糊题减去其显式配对题。正值表示模糊问法比原题更高。"""
    by_id = {row["case_id"]: row for row in records}
    per_metric: dict[str, list[float]] = {}
    pairs: list[dict[str, Any]] = []
    for row in records:
        paired_id = row.get("paired_case_id") or ""
        base = by_id.get(paired_id)
        if not paired_id or base is None:
            continue
        item: dict[str, Any] = {"case_id": row["case_id"], "paired_case_id": paired_id}
        for metric in ("recall_at_5", "mrr"):
            vague = (row.get("metrics") or {}).get(metric)
            explicit = (base.get("metrics") or {}).get(metric)
            delta = None
            if isinstance(vague, (int, float)) and isinstance(explicit, (int, float)):
                delta = round(float(vague) - float(explicit), 6)
                per_metric.setdefault(metric, []).append(delta)
            item[metric] = {"vague": vague, "pair": explicit, "delta": delta}
        pairs.append(item)
    return {
        "n": len(pairs),
        "note": "这 10 题与原题共享证据，差值不能并进独立正例的 Recall",
        "mean_delta": {
            metric: round(mean(values), 6) if values else None
            for metric, values in per_metric.items()
        },
        "pairs": pairs,
    }


def _fmt_metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def _render_retrieval_report(summary: dict[str, Any]) -> list[str]:
    independent = summary.get("independent_positive") or {}
    lines = [
        f"# 评测报告：{summary['strategy']}",
        "",
        f"- 模式：{summary['mode']}",
        f"- 样本数：{summary['n']}（含模糊配对；检索主指标看下面的独立正例）",
        f"- 完成/失败：{summary['completed']}/{summary['failed']}",
        "",
        "## 独立正例（explicit + cross_doc）",
        "",
        independent.get("note") or "",
        "",
        f"- 样本数：{independent.get('n', 0)}",
        "",
        "| 指标 | 均值 | 覆盖率 |",
        "|---|---:|---:|",
    ]
    for name, value in (independent.get("metrics") or {}).items():
        coverage = (independent.get("metric_coverage") or {}).get(name)
        lines.append(f"| {name} | {_fmt_metric(value)} | {'N/A' if coverage is None else f'{coverage:.1%}'} |")
    parent = independent.get("parent_restore") or {}
    if parent.get("hits"):
        lines.extend([
            "",
            f"- 父块还原 parent_restore_rate：{_fmt_metric(parent.get('rate'))} "
            f"（命中 {parent.get('hits')} 块，还原 {parent.get('restored')}，"
            f"未替换 {parent.get('not_replaced')}）",
        ])
    lines.extend(["", "## 分层", ""])
    strata = summary.get("strata") or {}
    if not strata:
        lines.append("无分层标签。")
    else:
        names = list(summary.get("metrics") or {})
        lines.extend(["| 层 | n | 指标 | 均值 | 覆盖率 |", "|---|---:|---|---:|---:|"])
        for name, bucket in strata.items():
            metrics = bucket.get("metrics") or {}
            for metric_name in names:
                coverage = (bucket.get("metric_coverage") or {}).get(metric_name)
                lines.append(f"| {name} | {bucket.get('n', 0)} | {metric_name} | {_fmt_metric(metrics.get(metric_name))} | {'N/A' if coverage is None else f'{coverage:.1%}'} |")
    vague = summary.get("vague_pairs") or {}
    if vague.get("n"):
        lines.extend([
            "",
            "## 模糊题相对配对题",
            "",
            vague.get("note") or "",
            "",
            f"- 配对数：{vague.get('n')}",
        ])
        for metric, delta in (vague.get("mean_delta") or {}).items():
            lines.append(f"- {metric} 平均差值（模糊 - 配对）：{_fmt_metric(delta)}")
    overall_parent = summary.get("parent_restore") or {}
    if overall_parent.get("hits"):
        lines.extend([
            "",
            "## 父块还原（全部命中证据块）",
            "",
            f"- rate：{_fmt_metric(overall_parent.get('rate'))}",
            f"- hits / restored / not_replaced / missing / no_parent："
            f"{overall_parent.get('hits')} / {overall_parent.get('restored')} / "
            f"{overall_parent.get('not_replaced')} / {overall_parent.get('missing')} / "
            f"{overall_parent.get('no_parent')}",
        ])
    lines.extend(["", "## 全量均值（含模糊配对，不作为检索主指标）", "", "| 指标 | 均值 | 覆盖率 |", "|---|---:|---:|"])
    for name, value in summary["metrics"].items():
        lines.append(f"| {name} | {_fmt_metric(value)} | {summary['metric_coverage'][name]:.1%} |")
    generation = summary.get("generation_metrics") or {}
    retrieval = summary.get("retrieval_metrics") or {}
    if generation:
        lines.extend(["", "## 生成侧 RAGAS 四指标", "", "| 指标 | 均值 |", "|---|---:|"])
        for name, value in generation.items():
            lines.append(f"| {name} | {_fmt_metric(value)} |")
    if retrieval:
        lines.extend([
            "",
            "## 检索侧客观指标（Recall@5 / MRR 等）",
            "",
            "块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；",
            "quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。",
            "与上面的 RAGAS judge 指标分开读。",
            "",
            "| 指标 | 均值 |",
            "|---|---:|",
        ])
        for name, value in retrieval.items():
            lines.append(f"| {name} | {_fmt_metric(value)} |")

    lines.extend(["", "## 检索阶段诊断", "", "| 阶段字段 | 均值 |", "|---|---:|"])
    for name, value in (summary.get("retrieval_diagnostics") or {}).get("means", {}).items():
        lines.append(f"| {name} | {_fmt_metric(value)} |")
    for label, bucket in [("全部", summary), ("独立正例", independent), *list(strata.items())]:
        diagnostic = bucket.get("retrieval_diagnostics") or {}
        lines.append(f"- {label} resolved fusion modes：{json.dumps(diagnostic.get('resolved_fusion_modes', {}), ensure_ascii=False)}；cliff enabled：{diagnostic.get('cliff_enabled_count', 0)}")
        for stage, value in diagnostic.get("survival_recall", {}).items():
            lines.append(f"- {label} {stage} evidence survival recall：{_fmt_metric(value)}；覆盖率：{diagnostic['survival_coverage'][stage]:.1%}")
    lines.extend(["", "- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。",
                  "- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。",
                  "", "## 运行可复现配置", ""])
    for name in ("dataset_fingerprint", "kb_fingerprint", "metrics_version", "annotation_schema_versions", "pdf_set_fingerprint", "chunk_content_fingerprint", "parser_clean_chunk_rules", "code_worktree_sha256", "code_diff_sha256", "embedding_model", "embedding_version", "embedding_dim", "embedding_pooling", "chunk_max_tokens", "chunk_overlap_tokens", "fusion_mode", "dense_weight", "sparse_weight", "git_head", "git_dirty"):
        lines.append(f"- {name}：{summary.get(name)}")
    return lines
