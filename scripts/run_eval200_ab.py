"""在 200 条评测集上跑检索链和 Agentic 两组只检索的 A/B。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.dataset import EvaluationDataset
from src.eval.eval200 import (
    classify_failure,
    is_transient_failure,
    judge_mediocre,
    parent_restore_summary,
    parent_status,
    precision_ceiling,
    render_report,
    score_retrieval,
    stage_counts,
    summarize_metric_rows,
)

DATASET_PATH = ROOT / "src" / "eval" / "data" / "research_report_eval200.jsonl"
RUN_DIR = ROOT / "scripts" / "results" / "research_report" / "20261016-eval200"
RETRIEVAL_PATH = RUN_DIR / "retrieval.jsonl"
AGENTIC_PATH = RUN_DIR / "agentic.jsonl"


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[str(row["case_id"])] = row
    return rows


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _case_view(case) -> dict:
    evidence = case.evidence
    provenance = case.provenance or {}
    return {
        "case_id": case.id,
        "stratum": str(provenance.get("stratum") or ""),
        "question_type": case.question_type,
        "query": case.query,
        "gold_chunk_ids": [item.chunk_id for item in evidence],
        "gold_sources": [item.source for item in evidence],
        "quotes": {item.chunk_id: item.quote for item in evidence},
        "evidence_meta": list(provenance.get("evidence_meta") or []),
        "entity_hints": dict(provenance.get("entity_hints") or {}),
    }


def _metrics(retrieved: list[str], gold: list[str]) -> dict:
    return score_retrieval(retrieved, gold, k=10)


def _dense_or_hybrid(kb, query: str, *, hybrid: bool) -> tuple[list[dict], dict]:
    docs = kb.hybrid_search(
        query,
        k=50,
        use_dense_only=not hybrid,
        use_rerank=hybrid,
        rerank_top_n=10,
        expand_parents=False,
        cliff_detection=False,
        fusion_mode="weighted",
    )
    diagnostics = dict(getattr(kb, "last_retrieval_diagnostics", None) or {})
    return list(docs), diagnostics


def _arm_from_docs(docs: list[dict], diagnostics: dict, gold_ids: list[str], *, error: str = "") -> dict:
    retrieved = [str(doc.get("child_id") or doc.get("id") or "") for doc in docs]
    coarse = [str(item) for item in diagnostics.get("coarse_candidate_chunk_ids") or []]
    return {
        "retrieved_chunk_ids": retrieved[:10],
        "retrieved_sources": [str(doc.get("source") or "") for doc in docs[:10]],
        "coarse_chunk_ids": coarse[:50],
        "coarse_top50": [coarse[:50]],
        "rerank_candidate_chunk_ids": [str(item) for item in diagnostics.get("rerank_candidate_chunk_ids") or []],
        "hard_filters": {},
        "rewritten": "",
        "attempted_queries": [],
        "parent_statuses": [],
        "metrics": _metrics(retrieved, gold_ids) if not error else {"precision_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0},
        "error": error,
        "rerank_observed": any(isinstance(doc.get("rerank_score"), (int, float)) for doc in docs),
    }


def _parent_statuses(kb, docs: list[dict], quotes: dict[str, str], retrieved: list[str]) -> list[str]:
    by_id = {str(doc.get("child_id") or doc.get("id") or ""): doc for doc in docs}
    statuses = []
    store = None
    for chunk_id in retrieved[:10]:
        if chunk_id not in quotes:
            continue
        doc = by_id.get(chunk_id) or {}
        parent_id = str(doc.get("parent_id") or "")
        if not parent_id:
            statuses.append(parent_status(quotes[chunk_id], None))
            continue
        try:
            store = store or kb._get_parent_store()
            row = store.get(parent_id)
        except Exception:
            statuses.append(parent_status(quotes[chunk_id], None, failed=True))
            continue
        content = None if not row else row.get("content")
        statuses.append(parent_status(quotes[chunk_id], None if content is None else str(content)))
    return statuses


def _agentic_arm(kb, query: str, gold_ids: list[str], *, enabled: bool) -> dict:
    delays = (0, 15, 40, 90)
    result: dict = {}
    for attempt, delay in enumerate(delays, 1):
        if delay:
            time.sleep(delay)
        result = _agentic_once(kb, query, gold_ids, enabled=enabled)
        blob = f"{result.get('error') or ''} {result.get('clarification') or ''}"
        if not is_transient_failure(blob):
            return result
        print(f"接口限流或超时，第 {attempt} 次后等待重试", flush=True)
    result["error"] = str(result.get("clarification") or result.get("error") or "transient")
    result["clarification"] = ""
    return result


def _agentic_once(kb, query: str, gold_ids: list[str], *, enabled: bool) -> dict:
    from src.eval.ab_runner import _pipeline_for_arm

    experiment = SimpleNamespace(name="agentic_on_off")
    arm = SimpleNamespace(
        retrieval="default",
        strategy_overrides={
            "rewrite": enabled,
            "route": enabled,
            "reflect": enabled,
            "retrieval_only": True,
            "k": 50,
            "rerank_top_n": 10,
            "use_rerank": True,
            "use_dense_only": False,
        },
    )
    try:
        pipeline = _pipeline_for_arm(experiment, arm, kb, query)
    except Exception as exc:
        failed = _arm_from_docs([], {}, gold_ids, error=f"{type(exc).__name__}: {exc}")
        return failed
    docs = list(pipeline.get("docs") or [])
    diagnostics = pipeline.get("diagnostics") or {}
    calls = diagnostics.get("retrieval_calls") or []
    coarse_groups = []
    rerank_ids: list[str] = []
    for call in calls:
        coarse = [str(item) for item in call.get("coarse_candidate_chunk_ids") or []]
        if coarse:
            coarse_groups.append(coarse[:50])
        rerank_ids.extend(str(item) for item in call.get("rerank_candidate_chunk_ids") or [])
    retrieved = [str(doc.get("child_id") or doc.get("id") or "") for doc in docs]
    hard = (diagnostics.get("filter_plan") or {}).get("hard") or pipeline.get("filters") or {}
    attempted = [str(item) for item in pipeline.get("attempted_queries") or [] if str(item).strip()]
    return {
        "retrieved_chunk_ids": retrieved[:10],
        "retrieved_sources": [str(doc.get("source") or "") for doc in docs[:10]],
        "coarse_chunk_ids": coarse_groups[0] if coarse_groups else [],
        "coarse_top50": coarse_groups,
        "rerank_candidate_chunk_ids": rerank_ids,
        "hard_filters": hard,
        "rewritten": str(pipeline.get("rewritten") or ""),
        "attempted_queries": attempted,
        "clarification": str(pipeline.get("clarification") or ""),
        "stop_reason": str(pipeline.get("stop_reason") or ""),
        "parent_statuses": [],
        "metrics": _metrics(retrieved, gold_ids),
        "error": "",
        "rerank_observed": any(isinstance(doc.get("rerank_score"), (int, float)) for doc in docs),
    }


def _complete(row: dict, keys: tuple[str, ...]) -> bool:
    for key in keys:
        arm = row.get(key) or {}
        if not arm or arm.get("error"):
            return False
        if is_transient_failure(str(arm.get("clarification") or "")):
            return False
    return True


def _run_retrieval(kb, cases: list[dict]) -> list[dict]:
    done = _load_jsonl(RETRIEVAL_PATH)
    rows = []
    checked_rerank = any((row.get("hybrid") or {}).get("rerank_observed") for row in done.values())
    for index, case in enumerate(cases, 1):
        existing = done.get(case["case_id"])
        if existing and _complete(existing, ("dense", "hybrid")):
            rows.append(existing)
            continue
        try:
            dense_docs, dense_diag = _dense_or_hybrid(kb, case["query"], hybrid=False)
            dense = _arm_from_docs(dense_docs, dense_diag, case["gold_chunk_ids"])
        except Exception as exc:
            dense = _arm_from_docs([], {}, case["gold_chunk_ids"], error=f"{type(exc).__name__}: {exc}")
        try:
            hybrid_docs, hybrid_diag = _dense_or_hybrid(kb, case["query"], hybrid=True)
            hybrid = _arm_from_docs(hybrid_docs, hybrid_diag, case["gold_chunk_ids"])
            hybrid["parent_statuses"] = _parent_statuses(kb, hybrid_docs, case["quotes"], hybrid["retrieved_chunk_ids"])
        except Exception as exc:
            hybrid = _arm_from_docs([], {}, case["gold_chunk_ids"], error=f"{type(exc).__name__}: {exc}")
        if not checked_rerank:
            if hybrid.get("error") or not hybrid.get("rerank_observed"):
                print("Reranker 没有在优化臂返回分数，停止，不估计 MRR。", file=sys.stderr)
                return []
            checked_rerank = True
        row = {key: case[key] for key in ("case_id", "stratum", "question_type", "query", "gold_chunk_ids", "gold_sources", "evidence_meta", "entity_hints")}
        row["dense"] = dense
        row["hybrid"] = hybrid
        row["parent_statuses"] = hybrid.get("parent_statuses") or []
        _append(RETRIEVAL_PATH, row)
        rows.append(row)
        print(f"检索 {index}/{len(cases)} {case['case_id']} MRR {hybrid['metrics']['mrr']}", flush=True)
    return rows


def _run_agentic(kb, cases: list[dict]) -> list[dict]:
    done = _load_jsonl(AGENTIC_PATH)
    rows = []
    for index, case in enumerate(cases, 1):
        existing = done.get(case["case_id"])
        if existing and _complete(existing, ("standard", "agentic")):
            rows.append(existing)
            continue
        standard = _agentic_arm(kb, case["query"], case["gold_chunk_ids"], enabled=False)
        agentic = _agentic_arm(kb, case["query"], case["gold_chunk_ids"], enabled=True)
        row = {key: case[key] for key in ("case_id", "stratum", "question_type", "query", "gold_chunk_ids", "gold_sources", "evidence_meta", "entity_hints")}
        row["standard"] = standard
        row["agentic"] = agentic
        _append(AGENTIC_PATH, row)
        rows.append(row)
        print(
            f"Agentic {index}/{len(cases)} {case['case_id']} "
            f"R {standard['metrics']['recall_at_10']} -> {agentic['metrics']['recall_at_10']}",
            flush=True,
        )
    return rows


def _metric_rows(records: list[dict], arm: str) -> list[dict]:
    return [{"stratum": row["stratum"], "metrics": row[arm]["metrics"]} for row in records]


def _examples(records: list[dict], arm: str, baseline: str) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = {}
    for row in records:
        stage = classify_failure(row, row[arm], row[baseline])
        if not stage:
            continue
        bucket = examples.setdefault(stage, [])
        if len(bucket) < 3:
            note = str((row.get(arm) or {}).get("clarification") or "")[:60]
            suffix = f" || {note}" if note else ""
            bucket.append(f"{row['case_id']} [{row['stratum']}] {row['query'][:80]}{suffix}")
    return examples


def _prefer_reachable_planner() -> None:
    """改写模型被限流时，改走主模型。这是改写客户端缺失时的既有回退。"""
    from src.core import config
    from src import engine as chat_service

    rewrite = config.get_rewrite_model()
    if rewrite is None:
        return
    try:
        rewrite.invoke("只回复 OK")
    except Exception as exc:
        if not is_transient_failure(str(exc)):
            raise
        print("改写模型不可用，规划与改写回退到主模型。", flush=True)
        config.get_rewrite_model = lambda: None
        chat_service.get_rewrite_model = lambda: None


def _planner_note() -> str:
    from src.core.config import DEEPSEEK_MODEL, get_rewrite_model

    if get_rewrite_model() is not None:
        return ""
    return (
        f"改写客户端不可用，规划与改写回退到主模型 {DEEPSEEK_MODEL}。"
        "qwen-turbo 在本次评测中返回 429，因此不使用它估计 Agentic 指标。"
    )


def _write_report(retrieval: list[dict], agentic: list[dict]) -> None:
    summary = {
        "precision_ceiling": precision_ceiling([
            {"evidence": [{"chunk_id": chunk_id} for chunk_id in row["gold_chunk_ids"]]}
            for row in retrieval
        ]),
        "dense": summarize_metric_rows(_metric_rows(retrieval, "dense")),
        "hybrid": summarize_metric_rows(_metric_rows(retrieval, "hybrid")),
        "standard": summarize_metric_rows(_metric_rows(agentic, "standard")),
        "agentic": summarize_metric_rows(_metric_rows(agentic, "agentic")),
        "parent_restore": parent_restore_summary([
            status for row in retrieval for status in row.get("parent_statuses") or []
        ]),
        "stage_counts": stage_counts(classify_failure(row, row["agentic"], row["standard"]) for row in agentic),
        "hybrid_stage_counts": stage_counts(classify_failure(row, row["hybrid"], row["dense"]) for row in retrieval),
        "examples": _examples(agentic, "agentic", "standard"),
        "planner_note": _planner_note(),
        "mediocre": [],
    }
    summary["mediocre"] = judge_mediocre(summary)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(summary)
    (RUN_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


def main() -> int:
    _configure_stdio()
    if not DATASET_PATH.exists():
        print(f"缺少评测集 {DATASET_PATH}", file=sys.stderr)
        return 1
    from src.core.config import get_model
    from src.index.milvus_kb import get_knowledge_base

    if get_model() is None:
        print("LLM 未配置，Agentic 改写无法运行，停止。", file=sys.stderr)
        return 1
    _prefer_reachable_planner()
    kb = get_knowledge_base()
    stats = kb.get_stats()
    if not stats.get("exists"):
        print(f"知识库不可用：{stats}", file=sys.stderr)
        return 1
    try:
        kb._get_parent_store()
    except Exception as exc:
        print(f"PostgreSQL 父块库不可用：{exc}", file=sys.stderr)
        return 1
    dataset = EvaluationDataset.load(DATASET_PATH)
    if len(dataset.cases) != 200:
        print(f"评测集不是 200 条：{len(dataset.cases)}", file=sys.stderr)
        return 1
    cases = [_case_view(case) for case in dataset.cases]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    retrieval = _run_retrieval(kb, cases)
    if len(retrieval) != 200:
        print("检索 A/B 未完成。", file=sys.stderr)
        return 1
    agentic = _run_agentic(kb, cases)
    if len(agentic) != 200:
        print("Agentic A/B 未完成。", file=sys.stderr)
        return 1
    _write_report(retrieval, agentic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
