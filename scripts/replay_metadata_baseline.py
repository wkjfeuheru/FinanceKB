"""只读回放 before PRODUCTION 的固定旧过滤条件，不新增生产策略。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.dataset import EvaluationDataset
from finance_rag.src.eval.retrieval import _run_async, adapt_production_output, knowledge_content_fingerprints
from finance_rag.src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput
from finance_rag.src.rag.retrieval.filter_policy import validate_explicit_filters


def _sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_baseline(baseline_dir: str | Path, dataset: EvaluationDataset) -> dict[str, Any]:
    """逐题校验原身份、原条件，失败样本保留在回放分母中。"""
    dataset.validate()
    baseline_dir = Path(baseline_dir)
    config = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    if config.get("strategy") != "PRODUCTION" or config.get("mode") != "retrieval":
        raise ValueError("基线必须是 before PRODUCTION 检索运行")
    snapshot = EvaluationDataset.load(baseline_dir / "dataset_snapshot.jsonl")
    original_fingerprint = EvaluationRunner._dataset_fingerprint(list(snapshot.cases))
    current_fingerprint = EvaluationRunner._dataset_fingerprint(list(dataset.cases))
    if original_fingerprint != current_fingerprint or config.get("dataset_fingerprint") != original_fingerprint:
        raise ValueError("数据集身份/标注与 before PRODUCTION snapshot 或指纹不符")

    records_bytes = (baseline_dir / "records.jsonl").read_bytes()
    records = [json.loads(line) for line in records_bytes.decode("utf-8").splitlines() if line.strip()]
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_id[row.get("case_id")].append(row)
    controls = {}
    for case in dataset.cases:
        if case.question_type == "negative":
            continue
        matches = by_id.get(case.id, [])
        row = matches[0] if len(matches) == 1 else {}
        diagnostics = row.get("diagnostics") or {}
        filters = deepcopy(diagnostics.get("filters"))
        control = {"case_id": case.id, "query": case.query, "original_filters": filters,
                   "original_filter_sha256": _sha(filters), "failure_reason": None}
        try:
            if len(matches) != 1:
                raise ValueError("基线 case_id 缺失或重复")
            if row.get("query") != case.query:
                raise ValueError("基线原问句与 case_id 对应不符")
            if row.get("status") != "completed":
                raise ValueError("基线样本未完成，不能证明原条件")
            if "filters" not in diagnostics or not isinstance(filters, dict):
                raise ValueError("基线缺少可证明的显式 filters 对象")
            # 验证但不替换原值，传给生产入口的是原条件的独立副本。
            validate_explicit_filters(filters)
            record_fingerprint = (row.get("run_fingerprints") or {}).get("dataset_fingerprint")
            if record_fingerprint and record_fingerprint != original_fingerprint:
                raise ValueError("基线 record 数据集指纹不符")
        except ValueError as exc:
            control["failure_reason"] = str(exc)
        controls[case.id] = control
    return {"baseline_run_dir": str(baseline_dir.resolve()),
            "baseline_records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "baseline_dataset_fingerprint": original_fingerprint,
            "baseline_kb_fingerprint": config.get("kb_fingerprint"),
            "baseline_provenance": config, "cases": controls}


def build_replay_strategy(plan: dict[str, Any], kb: Any) -> EvaluationStrategy:
    from finance_rag.src.services import chat_service

    by_query = {control["query"]: control for control in plan["cases"].values()}

    def retrieve(query: str) -> RetrievalOutput:
        control = by_query[query]
        if control["failure_reason"]:
            raise ValueError(control["failure_reason"])
        original = deepcopy(control["original_filters"])
        kwargs = {"filters": original, "infer_filters": False, "k": 50, "rerank_top_n": 5}
        control["actual_call"] = deepcopy(kwargs)
        try:
            pipeline = _run_async(lambda: chat_service.retrieve_documents(query, **kwargs))
            control["actual_filters"] = deepcopy(pipeline.get("filters"))
            diagnostic = pipeline.get("diagnostics") or {}
            control["filter_attempts"] = deepcopy(diagnostic.get("filter_attempts") or [])
            control["retrieval_calls"] = deepcopy(diagnostic.get("retrieval_calls") or [])
            expected = validate_explicit_filters(control["original_filters"])
            if pipeline.get("filters") != expected:
                raise ValueError("实际过滤条件与固定原条件不符")
            for attempt in control["filter_attempts"] + control["retrieval_calls"]:
                if "filters" in attempt and attempt["filters"] != expected:
                    raise ValueError("检索分支改变了固定原条件")
            output = adapt_production_output(pipeline)
            output.diagnostics["metadata_replay"] = {
                "baseline_records_sha256": plan["baseline_records_sha256"],
                "case_id": control["case_id"], "original_filters": deepcopy(control["original_filters"]),
                "original_filter_sha256": control["original_filter_sha256"],
                "actual_call": deepcopy(control["actual_call"]),
            }
            return output
        except Exception as exc:
            control["failure_reason"] = f"{type(exc).__name__}: {exc}"
            raise

    return EvaluationStrategy(
        name="METADATA_REPLAY", retrieve=retrieve,
        config={"k": 50, "top_k": 5, "use_rerank": True, "infer_filters": False,
                "production_path": "chat_service.retrieve_documents", "embedding_model": getattr(kb, "embedding_model", None),
                "baseline_records_sha256": plan["baseline_records_sha256"],
                "original_filter_sha256_by_case": {key: value["original_filter_sha256"] for key, value in plan["cases"].items()}},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定旧过滤条件的只读诊断回放（正例），不执行生产生成")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from finance_rag.src.eval.cli import _dataset_version
    from finance_rag.src.eval.testset_generator import knowledge_base_fingerprint
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    path = Path(args.dataset)
    version = _dataset_version(path, None)
    if any(Path(value).name != value or value in {".", ".."} for value in (version, args.run_id)):
        raise ValueError("dataset_version 和 run_id 必须是单个目录名")
    output = Path(args.outdir) / version / f"{args.run_id}-METADATA_REPLAY"
    plan: dict[str, Any] = {"baseline_run_dir": str(Path(args.baseline_run_dir).resolve()), "cases": {}}
    try:
        dataset = EvaluationDataset.load(path)
        invalid = [case.id for case in dataset.cases
                   if (case.tier == "gold" and case.review_status != "approved")
                   or (case.tier == "silver" and case.review_status != "auto_pass")]
        if invalid:
            raise ValueError(f"回放仅接受 approved Gold / auto_pass Silver：{invalid}")
        plan = load_baseline(args.baseline_run_dir, dataset)
        kb = get_knowledge_base()
        current_fingerprint = knowledge_base_fingerprint(kb)
        manifest_path = path.with_suffix(".manifest.json")
        expected_fingerprint = None
        if manifest_path.exists():
            expected_fingerprint = json.loads(manifest_path.read_text(encoding="utf-8")).get("kb_fingerprint")
        if expected_fingerprint and expected_fingerprint != current_fingerprint:
            raise ValueError(f"知识库指纹漂移：数据集={expected_fingerprint} 当前={current_fingerprint}")
        metadata = {key: deepcopy(plan[key]) for key in (
            "baseline_run_dir", "baseline_records_sha256", "baseline_dataset_fingerprint", "baseline_kb_fingerprint", "baseline_provenance",
        )}
        metadata.update(dataset_version=version, kb_fingerprint=current_fingerprint,
                        expected_kb_fingerprint=expected_fingerprint, comparable=False,
                        comparison_note="固定旧条件诊断；不同策略/KB/代码仅作逐题描述性对照，禁止直接显著性比较",
                        **knowledge_content_fingerprints(kb))
        run = EvaluationRunner().run(dataset, build_replay_strategy(plan, kb), "retrieval", output, metadata=metadata)
        dataset.save(output / "dataset_snapshot.jsonl")
        for row in run.records:
            control = plan["cases"][row["case_id"]]
            # 恢复检查点时从原结果恢复实际控制，不重新推断或重新调用。
            if row["status"] == "completed":
                diagnostic = row.get("diagnostics") or {}
                control.update(actual_call=(diagnostic.get("metadata_replay") or {}).get("actual_call"),
                               actual_filters=diagnostic.get("filters"),
                               filter_attempts=diagnostic.get("filter_attempts") or [],
                               retrieval_calls=diagnostic.get("retrieval_calls") or [])
            else:
                control["failure_reason"] = row.get("error") or control["failure_reason"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "replay_controls.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[METADATA_REPLAY] 正例={run.summary['n']} 完成={run.summary['completed']} 失败={run.summary['failed']} → {output}")
        return 1 if run.summary["failed"] else 0
    except Exception as exc:
        # 门禁/初始化失败同样落盘，不冒称有效评分或成功回放。
        failure = {**plan, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
        output.mkdir(parents=True, exist_ok=True)
        (output / "replay_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[METADATA_REPLAY] 失败：{failure['failure_reason']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
