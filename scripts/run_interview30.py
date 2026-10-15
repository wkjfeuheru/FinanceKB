"""实跑 30 题：检索四级、Agentic 对照、三个检索消融，并写出面试报告。"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.ab_catalog import select_experiments
from finance_rag.src.eval.ab_runner import run_experiment
from finance_rag.src.eval.dataset import EvaluationDataset
from finance_rag.src.eval.interview_dataset import INTERVIEW30_PATH
from finance_rag.src.eval.interview_report import render_interview_report
from finance_rag.src.eval.retrieval import LAYER_SPECS, build_retrieval_strategy
from finance_rag.src.eval.runner import EvaluationRunner

DATASET = INTERVIEW30_PATH
LAYERS = ["L0", "L1", "L2", "L3", "L5"]
ABLATIONS = ("route_on_off", "rewrite_on_off", "reflect_on_off")
COMMANDS = [
    "python scripts/build_interview30.py",
    "python scripts/evaluate.py dataset validate --dataset finance_rag/src/eval/data/research_report_interview30.jsonl --online",
    "python scripts/run_interview30.py",
]


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = ROOT / "scripts" / "results" / "eval" / "interview30" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "run_id": run_id,
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "commands": COMMANDS,
        "layers": [],
        "ablations": [],
        "note": "数字来自这次运行。简历里的 Recall@5=90%、MRR@5=1.0、上下文精确率 +14.1% 不是这套 18 篇研报、30 题协议的结果。",
    }
    try:
        from finance_rag.src.eval.testset_generator import build_evidence_resolver
        from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

        dataset = EvaluationDataset.load(DATASET)
        kb = get_knowledge_base()
        dataset.validate(evidence_resolver=build_evidence_resolver(kb))
    except Exception as exc:
        payload["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        payload["agentic_error"] = payload["retrieval_error"]
        payload["ablation_error"] = payload["retrieval_error"]
        _write(outdir, payload)
        print(payload["retrieval_error"], file=sys.stderr)
        traceback.print_exc()
        return 1

    runner = EvaluationRunner()
    try:
        for layer in LAYERS:
            strategy = build_retrieval_strategy(kb, layer, candidate_k=LAYER_SPECS[layer]["k"], top_k=5)
            result = runner.run(dataset, strategy, "retrieval", outdir / layer)
            payload["layers"].append(result.summary)
            print(f"[{layer}] n={result.summary['n']} independent={result.summary['independent_positive']['n']}")
    except Exception as exc:
        payload["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        print(payload["retrieval_error"], file=sys.stderr)
        traceback.print_exc()

    try:
        agentic = run_experiment(
            select_experiments(["agentic_on_off"])[0],
            testset_path=DATASET,
            fast=True,
            sample=0,
            seed=7,
            allow_missing=False,
            outdir=outdir / "ab",
            include_silver=False,
        )
        payload["agentic"] = _compact_ab(agentic)
    except Exception as exc:
        payload["agentic_error"] = f"{type(exc).__name__}: {exc}"
        print(payload["agentic_error"], file=sys.stderr)
        traceback.print_exc()

    for name in ABLATIONS:
        try:
            report = run_experiment(
                select_experiments([name])[0],
                testset_path=DATASET,
                fast=True,
                sample=0,
                seed=7,
                allow_missing=False,
                outdir=outdir / "ab",
                include_silver=False,
            )
            payload["ablations"].append({"name": name, **_compact_ab(report)})
        except Exception as exc:
            payload["ablations"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    path = _write(outdir, payload)
    print(path)
    failed = bool(payload.get("retrieval_error") or payload.get("agentic_error") or any(
        item.get("error") for item in payload["ablations"]
    ))
    return 1 if failed else 0


def _compact_ab(report: dict) -> dict:
    comparison = report.get("comparison") or {}
    arm_error = comparison.get("arm_a_error") or comparison.get("arm_b_error")
    return {
        "comparison": {"metrics": comparison.get("metrics") or {}},
        "arm_error": arm_error,
        "stratum_markdown": _stratum_from_report(report),
    }


def _stratum_from_report(report: dict) -> str:
    from finance_rag.src.eval.ab_runner import _render_stratum_breakdown

    return "\n".join(_render_stratum_breakdown(report))


def _write(outdir: Path, payload: dict) -> Path:
    (outdir / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    path = outdir / "report.md"
    path.write_text(render_interview_report(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
