"""补跑 interview30 里中断的 Agentic 臂和三个检索消融，并写出总报告。

检索层和标准臂沿用已有结果，不再重跑。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.ab_catalog import select_experiments
from finance_rag.src.eval.ab_runner import (
    _public_metrics,
    _render_stratum_breakdown,
    _with_public_metrics,
    compare_arms,
    run_experiment,
)
from finance_rag.src.eval.interview_dataset import INTERVIEW30_PATH
from finance_rag.src.eval.interview_report import render_interview_report

RUN_DIR = ROOT / "scripts" / "results" / "eval" / "interview30" / "20260929-222710"
AGENTIC_DIR = RUN_DIR / "ab" / "agentic_on_off" / "20260929-223042"
DATASET = INTERVIEW30_PATH
LAYERS = ["L0", "L1", "L2", "L3", "L5"]
ABLATIONS = ("route_on_off", "rewrite_on_off", "reflect_on_off")
COMMANDS = [
    "python scripts/build_interview30.py",
    "python scripts/evaluate.py dataset validate --dataset finance_rag/src/eval/data/research_report_interview30.jsonl --online",
    "python scripts/run_interview30.py",
    "python scripts/resume_interview30.py",
]


def main() -> int:
    agentic_out = AGENTIC_DIR / "agentic.json"
    if not agentic_out.exists():
        cmd = [
            sys.executable, "-m", "finance_rag.src.eval.ab_runner",
            "--experiment", "agentic_on_off",
            "--arm", "agentic",
            "--testset", str(DATASET),
            "--out", str(agentic_out),
            "--sample", "0",
            "--seed", "7",
            "--fast",
        ]
        print("running agentic arm", flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0 or not agentic_out.exists():
            print(f"agentic arm failed: {proc.returncode}", file=sys.stderr)
            return 1
        print("agentic arm done", flush=True)

    standard = json.loads((AGENTIC_DIR / "standard.json").read_text(encoding="utf-8"))
    agentic = json.loads(agentic_out.read_text(encoding="utf-8"))
    experiment = select_experiments(["agentic_on_off"])[0]
    comparison = compare_arms(
        _with_public_metrics(standard),
        _with_public_metrics(agentic),
    )
    agentic_report = _arm_report(experiment, standard, agentic, comparison)
    payload: dict = {
        "run_id": "20260929-222710",
        "dataset": "finance_rag/src/eval/data/research_report_interview30.jsonl",
        "commands": COMMANDS,
        "layers": [
            json.loads((RUN_DIR / layer / "summary.json").read_text(encoding="utf-8"))
            for layer in LAYERS
        ],
        "agentic": _compact_ab(agentic_report),
        "ablations": [],
        "note": "数字来自这次运行。简历里的 Recall@5=90%、MRR@5=1.0、上下文精确率 +14.1% 不是这套 18 篇研报、30 题协议的结果。",
    }

    for name in ABLATIONS:
        try:
            print(f"running {name}", flush=True)
            report = run_experiment(
                select_experiments([name])[0],
                testset_path=DATASET,
                fast=True,
                sample=0,
                seed=7,
                allow_missing=False,
                outdir=RUN_DIR / "ab",
                include_silver=False,
            )
            payload["ablations"].append({"name": name, **_compact_ab(report)})
            print(f"{name} done", flush=True)
        except Exception as exc:
            payload["ablations"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{name}: {exc}", file=sys.stderr)

    path = _write(payload)
    print(path, flush=True)
    failed = any(item.get("error") for item in payload["ablations"])
    return 1 if failed else 0


def _arm_report(experiment, standard: dict, agentic: dict, comparison: dict) -> dict:
    def _arm_block(arm, result: dict) -> dict:
        return {
            "label": arm.label,
            "metrics": _public_metrics(result),
            "metrics_by_stratum": result.get("metrics_by_stratum"),
            "metrics_by_stratum_counts": result.get("metrics_by_stratum_counts"),
            "entry_count": result.get("entry_count"),
            "error": result.get("error"),
        }

    return {
        "experiment": experiment.to_dict(),
        "comparison": comparison,
        "arms": {
            experiment.arm_a.name: _arm_block(experiment.arm_a, standard),
            experiment.arm_b.name: _arm_block(experiment.arm_b, agentic),
        },
    }


def _compact_ab(report: dict) -> dict:
    comparison = report.get("comparison") or {}
    arm_error = comparison.get("arm_a_error") or comparison.get("arm_b_error")
    return {
        "comparison": {"metrics": comparison.get("metrics") or {}},
        "arm_error": arm_error,
        "stratum_markdown": "\n".join(_render_stratum_breakdown(report)),
    }


def _write(payload: dict) -> Path:
    (RUN_DIR / "payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    path = RUN_DIR / "report.md"
    path.write_text(render_interview_report(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
