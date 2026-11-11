"""金融 RAG 统一评测命令行。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.config import PROJECT_ROOT

from .dataset import DatasetValidationError, EvaluationDataset
from .human_calibration import DEFAULT_HUMAN_SAMPLE_SIZE
from .statistics import EXPERIMENT_SPECS, compare_runs

ROOT = PROJECT_ROOT
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RESULTS = ROOT / "scripts" / "results" / "eval"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _records_path(path: str | Path) -> Path:
    value = Path(path)
    return value / "records.jsonl" if value.is_dir() else value


def _write_comparison(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# A/B 对比：{report['experiment']}", "",
        f"- 判定：{report['decision']}",
        f"- 可比性：{report.get('comparability_status', 'unavailable')}；原因：{', '.join(report.get('comparability_reasons') or []) or '无'}",
        f"- 主指标：{report['primary_metric']}",
        f"- 护栏：{', '.join(report['guardrails'])}",
        f"- Bootstrap：{report['bootstrap_samples']} 次（seed={report['seed']}）", "",
    ]
    for tier in ("gold", "silver"):
        section = report["tiers"][tier]
        lines += [f"## {tier.title()}（n={section['n']}）", "", "| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |", "|---|---:|---:|---:|---|---|---:|"]
        for name, metric in section["metrics"].items():
            ci = "N/A" if metric["ci95"] is None else f"[{metric['ci95'][0]:.4f}, {metric['ci95'][1]:.4f}]"
            fmt = lambda value: "N/A" if value is None else f"{value:.4f}"  # noqa: E731
            lines.append(
                f"| {name} | {fmt(metric['baseline'])} | {fmt(metric['candidate'])} | "
                f"{fmt(metric['delta'])} | {ci} | {metric['verdict']} | {metric['coverage']:.1%} |"
            )
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "largest_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tier", "metric", "case_id", "delta"])
        writer.writeheader()
        for tier, section in report["tiers"].items():
            for metric, values in section["metrics"].items():
                for change in values["largest_changes"]:
                    writer.writerow({"tier": tier, "metric": metric, **change})


def _dataset_version(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        return str(json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_version"])
    return path.stem


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.dataset_action == "generate":
        from .testset_generator import (
            build_evidence_resolver,
            generate_evaluation_dataset,
            knowledge_base_fingerprint,
        )

        output = Path(args.out)
        if args.from_markdown:
            from src.index.milvus_kb import get_knowledge_base

            from .migration import migrate_markdown_dataset

            kb = get_knowledge_base()
            dataset = migrate_markdown_dataset(
                args.from_markdown,
                build_evidence_resolver(kb),
                kb_fingerprint=knowledge_base_fingerprint(kb),
            )
            dataset.save(output)
        else:
            dataset, output = generate_evaluation_dataset(
                count=args.count, seed=args.seed, per_doc_limit=args.per_doc, out_path=output
            )
        review = Path(args.review_out) if args.review_out else output.with_suffix(".review.csv")
        dataset.export_review(review)
        print(f"已生成 {len(dataset.cases)} 条 JSONL 候选：{output}")
        print(f"人工审阅表：{review}")
        return 0

    dataset = EvaluationDataset.load(args.dataset)
    if args.dataset_action == "validate":
        resolver = None
        structured_resolver = None
        if args.online:
            from src.index.milvus_kb import get_knowledge_base

            from .testset_generator import build_evidence_resolver
            kb = get_knowledge_base()
            if any(case.schema_version == "2.0" for case in dataset.cases):
                from .retrieval import build_online_evidence_resolver
                structured_resolver = build_online_evidence_resolver(kb)
            else:
                resolver = build_evidence_resolver(kb)
        audit = dataset.validate(evidence_resolver=resolver, resolve_evidence=structured_resolver)
        print(f"校验通过：{audit.case_count} 条，在线核对证据 {audit.checked_evidence} 条")
        return 0
    if args.dataset_action == "export-review":
        dataset.export_review(args.out)
        print(f"已导出审阅表：{args.out}")
        return 0
    if args.dataset_action == "import-review":
        updated = dataset.import_review(args.review)
        updated.save(args.out)
        print(f"已导入审核结果：{args.out}")
        return 0
    if args.dataset_action == "freeze":
        fingerprint = args.kb_fingerprint
        if not fingerprint:
            from src.index.milvus_kb import get_knowledge_base

            from .testset_generator import knowledge_base_fingerprint
            fingerprint = knowledge_base_fingerprint(get_knowledge_base())
        manifest = dataset.freeze(
            args.out,
            kb_fingerprint=fingerprint,
            dataset_version=args.version,
            max_source_share=args.max_source_share,
        )
        print(f"已冻结 {manifest['case_count']} 条：{args.out}")
        print(f"数据集版本：{manifest['dataset_version']}；知识库指纹：{manifest['kb_fingerprint']}")
        return 0
    raise ValueError(args.dataset_action)


def _cmd_retrieval(args: argparse.Namespace) -> int:
    from src.index.milvus_kb import get_knowledge_base

    from .retrieval import LAYER_SPECS, build_retrieval_strategy, knowledge_content_fingerprints
    from .runner import EvaluationRunner
    from .testset_generator import knowledge_base_fingerprint

    dataset_path = Path(args.dataset)
    dataset = EvaluationDataset.load(dataset_path)
    invalid_status = [
        case.id for case in dataset.cases
        if (case.tier == "gold" and case.review_status != "approved")
        or (case.tier == "silver" and case.review_status != "auto_pass")
    ]
    if invalid_status and not args.diagnostic:
        raise DatasetValidationError(
            f"正式运行仅接受 approved Gold / auto_pass Silver；不合格条目：{invalid_status[:10]}"
        )
    if args.sample and args.sample < len(dataset.cases):
        cases = random.Random(args.seed).sample(list(dataset.cases), args.sample)
        dataset = EvaluationDataset(cases, source_path=dataset_path)

    kb = get_knowledge_base()
    current_fingerprint = knowledge_base_fingerprint(kb)
    content_fingerprints = knowledge_content_fingerprints(kb)
    manifest_path = dataset_path.with_suffix(".manifest.json")
    expected_fingerprint = None
    if manifest_path.exists():
        expected_fingerprint = json.loads(manifest_path.read_text(encoding="utf-8")).get("kb_fingerprint")
    if expected_fingerprint and expected_fingerprint != current_fingerprint and not args.allow_drift:
        raise DatasetValidationError(
            f"知识库指纹漂移：数据集={expected_fingerprint} 当前={current_fingerprint}"
        )
    comparable = not bool(expected_fingerprint and expected_fingerprint != current_fingerprint)

    version = _dataset_version(dataset_path, args.dataset_version)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    layers = [layer.upper() for layer in args.strategy]
    unknown = sorted(set(layers) - (set(LAYER_SPECS) | {"PRODUCTION"}))
    if unknown:
        raise ValueError(f"未知检索层：{unknown}")
    runner = EvaluationRunner()
    run_records: dict[str, list[dict[str, Any]]] = {}
    run_summaries: dict[str, dict[str, Any]] = {}
    for order, layer in enumerate(layers, 1):
        output = Path(args.outdir) / version / f"{run_id}-{layer}"
        strategy = build_retrieval_strategy(
            kb, layer, candidate_k=args.candidate_k, top_k=5
        )
        result = runner.run(
            dataset,
            strategy,
            "retrieval",
            output,
            metadata={
                "dataset_version": version,
                "kb_fingerprint": current_fingerprint,
                "expected_kb_fingerprint": expected_fingerprint,
                "comparable": comparable,
                "seed": args.seed,
                "execution_order": order,
                "execution_plan": layers,
                **content_fingerprints,
            },
        )
        dataset.save(output / "dataset_snapshot.jsonl")
        run_records[layer] = result.records
        run_summaries[layer] = result.summary
        print(f"[{layer}] n={result.summary['n']} → {output}")

    pairs = (
        ("L0", "L1", "hybrid_vs_dense"),
        ("L2", "L3", "rerank_on_off"),
        ("L3", "L5", "cliff_on_off"),
    )
    for baseline, candidate, experiment in pairs:
        if baseline not in run_records or candidate not in run_records:
            continue
        report = compare_runs(
            run_records[baseline], run_records[candidate], EXPERIMENT_SPECS[experiment],
            bootstrap_samples=args.bootstrap, seed=args.seed,
            baseline_config=run_summaries[baseline], candidate_config=run_summaries[candidate],
        )
        comparison_dir = Path(args.outdir) / version / f"{run_id}-{experiment}"
        _write_comparison(report, comparison_dir)
        print(f"[{experiment}] {report['decision']} → {comparison_dir}")
    if not comparable:
        print("警告：诊断运行检测到知识库指纹漂移，结果标记为不可比较")
    return 0


def _cmd_ab(args: argparse.Namespace) -> int:
    baseline = _read_jsonl(_records_path(args.baseline))
    candidate = _read_jsonl(_records_path(args.candidate))
    report = compare_runs(
        baseline,
        candidate,
        EXPERIMENT_SPECS[args.experiment],
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        baseline_config=_run_config(_records_path(args.baseline)),
        candidate_config=_run_config(_records_path(args.candidate)),
    )
    _write_comparison(report, Path(args.out))
    print(f"A/B 判定：{report['decision']} → {args.out}")
    return 0


def _cmd_ab_arm(args: argparse.Namespace) -> int:
    """把「单个实验臂」与冻结基线对比，产出三态判定。

    基线只冻结一次，候选臂由 ``scripts/ab_rag.py`` 跑出来（``<arm>.json``）；
    这样组件实验不必每次重跑对照组。
    """
    from .ab_runner import _paired_records
    from .baseline import load_baseline, verify_baseline
    from .statistics import compare_runs, select_experiment_spec

    baseline = load_baseline(args.profile, args.version)
    arm = json.loads(Path(args.arm_json).read_text(encoding="utf-8"))
    overrides = ((arm.get("arm") or {}).get("strategy_overrides") or {})
    retrieval_only = bool(overrides.get("retrieval_only"))
    spec = select_experiment_spec(args.experiment, retrieval_only=retrieval_only)
    verification = verify_baseline(args.profile, args.version)
    baseline_config = {
        **(baseline["manifest"].get("summary") or {}),
        "dataset_fingerprint": (baseline["manifest"].get("dataset") or {}).get("fingerprint"),
        "kb_fingerprint": (baseline["manifest"].get("kb") or {}).get("kb_fingerprint"),
        "protocol_fingerprint": (baseline["manifest"].get("protocol") or {}).get("fingerprint"),
        "judge_fingerprint": (baseline["manifest"].get("judge") or {}).get("fingerprint"),
        "comparable": bool(verification["ok"]),
    }
    report = compare_runs(
        _paired_records({"per_query": baseline["records"]}),
        _paired_records(arm),
        spec,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        baseline_config=baseline_config,
        candidate_config=arm.get("run_fingerprints"),
        noise_floor=(baseline["manifest"].get("noise_floor") or None),
    )
    report["baseline"] = {
        "profile": baseline["manifest"].get("profile"),
        "version": baseline["manifest"].get("version"),
        "path": str(baseline["path"]),
        "verified": verification["ok"],
        "drift": verification["drift"],
    }
    _write_comparison(report, Path(args.out))
    print(f"A/B 判定（基线 {args.profile}）：{report['decision']} → {args.out}")
    if not verification["ok"]:
        print("警告：基线指纹漂移，判定已降级为 no_decision。")
    return 0 if report["decision"] in {"pass", "inconclusive"} else 1


def _run_config(records_path: Path) -> dict[str, Any] | None:
    for name in ("run_config.json", "summary.json"):
        path = records_path.parent / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cmd_baseline(args: argparse.Namespace) -> int:
    """冻结基线的冻结 / 校验 / 列表 / 查看。"""
    from .baseline import (
        PROFILES,
        freeze_baseline,
        list_baselines,
        load_baseline,
        verify_baseline,
    )

    if args.baseline_action == "freeze":
        noise_floor = None
        if args.noise_floor:
            noise_floor = json.loads(Path(args.noise_floor).read_text(encoding="utf-8"))
        target = freeze_baseline(
            args.profile,
            dataset_path=args.dataset,
            version=args.version,
            sample=args.sample,
            seed=args.seed,
            include_silver=args.include_silver,
            outdir=args.outdir,
            noise_floor=noise_floor,
        )
        print(f"已冻结基线 {args.profile}：{target}")
        print(f"复现命令：python scripts/evaluate.py baseline freeze --profile {args.profile} "
              f"--dataset {args.dataset} --version {target.name}")
        return 0
    if args.baseline_action == "verify":
        try:
            report = verify_baseline(
                args.profile, args.version, deep=bool(getattr(args, "deep", False))
            )
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        if report["ok"]:
            print(f"基线可比：{report['profile']}/{report['version']}（{report['path']}）")
            return 0
        print(f"基线不可比：{report['profile']}/{report['version']}（{report['path']}）")
        for item in report["drift"]:
            print(f"  - {item['field']}: 冻结={item['expected']} 当前={item['actual']}")
        print("漂移后本次对比必须判 no_decision；确认无误再重新冻结。")
        return 1
    if args.baseline_action == "list":
        rows = list_baselines()
        if not rows:
            print(f"没有已冻结基线（{PROFILES.keys()}）。先执行 baseline freeze。")
            return 0
        for row in rows:
            print(f"{row['profile']:<4} {row['version']:<20} n={row['case_count']} "
                  f"{row['label']}  {row['path']}")
        return 0
    if args.baseline_action == "show":
        try:
            data = load_baseline(args.profile, args.version)
        except FileNotFoundError as exc:
            print(str(exc))
            return 1
        manifest = data["manifest"]
        print(json.dumps(
            {key: manifest.get(key) for key in ("profile", "label", "version", "strategy", "policy",
                                                "frozen_at", "dataset", "kb", "judge", "protocol",
                                                "metrics_version", "noise_floor")},
            ensure_ascii=False, indent=2, sort_keys=True,
        ))
        print(f"records={len(data['records'])} path={data['path']}")
        return 0
    raise ValueError(args.baseline_action)


def _cmd_gate(args: argparse.Namespace) -> int:
    """回归门禁：读报告、出退出码。"""
    from .gate import evaluate_gate, render_gate

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    result = evaluate_gate(
        report,
        policy=args.policy,
        max_regression=args.max_regression,
        latency_ratio=args.latency_ratio,
    )
    print(render_gate(result))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return int(result["exit_code"])


def _cmd_calibration_export(args: argparse.Namespace) -> int:
    """从已落盘的运行产物抽样导出待人工标注 CSV（不花 LLM 成本）。"""
    from .human_calibration import export_human_sample, load_run_arms

    run_dir = Path(args.run_dir)
    arms = load_run_arms(run_dir)
    if not arms:
        raise ValueError(f"运行目录里没有可用的评测产物（无 <arm>.json / records.jsonl）：{run_dir}")
    out = Path(args.out) if args.out else run_dir / "human_sample.csv"
    path = export_human_sample(
        arms, out_csv=out, n=args.n, seed=args.seed, arm=args.arm
    )
    print(f"已导出 {args.n} 条待人工标注样本（每臂）：{path}")
    print("请填写 human_faithfulness / human_answer_relevancy / "
          "human_context_precision / human_context_recall（0~1，空=未评），"
          "再执行 calibration import。")
    return 0


def _cmd_calibration_import(args: argparse.Namespace) -> int:
    """回填人工标注并输出 judge-人类一致性报告。"""
    from .human_calibration import (
        calibration_report,
        load_human_scores,
        load_run_arms,
        render_calibration_markdown,
        write_calibration_report,
    )

    run_dir = Path(args.run_dir)
    arms = load_run_arms(run_dir)
    if not arms:
        raise ValueError(f"运行目录里没有可用的评测产物（无 <arm>.json / records.jsonl）：{run_dir}")
    human_scores = load_human_scores(args.csv)
    if not human_scores:
        raise ValueError(f"人工标注 CSV 中没有已填写的 human_* 分数：{args.csv}")

    auto_rows = [row for rows in arms.values() for row in rows]
    report = calibration_report(auto_rows, human_scores)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
    else:
        target = write_calibration_report(run_dir, report)
    for line in render_calibration_markdown(report):
        print(line)
    print(f"\n校准报告：{target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="金融 RAG 统一评测")
    groups = parser.add_subparsers(dest="command", required=True)

    dataset = groups.add_parser("dataset", help="数据集生成、校验、审核和冻结")
    dataset_actions = dataset.add_subparsers(dest="dataset_action", required=True)
    generate = dataset_actions.add_parser("generate")
    generate.add_argument("--count", type=int, default=240)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--per-doc", type=int, default=40)
    generate.add_argument("--out", default=str(DEFAULT_DATA_DIR / "evaluation_candidates.jsonl"))
    generate.add_argument("--review-out")
    generate.add_argument("--from-markdown", help="只读迁移旧 Markdown；证据来源按 chunk 在线解析")
    validate = dataset_actions.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--online", action="store_true")
    export_review = dataset_actions.add_parser("export-review")
    export_review.add_argument("--dataset", required=True)
    export_review.add_argument("--out", required=True)
    import_review = dataset_actions.add_parser("import-review")
    import_review.add_argument("--dataset", required=True)
    import_review.add_argument("--review", required=True)
    import_review.add_argument("--out", required=True)
    freeze = dataset_actions.add_parser("freeze")
    freeze.add_argument("--dataset", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--kb-fingerprint")
    freeze.add_argument("--max-source-share", type=float, default=0.20)

    retrieval = groups.add_parser("retrieval")
    retrieval_actions = retrieval.add_subparsers(dest="retrieval_action", required=True)
    run = retrieval_actions.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--strategy", nargs="+", default=["L0", "L1", "L2", "L3", "L5"])
    run.add_argument("--candidate-k", type=int, default=50)
    run.add_argument("--sample", type=int, default=0)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--bootstrap", type=int, default=10_000)
    run.add_argument("--dataset-version")
    run.add_argument("--run-id")
    run.add_argument("--outdir", default=str(DEFAULT_RESULTS))
    run.add_argument("--allow-drift", action="store_true")
    run.add_argument("--diagnostic", action="store_true")

    ab = groups.add_parser("ab")
    ab_actions = ab.add_subparsers(dest="ab_action", required=True)
    compare = ab_actions.add_parser("run")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--experiment", choices=sorted(EXPERIMENT_SPECS), required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--bootstrap", type=int, default=10_000)
    compare.add_argument("--seed", type=int, default=7)
    arm = ab_actions.add_parser(
        "arm", help="单个实验臂对比冻结基线（基线只冻结一次）"
    )
    arm.add_argument("--profile", required=True, help="b0 / b1 / b2 或基线目录")
    arm.add_argument("--version", default=None)
    arm.add_argument("--arm-json", required=True, help="scripts/ab_rag.py 产出的 <arm>.json")
    arm.add_argument("--experiment", required=True, help="实验名（catalog 名或 spec 名）")
    arm.add_argument("--out", required=True)
    arm.add_argument("--bootstrap", type=int, default=10_000)
    arm.add_argument("--seed", type=int, default=7)

    baseline = groups.add_parser("baseline", help="冻结基线：冻结 / 校验 / 列表 / 查看")
    baseline_actions = baseline.add_subparsers(dest="baseline_action", required=True)
    b_freeze = baseline_actions.add_parser("freeze", help="跑一次基线并冻结元组")
    b_freeze.add_argument("--profile", required=True, help="b0 / b1 / b2")
    b_freeze.add_argument("--dataset", required=True)
    b_freeze.add_argument("--version", default=None)
    b_freeze.add_argument("--sample", type=int, default=0)
    b_freeze.add_argument("--seed", type=int, default=7)
    b_freeze.add_argument("--include-silver", action="store_true")
    b_freeze.add_argument("--outdir", default=None)
    b_freeze.add_argument("--noise-floor", default=None, help="test-retest 得到的噪声下限 JSON")
    b_verify = baseline_actions.add_parser("verify", help="校验当前环境是否仍与冻结基线可比")
    b_verify.add_argument("--profile", required=True)
    b_verify.add_argument("--version", default=None)
    b_verify.add_argument("--deep", action="store_true", help="额外扫描全集合 chunk 内容指纹（慢）")
    baseline_actions.add_parser("list", help="列出已冻结基线")
    b_show = baseline_actions.add_parser("show", help="查看基线 manifest")
    b_show.add_argument("--profile", required=True)
    b_show.add_argument("--version", default=None)

    gate = groups.add_parser("gate", help="回归门禁：回退即非 0 退出")
    gate.add_argument("--report", required=True, help="report.json 或 comparison.json")
    gate.add_argument("--policy", choices=["smoke", "release"], default="smoke")
    gate.add_argument("--max-regression", type=float, default=0.0)
    gate.add_argument("--latency-ratio", type=float, default=None)
    gate.add_argument("--out", default=None)

    calibration = groups.add_parser(
        "calibration", help="人工标注校准：导出待标注样本 / 回填一致性报告"
    )
    calibration_actions = calibration.add_subparsers(
        dest="calibration_action", required=True
    )
    cal_export = calibration_actions.add_parser("export")
    cal_export.add_argument("--run-dir", required=True, help="A/B 运行目录")
    cal_export.add_argument("--arm", default=None, help="只导出某一臂（默认全部）")
    cal_export.add_argument("--n", type=int, default=DEFAULT_HUMAN_SAMPLE_SIZE)
    cal_export.add_argument("--seed", type=int, default=7)
    cal_export.add_argument("--out", default=None)
    cal_import = calibration_actions.add_parser("import")
    cal_import.add_argument("--run-dir", required=True)
    cal_import.add_argument("--csv", required=True, help="已填写 human_* 的 CSV")
    cal_import.add_argument("--out", default=None, help="报告输出路径（默认写入运行目录）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "retrieval" and args.retrieval_action == "run":
        return _cmd_retrieval(args)
    if args.command == "ab" and args.ab_action == "run":
        return _cmd_ab(args)
    if args.command == "ab" and args.ab_action == "arm":
        return _cmd_ab_arm(args)
    if args.command == "baseline":
        return _cmd_baseline(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "calibration":
        if args.calibration_action == "export":
            return _cmd_calibration_export(args)
        if args.calibration_action == "import":
            return _cmd_calibration_import(args)
    raise ValueError("未知命令")
