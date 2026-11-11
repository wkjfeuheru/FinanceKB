"""RAG 优化点 A/B 实验命令行入口。

用法示例：

    # 列出全部内置实验
    python scripts/ab_rag.py --list

    # 基于当前知识库自动生成测试集（人工抽查后作为评测基准）
    python scripts/ab_rag.py --generate-testset --count 40

    # 跑单个实验（--fast = 3 项核心指标；--full = 6 项）
    python scripts/ab_rag.py --experiment hyde_on_off --fast --sample 5

    # 跑多个 / 全部实验
    python scripts/ab_rag.py --experiment rerank_on_off hyde_on_off --fast
    python scripts/ab_rag.py --all --fast --sample 5

    # 自定义测试集 / 抽样种子
    python scripts/ab_rag.py --experiment hyde_on_off \\
        --testset path/to/qa.md --sample 10 --seed 7

结果落盘 ``scripts/results/ab/<实验名>/<时间戳>/``。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 项目根目录（scripts/ 的上一级），保证以任意 CWD 运行时都能导入项目模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = (
    ROOT / "src" / "eval" / "data" / "evaluation_qa_generated.md"
)


def _resolve_testset(path: str | None) -> Path:
    """优先显式路径，其次自动生成测试集；缺失时抛错并提示生成。"""
    if path:
        return Path(path)
    if DEFAULT_TESTSET.exists():
        return DEFAULT_TESTSET
    raise FileNotFoundError(
        f"测试集不存在：{DEFAULT_TESTSET}\n"
        "请先运行 python scripts/ab_rag.py --generate-testset"
    )


def _cmd_list() -> int:
    from src.eval.ab_catalog import AB_EXPERIMENTS

    for exp in AB_EXPERIMENTS:
        print(f"{exp.name:24s} {exp.label}  [{exp.kind}]")
    return 0


def _cmd_generate_testset(args: argparse.Namespace) -> int:
    from src.eval.testset_generator import generate_evaluation_dataset

    out = Path(args.out) if args.out else (
        ROOT / "src" / "eval" / "data" / "evaluation_candidates.jsonl"
    )
    dataset, out = generate_evaluation_dataset(
        count=args.count,
        seed=args.seed,
        per_doc_limit=args.per_doc,
        out_path=out,
    )
    review = out.with_suffix(".review.csv")
    dataset.export_review(review)
    dist: dict[str, int] = {}
    for entry in dataset.cases:
        dist[entry.question_type] = dist.get(entry.question_type, 0) + 1
    print("提示：ab_rag.py 已进入兼容模式，候选生成已转交统一 JSONL 模块。")
    print(f"已生成 {len(dataset.cases)} 条 → {out}")
    print(f"人工审阅表：{review}")
    print(f"类型分布：{dist}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from src.eval.ab_catalog import AB_EXPERIMENTS, select_experiments
    from src.eval.ab_runner import run_experiment

    experiments = AB_EXPERIMENTS if args.all else select_experiments(args.experiment)
    testset_path = _resolve_testset(args.testset)
    if not testset_path.exists():
        print(f"测试集不存在：{testset_path}")
        return 1
    if testset_path.suffix.lower() == ".md":
        print("警告：Markdown 测试集已弃用；当前仅以兼容模式运行。")
    else:
        from src.eval.dataset import DatasetValidationError, EvaluationDataset
        dataset = EvaluationDataset.load(testset_path)
        pending_gold = [
            case.id for case in dataset.cases
            if case.tier == "gold" and case.review_status != "approved"
        ]
        if pending_gold:
            raise DatasetValidationError(
                f"Gold 冻结前必须人工批准；仍有 {len(pending_gold)} 条未批准"
            )

    for exp in experiments:
        print(f"\n=== 实验：{exp.name}（{exp.label}）===")
        report = run_experiment(
            exp,
            testset_path=testset_path,
            fast=not args.full,
            sample=args.sample,
            seed=args.seed,
            allow_missing=args.allow_missing,
            outdir=Path(args.outdir) if args.outdir else None,
            include_silver=args.include_silver,
            pairwise=args.pairwise,
            cross_check=args.cross_check,
            anchors=not args.no_anchors,
            anchor_strict=True if args.anchor_strict else None,
            export_human_sample=args.export_human_sample,
        )
        if args.report_diagnostics:
            _print_diagnostics(report)
    return 0


def _print_diagnostics(report: dict) -> None:
    """打印多跳/证据块归因明细，便于区分检索失败与评测集标注缺陷。"""
    print("\n--- 诊断明细（多跳与证据块） ---")
    arms = report.get("arms") or {}
    for arm_name, arm in sorted(arms.items()):
        if arm.get("error"):
            print(f"[{arm_name}] 执行失败：{arm['error']}")
            continue
        counts = (arm.get("metrics_by_type_counts") or {}).get("multi_hop", {})
        by_type = (arm.get("metrics_by_type") or {}).get("multi_hop", {})
        if not by_type:
            print(f"[{arm_name}] 无 multi_hop 样本")
            continue
        print(f"[{arm_name}] multi_hop n={counts.get('multi_hop_hit', 0)} "
              f"hit_rate={by_type.get('hit_rate')} "
              f"doc_coverage={by_type.get('doc_coverage')} "
              f"doc_recall={by_type.get('doc_recall')} "
              f"evidence_coverage={by_type.get('evidence_coverage')} "
              f"multi_hop_hit={by_type.get('multi_hop_hit')}")
        suspects = arm.get("suspect_annotations") or []
        if suspects:
            print(f"[{arm_name}] 疑似标注不可信 {len(suspects)} 条：")
            for item in suspects[:5]:
                print(f"    - 未召回 {item.get('missing_docs')} "
                      f"(实际召回：{item.get('retrieved_docs') or '空'})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 优化点 A/B 实验")
    parser.add_argument("--list", action="store_true", help="列出全部内置实验")
    parser.add_argument(
        "--generate-testset", action="store_true", help="基于知识库生成测试集"
    )
    parser.add_argument("--count", type=int, default=240, help="目标候选数（默认 240）")
    parser.add_argument("--per-doc", type=int, default=40, help="每文档纳入候选的最大块数（默认 40）")
    parser.add_argument("--out", default=None, help="测试集输出路径（默认自动）")
    parser.add_argument("--experiment", nargs="*", default=None, help="实验名（可多个）")
    parser.add_argument("--all", action="store_true", help="运行全部实验")
    parser.add_argument("--fast", action="store_true", help="快速模式（默认；四指标固定，与 --full 无差异）")
    parser.add_argument("--full", action="store_true", help="兼容开关；生成侧恒为 RAGAS 四指标")
    parser.add_argument("--sample", type=int, default=5, help="抽样条目数（默认 5）")
    parser.add_argument("--seed", type=int, default=7, help="随机种子（默认 7）")
    parser.add_argument("--testset", default=None, help="自定义测试集路径")
    parser.add_argument(
        "--allow-missing", action="store_true", help="保留相关文档不在知识库的条目"
    )
    parser.add_argument(
        "--include-silver", action="store_true",
        help="生成评估显式扩展到 Gold+Silver（JSONL 默认仅 Gold）",
    )
    parser.add_argument("--outdir", default=None, help="结果输出根目录")
    parser.add_argument(
        "--report-diagnostics",
        action="store_true",
        help="打印多跳/证据块诊断明细（区分检索失败与评测集标注缺陷）",
    )
    parser.add_argument(
        "--pairwise", nargs="?", type=int, const=20, default=0,
        help="成对裁判（顺序随机+换序复评）抽样题数，默认 0=关闭；不带值时取 20",
    )
    parser.add_argument(
        "--cross-check", type=int, default=0,
        help="第二裁判族交叉校验的题数，默认 0=关闭（需配置第二裁判模型）",
    )
    parser.add_argument(
        "--no-anchors", action="store_true", help="不跑锚点样本（默认跑，文件缺失则跳过）",
    )
    parser.add_argument(
        "--anchor-strict", action="store_true", help="锚点越界时判本次实验失败",
    )
    parser.add_argument(
        "--export-human-sample", type=int, default=0,
        help="导出 N 条待人工标注样本（默认 0=不导出），见 evaluate.py calibration",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.full:
        print("提示：生成侧指标集已固定为 RAGAS 四指标，--full 不再扩展指标。")

    if args.list:
        return _cmd_list()
    if args.generate_testset:
        return _cmd_generate_testset(args)
    if args.all or args.experiment:
        return _cmd_run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
