"""从已有 A/B 实验结果中统计**检索链路延迟**（报告聚合里缺失的维度）。

背景
----
``scripts/ab_rag.py`` 会把逐题延迟写进 ``per_query.csv``
（``retrieval_latency_ms`` / ``generation_latency_ms``），但 ``report.md`` /
``report.json`` 只聚合质量指标，不聚合延迟——于是"精度提升多少、代价是多少延迟"
这个最关键的权衡无法从报告直接读出。本脚本补齐这一层。

``retrieval_latency_ms`` 的口径是 ``ab_runner`` 中包住整段检索调用的墙钟耗时：
含查询改写（LLM）、多路子查询检索、可选 HyDE（LLM）、混合召回与 BGE 重排序，
即**检索链路端到端延迟**，不含答案生成与 Ragas 打分。

用法
----
    python scripts/ab_latency_stats.py                     # 扫描 scripts/results/ab 全部实验
    python scripts/ab_latency_stats.py --experiment rerank_on_off

注意
----
样本量很小（默认 --sample 5），且首次调用会加载 BGE 重排序模型，
延迟分布高度右偏：**p95 往往反映冷启动而非稳态成本**，解读时应以 p50 为主。
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "ab"


def _number(row: dict, key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def summarize(path: Path) -> dict[str, dict]:
    """按实验臂聚合一个 per_query.csv 的延迟与关键质量指标。"""
    # 必须用 utf-8-sig：CSV 带 BOM，否则首列名会带上 \ufeff 而识别不到 arm
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        has_arm = bool(reader.fieldnames and "arm" in reader.fieldnames)
    if not rows:
        return {}

    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(row.get("arm") or "single" if has_arm else "single", []).append(row)

    result: dict[str, dict] = {}
    for arm, items in arms.items():
        latencies = sorted(
            value for value in (_number(r, "retrieval_latency_ms") for r in items)
            if value is not None
        )
        if not latencies:
            continue
        ctx_precision = [
            value for value in (_number(r, "context_precision") for r in items)
            if value is not None
        ]
        result[arm] = {
            "n": len(latencies),
            "mean": st.mean(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "ctx_precision": st.mean(ctx_precision) if ctx_precision else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="统计 A/B 实验的检索链路延迟")
    parser.add_argument("--experiment", default="", help="只统计指定实验名（默认全部）")
    parser.add_argument("--root", default=str(RESULTS_ROOT), help="结果根目录")
    args = parser.parse_args()

    pattern = f"{args.root}/{args.experiment or '*'}/{'*' if not args.experiment else '*'}/per_query.csv"
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"未找到 per_query.csv（pattern={pattern}）")
        return 1

    for path in paths:
        csv_path = Path(path)
        experiment = csv_path.parents[1].name
        summary = summarize(csv_path)
        if not summary:
            continue
        print(f"== {experiment} ==")
        for arm, stats in summary.items():
            ctx = (
                f"{stats['ctx_precision']:.4f}"
                if stats["ctx_precision"] is not None
                else "n/a"
            )
            print(
                f"   {arm:<16} n={stats['n']:<3} "
                f"检索延迟 mean={stats['mean']:>9.1f}ms  "
                f"p50={stats['p50']:>9.1f}ms  p95={stats['p95']:>9.1f}ms  "
                f"| context_precision={ctx}"
            )
        print()

    print("提示：样本量小且首次调用会加载重排序模型，p95 常反映冷启动，请以 p50 解读。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
