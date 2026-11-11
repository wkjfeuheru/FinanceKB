"""检索层块级指标评估：Recall / Precision / MRR，支持多深度召回-K 曲线。

两种用途：

* 正式 Top-5 口径（默认 ``--top-k 5``）：JSONL 数据集转发给统一评测 CLI
  （``scripts/evaluate.py retrieval run``），保持冻结协议下的可比性；
* 召回-K 曲线（``--top-k 5 8 10``）：诊断工具。每个层只检索一次（深度取
  ``max(top_ks)``），在同一份 ranked list 上按各深度分别计分，用于判断
  生产 K 是否接近召回天花板（判读参考见输出末尾）。
"""

from __future__ import annotations

import argparse
import random
import socket
import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TESTSET = ROOT / "src" / "eval" / "data" / "evaluation_qa_generated.md"

METRIC_LABELS = {"recall": "Recall", "precision": "Prec", "mrr": "MRR"}


def _normalize_top_ks(raw: list[int]) -> list[int]:
    """去重升序；至少一个且全部 >=1。"""
    ks = sorted({int(k) for k in raw})
    if not ks or ks[0] < 1:
        raise SystemExit("错误：--top-k 需要至少一个 >=1 的深度")
    return ks


def _layer_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """构造 L0/L1/L2/L3/L5；每一对只改变被测优化手段。

    多深度时以 ``max(top_ks)`` 作为重排输出深度与 L2 的最终深度，保证高深度
    曲线点不缺数据；各深度指标共用同一份 ranked list，不重复检索。
    """
    candidate_k, top_ks = args.candidate_k, list(args.top_ks)
    max_k = max(top_ks)
    common = {"rerank_top_n": max_k, "expand_parents": True, "top_ks": top_ks}
    return [
        {"key": "L0", "name": "纯稠密检索", "note": "稠密基线", "k": candidate_k,
         "use_dense_only": True, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L1", "name": "混合检索(Weighted)", "note": "稠密 + BM25 加权融合", "k": candidate_k,
         "use_dense_only": False, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L2", "name": "混合检索，未重排序", "note": "与 L3 对照重排序", "k": max_k,
         "use_dense_only": False, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L3", "name": "混合检索 + BGE", "note": "关闭断崖检测", "k": candidate_k,
         "use_dense_only": False, "use_rerank": True, "cliff_detection": False, **common},
        {"key": "L5", "name": "混合检索 + BGE + 断崖检测", "note": "显式开启断崖检测", "k": candidate_k,
         "use_dense_only": False, "use_rerank": True, "cliff_detection": True, **common},
    ]


def _preflight() -> None:
    from src.core import config

    uri = config.MILVUS_URI
    parsed = urlparse(uri if "://" in uri else f"http://{uri}")
    host, port = parsed.hostname or "localhost", parsed.port or 19530
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        print(f"错误：Milvus 不可达（{host}:{port}）：{exc}")
        raise SystemExit(1) from exc
    print(f"Milvus 可达：{uri}")
    print(
        "断崖检测：绝对差距 >= "
        f"{config.RERANKER_CLIFF_ABS_THRESHOLD} 或相对跌幅 >= "
        f"{config.RERANKER_CLIFF_REL_THRESHOLD}"
        f"（至少保留 {config.RERANKER_CLIFF_MIN_TOP_K} 条）\n"
    )


def _load_jsonl_entries(path: Path) -> list[Any]:
    """EvalCase → 曲线诊断条目：chunk 级证据映射为 chunk_ids 与 related_docs。"""
    from src.eval.dataset import EvaluationDataset

    entries: list[Any] = []
    for case in EvaluationDataset.load(path).cases:
        if case.question_type == "negative":
            continue
        sources = list(dict.fromkeys(ref.source for ref in case.evidence if ref.source))
        entries.append(SimpleNamespace(
            query=case.query,
            chunk_ids=[ref.chunk_id for ref in case.evidence if ref.chunk_id],
            related_docs=" ".join(sources),
            question_type=case.question_type,
        ))
    return entries


def _load_entries(args: argparse.Namespace) -> tuple[list[Any], Any]:
    from src.eval.test_set import TestSetLoader
    from src.eval.testset_generator import filter_entries_by_kb
    from src.index.milvus_kb import get_knowledge_base

    path = Path(args.testset)
    if path.suffix.lower() == ".jsonl":
        raw = _load_jsonl_entries(path)
    else:
        raw = list(TestSetLoader(path).load_test_set())
    kb = get_knowledge_base()
    entries = [e for e in raw if getattr(e, "question_type", "") != "negative"]
    entries, removed = filter_entries_by_kb(entries, kb, allow_missing=args.allow_missing)
    if removed:
        print(f"剔除 {len(removed)} 条（相关文档不在知识库）")
    if args.sample and args.sample < len(entries):
        entries = random.Random(args.seed).sample(entries, args.sample)
        print(f"抽样 {len(entries)} 条（seed={args.seed}）")
    return entries, kb


def _retrieve(kb: Any, query: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    return kb.hybrid_search(
        query, k=spec["k"], use_rerank=spec["use_rerank"],
        rerank_top_n=spec["rerank_top_n"], expand_parents=spec["expand_parents"],
        use_dense_only=spec["use_dense_only"],
        cliff_detection=spec["cliff_detection"],
    )


def _evaluate_layer(kb: Any, entries: list[Any], spec: dict[str, Any]) -> dict[str, Any]:
    from src.eval.metrics import compute_chunk_retrieval_metrics

    top_ks = list(spec["top_ks"])
    values: dict[int, dict[str, list[float]]] = {
        k: {"recall": [], "precision": [], "mrr": []} for k in top_ks
    }
    errors, labeled_entries = 0, 0
    for entry in entries:
        try:
            docs = _retrieve(kb, entry.query, spec)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"    [warn] 检索失败：{entry.query[:30]}… {exc}")
            docs = []
        chunk_ids = [str(d.get("id") or d.get("child_id") or "") for d in docs]
        # 无证据标注的条目在任何深度都不可评，只检查一次
        first = compute_chunk_retrieval_metrics(chunk_ids, entry.chunk_ids, k=top_ks[0])
        if first["recall_at_k"] is None:
            continue
        labeled_entries += 1
        for k in top_ks:
            metrics = first if k == top_ks[0] else compute_chunk_retrieval_metrics(chunk_ids, entry.chunk_ids, k=k)
            values[k]["recall"].append(metrics["recall_at_k"])
            values[k]["precision"].append(metrics["precision_at_k"] or 0.0)
            values[k]["mrr"].append(metrics["mrr"] or 0.0)

    def _mean(k: int, key: str) -> float | None:
        return round(st.mean(values[k][key]), 4) if values[k][key] else None

    return {
        "key": spec["key"], "name": spec["name"], "note": spec["note"], "n": len(entries),
        "n_evidence_labeled": labeled_entries, "errors": errors,
        "per_k": {
            k: {"recall": _mean(k, "recall"), "precision": _mean(k, "precision"), "mrr": _mean(k, "mrr")}
            for k in top_ks
        },
    }


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _render(rows: list[dict[str, Any]], top_ks: list[int]) -> None:
    multi = len(top_ks) > 1
    width = 34 + 10 * len(top_ks) + (10 if multi else 0)
    print("=" * width)
    depths = "、".join(f"@{k}" for k in top_ks)
    print(f"检索层评估（块级）—— 深度 {depths}")
    print("=" * width)

    for metric in ("recall", "precision", "mrr"):
        print(f"\n{METRIC_LABELS[metric]}")
        print(f"{'层':<4}{'策略':<30}"
              + "".join(f"{f'{METRIC_LABELS[metric]}@{k}':>10}" for k in top_ks)
              + (f"{'Δ首末':>10}" if multi else ""))
        print("-" * width)
        for row in rows:
            cells = "".join(f"{_fmt(row['per_k'][k][metric]):>10}" for k in top_ks)
            delta = ""
            if multi:
                first, last = row["per_k"][top_ks[0]][metric], row["per_k"][top_ks[-1]][metric]
                delta = f"{last - first:>+10.4f}" if first is not None and last is not None else f"{'n/a':>10}"
            print(f"{row['key']:<4}{row['name']:<30}{cells}{delta}")
    print("-" * width)
    if rows:
        print(f"有效正例：{rows[0]['n']}；有证据块标注：{rows[0]['n_evidence_labeled']}；"
              f"每层检索一次，在同一份 ranked list 上按各深度分别聚合。")

    by_key = {row["key"]: row for row in rows}
    def _delta(left: str, right: str, metric: str) -> str:
        before, after = by_key.get(left), by_key.get(right)
        if not before or not after:
            return "n/a"
        cells = []
        for k in top_ks:
            b, a = before["per_k"][k][metric], after["per_k"][k][metric]
            if b is None or a is None:
                return "n/a"
            cells.append(f"{b:.4f} → {a:.4f}（{a - b:+.4f}）" if not multi else f"@{k} {a - b:+.4f}")
        return "  ".join(cells)

    for label, left, right in (("混合检索增益", "L0", "L1"), ("重排序影响", "L2", "L3"), ("断崖检测影响", "L3", "L5")):
        print(f"\n{label}（{left} → {right}）：")
        for metric in ("recall", "precision", "mrr"):
            print(f"  {METRIC_LABELS[metric]:<10}{_delta(left, right, metric)}")

    if multi:
        lo, hi = top_ks[0], top_ks[-1]
        print(
            f"\n判读参考（Recall {lo}→{hi} 的 Δ首末 列）：Δ < 0.05 → 生产 K 已近召回天花板，"
            "无需加大；Δ > 0.10 → 尾部有真证据，建议端到端确认："
            "python scripts/ab_rag.py --experiment rerank_topk_5_vs_10"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索层块级 Recall / Precision / MRR 评估（支持多深度召回-K 曲线）")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET), help="测试集路径")
    parser.add_argument("--candidate-k", type=int, default=50, help="L0/L1 与重排序前的候选深度")
    parser.add_argument("--top-k", type=int, nargs="+", default=[5],
                        help="评估深度，可传多个画召回-K 曲线，如 --top-k 5 8 10（默认 5）")
    parser.add_argument("--sample", type=int, default=0, help="抽样条目数（0=全量）")
    parser.add_argument("--seed", type=int, default=7, help="抽样随机种子")
    parser.add_argument("--allow-missing", action="store_true", help="保留知识库缺失的标注条目")
    parser.add_argument("--only", nargs="*", default=None, help="只评估指定层，如 --only L2 L3")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.top_ks = _normalize_top_ks(args.top_k)
    multi_k = len(args.top_ks) > 1
    if Path(args.testset).suffix.lower() == ".jsonl" and not multi_k:
        from src.eval.cli import main as unified_main

        print("提示：Top-5 正式口径建议直接使用 scripts/evaluate.py retrieval run；本脚本转发执行。")
        forwarded = [
            "retrieval", "run", "--dataset", args.testset,
            "--candidate-k", str(args.candidate_k), "--seed", str(args.seed),
        ]
        if args.sample:
            forwarded += ["--sample", str(args.sample)]
        if args.only:
            forwarded += ["--strategy", *args.only]
        return unified_main(forwarded)
    if Path(args.testset).suffix.lower() != ".jsonl":
        print("警告：Markdown 测试集已弃用；请迁移为 JSONL。当前使用旧版兼容执行路径。")
    if multi_k:
        print("多深度召回-K 曲线为诊断口径，与 Top-5 冻结协议的正式评测数字不可直接混用。\n")
    _preflight()
    entries, kb = _load_entries(args)
    if not entries:
        print("无可用测试条目")
        return 1
    specs = _layer_specs(args)
    if args.only:
        wanted = {key.upper() for key in args.only}
        specs = [spec for spec in specs if spec["key"] in wanted]
    if not specs:
        print("未选择任何有效评估层")
        return 1
    print(f"评估 {len(entries)} 条有效正例 × {len(specs)} 个检索策略层（深度 {'/'.join(str(k) for k in args.top_ks)}）\n")
    rows = []
    for spec in specs:
        print(f"  [{spec['key']}] {spec['name']} —— {spec['note']}")
        rows.append(_evaluate_layer(kb, entries, spec))
    print()
    _render(rows, args.top_ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
