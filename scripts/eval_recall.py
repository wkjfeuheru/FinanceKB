"""检索层块级 Top-5 指标评估（Recall@5、Precision@5、MRR）。"""

from __future__ import annotations

import argparse
import random
import socket
import statistics as st
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TESTSET = ROOT / "finance_rag" / "src" / "eval" / "data" / "evaluation_qa_generated.md"


def _layer_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """构造 L0/L1/L2/L3/L5；每一对只改变被测优化手段。"""
    candidate_k, top_k = args.candidate_k, args.top_k
    common = {"rerank_top_n": top_k, "expand_parents": True, "top_k": top_k}
    return [
        {"key": "L0", "name": "纯稠密检索", "note": "稠密基线", "k": candidate_k,
         "use_dense_only": True, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L1", "name": "混合检索(Weighted)", "note": "稠密 + BM25 加权融合", "k": candidate_k,
         "use_dense_only": False, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L2", "name": "混合检索，未重排序", "note": "与 L3 对照重排序", "k": top_k,
         "use_dense_only": False, "use_rerank": False, "cliff_detection": False, **common},
        {"key": "L3", "name": "混合检索 + BGE", "note": "关闭断崖检测", "k": candidate_k,
         "use_dense_only": False, "use_rerank": True, "cliff_detection": False, **common},
        {"key": "L5", "name": "混合检索 + BGE + 断崖检测", "note": "显式开启断崖检测", "k": candidate_k,
         "use_dense_only": False, "use_rerank": True, "cliff_detection": True, **common},
    ]


def _preflight() -> None:
    from finance_rag.src.core import config

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


def _load_entries(args: argparse.Namespace) -> tuple[list[Any], Any]:
    from finance_rag.src.eval.test_set import TestSetLoader
    from finance_rag.src.eval.testset_generator import filter_entries_by_kb
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

    entries = [e for e in TestSetLoader(Path(args.testset)).load_test_set() if e.question_type != "negative"]
    kb = get_knowledge_base()
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
    from finance_rag.src.eval.metrics import compute_chunk_retrieval_metrics

    values: dict[str, list[float]] = {"recall_at_5": [], "precision_at_5": [], "mrr": []}
    errors, labeled_entries = 0, 0
    for entry in entries:
        try:
            docs = _retrieve(kb, entry.query, spec)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"    [warn] 检索失败：{entry.query[:30]}… {exc}")
            docs = []
        chunk_ids = [str(d.get("id") or d.get("child_id") or "") for d in docs]
        metrics = compute_chunk_retrieval_metrics(chunk_ids, entry.chunk_ids, k=spec["top_k"])
        if metrics["recall_at_k"] is None:
            continue
        labeled_entries += 1
        values["recall_at_5"].append(metrics["recall_at_k"])
        values["precision_at_5"].append(metrics["precision_at_k"] or 0.0)
        values["mrr"].append(metrics["mrr"] or 0.0)

    def _mean(key: str) -> float | None:
        return round(st.mean(values[key]), 4) if values[key] else None

    return {
        "key": spec["key"], "name": spec["name"], "note": spec["note"], "n": len(entries),
        "n_evidence_labeled": labeled_entries, "errors": errors,
        "recall_at_5": _mean("recall_at_5"),
        "precision_at_5": _mean("precision_at_5"), "mrr": _mean("mrr"),
    }


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _render(rows: list[dict[str, Any]]) -> None:
    print("=" * 85)
    print("检索层 Top-5 评估（块级，三项指标）")
    print("=" * 85)
    print(f"{'层':<4}{'策略':<30}{'Recall@5':>14}{'Precision@5':>16}{'MRR':>12}")
    print("-" * 85)
    for row in rows:
        print(f"{row['key']:<4}{row['name']:<30}{_fmt(row['recall_at_5']):>14}"
              f"{_fmt(row['precision_at_5']):>16}{_fmt(row['mrr']):>12}")
    print("-" * 85)
    if rows:
        print(f"有效正例：{rows[0]['n']}；有证据块标注：{rows[0]['n_evidence_labeled']}；每层按 Top-5 聚合。")

    by_key = {row["key"]: row for row in rows}
    def _delta(left: str, right: str, metric: str) -> str:
        before, after = by_key.get(left), by_key.get(right)
        if not before or not after or before[metric] is None or after[metric] is None:
            return "n/a"
        return f"{before[metric]:.4f} → {after[metric]:.4f}（{after[metric] - before[metric]:+.4f}）"

    for label, left, right in (("混合检索增益", "L0", "L1"), ("重排序影响", "L2", "L3"), ("断崖检测影响", "L3", "L5")):
        print(f"\n{label}（{left} → {right}）：")
        for title, metric in (("Recall@5", "recall_at_5"), ("Precision@5", "precision_at_5"), ("MRR", "mrr")):
            print(f"  {title:<12}{_delta(left, right, metric)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索层块级 Recall@5 / Precision@5 / MRR 评估")
    parser.add_argument("--testset", default=str(DEFAULT_TESTSET), help="测试集路径")
    parser.add_argument("--candidate-k", type=int, default=50, help="L0/L1 与重排序前的候选深度")
    parser.add_argument("--top-k", type=int, default=5, choices=[5], help="固定评估深度")
    parser.add_argument("--sample", type=int, default=0, help="抽样条目数（0=全量）")
    parser.add_argument("--seed", type=int, default=7, help="抽样随机种子")
    parser.add_argument("--allow-missing", action="store_true", help="保留知识库缺失的标注条目")
    parser.add_argument("--only", nargs="*", default=None, help="只评估指定层，如 --only L2 L3")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if Path(args.testset).suffix.lower() == ".jsonl":
        from finance_rag.src.eval.cli import main as unified_main

        print("提示：eval_recall.py 已进入兼容模式，建议直接使用 scripts/evaluate.py retrieval run")
        forwarded = [
            "retrieval", "run", "--dataset", args.testset,
            "--candidate-k", str(args.candidate_k), "--seed", str(args.seed),
        ]
        if args.sample:
            forwarded += ["--sample", str(args.sample)]
        if args.only:
            forwarded += ["--strategy", *args.only]
        return unified_main(forwarded)
    print("警告：Markdown 测试集已弃用；请迁移为 JSONL。当前使用旧版兼容执行路径。")
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
    print(f"评估 {len(entries)} 条有效正例 × {len(specs)} 个检索策略层\n")
    rows = []
    for spec in specs:
        print(f"  [{spec['key']}] {spec['name']} —— {spec['note']}")
        rows.append(_evaluate_layer(kb, entries, spec))
    print()
    _render(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
