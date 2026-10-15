"""评测指标单测：证据块级 / 文档覆盖度 / 弱标注检测 / 分层采样。

背景：``multi_hop_hit`` 曾恒为 0.0000，而 ``hit_rate``/``context_recall`` 恒为 1.0。
根因有三层，本文件守护对应的修复：

1. 检索结果里**没有 chunk id**（``HybridRetriever`` 未取 id 字段）→ 块级指标无法计算；
2. 指标只有文档级（命中一篇即 1.0）与"整份文档集全覆盖"两档，中间没有梯度；
3. 测试集未标注证据的条目被算成假 0.0，污染均值。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from finance_rag.src.eval.ab_runner import (
    _flag_weak_multi_hop,
    _stratified_sample,
)
from finance_rag.src.eval.ragas_eval import (
    _compute_chunk_retrieval_metrics,
    _compute_doc_coverage,
    _compute_evidence_metrics,
    _compute_local_retrieval_metrics,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. 文档级覆盖度：区分「命中一篇」与「全部命中」
# ---------------------------------------------------------------------------


def test_doc_coverage_gives_gradient_on_multi_hop():
    related = "反洗钱特别预防措施管理办法.pdf, 银行保险机构数据安全管理办法.pdf"

    # 只召回一篇：hit_rate 会是 1.0（饱和），doc_coverage 才是 0.5
    one = _compute_doc_coverage(["反洗钱特别预防措施管理办法.pdf"], related)
    assert one == {"doc_coverage": 0.5, "doc_recall": 0.0}

    both = _compute_doc_coverage(
        ["反洗钱特别预防措施管理办法.pdf", "银行保险机构数据安全管理办法.pdf"],
        related,
    )
    assert both == {"doc_coverage": 1.0, "doc_recall": 1.0}

    none = _compute_doc_coverage(["无关文档.pdf"], related)
    assert none == {"doc_coverage": 0.0, "doc_recall": 0.0}


def test_hit_rate_saturates_where_doc_coverage_does_not():
    """同一份只召回 1/2 篇的检索结果：hit_rate=1.0 但 doc_recall=0.0。

    这正是「指标已饱和、multi_hop_hit 恒为 0」无法归因的根源。
    """
    related = "a.pdf, b.pdf"
    retrieved = ["a.pdf"]
    assert _compute_local_retrieval_metrics(retrieved, related, k=5)["hit_rate"] == 1.0
    assert _compute_doc_coverage(retrieved, related)["doc_recall"] == 0.0


def test_doc_coverage_returns_none_without_annotation():
    """无相关文档标注 → n/a，而不是假 0.0。"""
    assert _compute_doc_coverage(["a.pdf"], "") == {
        "doc_coverage": None,
        "doc_recall": None,
    }


def test_doc_coverage_respects_k_and_dedups():
    related = "a.pdf, b.pdf"
    # b 排在 k 之外 → 不计入覆盖
    assert _compute_doc_coverage(["a.pdf", "x.pdf", "b.pdf"], related, k=2)[
        "doc_coverage"
    ] == 0.5
    # 重复出现的同一文档只算一次
    assert _compute_doc_coverage(["a.pdf", "a.pdf"], "a.pdf, b.pdf")["doc_coverage"] == 0.5


# ---------------------------------------------------------------------------
# 2. 证据块级指标
# ---------------------------------------------------------------------------


def test_chunk_metrics_use_fixed_top5_precision_denominator():
    """不足五条结果时，Precision@5 仍以五为分母。"""
    result = _compute_chunk_retrieval_metrics(["x", "e1"], ["e1", "e2"], k=5)

    assert result == {
        "recall_at_k": 0.5,
        "precision_at_k": 0.2,
        "mrr": 0.5,
    }


def test_chunk_metrics_ignore_results_after_top5_and_allow_missing_labels():
    """Top-5 外的命中不得计分；无块标注返回 n/a。"""
    assert _compute_chunk_retrieval_metrics(
        ["x", "x2", "x3", "x4", "x5", "e1"], ["e1"], k=5
    ) == {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "mrr": 0.0,
    }
    assert _compute_chunk_retrieval_metrics(["e1"], []) == {
        "recall_at_k": None,
        "precision_at_k": None,
        "mrr": None,
    }


def test_evidence_metrics_partial_and_full_coverage():
    partial = _compute_evidence_metrics(["x", "e1", "y"], ["e1", "e2"], k=5)
    assert partial["evidence_hit"] == 1.0
    assert partial["evidence_coverage"] == 0.5
    assert partial["evidence_rank"] == 2.0
    assert partial["evidence_all_in_topk"] == 0.0

    full = _compute_evidence_metrics(["e1", "e2"], ["e1", "e2"], k=5)
    assert full["evidence_coverage"] == 1.0
    assert full["evidence_all_in_topk"] == 1.0


def test_evidence_rank_uses_k_plus_one_when_missing():
    """未命中记 k+1，避免与「排第 1」混淆。"""
    miss = _compute_evidence_metrics(["x", "y"], ["e1"], k=5)
    assert miss["evidence_hit"] == 0.0
    assert miss["evidence_rank"] == 6.0
    first = _compute_evidence_metrics(["e1"], ["e1"], k=5)
    assert first["evidence_rank"] == 1.0


def test_evidence_metrics_na_without_annotation():
    """未标注证据块 → 全部 None（n/a），不产生假 0.0 污染均值。"""
    result = _compute_evidence_metrics(["x", "y"], [], k=5)
    assert result == {
        "evidence_hit": None,
        "evidence_coverage": None,
        "evidence_rank": None,
        "evidence_all_in_topk": None,
    }
    assert _compute_evidence_metrics(["x"], None, k=5)["evidence_hit"] is None


def test_evidence_metrics_dedups_repeated_evidence():
    """证据块重复标注不应把覆盖率算超 1.0。"""
    result = _compute_evidence_metrics(["e1"], ["e1", "e1"], k=5)
    assert result["evidence_coverage"] == 1.0
    assert result["evidence_all_in_topk"] == 1.0


# ---------------------------------------------------------------------------
# 3. 弱标注检测：把「检索失败」与「评测集缺陷」分开
# ---------------------------------------------------------------------------


def test_flag_weak_multi_hop_detects_unretrieved_document():
    diagnostics = [
        {   # 标注的第二篇文档完全没被召回 → 疑似标注不可信
            "query": "立法目的是什么",
            "expected_docs": "反洗钱特别预防措施管理办法.pdf, 银行保险机构数据安全管理办法.pdf",
            "retrieved_docs": "反洗钱特别预防措施管理办法",
            "doc_coverage": 0.5,
            "evidence_coverage": 0.5,
            "evidence_rank": 1.0,
        },
        {   # 一篇都没召回 → 命中「疑似」判定
            "query": "黄油与大炮",
            "expected_docs": "宏观经济报告.pdf, 内部内控与组织权责管理制度.md",
            "retrieved_docs": "完全无关文档",
            "doc_coverage": 0.0,
            "evidence_coverage": 0.0,
            "evidence_rank": 6.0,
        },
    ]
    flagged = _flag_weak_multi_hop(diagnostics)
    assert len(flagged) == 1
    assert flagged[0]["query"] == "黄油与大炮"
    # missing_docs 以去扩展名的 stem 形式给出
    assert "内部内控与组织权责管理制度" in flagged[0]["missing_docs"]
    assert "宏观经济报告" in flagged[0]["missing_docs"]


def test_flag_weak_multi_hop_empty_when_all_retrieved():
    diagnostics = [{
        "query": "q",
        "expected_docs": "a.pdf, b.pdf",
        "retrieved_docs": "a;b",
        "doc_coverage": 1.0,
        "evidence_coverage": 1.0,
        "evidence_rank": 1.0,
    }]
    assert _flag_weak_multi_hop(diagnostics) == []


# ---------------------------------------------------------------------------
# 4. 分层采样
# ---------------------------------------------------------------------------


class _Entry:
    def __init__(self, question_type: str, name: str):
        self.question_type = question_type
        self.name = name


def _entries(spec: dict[str, int]) -> list[_Entry]:
    out: list[_Entry] = []
    for qtype, n in spec.items():
        out.extend(_Entry(qtype, f"{qtype}-{i}") for i in range(n))
    return out


def test_stratified_sample_covers_every_type():
    """三类各 1 题时，sample=3 必须每类都抽到（旧实现可能整类缺失）。"""
    entries = _entries({"single_hop": 10, "multi_hop": 2, "negative": 2})
    picked = _stratified_sample(entries, 3, seed=7)
    assert {e.question_type for e in picked} == {"single_hop", "multi_hop", "negative"}
    assert len(picked) == 3


def test_stratified_sample_is_reproducible_and_exact():
    entries = _entries({"single_hop": 20, "multi_hop": 8, "negative": 4})
    first = [e.name for e in _stratified_sample(entries, 9, seed=42)]
    second = [e.name for e in _stratified_sample(entries, 9, seed=42)]
    assert first == second
    assert len(first) == 9


def test_stratified_sample_never_exceeds_bucket_size():
    entries = _entries({"single_hop": 2, "multi_hop": 1})
    picked = _stratified_sample(entries, 10, seed=1)
    assert len(picked) == 3  # 不足时按实际题量返回，不报错


def test_stratified_sample_distributes_surplus():
    """余量补给题量最多的类型，且不超出该类型题量。"""
    entries = _entries({"single_hop": 5, "multi_hop": 1})
    picked = _stratified_sample(entries, 4, seed=3)
    counts = {}
    for entry in picked:
        counts[entry.question_type] = counts.get(entry.question_type, 0) + 1
    assert counts["multi_hop"] == 1
    assert counts["single_hop"] == 3


# ---------------------------------------------------------------------------
# 5. 既有报告回归：坏数据的表现必须可解释
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "csv_path",
    sorted((_REPO_ROOT / "scripts" / "results" / "ab").glob("*/*/per_query.csv")),
    ids=lambda p: p.parent.parent.name,
)
def test_existing_reports_expose_multi_hop_state(csv_path: Path):
    """对既有报告 CSV：多跳行的 hit_rate 已饱和、chunk_ids 已标注。

    该测试固化「问题确实存在」这一事实，防止将来悄悄改动指标口径后无人察觉。
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    multi = [r for r in rows if r.get("question_type") == "multi_hop"]
    if not multi:
        pytest.skip("该报告无 multi_hop 样本")
    for row in multi:
        # 相关文档已标注 → 文档级指标有值
        assert row.get("related_docs")
        # 证据块也已标注（说明 id 数据是有的，只是此前没被用于指标）
        assert (row.get("chunk_ids") or "").strip()
        assert float(row["hit_rate"]) == 1.0
    # 旧报告的 multi_hop_hit 全为 0（这正是要修的现象）
    assert all(float(r["multi_hop_hit"]) == 0.0 for r in multi)
