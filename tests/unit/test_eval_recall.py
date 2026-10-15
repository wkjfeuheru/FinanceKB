"""检索 Top-5 评估脚本的层级隔离和聚合口径。"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from scripts.eval_recall import _evaluate_layer, _layer_specs
from finance_rag.src.eval.retrieval import build_retrieval_strategy


class _FakeKnowledgeBase:
    def hybrid_search(self, *_args, **_kwargs):
        return [{"id": "e1", "source": "doc-a.pdf"}, {"id": "x", "source": "doc-b.pdf"}]


def test_layers_keep_only_requested_controls():
    specs = {
        item["key"]: item
        for item in _layer_specs(
            Namespace(candidate_k=50, rerank_top_n=5, top_k=5, cliff_threshold=0.1)
        )
    }

    assert set(specs) == {"L0", "L1", "L2", "L3", "L5"}
    assert specs["L0"]["use_dense_only"] is True
    assert specs["L1"]["use_dense_only"] is False
    assert specs["L2"]["use_rerank"] is False
    assert specs["L3"]["use_rerank"] is True
    assert specs["L3"]["cliff_detection"] is False
    assert specs["L5"]["cliff_detection"] is True
    # L2 无重排层，其 k 就是最终深度；其余层用粗排候选深度
    assert specs["L2"]["k"] == 5
    assert {specs[key]["k"] for key in ("L0", "L1", "L3", "L5")} == {50}


def test_evaluate_layer_aggregates_only_three_chunk_metrics():
    entry = SimpleNamespace(
        query="q",
        related_docs="doc-a.pdf",
        chunk_ids=["e1", "e2"],
    )
    spec = _layer_specs(
        Namespace(candidate_k=50, rerank_top_n=5, top_k=5, cliff_threshold=0.1)
    )[0]

    row = _evaluate_layer(_FakeKnowledgeBase(), [entry], spec)

    assert row["recall_at_5"] == 0.5
    assert row["precision_at_5"] == 0.2
    assert row["mrr"] == 1.0
    assert "doc_coverage" not in row
    assert "hit_rate" not in row


def test_unified_block_retrieval_disables_parent_expansion():
    captured = {}

    class Kb:
        def hybrid_search(self, _query, **kwargs):
            captured.update(kwargs)
            return [{"id": "e1"}]

    strategy = build_retrieval_strategy(Kb(), "L0")
    strategy.retrieve("q")

    assert captured["expand_parents"] is False
    # 统一粗排深度：默认 candidate_k=50（L2 例外，见下）
    assert captured["k"] == 50


def test_l2_keeps_final_depth_as_k():
    """L2 无重排层：它的 k 就是最终深度，不能被粗排深度顶替。"""
    captured = {}

    class Kb:
        def hybrid_search(self, _query, **kwargs):
            captured.update(kwargs)
            return [{"id": "e1"}]

    build_retrieval_strategy(Kb(), "L2").retrieve("q")

    assert captured["k"] == 5


def test_l3_requests_unified_coarse_depth():
    captured = {}

    class Kb:
        def hybrid_search(self, _query, **kwargs):
            captured.update(kwargs)
            return [{"id": "e1"}]

    build_retrieval_strategy(Kb(), "L3").retrieve("q")

    assert captured["k"] == 50
    assert captured["rerank_top_n"] == 5


def test_l5_records_coarse_cliff_not_final_top_k():
    class Kb:
        last_retrieval_cliff = {"pre": 20, "post": 8}

        def hybrid_search(self, *_args, **_kwargs):
            return [{"id": str(i)} for i in range(5)]

    output = build_retrieval_strategy(Kb(), "L5").retrieve("q")

    assert output.cliff_triggered is True
    assert output.pre_cliff_depth == 20
    assert output.post_cliff_depth == 8
    assert len(output.results) == 5
