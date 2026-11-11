"""检索评估脚本的层级隔离、多深度曲线聚合口径与 A/B 实验目录。"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from scripts.eval_recall import _evaluate_layer, _layer_specs
from src.eval.ab_catalog import select_experiments
from src.eval.retrieval import build_retrieval_strategy


class _FakeKnowledgeBase:
    def hybrid_search(self, *_args, **_kwargs):
        return [{"id": "e1", "source": "doc-a.pdf"}, {"id": "x", "source": "doc-b.pdf"}]


def test_layers_keep_only_requested_controls():
    specs = {
        item["key"]: item
        for item in _layer_specs(Namespace(candidate_k=50, top_ks=[5]))
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
    assert specs["L3"]["rerank_top_n"] == 5


def test_multi_depth_uses_max_k_as_output_depth():
    """多深度曲线：重排输出深度与 L2 最终深度取 max(top_ks)，高深度点不缺数据。"""
    specs = {
        item["key"]: item
        for item in _layer_specs(Namespace(candidate_k=50, top_ks=[5, 8, 10]))
    }

    assert specs["L2"]["k"] == 10
    assert specs["L3"]["rerank_top_n"] == 10
    assert {specs[key]["k"] for key in ("L0", "L1", "L3", "L5")} == {50}
    assert all(spec["top_ks"] == [5, 8, 10] for spec in specs.values())


def test_evaluate_layer_aggregates_only_three_chunk_metrics():
    entry = SimpleNamespace(
        query="q",
        related_docs="doc-a.pdf",
        chunk_ids=["e1", "e2"],
    )
    spec = _layer_specs(Namespace(candidate_k=50, top_ks=[5]))[0]

    row = _evaluate_layer(_FakeKnowledgeBase(), [entry], spec)

    assert row["per_k"][5]["recall"] == 0.5
    assert row["per_k"][5]["precision"] == 0.2
    assert row["per_k"][5]["mrr"] == 1.0
    assert 8 not in row["per_k"]
    assert "doc_coverage" not in row
    assert "hit_rate" not in row


def test_multi_depth_curve_scores_one_retrieval_pass():
    """多深度：每条只检索一次，各深度在同一份 ranked list 上计分。"""
    calls = []

    class CountingKb:
        def hybrid_search(self, *_args, **_kwargs):
            calls.append(1)
            return [{"id": "e1", "source": "doc-a.pdf"}, {"id": "x", "source": "doc-b.pdf"}]

    entry = SimpleNamespace(query="q", related_docs="doc-a.pdf", chunk_ids=["e1", "e2"])
    spec = _layer_specs(Namespace(candidate_k=50, top_ks=[5, 8]))[0]

    row = _evaluate_layer(CountingKb(), [entry], spec)

    assert len(calls) == 1
    assert set(row["per_k"]) == {5, 8}
    # 固定分母 Precision 随深度稀释；Recall 与 MRR 不变
    assert row["per_k"][8]["recall"] == row["per_k"][5]["recall"] == 0.5
    assert row["per_k"][8]["precision"] == 0.125 < row["per_k"][5]["precision"] == 0.2
    assert row["per_k"][8]["mrr"] == row["per_k"][5]["mrr"] == 1.0


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


def test_rerank_topk_experiment_switches_context_size_via_env():
    """K=5 vs K=10：双臂只差 CHAT_RERANK_TOP_K，运行时实验共享默认集合。"""
    experiment = select_experiments(["rerank_topk_5_vs_10"])[0]

    assert experiment.label == "重排后上下文条数（K=5 vs K=10）"
    assert experiment.kind == "runtime"
    assert experiment.arm_a.env_overrides == {"CHAT_RERANK_TOP_K": "5"}
    assert experiment.arm_b.env_overrides == {"CHAT_RERANK_TOP_K": "10"}
    assert experiment.arm_a.strategy_overrides == {}
    assert experiment.arm_b.strategy_overrides == {}
