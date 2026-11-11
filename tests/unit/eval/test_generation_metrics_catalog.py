"""生成侧指标目录：固定为 RAGAS 四指标（fast/full 不再分档的守护）。

历史口径里 ``answer_correctness`` / ``context_entity_recall`` 会在 fast/full 之间
切换；现在两条链路各两项、恒为四项。这个文件守护该口径不被悄悄改回去。
"""

from __future__ import annotations

from src.eval.metrics import GENERATION_METRICS


def test_metric_catalog_is_the_four_ragas_metrics():
    assert GENERATION_METRICS == (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


def test_ragas_metric_set_is_identical_for_fast_and_full():
    from src.eval.ragas_eval import GENERATION_CHAIN_METRICS, _get_metrics

    fast_names = [metric.name for metric in _get_metrics(llm=None, embeddings=None, fast=True)]
    full_names = [metric.name for metric in _get_metrics(llm=None, embeddings=None, fast=False)]

    assert fast_names == full_names
    assert set(fast_names) == set(GENERATION_METRICS)
    assert GENERATION_CHAIN_METRICS == GENERATION_METRICS
    # 已下线的历史指标不得再出现在指标集里
    assert "answer_correctness" not in fast_names
    assert "context_entity_recall" not in fast_names
    assert "noise_sensitivity" not in fast_names


def test_judge_model_name_is_reported_for_provenance():
    from src.eval.ragas_eval import _judge_model_name

    class _Model:
        model_name = "qwen-turbo"

    class _Anonymous:
        pass

    assert _judge_model_name(_Model()) == "qwen-turbo"
    assert _judge_model_name(_Anonymous()) == "_Anonymous"


def test_statistics_specs_keep_retrieval_metrics_as_guardrails():
    """生成侧 A/B 仍以 recall@5 / mrr 作护栏；检索三件套主指标不变。"""
    from src.eval.statistics import EXPERIMENT_SPECS, select_experiment_spec

    generation = EXPERIMENT_SPECS["generation_ab"]
    assert generation.primary_metric == "faithfulness"
    assert set(generation.guardrails) == {
        "answer_relevancy", "context_recall", "recall_at_5", "mrr",
    }
    assert EXPERIMENT_SPECS["rerank_on_off"].primary_metric == "mrr"
    assert EXPERIMENT_SPECS["hybrid_vs_dense"].primary_metric == "recall_at_5"
    assert EXPERIMENT_SPECS["cliff_on_off"].primary_metric == "precision_at_5"

    assert select_experiment_spec("hybrid_vs_dense").name == "hybrid_vs_dense"
    assert select_experiment_spec("unknown", retrieval_only=True).name == "retrieval_only_ab"
    assert select_experiment_spec("unknown").name == "generation_ab"
