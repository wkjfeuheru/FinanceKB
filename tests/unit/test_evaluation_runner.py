"""统一运行器的检索、生成、检查点和报告行为。"""

from __future__ import annotations

import json

from finance_rag.src.eval.dataset import EvaluationDataset
from finance_rag.src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput


def _rows():
    positive = {
        "id": "gold-positive",
        "query": "问题？",
        "question_type": "single_hop",
        "difficulty": "medium",
        "answer_type": "factual",
        "reference_answer": "答案",
        "required_facts": ["事实"],
        "evidence": [{
            "source": "a.pdf", "document_version": "v1",
            "chunk_id": "e1", "quote": "证据",
        }],
        "expected_behavior": "", "tier": "gold", "review_status": "approved",
        "reviewer": "alice", "review_note": "", "provenance": {},
    }
    negative = {
        **positive,
        "id": "gold-negative",
        "query": "无答案问题？",
        "question_type": "negative",
        "reference_answer": "",
        "required_facts": [],
        "evidence": [],
        "expected_behavior": "应拒绝回答",
    }
    silver = {
        **positive,
        "id": "silver-positive",
        "query": "另一个问题？",
        "tier": "silver",
        "review_status": "auto_pass",
    }
    return [positive, negative, silver]


def test_retrieval_run_uses_positive_cases_and_writes_artifacts(tmp_path):
    dataset = EvaluationDataset.from_dicts(_rows())
    strategy = EvaluationStrategy(
        name="fake",
        retrieve=lambda _query: RetrievalOutput(
            candidates=[{"id": "e1", "source": "a.pdf"}],
            results=[{"id": "e1", "source": "a.pdf"}],
            cliff_triggered=False,
            pre_cliff_depth=1,
            post_cliff_depth=1,
        ),
    )

    result = EvaluationRunner().run(dataset, strategy, "retrieval", tmp_path)

    assert [row["case_id"] for row in result.records] == ["gold-positive", "silver-positive"]
    assert {key: result.records[0]["metrics"][key] for key in ("recall_at_5", "precision_at_5", "mrr")} == {
        "recall_at_5": 1.0, "precision_at_5": 0.2, "mrr": 1.0,
    }
    assert result.records[0]["diagnostics"]["candidate_chunk_ids"] == ["e1"]
    assert (tmp_path / "records.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "per_query.csv").exists()


def test_runner_resumes_completed_cases_without_retrieving_again(tmp_path):
    dataset = EvaluationDataset.from_dicts([_rows()[0]])
    calls = []

    def retrieve(query):
        calls.append(query)
        return RetrievalOutput([], [{"id": "e1", "source": "a.pdf"}])

    strategy = EvaluationStrategy(name="fake", retrieve=retrieve)
    runner = EvaluationRunner()
    runner.run(dataset, strategy, "retrieval", tmp_path)
    runner.run(dataset, strategy, "retrieval", tmp_path)

    assert calls == ["问题？"]
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_generation_defaults_to_gold_and_reports_negative_rejection(tmp_path):
    dataset = EvaluationDataset.from_dicts(_rows())
    strategy = EvaluationStrategy(
        name="generation",
        retrieve=lambda _query: RetrievalOutput([], [{"id": "e1", "content": "证据"}]),
        generate=lambda case, _contexts: "根据知识库无法回答" if case.question_type == "negative" else "答案",
        evaluate_answer=lambda _case, _contexts, _answer: {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        },
    )

    result = EvaluationRunner().run(dataset, strategy, "generation", tmp_path)

    assert {row["case_id"] for row in result.records} == {"gold-positive", "gold-negative"}
    negative = next(row for row in result.records if row["case_id"] == "gold-negative")
    assert negative["metrics"] == {"negative_rejection": 1.0}
    # 生成侧固定 RAGAS 四指标：answer_correctness 已下线，不得再出现在记录里
    positive = next(row for row in result.records if row["case_id"] == "gold-positive")
    assert set(positive["metrics"]) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    }
    assert positive["diagnostics"]["answer_chars"] == 2
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["metric_coverage"]["faithfulness"] == 0.5
    assert set(summary["generation_metrics"]) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    }
    assert "answer_correctness" not in summary["metrics"]
