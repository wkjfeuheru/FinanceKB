"""生产与评测共享执行契约，模型和存储全部离线替代。"""

from types import SimpleNamespace

import pytest

from src.eval.ab_catalog import ABArm, ABExperiment
from src.eval.retrieval import adapt_production_output, build_retrieval_strategy
from src.answer import generator as answer_service
from src.retrieve import pipeline as retrieval_service


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    from src.core import config
    from src import engine as chat_service
    monkeypatch.setattr(config, "get_model", lambda: None)
    monkeypatch.setattr(chat_service, "get_model", lambda: None)
    monkeypatch.setattr(chat_service, "get_rewrite_model", lambda: None)
    async def retrieve(query, **kwargs):
        return _pipeline()
    async def stream(query, **kwargs):
        yield {"type": "token", "content": "无法回答"}
        yield {"type": "done", "answer_rejected": True}
    monkeypatch.setattr(chat_service, "retrieve_documents", retrieve)
    monkeypatch.setattr(chat_service, "chat_stream", stream)


def _pipeline():
    return {"docs": [{"id": "child", "source": "doc.pdf", "content": "原始摘要", "covered_child_ids": ["other"]}],
            "evidence_pool": [{"id": "child", "source": "doc.pdf", "content": "原始摘要"}],
            "context_docs": [{"id": "child", "source": "doc.pdf", "content": "预算内完整表格"}],
            "context_text": "[1] 原文预算快照", "sources": [{"source": "doc.pdf"}],
            "decision": "generate", "plan": {}, "coverage": {}, "diagnostics": {"retry_count": 1},
            "retrieval_error": None}


def test_adapt_consumes_context_snapshot_without_reexpansion(monkeypatch):
    from src import engine as chat_service
    monkeypatch.setattr(chat_service, "expand_table_docs", lambda docs: (_ for _ in ()).throw(AssertionError("不能重复展开")))
    pipeline = _pipeline()
    output = adapt_production_output(pipeline)
    assert output.context_results == pipeline["context_docs"]
    assert output.diagnostics["context_text"] == pipeline["context_text"]
    output.context_results[0]["content"] = "改动输出"
    assert pipeline["context_docs"][0]["content"] == "预算内完整表格"


def test_production_positive_negative_share_policy_executor_and_injected_kb(monkeypatch):
    calls = []
    kb = object()
    policy = retrieval_service.RetrievalPolicy(mode="agentic")
    monkeypatch.setattr(retrieval_service, "resolve_policy", lambda **kwargs: policy)
    async def execute(request, selected_policy, dependencies=None, **kwargs):
        calls.append((request.query, selected_policy, kwargs["kb"]))
        return _pipeline()
    async def answer(query, pipeline, **kwargs):
        assert pipeline["context_text"] == "[1] 原文预算快照"
        return {"answer": "无法回答", "answer_rejected": True, "answer_status": "refused"}
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)
    monkeypatch.setattr(answer_service, "answer_pipeline", answer)
    strategy = build_retrieval_strategy(kb, "PRODUCTION")
    assert strategy.retrieve("正题").results[0]["id"] == "child"
    outcome = strategy.evaluate_negative(SimpleNamespace(query="负题"))
    assert outcome["answer_rejected"] is True
    assert calls == [("正题", policy, kb), ("负题", policy, kb)]


def test_ab_arm_uses_shared_executor_and_retrieval_only_disables_grading(monkeypatch):
    from src.eval import ab_runner
    calls = []
    async def execute(request, policy, dependencies=None, **kwargs):
        calls.append((request, policy, kwargs["kb"]))
        return _pipeline()
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)
    arm = ABArm("a", "离线", strategy_overrides={"retrieval_only": True, "rewrite": False, "route": True, "reflect": False})
    experiment = ABExperiment("x", "x", arm, arm)
    kb = object()
    pipeline = ab_runner._pipeline_for_arm(experiment, arm, kb, "问题")
    assert pipeline["context_text"] == "[1] 原文预算快照"
    assert calls[0][1].grade is False
    assert calls[0][1].use_tools is False
    assert calls[0][2] is kb


def test_ab_answer_calls_shared_validation_exit(monkeypatch):
    from src.eval import ab_runner
    calls = []
    async def answer(query, pipeline, **kwargs):
        calls.append((query, pipeline))
        return {"answer": "经过声明核验的答案", "answer_status": "complete"}
    monkeypatch.setattr(answer_service, "answer_pipeline", answer)
    pipeline = _pipeline()
    assert ab_runner._generate_answer("问题", pipeline) == "经过声明核验的答案"
    assert calls == [("问题", pipeline)]


def test_agentic_compatibility_wrapper_delegates_to_shared_executor(monkeypatch):
    from src.eval.agentic_retrieval import retrieve_agentic
    calls = []
    async def execute(request, policy, dependencies=None, **kwargs):
        calls.append((request, policy, dependencies, kwargs["kb"]))
        return _pipeline()
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)
    kb = object()
    assert retrieve_agentic("问题", rewrite=False, route=False, reflect=False, kb=kb)[0]["id"] == "child"
    assert calls[0][3] is kb
    assert calls[0][1].grade is False


@pytest.mark.parametrize("retrieval_only", [True, False])
@pytest.mark.parametrize("judge_name", ["offline-judge", "unavailable", None])
def test_execute_arm_keeps_snapshot_rows_groups_and_run_fingerprints(monkeypatch, tmp_path, retrieval_only, judge_name):
    from src.eval import ab_runner, ragas_eval, testset_generator
    from src.eval.test_set import TestSetEntry, TestSetLoader
    from src.index import milvus_kb
    from src import engine as chat_service

    entry = TestSetEntry(query="问题", unique_id="a", ground_truth="原始摘要", related_docs="doc.pdf", chunk_ids=("child",), evidence_snippets=("预算内完整表格",))
    kb = SimpleNamespace(collection_name="offline", list_documents=lambda: [{"source": "doc.pdf"}], get_fingerprint_store=lambda: {"doc.pdf": {"sha256": "f" * 64}})
    monkeypatch.setattr(milvus_kb, "get_knowledge_base", lambda: kb)
    monkeypatch.setattr(TestSetLoader, "load_test_set", lambda self: [entry])
    monkeypatch.setattr(testset_generator, "filter_entries_by_kb", lambda entries, kb, **kwargs: (entries, []))
    def retrieve(*args):
        pipeline = _pipeline()
        pipeline["docs"][0].update(id="parent", child_id="child")
        return pipeline
    monkeypatch.setattr(ab_runner, "_pipeline_for_arm", retrieve)
    monkeypatch.setattr(ab_runner, "_answer_for_arm", lambda query, pipeline: {"answer": "核验通过[1]", "answer_status": "complete", "answer_rejected": False, "citation_validation": {"status": "verified"}})
    monkeypatch.setattr(chat_service, "expand_table_docs", lambda docs: (_ for _ in ()).throw(AssertionError("不得重做上下文")))
    class Evaluator:
        def __init__(self, kb):
            pass
        def evaluate_batch(self, rows, **kwargs):
            return {"per_query": [{**row, "metrics": {"faithfulness": 1.0}} for row in rows], "metric_errors": {}, "fallback_metrics": [], "judge_model_name": judge_name}
    monkeypatch.setattr(ragas_eval, "RagasBatchEvaluator", Evaluator)
    arm = ABArm("a", "离线", strategy_overrides={"retrieval_only": retrieval_only})
    path = tmp_path / "data.md"
    path.write_text("只读数据", encoding="utf-8")
    result = ab_runner.execute_arm(ABExperiment("x", "x", arm, arm), arm, testset_path=path, fast=True, sample=0, seed=7, allow_missing=True)
    row = result["per_query"][0]
    assert row["retrieved_contexts"] == ["预算内完整表格"]
    assert row["sources"][0]["id"] == "parent"
    assert row["sources"][0]["child_id"] == "child"
    assert row["sources"][0]["covered_child_ids"] == ["other"]
    assert row["metrics"]["recall_at_5"] == 1
    assert row["metrics"]["context_evidence_recall_at_5"] == 1
    assert row["sample_group_id"]
    assert row["run_fingerprints"]["protocol_version"] == "agentic-v2"
    assert result["run_fingerprints"]["dataset_fingerprint"]
    assert result["run_fingerprints"]["kb_fingerprint"]
    if retrieval_only or judge_name == "offline-judge":
        assert result["run_fingerprints"]["judge_fingerprint"]
    else:
        assert result["run_fingerprints"]["judge_fingerprint"] is None
        assert result["run_fingerprints"]["comparable"] is False
    if not retrieval_only:
        assert row["answer_status"] == "complete"
        assert row["citation_validation"]["status"] == "verified"


def test_ab_statistics_transports_groups_and_strict_fingerprints():
    from src.eval.ab_runner import _run_statistics

    arm_a, arm_b = ABArm("before", "before"), ABArm("after", "after")
    fingerprints = {"protocol_version": "agentic-v2", "protocol_fingerprint": "p", "dataset_fingerprint": "d", "kb_fingerprint": "k", "judge_fingerprint": "j"}
    def result(value):
        return {"protocol_version": "agentic-v2", "run_fingerprints": fingerprints,
                "per_query": [{"case_id": case, "sample_group_id": group, "tier": "gold", "metrics": {"faithfulness": value, "answer_relevancy": 1, "context_recall": 1, "recall_at_5": 1, "mrr": 1}}
                              for case, group in [("a", "g1"), ("variant", "g1"), ("b", "g2")]]}
    experiment = ABExperiment("x", "x", arm_a, arm_b)
    report = _run_statistics(experiment, {"before": result(0), "after": result(1)}, [arm_a, arm_b], seed=7, bootstrap=30, retrieval_only=False)
    assert report["tiers"]["gold"]["metrics"]["faithfulness"]["n"] == 2
    assert report["minimum_gold_coverage"] == 0.95
    assert report["can_claim_improvement"] is True


def test_legacy_planner_over_limit_returns_no_answerable_docs():
    from src.eval.agentic_retrieval import retrieve_agentic

    searches = []
    def search(queries, **kwargs):
        searches.append(queries)
        return [{"id": "a", "content": "证据"}]
    result = retrieve_agentic("完整问题", rewrite=True, route=False, reflect=False, search=search,
                              rewrite_query_fn=lambda query: {"rewritten": "一", "sub_queries": ["二", "三", "四"]})
    assert result == []
    assert searches == [["完整问题"]]


def test_legacy_adapter_preserves_successful_retrieval():
    from src.eval.agentic_retrieval import retrieve_agentic

    searches = []
    docs = [{"id": "a", "content": "证据"}]

    def search(queries, **kwargs):
        searches.append(queries)
        return docs

    result = retrieve_agentic("完整问题", rewrite=True, route=False, reflect=False, search=search,
                              rewrite_query_fn=lambda query: {"rewritten": "改写问题", "sub_queries": []})
    assert result == docs
    assert searches == [["完整问题", "改写问题"]]


def test_legacy_adapter_preserves_retrieval_errors():
    from src.core.exceptions import VectorStoreError
    from src.eval.agentic_retrieval import retrieve_agentic

    class UnavailableKB:
        collection_name = "offline"

        def hybrid_search(self, query, **kwargs):
            raise VectorStoreError("检索服务不可用")

    with pytest.raises(RuntimeError, match="检索"):
        retrieve_agentic("完整问题", kb=UnavailableKB())


@pytest.mark.parametrize("limit", [-1, 3])
def test_legacy_adapter_rejects_out_of_protocol_retry_limit(limit):
    from src.eval.agentic_retrieval import retrieve_agentic

    with pytest.raises(ValueError):
        retrieve_agentic("问题", max_reflect_rounds=limit)


@pytest.mark.parametrize("answer", [{"answer": "拒答"}, {"answer_status": "refused"}])
def test_production_negative_requires_complete_answer_result(monkeypatch, answer):
    async def execute(*args, **kwargs):
        return _pipeline()
    async def generate(*args, **kwargs):
        return answer
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)
    monkeypatch.setattr(answer_service, "answer_pipeline", generate)
    with pytest.raises(RuntimeError, match="生产回答缺少完成结果"):
        build_retrieval_strategy(object(), "PRODUCTION").evaluate_negative(SimpleNamespace(query="问题"))


def test_production_unverified_answer_is_not_a_verified_rejection(monkeypatch):
    async def execute(*args, **kwargs):
        return _pipeline()
    async def generate(*args, **kwargs):
        return {"answer": "暂时无法回答", "answer_status": "unverified", "answer_rejected": True}
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)
    monkeypatch.setattr(answer_service, "answer_pipeline", generate)
    outcome = build_retrieval_strategy(object(), "PRODUCTION").evaluate_negative(SimpleNamespace(query="问题"))
    assert outcome["answer_rejected"] is None
    assert outcome["evaluation_status"] == "unverified"
    assert outcome["answer"] == "暂时无法回答"


@pytest.mark.parametrize("sha", [None, "", "unknown", "unavailable", "sha-a", "x" * 64])
def test_pdf_fingerprint_rejects_missing_or_invalid_identity(sha):
    from src.eval.retrieval import knowledge_content_fingerprints
    kb = SimpleNamespace(list_documents=lambda: [{"source": "a.pdf"}], get_fingerprint_store=lambda: {"a.pdf": {"sha256": sha}})
    assert knowledge_content_fingerprints(kb)["pdf_set_fingerprint"] == "unavailable"


def test_pdf_fingerprint_checks_real_document_set_not_store_keys():
    from src.eval.retrieval import knowledge_content_fingerprints
    kb = SimpleNamespace(list_documents=lambda: [{"source": "a.pdf"}, {"source": "b.pdf"}], get_fingerprint_store=lambda: {"a.pdf": {"sha256": "a" * 64}})
    assert knowledge_content_fingerprints(kb)["pdf_set_fingerprint"] == "unavailable"


def test_pdf_fingerprint_can_establish_complete_sources_from_vector_scan():
    from src.eval.retrieval import knowledge_content_fingerprints
    class Iterator:
        def __init__(self):
            self.rows = [[{"id": "a", "source": "a.pdf", "content": "证据"}], []]
        def next(self):
            return self.rows.pop(0)
        def close(self):
            pass
    kb = SimpleNamespace(collection_name="offline", _get_client=lambda: SimpleNamespace(query_iterator=lambda **kwargs: Iterator()), get_fingerprint_store=lambda: {"a.pdf": {"sha256": "a" * 64}})
    result = knowledge_content_fingerprints(kb)
    assert len(result["pdf_set_fingerprint"]) == 64
    assert result["chunk_content_count"] == 1


def test_parent_evidence_coverage_uses_slots_and_visible_same_source_quotes():
    from src.eval.ab_runner import _evidence_slot_metrics
    refs = [{"chunk_id": "c1", "source": "doc.pdf", "quote": "事实一"},
            {"chunk_id": "c2", "source": "doc.pdf", "quote": "事实二"}]
    docs = [{"id": "representative", "source": "doc.pdf", "content": "事实一，事实二", "covered_child_ids": ["c1", "c2"]}]
    metrics = _evidence_slot_metrics(docs, refs, k=5)
    assert metrics["recall_at_5"] == 1
    assert metrics["precision_at_5"] == 0.2
    assert metrics["chunk_mrr"] == 1
    assert metrics["evidence_rank"] == 1
    assert _evidence_slot_metrics([{**docs[0], "content": "事实一"}], refs)["recall_at_5"] == 0.5
    assert _evidence_slot_metrics([{**docs[0], "source": "other.pdf"}], refs)["recall_at_5"] == 0
    assert _evidence_slot_metrics([{**docs[0], "content": "标签覆盖但没有原文"}], refs)["recall_at_5"] == 0


def test_parent_after_fifth_slot_is_not_counted_as_top5_evidence():
    from src.eval.ab_runner import _evidence_slot_metrics
    docs = [{"id": str(i), "source": "doc.pdf", "content": "无关"} for i in range(5)]
    docs.append({"id": "parent", "source": "doc.pdf", "content": "事实一", "covered_child_ids": ["gold"]})
    assert _evidence_slot_metrics(docs, [{"chunk_id": "gold", "source": "doc.pdf", "quote": "事实一"}])["recall_at_5"] == 0
