"""真实检索编排；仅替换外部 Milvus 与模型推理边界。"""

import asyncio

import pytest

from finance_rag.src.eval.retrieval import build_retrieval_strategy
from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever
from finance_rag.src.services import chat_service


def hit(cid, source, score):
    return {"id": cid, "distance": score, "entity": {
        "id": cid, "content": "相同模板", "source": source,
        "parent_id": "", "security_name": "贵州茅台", "heading_path": "经营分析",
    }}


class Client:
    def __init__(self, hits=None):
        self.hits = hits or []
        self.calls = []

    def describe_collection(self, _name):
        return {"fields": [{"name": "heading_path"}, {"name": "security_name"}]}

    def search(self, **kwargs):
        self.calls.append(("dense", kwargs))
        return [self.hits]

    def hybrid_search(self, **kwargs):
        self.calls.append(("weighted", kwargs))
        return [self.hits]


class Reranker:
    def __init__(self):
        self.seen = []
        self.texts = []

    def rerank(self, _query, candidates, top_n, text_fn):
        self.seen = list(candidates)
        self.texts = [text_fn(c) for c in candidates]
        return candidates[:top_n]


def retriever(client, reranker=None):
    return HybridRetriever(client, "finance_kb", lambda _q: [0.1], reranker=reranker)


def test_default_fusion_uses_explicit_dense_heavy_weights():
    client = Client()
    retriever(client).search("q", expand_parents=False)
    ranker = client.calls[0][1]["ranker"]
    assert ranker.dict()["strategy"] == "weighted"
    assert ranker.dict()["params"]["weights"] == [0.8, 0.2]


@pytest.mark.parametrize("weights", [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)])
def test_evaluation_can_pass_weights(weights):
    client = Client()
    retriever(client).search("q", fusion_mode="weighted", dense_weight=weights[0], sparse_weight=weights[1])
    assert client.calls[0][1]["ranker"].dict()["params"]["weights"] == list(weights)


@pytest.mark.parametrize("query, mode", [
    ("行业景气变化原因", "dense"), ("2026年盈利", "weighted"),
    ("600519评级", "weighted"), ("AIDC订单", "weighted"),
    ('比较“确定性增长”观点', "weighted"), ('比较"投资要点"', "weighted"),
])
def test_auto_routes_only_exact_signals(query, mode):
    client = Client()
    r = retriever(client)
    r.search(query, fusion_mode="auto")
    assert client.calls[0][0] == mode
    assert r._last_diagnostics["fusion_mode"] == mode


def test_legacy_dense_switch_overrides_weighted_mode():
    client = Client()
    retriever(client).search("2026", use_dense_only=True, fusion_mode="weighted")
    assert client.calls[0][0] == "dense"


def test_source_balance_happens_before_rerank_and_preserves_context():
    hits = [hit(f"a{i}", "a.pdf", 1 - i * .001) for i in range(30)]
    hits += [hit(f"b{i}", "b.pdf", .9 - i * .001) for i in range(10)]
    hits += [hit(f"c{i}", "c.pdf", .8 - i * .001) for i in range(5)]
    model = Reranker()
    r = retriever(Client(hits), model)
    result = r.search("q", k=5, use_dense_only=True, expand_parents=False, use_rerank=True, rerank_top_n=5)
    assert len(model.seen) == 15
    assert [c["id"] for c in model.seen[:6]] == ["a0", "b0", "c0", "a1", "b1", "c1"]
    assert {c["source"] for c in result} == {"a.pdf", "b.pdf", "c.pdf"}
    assert "贵州茅台；经营分析" in model.texts[0]
    assert r._last_diagnostics["coarse_source_count"] == 3
    assert r._last_diagnostics["coarse_duplicate_source_slots"] == 42


def test_diagnostics_preserve_coarse_ids_and_actual_rerank_input_ids():
    hits = [hit(f"a{i}", "a.pdf", 1 - i * .001) for i in range(30)]
    hits += [hit("gold", "b.pdf", .9)]
    model = Reranker()
    r = retriever(Client(hits), model)
    r.search("q", k=5, use_dense_only=True, expand_parents=False,
             use_rerank=True, rerank_top_n=5)
    assert r._last_diagnostics.get("coarse_candidate_chunk_ids") == [f"a{i}" for i in range(30)] + ["gold"]
    assert r._last_diagnostics.get("rerank_candidate_chunk_ids") == ["a0", "gold"] + [f"a{i}" for i in range(1, 14)]


def test_source_routing_uses_best_score_and_backfills_original_order():
    hits = [hit(f"a{i}", "a.pdf", .8 - i * .001) for i in range(20)]
    hits += [hit("best", "b.pdf", .99), hit("last", "c.pdf", .1)]
    hits.insert(1, hit("a0", "a.pdf", .8))
    model = Reranker()
    r = retriever(Client(hits), model)
    r.search("q", k=5, use_dense_only=True, expand_parents=False, use_rerank=True,
             rerank_top_n=5, candidate_source_limit=2, candidate_max_per_source=2)
    assert [c["id"] for c in model.seen[:4]] == ["best", "a0", "a1", "a2"]
    assert [c["id"] for c in model.seen[4:]] == [f"a{i}" for i in range(3, 14)]
    assert len({c["id"] for c in model.seen}) == len(model.seen) == 15


@pytest.mark.parametrize("enabled, expected", [(None, 8), (False, 8), (True, 5)])
def test_cliff_is_explicit_and_defaults_off(enabled, expected):
    scores = [1, .95, .2, .05, .04, .03, .02, .01]
    model = Reranker()
    r = retriever(Client([hit(str(i), "a.pdf", s) for i, s in enumerate(scores)]), model)
    kwargs = {} if enabled is None else {"cliff_detection": enabled}
    r.search("q", k=20, use_dense_only=True, expand_parents=False, use_rerank=True, **kwargs)
    assert len(model.seen) == expected
    assert r._last_diagnostics["cliff_enabled"] is (enabled is True)


def test_eval_layers_pass_cliff_without_mutating_global_function():
    from finance_rag.src.rag.retrieval import hybrid_retriever
    original = hybrid_retriever._apply_cliff_detection
    calls = []

    class KB:
        def hybrid_search(self, _query, **kwargs):
            calls.append(kwargs)
            return []

    for layer in ["L3", "L5", "L3"]:
        build_retrieval_strategy(KB(), layer).retrieve("q")
        assert hybrid_retriever._apply_cliff_detection is original
    assert [c["cliff_detection"] for c in calls] == [False, True, False]


def test_merge_uses_chunk_identity_and_preserves_sources():
    docs = [{"id": f"a{i}", "content": "同一模板", "source": "a.pdf", "score": 1 - i * .01} for i in range(6)]
    docs += [{"id": "b0", "content": "同一模板", "source": "b.pdf", "score": .4},
             {"id": "a0", "content": "修改后的同一个块", "source": "a.pdf", "score": .1}]
    merged = chat_service.merge_docs(docs, [], 5)
    assert len(merged) == 5
    assert len({d["id"] for d in merged}) == 5
    assert "b0" in {d["id"] for d in merged}


def test_pipeline_rewrite_failure_falls_back_to_original_query(monkeypatch):
    calls = []

    class KB:
        def hybrid_search(self, query, **_kwargs):
            calls.append(query)
            return [{"id": "c", "source": "a.pdf", "content": "证据", "score": 1}]

    def fail(*_args):
        raise RuntimeError("模型失败")

    monkeypatch.setattr(chat_service, "CHAT_ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(chat_service, "rewrite_query", fail)
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("finance_kb", KB())])
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    result = asyncio.run(chat_service.retrieve_pipeline("原问题", infer_filters=False))
    assert calls == ["原问题"]
    assert result["rewritten"] == "原问题"


@pytest.mark.parametrize("filters, expected", [
    ({"report_type": "个股", "security_name": "贵州茅台", "security_code": "600519"},
     {"report_type": ["个股"], "security_name": ["贵州茅台"]}),
    ({"report_type": "行业", "industry_l1": "电子", "industry_l2": "半导体", "security_name": "贵州茅台"},
     {"report_type": ["行业"], "industry_l1": ["电子"], "industry_l2": ["半导体"]}),
    ({"report_type": "宏观", "industry_l1": "电子", "security_name": "贵州茅台"},
     {"report_type": ["宏观"]}),
])
def test_query_filters_follow_report_type_contract(filters, expected):
    assert chat_service.normalize_metadata_filters(filters) == expected


def test_kb_adapter_forwards_routing_and_exposes_diagnostics():
    from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    client = Client([hit("c", "a.pdf", 1)])
    r = retriever(client)
    kb.ensure_collection = lambda: None
    kb._get_retriever = lambda: r
    docs = kb.hybrid_search("行业趋势", fusion_mode="auto", dense_weight=.7,
                            sparse_weight=.3, cliff_detection=False,
                            candidate_source_limit=3, candidate_max_per_source=4)
    assert docs[0]["id"] == "c"
    assert client.calls[0][0] == "dense"
    assert kb.last_retrieval_diagnostics["coarse_source_count"] == 1


def test_eval_adapter_accepts_fusion_weights():
    client = Client()
    r = retriever(client)

    class KB:
        def hybrid_search(self, query, **kwargs):
            return r.search(query, **kwargs)

    strategy = build_retrieval_strategy(KB(), "L1", fusion_mode="weighted", dense_weight=.9, sparse_weight=.1)
    strategy.retrieve("query")
    assert client.calls[0][1]["ranker"].dict()["params"]["weights"] == [.9, .1]
    assert strategy.config["dense_weight"] == .9


def test_pipeline_retrieves_necessary_subqueries_and_keeps_both_sources(monkeypatch):
    calls = []

    class KB:
        def hybrid_search(self, query, **kwargs):
            calls.append(query)
            if query == "乙公司":
                return [{"id": "b", "source": "b.pdf", "content": "模板", "score": .6},
                        {"id": "a0", "source": "a.pdf", "content": "另一写法", "score": .5}]
            return [{"id": f"a{i}", "source": "a.pdf", "content": "模板", "score": 1 - i * .01} for i in range(5)]

    monkeypatch.setattr(chat_service, "CHAT_ENABLE_QUERY_REWRITE", True)
    from finance_rag.src.agent.planning import QueryPlan, Subproblem
    from finance_rag.src.services import retrieval_service
    monkeypatch.setattr(retrieval_service, "plan_query", lambda *a, **kw: QueryPlan(
        needs_decomposition=True,
        subproblems=[Subproblem(id="q1", query="甲公司"), Subproblem(id="q2", query="乙公司")],
    ))
    monkeypatch.setattr(chat_service, "rewrite_query", lambda *_: {
        "rewritten": "公司对比", "sub_queries": ["甲公司", "乙公司", "乙公司"], "keywords": []})
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: [("finance_kb", KB())])
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    result = asyncio.run(chat_service.retrieve_pipeline("对比", infer_filters=False))
    assert calls == ["甲公司", "乙公司"]
    assert len(result["docs"]) == len({d["id"] for d in result["docs"]}) == 5
    assert {d["source"] for d in result["docs"]} == {"a.pdf", "b.pdf"}


def test_rewrite_without_model_uses_original_only(monkeypatch):
    monkeypatch.setattr(chat_service, "get_rewrite_model", lambda: None)
    monkeypatch.setattr(chat_service, "get_model", lambda: None)
    info = chat_service.rewrite_query("甲公司的评级与盈利增速和乙公司的订单")
    assert info["rewritten"] == "甲公司的评级与盈利增速和乙公司的订单"
    assert info["sub_queries"] == ["甲公司的评级与盈利增速和乙公司的订单"]


def test_legacy_eval_passes_cliff_without_global_mutation():
    from argparse import Namespace
    from types import SimpleNamespace
    from scripts.eval_recall import _evaluate_layer, _layer_specs
    from finance_rag.src.rag.retrieval import hybrid_retriever

    original = hybrid_retriever._apply_cliff_detection
    seen = []

    class KB:
        def hybrid_search(self, query, **kwargs):
            seen.append(kwargs)
            return []

    entry = SimpleNamespace(query="q", chunk_ids=["c"])
    for spec in _layer_specs(Namespace(candidate_k=50, top_k=5)):
        _evaluate_layer(KB(), [entry], spec)
        assert hybrid_retriever._apply_cliff_detection is original
    assert [s["cliff_detection"] for s in seen] == [False, False, False, False, True]


def test_infer_filter_model_receives_security_name_contract(monkeypatch):
    from langchain_core.runnables import RunnableLambda

    prompts = []

    def answer(prompt):
        prompts.append(prompt.to_string())
        return '{"report_type": "个股", "security_name": "贵州茅台"}'

    monkeypatch.setattr(chat_service, "get_model", lambda: RunnableLambda(answer))
    result = chat_service.infer_metadata_filters("贵州茅台的评级")
    assert result == {"report_type": ["个股"], "security_name": ["贵州茅台"]}
    assert "security_name" in prompts[0]
    assert "security_code" not in prompts[0]


def test_unavailable_rewrite_model_does_not_generate_spurious_subqueries(monkeypatch):
    monkeypatch.setattr(chat_service, "get_rewrite_model", lambda: None)
    monkeypatch.setattr(chat_service, "get_model", lambda: None)
    assert chat_service.rewrite_query("2026年银行监管政策")["sub_queries"] == ["2026年银行监管政策"]


def test_security_name_filter_cannot_escape_milvus_string_literal():
    expression = HybridRetriever._build_filter({"security_name": ['公司" or id != "']})
    assert expression == r'security_name in ["公司\" or id != \""]'


def test_mixed_report_filters_use_type_specific_or_branches():
    filters = chat_service.normalize_metadata_filters({
        "report_type": ["个股", "行业", "宏观"], "security_name": "贵州茅台",
        "industry_l1": "食品饮料", "industry_l2": "白酒Ⅱ", "broker": "中信证券",
    })
    expression = HybridRetriever._build_filter(filters)
    assert expression == (
        'broker in ["中信证券"] and '
        '((report_type == "个股" and security_name in ["贵州茅台"]) or '
        '(report_type == "行业" and industry_l1 in ["食品饮料"] and industry_l2 in ["白酒Ⅱ"]) or '
        '(report_type == "宏观"))'
    )


@pytest.mark.parametrize("report_type, fields, expected", [
    ("个股", {"security_name": "贵州茅台", "industry_l1": "电子"},
     'security_name in ["贵州茅台"] and report_type in ["个股"]'),
    ("行业", {"security_name": "贵州茅台", "industry_l1": "电子", "industry_l2": "半导体"},
     'industry_l1 in ["电子"] and industry_l2 in ["半导体"] and report_type in ["行业"]'),
    ("宏观", {"security_name": "贵州茅台", "industry_l1": "电子"},
     'report_type in ["宏观"]'),
])
def test_single_report_filters_stay_strict(report_type, fields, expected):
    filters = chat_service.normalize_metadata_filters({"report_type": report_type, **fields})
    assert HybridRetriever._build_filter(filters) == expected


@pytest.mark.parametrize("weights", [(-.1, 1.1), (.8, -.1), (1.1, .2), (float("nan"), .2),
                                    (.8, float("nan")), (float("inf"), .2), (.8, float("-inf")), (0, 0)])
def test_invalid_weights_fail_before_any_embedding_or_database_call(weights):
    embedded = []
    client = Client()
    r = HybridRetriever(client, "finance_kb", lambda q: embedded.append(q) or [.1])
    with pytest.raises(ValueError, match="权重"):
        r.search("query", dense_weight=weights[0], sparse_weight=weights[1])
    assert embedded == []
    assert client.calls == []
    assert r._schema_fields is None


def test_valid_nonunit_weights_are_normalized():
    client = Client()
    retriever(client).search("q", dense_weight=.2, sparse_weight=.2)
    assert client.calls[0][1]["ranker"].dict()["params"]["weights"] == [.5, .5]


def test_invalid_environment_weight_is_rejected_when_config_loads():
    import os
    import subprocess
    import sys

    env = dict(os.environ, RETRIEVAL_DENSE_WEIGHT="nan", RETRIEVAL_SPARSE_WEIGHT="0.2")
    result = subprocess.run([sys.executable, "-c", "import finance_rag.src.core.config"],
                            env=env, capture_output=True, encoding="utf-8", errors="replace")
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_kb_invalid_weights_do_not_ensure_collection():
    from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase

    touched = []
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.ensure_collection = lambda: touched.append("ensure")
    kb._get_retriever = lambda: retriever(Client())
    with pytest.raises(ValueError, match="权重"):
        kb.hybrid_search("q", dense_weight=float("nan"), sparse_weight=.2)
    assert touched == []
