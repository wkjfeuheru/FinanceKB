"""路由后的文档内重新检索；仅替换 Milvus 与模型推理边界。"""

import json

import pytest

from src.retrieve.hybrid_retriever import HybridRetriever


def hit(cid, source, score=.9, **metadata):
    return {"id": cid, "distance": score, "entity": {
        "id": cid, "source": source, "content": "证据", "parent_id": "",
        "broker": "东吴证券", "heading_path": "投资评级", **metadata,
    }}


class RoutedClient:
    def __init__(self, global_hits, by_source):
        self.global_hits = global_hits
        self.by_source = by_source
        self.calls = []

    def describe_collection(self, _name):
        return {"fields": [{"name": "broker"}, {"name": "heading_path"}]}

    def _search(self, mode, kwargs):
        self.calls.append((mode, kwargs))
        expression = kwargs.get("filter", "") if mode == "dense" else kwargs["reqs"][0].expr
        for source, hits in self.by_source.items():
            if f"source == {json.dumps(source, ensure_ascii=False)}" in (expression or ""):
                if isinstance(hits, Exception):
                    raise hits
                return [hits]
        return [self.global_hits]

    def search(self, **kwargs):
        return self._search("dense", kwargs)

    def hybrid_search(self, **kwargs):
        return self._search("weighted", kwargs)


class Reranker:
    def __init__(self):
        self.seen = []
        self.calls = 0
        self.texts = []

    def rerank(self, query, candidates, top_n, text_fn):
        self.calls += 1
        self.seen = list(candidates)
        self.texts = [text_fn(c) for c in candidates]
        return sorted(candidates, key=lambda c: c["id"] == "gold", reverse=True)[:top_n]


@pytest.mark.parametrize("mode", ["dense", "weighted"])
def test_source_search_recovers_unseen_chunk_before_single_rerank(mode):
    global_hits = [hit(f"a{i}", "a.pdf", 1 - i * .001) for i in range(30)] + [hit("b0", "b.pdf", .85)]
    client = RoutedClient(global_hits, {
        "a.pdf": [hit("a0", "a.pdf", 1), hit("a1", "a.pdf", .99)],
        "b.pdf": [hit("b0", "b.pdf", .85), hit("gold", "b.pdf", .8)],
    })
    model = Reranker()
    embedded = []
    r = HybridRetriever(client, "kb", lambda q: embedded.append(q) or [.1], reranker=model)
    docs = r.search("2026评级", k=5, rerank_top_n=5, use_rerank=True, expand_parents=False,
                    fusion_mode=mode, dense_weight=.7, sparse_weight=.3)
    assert "gold" in [c["id"] for c in model.seen]
    assert "gold" in [c["id"] for c in docs]
    assert [c["id"] for c in model.seen[:4]] == ["a0", "b0", "a1", "gold"]
    assert len({c["id"] for c in model.seen}) == len(model.seen) == 15
    assert embedded == ["2026评级"]
    assert model.calls == 1
    assert [mode_name for mode_name, _ in client.calls] == [mode, mode, mode]
    assert r._last_diagnostics["coarse_candidate_chunk_ids"] == [f"a{i}" for i in range(30)] + ["b0"]
    assert "gold" not in r._last_diagnostics["coarse_candidate_chunk_ids"]
    assert "gold" in r._last_diagnostics["document_candidate_chunk_ids"]
    assert r._last_diagnostics["rerank_candidate_chunk_ids"] == [c["id"] for c in model.seen]
    # 重排候选文本只注入 heading_path 上下文（不再拼标的名称）
    assert "投资评级" in model.texts[0]
    if mode == "weighted":
        assert all(kwargs["ranker"].dict()["params"]["weights"] == [.7, .3] for _, kwargs in client.calls)


@pytest.mark.parametrize("mode", ["dense", "weighted"])
def test_source_search_retains_filters_version_and_safe_source_literal(mode, monkeypatch):
    from src.retrieve import hybrid_retriever

    monkeypatch.setattr(hybrid_retriever, "ENABLE_VERSIONING", True)
    source = '研报" or id != "\\.pdf'
    client = RoutedClient([hit("global", source)], {source: [hit("gold", source)]})
    r = HybridRetriever(client, "kb", lambda q: [.1], reranker=Reranker())
    r.search("评级", use_rerank=True, expand_parents=False, fusion_mode=mode,
             filters={"report_type": ["个股"], "broker": ["东吴证券"]})
    assert len(client.calls) == 2
    scoped = client.calls[1][1]
    expected = (
        '(report_type in ["个股"] and broker in ["东吴证券"]) and (is_current == true)'
    )
    expressions = [scoped["filter"]] if mode == "dense" else [request.expr for request in scoped["reqs"]]
    assert expressions == [f"({expected}) and (source == {json.dumps(source, ensure_ascii=False)})"] * len(expressions)


def test_document_search_failure_and_shortage_backfill_global_in_original_order():
    global_hits = [hit("a0", "a.pdf", .99), hit("a1", "a.pdf", .98),
                   hit("b0", "b.pdf", .9), hit("b1", "b.pdf", .89), hit("c0", "c.pdf", .8)]
    client = RoutedClient(global_hits, {
        "a.pdf": RuntimeError("单文档请求失败"), "b.pdf": [], "c.pdf": [hit("gold", "c.pdf", .7)],
    })
    model = Reranker()
    r = HybridRetriever(client, "kb", lambda q: [.1], reranker=model)
    r.search("评级", use_rerank=True, k=5, rerank_top_n=5, expand_parents=False,
             fusion_mode="dense", candidate_max_per_source=1)
    assert [c["id"] for c in model.seen] == ["a0", "b0", "gold", "a1", "b1", "c0"]
    assert len(client.calls) == 4
    assert r._last_diagnostics["document_search_failures"] == 1


def test_source_route_is_global_best_score_not_local_score_and_respects_limit():
    client = RoutedClient([hit("a0", "a.pdf", .9), hit("b0", "b.pdf", .99), hit("c0", "c.pdf", .8)], {
        "a.pdf": [hit("a1", "a.pdf", .95)], "b.pdf": [hit("b1", "b.pdf", .1)],
        "c.pdf": [hit("c1", "c.pdf", .99)],
    })
    model = Reranker()
    r = HybridRetriever(client, "kb", lambda q: [.1], reranker=model)
    r.search("评级", use_rerank=True, expand_parents=False, fusion_mode="dense", candidate_source_limit=2)
    assert len(client.calls) == 3
    assert client.calls[1][1]["filter"] == 'source == "b.pdf"'
    assert client.calls[2][1]["filter"] == 'source == "a.pdf"'
    assert [c["id"] for c in model.seen[:2]] == ["b1", "a1"]
    assert "c1" not in [c["id"] for c in model.seen]


def test_parent_expansion_processes_new_document_candidates():
    client = RoutedClient([hit("global", "a.pdf")], {
        "a.pdf": [hit("gold", "a.pdf", parent_id="p1")],
    })

    class Parents:
        def get_batch(self, ids):
            return [{"id": "p1", "content": "证据" + "父块全文" * 30, "heading_path": "投资评级"}]

    model = Reranker()
    r = HybridRetriever(client, "kb", lambda q: [.1], reranker=model, parent_store=Parents())
    r.search("评级", use_rerank=True, fusion_mode="dense")
    assert model.seen[0]["id"] == "gold"
    assert model.seen[0]["content"] == "证据" + "父块全文" * 30


def test_without_rerank_keeps_one_global_search():
    client = RoutedClient([hit("global", "a.pdf")], {"a.pdf": [hit("gold", "a.pdf")]})
    r = HybridRetriever(client, "kb", lambda q: [.1])
    result = r.search("评级", fusion_mode="dense", use_rerank=False)
    assert [c["id"] for c in result] == ["global"]
    assert len(client.calls) == 1


def test_source_routing_precedes_parent_deduplication():
    client = RoutedClient([hit("a", "a.pdf", .8, parent_id="same"),
                           hit("b", "b.pdf", .99, parent_id="same")], {
        "a.pdf": [hit("local-a", "a.pdf")], "b.pdf": [hit("gold", "b.pdf")],
    })

    class Parents:
        def get_batch(self, ids):
            return []

    r = HybridRetriever(client, "kb", lambda q: [.1], reranker=Reranker(), parent_store=Parents())
    r.search("评级", use_rerank=True, fusion_mode="dense", candidate_source_limit=1)
    assert client.calls[1][1]["filter"] == 'source == "b.pdf"'


def test_exact_publication_date_set_builds_safe_in_expression():
    assert HybridRetriever._build_filter({
        "date": ["2026-09-24", "2026-09-27"], "broker": ["东吴证券", "太平洋证券"]
    }) == 'date in ["2026-09-24", "2026-09-27"] and broker in ["东吴证券", "太平洋证券"]'
