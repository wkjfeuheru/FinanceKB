"""断崖检测发生在粗召回之后、重排序之前。"""

from finance_rag.src.eval.retrieval import build_retrieval_strategy
from finance_rag.src.rag.retrieval.hybrid_retriever import BGEReranker, HybridRetriever


def _client(scores: list[float]):
    hits = [
        {
            "id": f"c{i}",
            "distance": score,
            "entity": {"id": f"c{i}", "content": "body", "parent_id": f"p{i}"},
        }
        for i, score in enumerate(scores)
    ]

    class Client:
        def search(self, **_kwargs):
            return [hits]

        def describe_collection(self, _name):
            return {"fields": []}

    return Client()


def test_coarse_cliff_drops_tail_before_rerank():
    """断崖在重排前生效，但受兜底下限约束（floor = max(min_top_k=5, rerank_top_n=3)）。"""
    seen: dict[str, list[str]] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3, text_fn=None):
            seen["ids"] = [item["id"] for item in candidates]
            return candidates[:top_n]

    retriever = HybridRetriever(
        client=_client([1.0, 0.95, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
    )
    results = retriever.search(
        "q", k=20, use_dense_only=True, expand_parents=False,
        use_rerank=True, rerank_top_n=3, cliff_detection=True,
    )

    # 断崖落在 i=1（0.95 → 0.2，跌幅 79%），但兜底要求至少 5 条进重排
    assert seen["ids"] == ["c0", "c1", "c2", "c3", "c4"]
    assert [item["id"] for item in results] == ["c0", "c1", "c2"]
    assert retriever._last_cliff == {"pre": 8, "post": 5}


def test_cliff_floor_never_below_requested_final_count():
    """客户端把 rerank_top_n 调到 10 时，断崖也必须留下 >=10 条候选。"""
    seen: dict[str, int] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3, text_fn=None):
            seen["n"] = len(candidates)
            return candidates[:top_n]

    retriever = HybridRetriever(
        client=_client([1.0] + [0.005] * 19),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
    )
    results = retriever.search(
        "q", k=20, use_dense_only=True, expand_parents=False,
        use_rerank=True, rerank_top_n=10, cliff_detection=True,
    )

    assert seen["n"] == 10                      # floor = max(5, 10)
    assert retriever._last_cliff == {"pre": 20, "post": 10}
    assert len(results) == 10


def test_rerank_does_not_apply_a_second_cliff():
    reranker = BGEReranker()

    class Model:
        def predict(self, pairs):
            assert len(pairs) == 4
            return [0.95, 0.90, 0.40, 0.38]

    reranker._get_model = lambda: Model()
    results = reranker.rerank(
        "q",
        [{"content": str(i), "score": 1 - i * 0.01} for i in range(4)],
        top_n=4,
    )

    assert len(results) == 4


def test_parent_expansion_keeps_coarse_pool_when_reranking():
    seen: dict[str, int] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3, text_fn=None):
            seen["n"] = len(candidates)
            return candidates[:top_n]

    class Parents:
        def get_batch(self, parent_ids):
            return [
                {"id": pid, "content": "x" * 80, "heading": "", "heading_path": ""}
                for pid in parent_ids
            ]

    retriever = HybridRetriever(
        client=_client([1.0, 0.98, 0.96, 0.94]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
        parent_store=Parents(),
    )
    retriever.search(
        "q", k=1, use_dense_only=True, expand_parents=True,
        use_rerank=True, rerank_top_n=2,
    )

    assert seen["n"] == 4


def test_keyword_boost_keeps_coarse_pool_when_reranking():
    seen: dict[str, int] = {}

    class Reranker:
        def rerank(self, _query, candidates, top_n=3, text_fn=None):
            seen["n"] = len(candidates)
            return candidates[:top_n]

    retriever = HybridRetriever(
        client=_client([1.0, 0.98, 0.96, 0.94]),
        collection_name="finance_kb",
        embed_query_fn=lambda _text: [0.1],
        reranker=Reranker(),
    )
    retriever.search(
        "q", k=1, use_dense_only=True, expand_parents=False,
        use_rerank=True, rerank_top_n=2, keywords=["missing-token"],
    )

    assert seen["n"] == 4


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
