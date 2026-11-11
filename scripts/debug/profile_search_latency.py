"""Read-only in-process search profiler. Run in the app container via stdin.

Wrappers live only in this diagnostic process; the serving process is untouched.
"""

import argparse
import functools
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="新能源汽车")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--benchmark-rerank", action="store_true")
    args = parser.parse_args()
    from src.api.schemas.search import SearchRequest
    from src.index.milvus_kb import KnowledgeBase
    from src.index.onnx_embedder import OnnxEmbedder
    from src.retrieve.hybrid_retriever import BGEReranker, HybridRetriever
    from src.retrieve.search_service import search_research

    timings = {}
    captured = []

    def instrument(cls, name):
        original = getattr(cls, name)

        @functools.wraps(original)
        def timed(*a, **kw):
            if cls is BGEReranker and name == "rerank":
                text_fn = kw.get("text_fn")
                captured[:] = [[a[1], text_fn(c) if text_fn else c["content"][:2000]] for c in a[2]]
            started = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                key = cls.__name__ + "." + name
                timings.setdefault(key, []).append(time.perf_counter() - started)

        setattr(cls, name, timed)

    for cls, name in [
        (KnowledgeBase, "has_rows"), (KnowledgeBase, "ensure_collection"),
        (OnnxEmbedder, "embed_query"), (OnnxEmbedder, "_load_session_and_tokenizer"),
        (HybridRetriever, "_search_candidates"), (HybridRetriever, "_expand_to_parents"),
        (BGEReranker, "_get_model"), (BGEReranker, "rerank"),
    ]:
        instrument(cls, name)

    for run in range(args.runs):
        timings.clear()
        started = time.perf_counter()
        result = search_research(SearchRequest(query=args.query, page_size=3))
        total = time.perf_counter() - started
        import torch

        from src.index.milvus_kb import get_knowledge_base
        kb = get_knowledge_base(result.collection)
        diag = kb.last_retrieval_diagnostics
        print(json.dumps({
            "run": run + 1, "total_s": round(total, 3), "returned": result.returned,
            "torch_threads": torch.get_num_threads(),
            "reranker_device": str(BGEReranker._model_instance.device),
            "cuda_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None,
            "stages": {k: {"calls": len(v), "total_s": round(sum(v), 3)} for k, v in timings.items()},
            "counts": {k: v for k, v in diag.items() if k.endswith("count")},
        }), flush=True)

    if args.benchmark_rerank:
        import numpy as np
        model = BGEReranker._model_instance
        reference = None
        for threads, batch_size in [(4, 32), (1, 32), (2, 32), (4, 4), (2, 4)]:
            torch.set_num_threads(threads)
            started = time.perf_counter()
            scores = np.asarray(model.predict(captured, batch_size=batch_size, show_progress_bar=False))
            if reference is None:
                reference = scores
            print(json.dumps({
                "benchmark": "rerank", "threads": threads, "batch_size": batch_size,
                "pairs": len(captured), "elapsed_s": round(time.perf_counter() - started, 3),
                "max_score_delta": float(np.max(np.abs(scores - reference))),
                "same_order": bool(np.array_equal(np.argsort(-scores), np.argsort(-reference))),
            }), flush=True)


if __name__ == "__main__":
    main()
