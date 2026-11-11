"""多查询合并排序协议回归（RAG-10）。

``docs/rag-review.md`` 的问题：``hybrid_retriever.search`` 对**每条查询各自**
归一化（各自首条都是 1.0），``merge_docs`` 又按归一化后的 ``score`` 加日期排序，
于是重排分 0.05 的新报告可以挤掉 0.95 的旧报告——被截断的强证据再也找不回来。

验收标准：
* 0.95 与 0.05 示例优先保留强证据，日期仅在相关度同分时生效；
* 覆盖多子查询、过滤回退、HyDE 合并以及无重排模式；
* 最终 TopK、去重与单来源多样性约束仍满足既有契约。
"""

from __future__ import annotations

from src.engine import merge_docs, merge_ranked_groups


def _doc(doc_id: str, *, date: str, content: str = "", merge_score=None,
         rerank_score=None, source: str | None = None, block_type: str = "text") -> dict:
    doc = {
        "id": doc_id,
        "chunk_key": doc_id,
        "collection": "kb",
        "source": source or f"{doc_id}.pdf",
        "content": content or f"{doc_id} 正文",
        "date": date,
        "score": 1.0,
        "block_type": block_type,
    }
    if merge_score is not None:
        doc["merge_score"] = merge_score
    if rerank_score is not None:
        doc["rerank_score"] = rerank_score
    return doc


# ---------------------------------------------------------------------------
# 重排协议：原始分跨查询可比，弱证据不得因日期新而挤掉强证据
# ---------------------------------------------------------------------------


def test_strong_evidence_beats_newer_but_weaker_evidence():
    """0.95（旧报告）优先于 0.05（新报告）：修复前 top_k=1 会选中后者。"""
    weak = _doc("weak", date="2026-08-15", content="弱证据", rerank_score=0.05, merge_score=0.05)
    strong = _doc("strong", date="2023-01-01", content="强证据", rerank_score=0.95, merge_score=0.95)

    top1 = merge_ranked_groups([([weak], 1.0), ([strong], 1.0)], top_k=1)

    assert [doc["id"] for doc in top1] == ["strong"]


def test_date_still_breaks_exact_score_ties():
    """相关度同分时日期仍然是明确的次键（新报告优先）。"""
    old = _doc("old", date="2023-01-01", merge_score=0.8)
    new = _doc("new", date="2026-08-15", merge_score=0.8)

    assert [doc["id"] for doc in merge_ranked_groups([([old, new], 1.0)], top_k=2)] == ["new", "old"]


def test_display_score_is_normalized_after_selection():
    """展示分数在最终选取后归一化（首条 1.0），且不参与排序。"""
    docs = [
        _doc("a", date="2026-01-01", merge_score=0.95),
        _doc("b", date="2026-01-01", merge_score=0.05),
    ]
    selected = merge_ranked_groups([(docs, 1.0)], top_k=2)

    assert [doc["id"] for doc in selected] == ["a", "b"]
    assert selected[0]["score"] == 1.0
    assert selected[1]["score"] == round(0.05 / 0.95, 4)
    # 原始分保留在 merge_score 上，便于排障与后续合并
    assert selected[0]["merge_score"] == 0.95


def test_no_rerank_mode_uses_raw_fusion_scores_across_queries():
    """无重排模式：同一融合公式的原始粗排分同样跨查询可比。"""
    weak = _doc("weak", date="2026-08-15", merge_score=0.12)
    strong = _doc("strong", date="2023-01-01", merge_score=0.91)

    assert [doc["id"] for doc in merge_ranked_groups([([weak], 1.0), ([strong], 1.0)], top_k=1)] == ["strong"]


def test_mixed_score_protocols_fall_back_to_rank_fusion():
    """评分协议不一致（部分候选没有统一分）时退化为秩融合，不混用分数尺度。"""
    with_score = _doc("with", date="2023-01-01", merge_score=0.02)
    without_score = _doc("without", date="2026-08-15")

    # 各自的组内排名都是第 1 → 秩融合同分 → 日期次键判定
    assert [doc["id"] for doc in merge_ranked_groups(
        [([with_score], 1.0), ([without_score], 1.0)], top_k=2
    )] == ["without", "with"]


# ---------------------------------------------------------------------------
# 多子查询 / 过滤回退 / HyDE 合并
# ---------------------------------------------------------------------------


def test_multi_subquery_groups_respect_group_boundaries():
    """多子查询：每个子查询是独立一组，组内名次参与秩融合而不是各自归一化分。"""
    sub_a = [_doc("a1", date="2026-01-01", merge_score=0.9), _doc("a2", date="2026-01-01", merge_score=0.1)]
    sub_b = [_doc("b1", date="2026-01-01", merge_score=0.8)]

    merged = merge_ranked_groups([(sub_a, 1.0), (sub_b, 1.0)], top_k=3)

    assert [doc["id"] for doc in merged] == ["a1", "b1", "a2"]


def test_duplicate_across_groups_keeps_highest_rank():
    """跨组重复的块必须去重，并保留最高的排名分。"""
    group_a = [_doc("dup", date="2026-01-01", merge_score=0.3)]
    group_b = [_doc("dup", date="2026-01-01", merge_score=0.95)]

    merged = merge_ranked_groups([(group_a, 1.0), (group_b, 1.0)], top_k=2)

    assert len(merged) == 1
    assert merged[0]["merge_score"] == 0.95


def test_hyde_group_is_damped_by_weight():
    """HyDE 组按权重阻尼：同分时主检索证据必须排在 HyDE 证据之前。"""
    main = [_doc("main", date="2026-01-01", merge_score=0.5)]
    hyde = [_doc("hyde", date="2026-01-01", merge_score=0.5)]

    merged = merge_ranked_groups([(main, 1.0), (hyde, 0.3)], top_k=2)

    assert [doc["id"] for doc in merged] == ["main", "hyde"]


def test_filter_fallback_groups_are_merged_without_scale_mixing():
    """过滤回退：硬过滤组与回退组一起合并，仍按统一协议排序。"""
    hard = [_doc("hard", date="2026-01-01", merge_score=0.2)]
    fallback = [_doc("fallback", date="2026-01-01", merge_score=0.85)]

    assert [doc["id"] for doc in merge_ranked_groups(
        [(hard, 1.0), (fallback, 1.0)], top_k=1
    )] == ["fallback"]


def test_each_group_keeps_its_top_hit_inside_top_10():
    """低分一侧的第一条不能被高分一侧占满 top-10。"""
    strong = [_doc(f"a{i}", date="2026-01-01", source="a.pdf", merge_score=0.9 - i * 0.01) for i in range(10)]
    weak = [_doc("b0", date="2026-01-01", source="b.pdf", merge_score=0.05)]

    merged = merge_ranked_groups([(strong, 1.0), (weak, 1.0)], top_k=10)

    assert "b0" in [doc["id"] for doc in merged]


def test_diversity_constraints_still_enforced():
    """最终 TopK 仍受单来源多样性约束（同一文档不得霸榜）。"""
    same_source = [_doc(f"s{i}", date="2026-01-01", source="same.pdf", merge_score=0.9 - i * 0.01)
                   for i in range(5)]
    other = [_doc("other", date="2026-01-01", source="other.pdf", merge_score=0.1)]

    merged = merge_ranked_groups([(same_source, 1.0), (other, 1.0)], top_k=4)

    assert len(merged) == 4
    assert sum(1 for doc in merged if doc["source"] == "same.pdf") < 4


def test_legacy_merge_docs_keeps_score_primary_semantics():
    """外部直接构造的候选（无统一分）沿用旧语义：分数为主键、日期为次键。"""
    high = _doc("high", date="2023-01-01")
    low = _doc("low", date="2026-08-15")
    high["score"], low["score"] = 1.0, 0.4

    assert [doc["id"] for doc in merge_docs([high], [low], top_k=2)] == ["high", "low"]

    equal_old, equal_new = _doc("old", date="2023-01-01"), _doc("new", date="2026-08-15")
    assert [doc["id"] for doc in merge_docs([equal_old], [equal_new], top_k=2)] == ["new", "old"]


def test_search_exposes_raw_score_as_merge_score(monkeypatch):
    """检索层必须把归一化前的原始分保留在 merge_score 上（合并的唯一依据）。"""
    from src.retrieve import hybrid_retriever

    class _Client:
        def describe_collection(self, name):
            return {"fields": []}

    retriever = hybrid_retriever.HybridRetriever(_Client(), "kb", lambda q: [0.1])
    raw_matches = [
        {"id": "a", "content": "x", "source": "s.pdf", "score": 0.42},
        {"id": "b", "content": "y", "source": "s.pdf", "score": 0.21},
    ]
    monkeypatch.setattr(
        retriever, "_search_candidates", lambda *args, **kwargs: [dict(m) for m in raw_matches]
    )

    docs = retriever.search("查询", k=2, use_rerank=False, expand_parents=False)

    # 展示分归一化到首条 1.0，但合并依据仍是原始分
    assert [doc["merge_score"] for doc in docs] == [0.42, 0.21]
    assert [doc["score"] for doc in docs] == [1.0, round(0.21 / 0.42, 4)]
