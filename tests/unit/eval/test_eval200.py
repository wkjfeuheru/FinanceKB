"""200 条评测规则、@10 指标和归因门槛。不连接知识库。"""

from __future__ import annotations

from src.eval.eval200 import (
    audit_cases,
    classify_failure,
    copies_span,
    fill_quotas,
    filters_exclude,
    judge_mediocre,
    locate_quote,
    mrr_phrase,
    parse_generation,
    precision_ceiling,
    render_report,
    score_retrieval,
    vague_rejection_reason,
)

QUOTE = "营收增长百分之十，净利润同步改善。"


def _chunk(chunk_id: str, source: str, content: str | None = None, **extra) -> dict:
    return {
        "id": chunk_id,
        "source": source,
        "version": "v1",
        "content": content or f"甲公司报告正文。{QUOTE}后续为补充说明，不涉及排版目录。",
        "broker": extra.get("broker", ""),
        "date": extra.get("date", ""),
    }


def _complete(kind: str, items: list[dict]) -> list[str]:
    if kind == "cross_doc":
        return [
            "\n".join([
                "## 问题",
                f"比较两家公司的经营变化？{index}",
                "### 标准答案",
                "两侧都给出了经营数据。",
                "### 原文引文A",
                QUOTE,
                "### 原文引文B",
                QUOTE,
            ])
            for index, _item in enumerate(items)
        ]
    question = "最新一期营业收入同比变化是多少？" if kind == "explicit" else "这家公司最新一期营收变化怎么样？"
    return [
        "\n".join([
            "## 问题",
            f"{question}{index}",
            "### 标准答案",
            "营收有所增长。",
            "### 原文引文",
            QUOTE,
        ])
        for index, _item in enumerate(items)
    ]


def _pool() -> list[dict]:
    chunks = []
    for source_index, source in enumerate(("a.pdf", "b.pdf", "c.pdf")):
        for index in range(6):
            chunks.append(_chunk(f"{source}-{index}", source, broker="甲证券" if source_index == 0 else ""))
    return chunks


def test_locate_quote_uses_exact_span_inside_hint():
    content = _chunk("c", "a.pdf")["content"]
    found = locate_quote(content, f"前缀{QUOTE}后缀")
    assert found is not None and QUOTE in found and found in content
    assert locate_quote(content, "太短") is None
    spaced = "营收 增长百分之十，净利润同步改善。"
    assert locate_quote(spaced, QUOTE) == spaced


def test_copies_span_and_vague_guard():
    content = _chunk("c", "a.pdf")["content"]
    assert copies_span("请解释" + QUOTE, content)
    assert not copies_span("最新一期营业收入同比变化是多少？", content)
    assert vague_rejection_reason("太平洋证券2026年怎么看600335？")
    assert vague_rejection_reason("这家公司最新一期营收变化怎么样？") is None


def test_parse_generation_rejects_skip_and_accepts_pair():
    assert parse_generation("SKIP") is None
    parsed = parse_generation(
        f"## 问题\n比较两家公司的经营变化？\n### 标准答案\n两侧不同。\n### 原文引文A\n{QUOTE}\n### 原文引文B\n{QUOTE}",
        multi=True,
    )
    assert parsed is not None
    assert parsed["quotes"] == [QUOTE, QUOTE]
    loose = parse_generation(
        f"<问题> 比较两家公司的营收增速？\n### 标准答案\n两侧不同。\n### 原文引文A\n{QUOTE}\n### 原文引文B\n{QUOTE}",
        multi=True,
    )
    assert loose is not None
    assert loose["query"].startswith("比较两家公司")
    inline = parse_generation(
        f"<问题> 这项涂层技术解决什么问题？ ### 标准答案 形成一致涂层。 ### 原文引文 {QUOTE}"
    )
    assert inline is not None
    assert inline["quotes"] == [QUOTE]


def test_fill_quotas_is_disjoint_and_respects_source_cap():
    cases, rejections = fill_quotas(
        _pool(),
        _complete,
        targets={"explicit": 2, "cross_doc": 1, "vague": 1},
        cap=2,
        batch_size=2,
    )
    by_id = {chunk["id"]: chunk for chunk in _pool()}
    assert rejections["问题抄录了证据正文"] == 0
    assert audit_cases(cases, by_id, targets={"explicit": 2, "cross_doc": 1, "vague": 1}, cap=2) == []
    assert precision_ceiling(cases) == 0.125


def test_score_retrieval_uses_fixed_precision_denominator():
    metrics = score_retrieval(["gold", "other"], ["gold"], k=10)
    assert metrics == {"precision_at_10": 0.1, "recall_at_10": 1.0, "mrr": 1.0}
    missed = score_retrieval(["other"], ["gold"], k=10)
    assert missed["precision_at_10"] == 0.0
    assert missed["mrr"] == 0.0


def test_filter_and_failure_stages():
    assert filters_exclude({"broker": ["乙证券"]}, [{"broker": "甲证券", "date": ""}])
    assert filters_exclude({"date": {"gte": "2020-01-01", "lte": "2020-01-02"}}, [{"broker": "", "date": "2026-09-24"}])
    assert not filters_exclude({"date": {"gte": "2026-09-01", "lte": "2026-09-30"}}, [{"date": "2026-09-24"}])

    record = {
        "query": "营收同比增长8.71%，代码600335怎么看？",
        "stratum": "explicit",
        "question_type": "single_hop",
        "gold_chunk_ids": ["gold"],
        "gold_sources": ["a.pdf"],
        "evidence_meta": [{"broker": "甲证券", "date": "2026-09-24"}],
        "entity_hints": {"brokers": ["甲证券"], "codes": ["600335"]},
    }
    baseline = {"metrics": {"precision_at_10": 0.1, "recall_at_10": 1.0, "mrr": 1.0}}
    missed = {"metrics": {"precision_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0}, "retrieved_chunk_ids": [], "retrieved_sources": []}
    assert classify_failure(record, {**missed, "error": "timeout"}, baseline) == "error"
    assert classify_failure(record, {**missed, "clarification": "请缩小比较范围。", "stop_reason": "invalid_plan"}, baseline) == "not_searched"
    assert classify_failure(record, {**missed, "clarification": "Error code: 429 limit_requests"}, baseline) == "error"
    assert classify_failure(record, {**missed, "hard_filters": {"broker": ["乙证券"]}}, baseline) == "filter_dropped"
    assert classify_failure(
        record,
        {**missed, "attempted_queries": ["营收怎么看？"]},
        baseline,
    ) == "rewrite_drift"
    vague = {**record, "stratum": "vague", "query": "这家公司最新一期营收变化怎么样？"}
    assert classify_failure(vague, {**missed, "attempted_queries": ["这家公司最新一期营收变化怎么样？"]}, baseline) == "vague_unresolved"
    partial = {
        **record,
        "question_type": "multi_hop",
        "stratum": "cross_doc",
        "gold_chunk_ids": ["left", "right"],
        "gold_sources": ["a.pdf", "b.pdf"],
    }
    partial_hit = {
        "metrics": {"precision_at_10": 0.1, "recall_at_10": 0.5, "mrr": 1.0},
        "retrieved_chunk_ids": ["left"],
        "retrieved_sources": ["a.pdf"],
        "coarse_top50": [["left"]],
    }
    assert classify_failure(partial, partial_hit, baseline) == "cross_doc_partial"
    assert classify_failure(record, {**missed, "coarse_top50": [[]]}, baseline) == "miss_coarse"
    assert classify_failure(
        record,
        {**missed, "coarse_top50": [["gold"]], "rerank_candidate_chunk_ids": ["gold"]},
        baseline,
    ) == "rerank_demoted"
    ranked = {
        "metrics": {"precision_at_10": 0.1, "recall_at_10": 1.0, "mrr": 0.5},
        "retrieved_chunk_ids": ["gold"],
        "retrieved_sources": ["a.pdf"],
        "parent_statuses": ["missing"],
    }
    assert classify_failure(record, ranked, baseline) == "parent_split"


def _summary(mrr: float, half: float, recall: float, gain: float) -> dict:
    stratum = {
        "n": 10,
        "precision_at_10": 0.05 + gain,
        "recall_at_10": 0.50 + gain,
        "mrr": mrr,
    }
    base = {"n": 10, "precision_at_10": 0.05, "recall_at_10": 0.50, "mrr": 0.4}
    return {
        "precision_ceiling": 0.13,
        "dense": {"mrr": 0.2, "precision_at_10": 0.04, "recall_at_10": 0.3},
        "hybrid": {"mrr": mrr, "mrr_ci": [mrr - half, mrr + half], "mrr_half_width": half, "precision_at_10": 0.08, "recall_at_10": 0.6},
        "standard": {"precision_at_10": 0.05, "recall_at_10": 0.50, "by_stratum": {"explicit": base, "cross_doc": base, "vague": base}},
        "agentic": {"precision_at_10": 0.05 + gain, "recall_at_10": recall, "by_stratum": {"explicit": stratum, "cross_doc": stratum, "vague": stratum}},
        "parent_restore": {"rate": 0.5},
        "stage_counts": {"miss_coarse": 3},
        "examples": {"miss_coarse": ["rr200-explicit-001 原题"]},
    }


def test_mediocre_gate_and_report_wording():
    weak = _summary(0.33, 0.08, 0.40, 0.0)
    assert judge_mediocre(weak)
    assert "不写稳定" in mrr_phrase(weak["hybrid"])
    report = render_report(weak)
    assert "改进建议" in report
    assert "miss_coarse" in report
    strong = _summary(0.62, 0.04, 0.70, 0.08)
    assert judge_mediocre(strong) == []
    assert mrr_phrase(strong["hybrid"]).startswith("MRR 稳定为")
    assert "改进建议" not in render_report(strong)
