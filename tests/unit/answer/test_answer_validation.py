"""答案出口回归：编号、数字与图片证据必须绑定到实际来源。"""

import json

import pytest

from src.answer import citations


class ReportModel:
    def __init__(self, claim, quote, indices=None):
        self.claim = claim
        self.quote = quote
        self.indices = indices or [1]

    def invoke(self, payload):
        return json.dumps(
            {
                "checks": [
                    {
                        "claim": self.claim,
                        "citation_indices": self.indices,
                        "status": "supported",
                        "supporting_quote": self.quote,
                        "reason": "原文支持",
                    }
                ]
            },
            ensure_ascii=False,
        )


@pytest.mark.parametrize(
    "claim,quote",
    [
        ("营业收入10.0亿元", "营业收入10亿元"),
        ("营业收入１，０００．００万元", "营业收入1000万元"),
        ("营业收入10亿元，营业收入10亿元", "营业收入10亿元"),
    ],
)
def test_equivalent_numbers_and_repeated_facts(claim, quote):
    assert citations._detail_mismatch(claim, quote) is False


@pytest.mark.asyncio
async def test_missing_original_image_cannot_support_invented_transcription():
    claim = "营业收入999亿元"
    result = await citations.validate_claims(
        claim + "[1]",
        [{"content": "收入图表", "block_type": "image", "image_key": "missing.png"}],
        model=ReportModel(claim, claim),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_numeric_report_uses_only_cited_evidence():
    claim = "营业收入20亿元"
    result = await citations.validate_claims(
        claim + "[1]",
        [{"content": "营业收入10亿元"}, {"content": claim}],
        model=ReportModel(claim, "营业收入10亿元"),
        enabled=True,
    )
    assert result["status"] == "failed"
    assert result["numbers"]["status"] == "failed"
    assert result["numbers"]["checks"][0]["unsupported"] == ["20亿元"]


@pytest.mark.asyncio
async def test_success_has_independent_citation_and_numeric_reports():
    claim = "营业收入10.0亿元"
    result = await citations.validate_claims(
        claim + "[1]",
        [{"content": "营业收入10亿元"}],
        model=ReportModel(claim, "营业收入10亿元"),
        enabled=True,
    )
    assert result["status"] == "verified"
    assert result["citations"]["used"] == [1]
    assert result["numbers"]["status"] == "verified"


def test_ledger_snapshots_sources_and_never_rewrites_dangling_citations():
    from src.answer.citations import CitationLedger

    docs = [{"id": "a", "content": "收入10亿元"}, {"id": "b", "content": "利润5亿元"}]
    ledger = CitationLedger(docs)
    docs[0]["content"] = "被修改"
    assert ledger.documents[0]["content"] == "收入10亿元"
    result = ledger.validate("利润5亿元[2]，收入10亿元[99]")
    assert result["used"] == [2]
    assert result["dangling"] == [99]
    assert result["unused"] == [1]
    assert result["status"] == "failed"


@pytest.mark.parametrize(
    "claim,quote",
    [
        ("收入-10亿元", "收入10亿元"),
        ("毛利率提高2个百分点", "毛利率提高2%"),
        ("2024年收入10亿元", "2023年收入10亿元"),
        ("收入20亿元，利润10亿元", "收入10亿元，利润20亿元"),
        ("收入1亿元", "收入10000万元"),
        ("收入10.1亿元", "收入10.09亿元"),
        ("利润为5亿元", "利润为15亿元"),
        ("2026年收入10亿元", "2026H1收入10亿元"),
    ],
)
def test_numeric_conflicts_remain_failures(claim, quote):
    assert citations._detail_mismatch(claim, quote) is True


def test_same_values_cannot_be_swapped_between_years():
    assert (
        citations._detail_mismatch(
            "2024年收入20亿元，2025年收入10亿元",
            "2024年收入10亿元，2025年收入20亿元",
        )
        is True
    )


@pytest.mark.asyncio
async def test_attached_image_with_wrong_key_cannot_support_transcription():
    claim = "营业收入999亿元"
    result = await citations.validate_claims(
        claim + "[1]",
        [{"content": "收入图表", "block_type": "image", "image_key": "a.png"}],
        model=ReportModel(claim, claim),
        enabled=True,
        images=[{"index": 1, "image_key": "b.png", "data_url": "data:image/png;base64,YQ=="}],
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_sse_never_releases_numeric_failure_even_if_model_reports_supported():
    from src.answer.generator import answer_events, answer_pipeline

    class Model:
        def invoke(self, prompt):
            if "supplied_claims" in str(prompt):
                return ReportModel("收入20亿元", "收入10亿元").invoke(prompt)
            return "收入20亿元[1]"

    result = await answer_pipeline(
        "收入",
        {
            "decision": "generate",
            "coverage": {"supported_items": ["q1"], "missing_items": []},
            "context_text": "[1] 收入10亿元",
            "context_docs": [{"id": "a", "content": "收入10亿元"}],
            "sources": [{"index": 1, "id": "a"}],
        },
        model=Model(),
        enabled=True,
    )
    events = [event async for event in answer_events(result)]
    assert result["answer_status"] == "refused"
    assert all("收入20亿元" not in event.get("content", "") for event in events)
    assert events[-1]["citation_validation"]["numbers"]["status"] == "failed"
    assert "supporting_quote" not in str(events)


@pytest.mark.asyncio
async def test_mismatched_source_numbers_stop_before_generation():
    from src.answer.generator import answer_pipeline

    class Model:
        def invoke(self, prompt):
            raise AssertionError("编号不一致时不能调用生成模型")

    result = await answer_pipeline(
        "收入",
        {
            "decision": "generate",
            "coverage": {"supported_items": ["q1"], "missing_items": []},
            "context_text": "[1] 收入10亿元",
            "context_docs": [{"id": "a", "content": "收入10亿元"}],
            "sources": [{"index": 2, "id": "a"}],
        },
        model=Model(),
        enabled=True,
    )
    assert result["answer_status"] == "unverified"
