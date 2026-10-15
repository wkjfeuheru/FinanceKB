"""声明核验门禁的离线行为测试。"""

import json
from types import SimpleNamespace

import pytest

from finance_rag.src.services import citation_validator as validator


class Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


def response(
    claim="2024年收入增长10%", quote="2024年收入增长10%", indices=None, status="supported", **extra
):
    return json.dumps(
        {
            "checks": [
                {
                    "claim": claim,
                    "citation_indices": indices or [1],
                    "status": status,
                    "supporting_quote": quote,
                    "reason": "原文支持",
                    **extra,
                }
            ]
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_supported_claim_uses_actual_content():
    model = Model(response())
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。", [{"content": "2024年收入增长10%"}], model=model, enabled=True
    )
    assert result["status"] == "verified"
    assert result["valid"] is True
    assert result["score"] == 1.0
    assert result["verified_citations"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"enabled": False}, {"enabled": True}])
async def test_unavailable_validation_never_passes(kwargs):
    result = await validator.validate_claims(
        "收入增长10%[1]", [{"content": "收入增长10%"}], **kwargs
    )
    assert result["status"] == "unknown"
    assert result["score"] is None
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_uncited_financial_fact_fails():
    result = await validator.validate_claims(
        "2024年收入增长10%",
        [{"content": "2024年收入增长10%"}],
        model=Model(response()),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_model_cannot_omit_uncited_second_claim():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。利润增长20%。",
        [{"content": "2024年收入增长10%"}],
        model=Model(response()),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_model_cannot_omit_cited_second_claim():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。利润增长20%[1]。",
        [{"content": "2024年收入增长10%，利润增长20%"}],
        model=Model(response()),
        enabled=True,
    )
    assert result["status"] == "unknown"
    assert result["score"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["", "2024年收入下降10%", "2023年收入增长10%", "2024年收入增长20%", "2024年收入增长10亿元"],
)
async def test_fabricated_quote_and_preview_never_pass(content):
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": content, "preview": "2024年收入增长10%"}],
        model=Model(response()),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quote", ["2023年收入增长10%", "2024年收入增长20%", "2024年收入下降10%", "2024年收入增长10亿元"]
)
async def test_real_quote_with_conflicting_details_fails(quote):
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": quote}],
        model=Model(response(quote=quote)),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_invalid_index_fails():
    result = await validator.validate_claims(
        "2024年收入增长10%[2]。",
        [{"content": "2024年收入增长10%"}],
        model=Model(response(indices=[2])),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    ["garbage", response(secret="unexpected"), response(status="passed"), '{"checks": []}'],
)
async def test_invalid_schema_is_unknown_after_one_retry(payload):
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%"}],
        model=Model(payload, payload),
        enabled=True,
    )
    assert result["status"] == "unknown"
    assert result["score"] is None


@pytest.mark.asyncio
async def test_json_parse_retry_can_recover():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%"}],
        model=Model("bad json", response()),
        enabled=True,
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_model_failure_is_unknown():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%"}],
        model=Model(RuntimeError("offline")),
        enabled=True,
    )
    assert result["status"] == "unknown"
    assert result["score"] is None


def test_legacy_embedding_failure_does_not_pass():
    def broken_embed(text):
        raise RuntimeError("offline")

    result = validator.CitationValidator(broken_embed).validate(
        "收入增长[1]", [{"content": "收入增长"}]
    )
    assert result["valid"] is False


def test_legacy_no_citations_does_not_pass():
    result = validator.CitationValidator(lambda text: [1.0]).validate(
        "收入增长", [{"content": "收入增长"}]
    )
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_missing_model_with_uncited_answer_is_unknown():
    result = await validator.validate_claims(
        "收入增长10%", [{"content": "收入增长10%"}], enabled=True
    )
    assert result["status"] == "unknown"
    assert result["score"] is None


@pytest.mark.asyncio
async def test_sync_invoke_and_citation_after_period():
    model = SimpleNamespace(invoke=lambda prompt: response())
    result = await validator.validate_claims(
        "2024年收入增长10%。[1]", [{"content": "2024年收入增长10%"}], model=model, enabled=True
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_unknown_model_check_keeps_score_unknown():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%"}],
        model=Model(response(status="unknown")),
        enabled=True,
    )
    assert result["status"] == "unknown"
    assert result["score"] is None


@pytest.mark.asyncio
async def test_swapped_numeric_values_cannot_pass():
    claim = "2024年收入10亿元，利润20亿元"
    quote = "2024年收入20亿元，利润10亿元"
    result = await validator.validate_claims(
        claim + "[1]。",
        [{"content": quote}],
        model=Model(response(claim=claim, quote=quote)),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_citation_report_cannot_change_original_index():
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%"}] * 2,
        model=Model(response(indices=[2]), response(indices=[2])),
        enabled=True,
    )
    assert result["status"] == "unknown"


@pytest.mark.asyncio
async def test_multi_source_quote_must_exist_in_each_cited_document():
    result = await validator.validate_claims(
        "2024年收入增长10%[1][2]。",
        [{"content": "2024年收入增长10%"}, {"content": "别的来源"}],
        model=Model(response(indices=[1, 2])),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_preview_is_not_sent_to_model():
    model = Model(response())
    result = await validator.validate_claims(
        "2024年收入增长10%[1]。",
        [{"content": "2024年收入增长10%", "preview": "PREVIEW_SECRET"}],
        model=model,
        enabled=True,
    )
    assert result["status"] == "verified"
    assert "PREVIEW_SECRET" not in model.prompts[0]


def test_legacy_uses_content_instead_of_preview():
    result = validator.CitationValidator(lambda text: [1.0]).validate(
        "收入增长[1]", [{"content": "", "preview": "收入增长"}]
    )
    assert result["valid"] is False


@pytest.mark.asyncio
async def test_multi_source_claim_accepts_individual_real_quotes():
    claim = "甲公司收入10亿元，乙公司收入20亿元"
    quotes = [
        {"citation_index": 1, "quote": "甲公司收入10亿元"},
        {"citation_index": 2, "quote": "乙公司收入20亿元"},
    ]
    result = await validator.validate_claims(
        claim + "[1][2]。",
        [{"content": quote["quote"]} for quote in quotes],
        model=Model(response(claim=claim, quote="", indices=[1, 2], supporting_quotes=quotes)),
        enabled=True,
    )
    assert result["status"] == "verified"
    assert result["verified_citations"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quotes",
    [
        [{"citation_index": 1, "quote": "甲公司收入10亿元"}],
        [
            {"citation_index": 1, "quote": "甲公司收入10亿元"},
            {"citation_index": 2, "quote": "乙公司收入30亿元"},
        ],
        [
            {"citation_index": 1, "quote": "甲公司收入10亿元"},
            {"citation_index": 3, "quote": "乙公司收入20亿元"},
        ],
    ],
)
async def test_multi_source_map_rejects_missing_fabricated_or_wrong_index(quotes):
    claim = "甲公司收入10亿元，乙公司收入20亿元"
    result = await validator.validate_claims(
        claim + "[1][2]。",
        [{"content": "甲公司收入10亿元"}, {"content": "乙公司收入20亿元"}],
        model=Model(response(claim=claim, quote="", indices=[1, 2], supporting_quotes=quotes)),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "doc",
    [
        {"content": "乙公司收入10亿元"},
        {"content": "乙公司收入10亿元", "security_name": "乙公司"},
    ],
)
async def test_same_number_from_different_company_fails(doc):
    result = await validator.validate_claims(
        "甲公司收入10亿元[1]。",
        [doc],
        model=Model(response(claim="甲公司收入10亿元", quote="乙公司收入10亿元")),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_security_name_does_not_justify_explicit_other_entity_in_quote():
    result = await validator.validate_claims(
        "甲公司收入10亿元[1]。",
        [{"content": "乙公司收入10亿元", "security_name": "甲公司"}],
        model=Model(response(claim="甲公司收入10亿元", quote="乙公司收入10亿元")),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_multi_source_cannot_swap_company_value_associations():
    claim = "甲公司收入10亿元，乙公司收入20亿元"
    quotes = [
        {"citation_index": 1, "quote": "乙公司收入10亿元"},
        {"citation_index": 2, "quote": "甲公司收入20亿元"},
    ]
    result = await validator.validate_claims(
        claim + "[1][2]。",
        [{"content": quote["quote"]} for quote in quotes],
        model=Model(response(claim=claim, quote="", indices=[1, 2], supporting_quotes=quotes)),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_security_name_can_resolve_unnamed_company_reference():
    result = await validator.validate_claims(
        "甲公司收入10亿元[1]。",
        [{"content": "公司收入10亿元", "security_name": "甲公司"}],
        model=Model(response(claim="甲公司收入10亿元", quote="公司收入10亿元")),
        enabled=True,
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_reordered_metrics_with_same_values_are_supported():
    claim = "2024年利润20亿元，收入10亿元"
    quote = "2024年收入10亿元，利润20亿元"
    result = await validator.validate_claims(
        claim + "[1]。",
        [{"content": quote}],
        model=Model(response(claim=claim, quote=quote)),
        enabled=True,
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_repeated_numbers_need_enough_occurrences_in_evidence():
    claim = "收入10亿元，利润10亿元"
    result = await validator.validate_claims(
        claim + "[1]。",
        [{"content": "收入10亿元"}],
        model=Model(response(claim=claim, quote="收入10亿元")),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim", ["营业收入2024年为10亿元", "营业收入（2024年）：10亿元", "营业收入2024年度为10亿元"]
)
async def test_year_after_metric_is_period_not_amount(claim):
    quote = "2024年营业收入10亿元"
    result = await validator.validate_claims(
        claim + "[1]。",
        [{"content": quote}],
        model=Model(response(claim=claim, quote=quote)),
        enabled=True,
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_year_shaped_amount_is_not_removed_from_metric_check():
    claim = "2024年营业收入2024元，利润10元"
    quote = "2024年营业收入10元，利润2024元"
    result = await validator.validate_claims(
        claim + "[1]。",
        [{"content": quote}],
        model=Model(response(claim=claim, quote=quote)),
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_partial_draft_cannot_publish_blocked_metric_under_allowed_item_label():
    report = {
        "checks": [
            {
                "claim": "Revenue10",
                "citation_indices": [1],
                "status": "supported",
                "supporting_quote": "Revenue10",
                "reason": "原文支持",
                "item_id": "q1",
            },
            {
                "claim": "Profit5",
                "citation_indices": [1],
                "status": "supported",
                "supporting_quote": "Profit5",
                "reason": "原文支持",
                "item_id": "q1",
            },
        ]
    }
    result = await validator.validate_claims(
        "Revenue10[1]. Profit5[1]",
        [{"content": "Revenue10 Profit5"}, {"content": "Profit7"}],
        model=Model(json.dumps(report)),
        enabled=True,
        allowed_items=[{"id": "q1", "query": "Revenue", "metric": ""}],
        blocked_items=[{"id": "q2", "query": "Profit", "metric": ""}],
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_id, expected", [(None, "unknown"), ("q2", "failed"), ("fake", "failed")]
)
async def test_partial_claim_requires_known_allowed_item_id(item_id, expected):
    result = await validator.validate_claims(
        "营业收入10亿元[1]。",
        [{"content": "营业收入10亿元"}],
        model=Model(response(claim="营业收入10亿元", quote="营业收入10亿元", item_id=item_id)),
        enabled=True,
        allowed_items=[{"id": "q1", "query": "营业收入"}],
        blocked_items=[{"id": "q2", "query": "净利润"}],
    )
    assert result["status"] == expected


@pytest.mark.asyncio
async def test_partial_supported_metric_can_publish_with_allowed_item_mapping():
    result = await validator.validate_claims(
        "营业收入10亿元[1]。",
        [{"content": "营业收入10亿元"}],
        model=Model(response(claim="营业收入10亿元", quote="营业收入10亿元", item_id="q1")),
        enabled=True,
        allowed_items=[{"id": "q1", "query": "Revenue"}],
        blocked_items=[{"id": "q2", "query": "Profit"}],
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_partial_same_metric_different_entity_can_publish_allowed_entity():
    result = await validator.validate_claims(
        "甲公司收入10亿元[1]。",
        [{"content": "甲公司收入10亿元"}],
        model=Model(response(claim="甲公司收入10亿元", quote="甲公司收入10亿元", item_id="q1")),
        enabled=True,
        allowed_items=[{"id": "q1", "query": "甲公司收入", "entity": "甲公司", "metric": "收入"}],
        blocked_items=[{"id": "q2", "query": "乙公司收入", "entity": "乙公司", "metric": "收入"}],
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_partial_draft_cannot_publish_blocked_period():
    result = await validator.validate_claims(
        "2023年营业收入10亿元[1]。",
        [{"content": "2023年营业收入10亿元"}],
        model=Model(
            response(claim="2023年营业收入10亿元", quote="2023年营业收入10亿元", item_id="q1")
        ),
        enabled=True,
        allowed_items=[{"id": "q1", "query": "2024年营业收入", "period": "2024年"}],
        blocked_items=[{"id": "q2", "query": "2023年营业收入", "period": "2023年"}],
    )
    assert result["status"] == "failed"
