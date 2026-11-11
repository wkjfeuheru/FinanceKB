"""核验延迟回归：格式容错不能依赖再调用一次模型。"""

import asyncio
import json

import pytest

from src.answer import claims


class Model:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


@pytest.mark.asyncio
async def test_literal_newline_in_quote_is_repaired_locally_without_retry():
    model = Model(
        '{"checks":[{"claim":"主营磁材和光伏","citation_indices":[1],'
        '"status":"supported","supporting_quote":"主营磁材\n和光伏","reason":""}]}'
    )
    result = await claims.validate_claims(
        "主营磁材和光伏[1]",
        [{"content": "主营磁材\n和光伏"}],
        model=model,
        enabled=True,
    )
    assert result["status"] == "verified"
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_verification_sends_only_referenced_sources_with_original_numbers():
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim": "主营磁材",
                        "citation_indices": [2],
                        "status": "supported",
                        "supporting_quote": "主营磁材",
                        "reason": "",
                    }
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "主营磁材[2]",
        [{"content": "UNUSED_SECRET"}, {"content": "主营磁材"}],
        model=model,
        enabled=True,
    )
    assert result["status"] == "verified"
    assert "UNUSED_SECRET" not in model.prompts[0]
    assert '"citation_index": 2' in model.prompts[0]


@pytest.mark.asyncio
async def test_generation_and_verification_share_deadline(monkeypatch):
    from src.answer.generator import answer_pipeline
    from src.core import config

    monkeypatch.setattr(config, "ANSWER_TIMEOUT_SECONDS", 0.02, raising=False)

    class SlowModel:
        async def ainvoke(self, prompt):
            await asyncio.sleep(1)
            return "主营磁材[1]"

    pipeline = {
        "decision": "generate",
        "context_text": "[1] 主营磁材",
        "context_docs": [{"content": "主营磁材"}],
        "sources": [{"index": 1}],
        "coverage": {"supported_items": ["q1"], "missing_items": []},
    }
    result = await asyncio.wait_for(
        answer_pipeline("主要业务有哪些", pipeline, model=SlowModel(), enabled=True), 0.2
    )
    assert result["answer_status"] == "unverified"
    assert "超时" in str(result["citation_validation"]["issues"])


@pytest.mark.asyncio
async def test_verification_requests_json_mode_without_changing_generation():
    options = []

    class JsonModel(Model):
        def bind(self, **kwargs):
            options.append(kwargs)
            return self

    model = JsonModel(
        json.dumps(
            {
                "checks": [
                    {
                        "claim": "主营磁材",
                        "citation_indices": [1],
                        "status": "supported",
                        "supporting_quote": "主营磁材",
                        "reason": "",
                    }
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "主营磁材[1]", [{"content": "主营磁材"}], model=model, enabled=True
    )
    assert result["status"] == "verified"
    assert options == [{"response_format": {"type": "json_object"}}]


@pytest.mark.asyncio
async def test_compact_report_resolves_exact_claim_and_citations():
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": 1,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 2, "quote": "主营磁材"}],
                        "reason": "",
                    }
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "主营磁材[2]", [{"content": "无关"}, {"content": "主营磁材"}], model=model, enabled=True
    )
    assert result["status"] == "verified"
    assert result["checks"][0]["claim"] == "主营磁材"
    assert result["checks"][0]["citation_indices"] == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[0], [2], [True], ["1"], [1, 1]])
async def test_compact_report_cannot_invent_duplicate_or_coerce_ids(ids):
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": value,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 1, "quote": "主营磁材"}],
                        "reason": "",
                    }
                    for value in ids
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "主营磁材[1]", [{"content": "主营磁材"}], model=model, enabled=True
    )
    assert result["status"] == "unknown"


@pytest.mark.asyncio
async def test_compact_report_still_requires_all_cited_sources():
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": 1,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 1, "quote": "主营磁材"}],
                        "reason": "",
                    }
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "主营磁材[1][2]",
        [{"content": "主营磁材"}, {"content": "主营磁材"}],
        model=model,
        enabled=True,
    )
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_slow_verifier_cannot_publish_draft_after_deadline(monkeypatch):
    from src.answer.generator import answer_pipeline
    from src.core import config

    monkeypatch.setattr(config, "ANSWER_TIMEOUT_SECONDS", 0.02)

    async def slow_validator(*args, **kwargs):
        await asyncio.sleep(1)
        return {"status": "verified"}

    result = await asyncio.wait_for(
        answer_pipeline(
            "主要业务有哪些",
            {
                "decision": "generate",
                "context_text": "[1] 主营磁材",
                "context_docs": [{"content": "主营磁材"}],
                "sources": [{"index": 1}],
                "coverage": {"supported_items": ["q1"], "missing_items": []},
            },
            model=Model("主营磁材[1]"),
            validator=slow_validator,
        ),
        0.2,
    )
    assert result["answer_status"] == "unverified"
    assert "主营磁材" not in result["answer"]


@pytest.mark.asyncio
async def test_text_only_answer_uses_configured_text_grading_model(monkeypatch):
    from src.answer.generator import answer_pipeline
    from src.core import config

    text_model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": 1,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 1, "quote": "主营磁材"}],
                        "reason": "",
                    }
                ]
            }
        )
    )
    generation_model = Model("主营磁材[1]")
    monkeypatch.setattr(config, "get_grading_model", lambda: text_model)
    result = await answer_pipeline(
        "主要业务有哪些",
        {
            "decision": "generate",
            "context_text": "[1] 主营磁材",
            "context_docs": [{"content": "主营磁材"}],
            "sources": [{"index": 1}],
            "coverage": {"supported_items": ["q1"], "missing_items": []},
        },
        model=generation_model,
        enabled=True,
    )
    assert result["answer_status"] == "complete"
    assert len(generation_model.prompts) == 1
    assert len(text_model.prompts) == 1


def test_colon_intro_inherits_from_named_business_items_but_still_gets_verified():
    from src.answer.citations import CitationLedger, _answer_claims

    answer = "公司主营磁材和光伏，具体如下：\n\n**磁材业务**：生产磁材[1]。\n\n**光伏业务**：生产组件[2]。"
    parsed = _answer_claims(answer)
    assert parsed[0]["claim"] == "公司主营磁材和光伏，具体如下："
    assert parsed[0]["citation_indices"] == [1, 2]
    assert (
        CitationLedger([{"content": "生产磁材"}, {"content": "生产组件"}]).validate(answer)[
            "status"
        ]
        == "verified"
    )


def test_ordinary_uncited_fact_is_not_mistaken_for_named_list_intro():
    from src.answer.citations import _answer_claims

    parsed = _answer_claims("收入999亿元。\n\n**磁材业务**：生产磁材[1]。")
    assert parsed[0]["citation_indices"] == []


@pytest.mark.asyncio
async def test_named_list_intro_is_still_subject_to_numeric_verification():
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": 1,
                        "status": "supported",
                        "supporting_quotes": [
                            {"citation_index": 1, "quote": "收入10亿元，生产磁材"}
                        ],
                        "reason": "",
                    },
                    {
                        "claim_id": 2,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 1, "quote": "生产磁材"}],
                        "reason": "",
                    },
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "收入999亿元，具体如下：\n\n**磁材业务**：生产磁材[1]。",
        [{"content": "收入10亿元，生产磁材"}],
        model=model,
        enabled=True,
    )
    assert result["status"] == "failed"
    assert result["numbers"]["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,quote,expected",
    [
        (
            "2026H1磁材收入21.8亿元，器件收入8.1亿元，同比增长86.5%。",
            "器件收入8.1亿元，同比增长86.5%",
            "verified",
        ),
        (
            "2025H1磁材收入21.8亿元，器件收入8.1亿元，同比增长86.5%。",
            "器件收入8.1亿元，同比增长86.5%",
            "failed",
        ),
        (
            "2026H1磁材收入21.8亿元。器件收入8.1亿元，同比增长86.5%。",
            "器件收入8.1亿元，同比增长86.5%",
            "failed",
        ),
        (
            "2026H1磁材收入21.8亿元，2025H1收入20亿元，器件收入8.1亿元，同比增长86.5%。",
            "器件收入8.1亿元，同比增长86.5%",
            "failed",
        ),
    ],
)
async def test_short_quote_inherits_only_unique_period_in_same_sentence(content, quote, expected):
    model = Model(
        json.dumps(
            {
                "checks": [
                    {
                        "claim_id": 1,
                        "status": "supported",
                        "supporting_quotes": [{"citation_index": 1, "quote": quote}],
                        "reason": "",
                    }
                ]
            }
        )
    )
    result = await claims.validate_claims(
        "2026H1器件收入8.1亿元，同比增长86.5%[1]", [{"content": content}], model=model, enabled=True
    )
    assert result["status"] == expected


@pytest.mark.asyncio
async def test_regeneration_keeps_verified_claim_source_mapping(monkeypatch):
    from src.answer.generator import answer_pipeline
    from src.core import config

    monkeypatch.setattr(config, "get_grading_model", lambda: None)
    model = Model("主营磁材和器件[1][2]")
    calls = []

    async def validator(*args, **kwargs):
        calls.append(1)
        return (
            {
                "status": "failed",
                "issues": ["其他事实不受支持"],
                "checks": [
                    {"claim": "主营磁材和器件", "citation_indices": [1, 2], "status": "supported"},
                ],
            }
            if len(calls) == 1
            else {"status": "verified"}
        )

    await answer_pipeline(
        "主营业务有哪些",
        {
            "decision": "generate",
            "context_text": "[1] 磁材；[2] 器件",
            "context_docs": [{"content": "磁材"}, {"content": "器件"}],
            "sources": [{"index": 1}, {"index": 2}],
            "coverage": {"supported_items": ["q1"], "missing_items": []},
        },
        model=model,
        validator=validator,
    )
    assert '"citation_indices": [1, 2]' in model.prompts[1]


@pytest.mark.parametrize("marker", ["[1, 2]", "[1，2]", "[1、2]"])
def test_grouped_citations_keep_every_original_number(marker):
    from src.answer.citations import CitationLedger

    result = CitationLedger([{"content": "主营磁材"}, {"content": "主营磁材"}]).validate("主营磁材" + marker)
    assert result["status"] == "verified"
    assert result["used"] == [1, 2]


@pytest.mark.asyncio
async def test_grouped_citations_are_canonicalized_before_validation_and_publishing():
    from src.answer.generator import answer_pipeline

    async def validator(answer, *args, **kwargs):
        assert answer == "主营磁材[1][2]"
        return {"status": "verified"}

    result = await answer_pipeline("主营业务有哪些", {
        "decision": "generate", "context_text": "[1] 磁材；[2] 磁材",
        "context_docs": [{"content": "磁材"}, {"content": "磁材"}],
        "sources": [{"index": 1}, {"index": 2}], "coverage": {"supported_items": ["q1"], "missing_items": []},
    }, model=Model("主营磁材[1, 2]"), validator=validator)
    assert result["answer"] == "主营磁材[1][2]"
