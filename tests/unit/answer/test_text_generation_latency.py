import pytest


@pytest.mark.asyncio
async def test_production_text_route_uses_text_model_and_still_validates(monkeypatch):
    from src.answer.generator import answer_pipeline
    from src.core import config

    class Vision:
        def invoke(self, prompt):
            raise AssertionError("纯文本不应调用视觉生成")

    class Text:
        def invoke(self, prompt):
            return "主营磁材[1]"

    monkeypatch.setattr(config, "get_grading_model", lambda: Text())
    calls = []

    async def validator(answer, docs, **options):
        calls.append(answer)
        return {"status": "verified", "numbers": {"status": "verified"}}

    result = await answer_pipeline(
        "主营业务",
        {
            "context_text": "[1] 主营磁材",
            "context_docs": [{"content": "主营磁材"}],
            "sources": [{"index": 1}],
            "coverage": {"supported_items": ["q1"], "missing_items": []},
        },
        model=Vision(),
        validator=validator,
        prefer_text=True,
    )
    assert result["answer_status"] == "complete"
    assert calls == ["主营磁材[1]"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,quotes",
    [
        ("主营磁材[1]及器件[1]", ["主营磁材及器件"]),
        ("主营磁材及器件[1]", ["主营磁材", "及器件"]),
    ],
)
async def test_quote_coverage_tracks_sources_instead_of_occurrence_counts(answer, quotes):
    import json

    from src.answer.claims import validate_claims

    class Model:
        def invoke(self, prompt):
            return json.dumps(
                {
                    "checks": [
                        {
                            "claim_id": 1,
                            "status": "supported",
                            "reason": "",
                            "supporting_quotes": [
                                {"citation_index": 1, "quote": quote} for quote in quotes
                            ],
                        }
                    ]
                }
            )

    result = await validate_claims(
        answer, [{"content": "主营磁材及器件"}], model=Model(), enabled=True
    )
    assert result["status"] == "verified"
    assert result["numbers"]["status"] == "verified"
