import pytest

from src.answer.citations import CitationLedger, _answer_claims, _quote_in_source
from src.engine import build_context

TABLE = """| 铜箔类别 | 主要厂商 |
| --- | --- |
| HTE | 铜冠、德福[1] |
| HVLP5（研发中） | 三井（试产）、SKNexilis（研发）[1] |
"""


def test_answer_table_header_is_not_an_uncited_fact_and_rows_keep_association():
    ledger = CitationLedger([{"content": "HTE 铜冠、德福；HVLP5 三井（试产）、SKNexilis（研发）"}])
    assert ledger.validate(TABLE)["status"] == "verified"
    claims = _answer_claims(TABLE)
    assert len(claims) == 2
    assert all(claim["citation_indices"] == [1] for claim in claims)
    assert "HVLP5" in claims[1]["claim"] and "三井（试产）" in claims[1]["claim"]


def test_missing_table_row_citation_cannot_borrow_from_another_row():
    result = CitationLedger([{"content": "铜箔分类"}]).validate(
        TABLE.replace("铜冠、德福[1]", "铜冠、德福")
    )
    assert result["status"] == "failed"
    assert any("HTE" in claim for claim in result["missing"])


def test_embedded_html_is_normalized_before_context_clipping():
    rows = "".join(
        f'<tr><td rowspan="1" colspan="1">HVLP{i}</td><td rowspan="1" colspan="1">三井（试产）</td></tr>'
        for i in range(25)
    )
    html = "说明\n<table><tr><th>铜箔类别</th><th>主要厂商</th></tr>" + rows + "</table>"
    retained = []
    text, _ = build_context([{"content": html, "block_type": "text"}], context_docs=retained)
    assert "HVLP24" in text
    assert "<td" not in text
    assert retained[0]["content"] in text
    assert _quote_in_source(1, "| HVLP24 | 三井（试产） |", retained)
    assert not _quote_in_source(1, "| HVLP24 | 诺德（量产） |", retained)


@pytest.mark.parametrize("row", ["| HVLP5 | 三井 |", "| HVLP5 | 三井[9] |"])
def test_table_row_still_requires_valid_source(row):
    assert (
        CitationLedger([{"content": "来源"}]).validate("| 类别 | 厂商 |\n| --- | --- |\n" + row)[
            "status"
        ]
        == "failed"
    )


def test_invalid_html_table_is_preserved_instead_of_inventing_cells():
    from src.chunking.chunker import normalize_html_tables

    broken = "<table><tr><td>HVLP5</tr></table>"
    assert normalize_html_tables(broken).strip() == broken


def test_table_headers_stay_in_each_claim_including_numeric_conditions():
    claims = _answer_claims("| 2026年产量（吨） | 企业 |\n| --- | --- |\n| 100 | 铜冠[1] |")
    assert len(claims) == 1
    assert "2026年产量（吨）：100" in claims[0]["claim"]
