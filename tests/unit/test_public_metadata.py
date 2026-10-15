"""公共元数据回归：真实解析，不让暂存文件名、报告期或引用污染字段。"""

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from finance_rag.src.services.document_service import resolve_document_date


def _resolve(source="报告.pdf", markdown="", **kwargs):
    from finance_rag.src.rag.ingestion.public_metadata import resolve_public_metadata

    return resolve_public_metadata(source, markdown, as_of=date(2026, 10, 3), **kwargs)


def test_publication_date_not_controlled_by_temp_uuid():
    results = []
    for filename in (".abcdef2053abcdef.uploading.pdf", ".abcdef2006abcdef.uploading.pdf"):
        parsed = SimpleNamespace(path=Path(filename), source="报告.pdf", metadata={},
                                 chunks=SimpleNamespace(markdown="发布日期：2026年9月27日。"))
        results.append(resolve_document_date(parsed))
    assert results == ["2026-09-27", "2026-09-27"]


def test_reporting_period_is_not_publication_date():
    result = _resolve(markdown="报告期：2026-06-30\n发布日期：2026-09-24")
    assert result.date == "2026-09-24"
    assert [item["value"] for item in result.audit["fields"]["date"]["evidence"]] == ["2026-09-24"]


def test_publication_label_on_same_line_as_reporting_period_is_kept():
    result = _resolve(markdown="报告期：2026年6月30日。发布日期：2026年9月24日。")
    assert result.date == "2026-09-24"
    assert [item["value"] for item in result.audit["fields"]["date"]["evidence"]] == ["2026-09-24"]


def test_later_financial_section_does_not_cancel_publication_label():
    result = _resolve(markdown="发布日期：2026-09-24。财务预测：2026-12-31")
    assert result.date == "2026-09-24"


def test_related_research_heading_does_not_publish_this_report():
    result = _resolve(markdown="相关研究\n发布日期：2026-09-24")
    assert result.date == ""


def test_related_research_entry_with_title_does_not_publish_this_report():
    result = _resolve(markdown="相关研究\n某行业跟踪\n发布日期：2026-09-24")
    assert result.date == ""


def test_explicit_own_report_label_after_related_entry_restores_publication_context():
    result = _resolve(markdown="相关研究\n旧报告\n发布日期：2025-09-01\n本报告发布日期：2026-09-24")
    assert result.date == "2026-09-24"
    assert [item["value"] for item in result.audit["fields"]["date"]["evidence"]] == ["2026-09-24"]


@pytest.mark.parametrize("text", ["2026年", "2026年9月", "报告期：2026-06-30", "暂无日期",
                                      "相关研究\n2026-09-24", "表格\n2026-09-24"])
def test_incomplete_or_unlabelled_markdown_does_not_invent_day(text):
    result = _resolve(markdown=text)
    assert result.date == ""
    assert result.audit["fields"]["date"]["status"] == "unknown"
    assert result.needs_review is True


def test_invalid_calendar_date_is_unknown_and_audited():
    result = _resolve(markdown="发布日期：2026-02-31")
    assert result.date == ""
    assert "invalid_calendar_date" in result.audit["fields"]["date"]["reasons"]


def test_conflicting_reliable_dates_are_not_ranked_or_replaced():
    result = _resolve("报告-20260923.pdf", "发布日期：2026-09-24")
    assert result.date == ""
    assert result.audit["fields"]["date"]["status"] == "conflict"
    assert {item["value"] for item in result.audit["fields"]["date"]["evidence"]} == {"2026-09-23", "2026-09-24"}


def test_future_date_requires_review_without_alternate_fallback():
    result = _resolve(markdown="发布日期：2026-10-04")
    assert result.date == ""
    assert "future_date" in result.audit["fields"]["date"]["reasons"]


def test_distribution_id_date_is_not_publication_date():
    result = _resolve("H3_AP202609211829713285_1.pdf", "发布日期：2026-09-19")
    assert result.date == "2026-09-19"


def test_first_page_layout_date_beyond_markdown_prefix(tmp_path):
    import pymupdf

    path = tmp_path / ".abcdef2053.uploading.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 60), "2026-09-24")
        page.insert_text((60, 130), "Research report")
        document.save(path)
    result = _resolve(markdown="正文" * 2000, pdf_path=path)
    assert result.date == "2026-09-24"
    field = result.audit["fields"]["date"]
    assert field["status"] == "known"
    assert field["evidence"][0]["page"] == 1
    assert len(field["evidence"][0]["location"]) == 4
    assert len(result.audit["pdf_sha256"]) == 64


def test_landscape_original_cover_date_below_sixty_percent_is_kept():
    path = Path(__file__).resolve().parents[2] / "assets/reports/H3_AP202609241829864267_1.pdf"
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=path.name, pdf_path=path)
    assert result.date == "2026-09-24"
    assert result.audit["fields"]["date"]["status"] == "known"
    assert any(item["page"] == 1 and item["source"] == "pdf_first_page_cover"
               for item in result.audit["fields"]["date"]["evidence"])


@pytest.mark.parametrize(("filename", "published"), [
    ("H3_AP202609261829925702_1.pdf", "2026-09-24"),
    ("H3_AP202609231829798847_1.pdf", "2026-09-22"),
    ("H3_AP202609281829951050_1.pdf", "2026-09-28"),
    ("H3_AP202609251829896818_1.pdf", "2026-09-12"),
])
def test_original_cover_dates_use_report_title_context(filename, published):
    path = Path(__file__).resolve().parents[2] / "assets/reports" / filename
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=filename, pdf_path=path)
    assert result.date == published
    assert result.audit["fields"]["date"]["status"] == "known"
    assert any(item["page"] == 1 and item["source"] == "pdf_first_page_cover"
               for item in result.audit["fields"]["date"]["evidence"])


def test_pdf_table_date_not_treated_as_cover_date(tmp_path):
    import pymupdf

    path = tmp_path / "table.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 60), "Research report")
        page.insert_text((60, 650), "2026-06-30  10.2  12.3")
        document.save(path)
    assert _resolve(pdf_path=path).date == ""


def test_pdf_isolated_upper_page_date_without_cover_context_is_unknown(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((60, 140), "2026-06-30")
        document.save(path)
    assert _resolve(pdf_path=path).date == ""


@pytest.mark.parametrize("heading", ["相关研究", "基本数据", "Related research", "Basic data"])
def test_pdf_adjacent_section_date_is_not_cover_date(tmp_path, heading):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 60), "Research report")
        page.insert_text((60, 140), heading, fontname="china-s")
        page.insert_text((60, 160), "2026-06-30")
        document.save(path)
    assert _resolve(pdf_path=path).date == ""


def test_pdf_related_study_date_stays_excluded_beyond_footer_gap(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((60, 20), "相关研究", fontname="china-s")
        page.insert_text((60, 65), "Research report")
        page.insert_text((60, 250), "2026-06-30")
        document.save(path)
    assert _resolve(pdf_path=path).date == ""


def test_publisher_signature_wins_over_body_citation():
    result = _resolve(markdown="发布日期：2026-09-24\n参考中信证券研究观点。\n东吴证券研究所")
    assert result.broker == "东吴证券"
    assert result.audit["fields"]["broker"]["status"] == "known"


def test_reliable_publisher_conflict_is_preserved():
    result = _resolve("宏观报告-中信证券.pdf", "东吴证券研究所")
    assert result.broker == ""
    assert result.audit["fields"]["broker"]["status"] == "conflict"


@pytest.mark.parametrize(("source", "expected"), [
    ("格林大华期货有限公司-20260924.pdf", "格林大华期货"),
    ("国信证券经济研究所-20260927.pdf", "国信证券"),
])
def test_controlled_full_legal_and_research_institute_names_normalize(source, expected):
    result = _resolve(source=source)
    assert result.broker == expected
    assert result.audit["fields"]["broker"]["status"] == "known"


@pytest.mark.parametrize(("filename", "publisher", "page_number", "signature"), [
    ("H3_AP202609241829864267_1.pdf", "格林大华期货", 34, "格林大华期货有限公司"),
    ("H3_AP202609271829939976_1.pdf", "国信证券", 14, "国信证券经济研究所"),
])
def test_original_pdf_last_page_full_signature_is_attributed(filename, publisher, page_number, signature):
    path = Path(__file__).resolve().parents[2] / "assets/reports" / filename
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=filename, pdf_path=path)
    assert result.broker == publisher
    assert result.audit["fields"]["broker"]["status"] == "known"
    assert any(item["page"] == page_number and signature in item["raw"]
               for item in result.audit["fields"]["broker"]["evidence"])


@pytest.mark.parametrize("filename", [
    "H3_AP202609271829940631_1.pdf",
    "2026中报点评：短期扰动不改基本面，燃机订单超预期+AIDC一体化打开成长空间.pdf",
])
def test_original_east_wu_footer_is_publisher(filename):
    path = Path(__file__).resolve().parents[2] / "assets/reports" / filename
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=filename, pdf_path=path)
    assert result.broker == "东吴证券"
    assert result.audit["fields"]["broker"]["status"] == "known"
    assert any(item["page"] == 1 and "东吴证券研究所" in item["raw"]
               for item in result.audit["fields"]["broker"]["evidence"])
    assert any(item["page"] == 3 and "东吴证券研究所" in item["raw"]
               for item in result.audit["fields"]["broker"]["evidence"])


@pytest.mark.parametrize(("filename", "published"), [
    ("H3_AP202609261829925702_1.pdf", "2026-09-24"),
    ("H3_AP202609271829940631_1.pdf", "2026-09-27"),
    ("2026中报点评：短期扰动不改基本面，燃机订单超预期+AIDC一体化打开成长空间.pdf", "2026-08-14"),
])
def test_original_three_source_publication_dates(filename, published):
    path = Path(__file__).resolve().parents[2] / "assets/reports" / filename
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=filename, pdf_path=path)
    assert result.date == published
    assert result.audit["fields"]["date"]["status"] == "known"


@pytest.mark.parametrize(("filename", "publisher"), [
    ("H3_AP202609261829925702_1.pdf", "太平洋证券"),
    ("H3_AP202609271829940631_1.pdf", "东吴证券"),
    ("2026中报点评：短期扰动不改基本面，燃机订单超预期+AIDC一体化打开成长空间.pdf", "东吴证券"),
])
def test_original_three_source_publishers(filename, publisher):
    path = Path(__file__).resolve().parents[2] / "assets/reports" / filename
    if not path.is_file():
        pytest.skip("原始 PDF 未随测试工作区提供")
    result = _resolve(source=filename, pdf_path=path)
    assert result.broker == publisher
    assert result.audit["fields"]["broker"]["status"] == "known"


def test_broker_mention_alone_is_not_publisher_identity():
    assert _resolve(markdown="中信证券认为需求增加。").broker == ""


def test_related_research_standalone_institution_is_not_publisher():
    result = _resolve(markdown="相关研究\n某行业跟踪\n中信证券")
    assert result.broker == ""
    assert result.audit["fields"]["broker"]["status"] == "unknown"


@pytest.mark.parametrize("citation_page", [1, 2])
def test_pdf_adjacent_citation_institution_is_not_publisher(tmp_path, citation_page):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.new_page()
        page = document[citation_page - 1]
        page.insert_text((60, 140), "相关研究", fontname="china-s")
        page.insert_text((60, 170), "中信证券", fontname="china-s")
        document.save(path)
    result = _resolve(pdf_path=path)
    assert result.broker == ""
    assert result.audit["fields"]["broker"]["status"] == "unknown"


@pytest.mark.parametrize("citation_page", [1, 2])
def test_pdf_uncontextualized_body_institution_is_not_publisher(tmp_path, citation_page):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.new_page()
        document[citation_page - 1].insert_text((60, 350), "中信证券", fontname="china-s")
        document.save(path)
    assert _resolve(pdf_path=path).broker == ""


def test_pdf_lonely_last_page_footer_without_report_context_is_unknown(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page()
        last_page = document.new_page()
        last_page.insert_text((60, 50), "免责声明", fontname="china-s")
        last_page.insert_text((60, 790), "中信证券研究所", fontname="china-s")
        document.save(path)
    assert _resolve(pdf_path=path).broker == ""


def test_explicit_date_is_validated_and_audit_cannot_be_forged():
    with pytest.raises(ValueError, match="date"):
        _resolve(explicit={"date": "2026-02-31"})
    result = _resolve(explicit={"date": "2026-09-24", "public_metadata_audit": {"rules_version": "manual"}})
    assert result.date == "2026-09-24"
    assert result.audit["rules_version"] == "public-metadata-v1"
    assert result.audit["fields"]["date"]["status"] == "manual"


def test_pdf_read_failure_has_unknown_diagnostic_without_mtime(tmp_path):
    path = tmp_path / "2026-09-24.pdf"
    path.write_bytes(b"not a PDF")
    result = _resolve(pdf_path=path)
    assert result.date == ""
    assert "pdf_read_error" in result.audit["fields"]["date"]["reasons"]


def test_public_review_survives_canonical_enrichment_and_partial_patch():
    from finance_rag.src.rag.ingestion.metadata_extractor import ExtractedMetadata, apply_metadata, merge_metadata_patch

    original = _resolve(markdown="发布日期：2026-09-24\n发布日期：2026-09-25").as_metadata()
    merged = apply_metadata(original, ExtractedMetadata(report_type="宏观", broker="中信证券"))
    assert merged["date"] == ""
    assert merged["broker"] == ""
    assert merged["needs_review"] is True
    patched = merge_metadata_patch(merged, {"broker": "东吴证券"})
    assert patched["needs_review"] is True
    assert patched["public_metadata_audit"]["fields"]["date"]["status"] == "conflict"
    assert patched["public_metadata_audit"]["fields"]["broker"]["status"] == "manual"
    assert original["public_metadata_audit"]["fields"]["broker"]["status"] == "unknown"


def test_patch_does_not_clear_type_review_by_fixing_only_broker():
    from finance_rag.src.rag.ingestion.metadata_extractor import build_canonical_metadata, merge_metadata_patch

    current = build_canonical_metadata({"report_type": "行业", "industry_l1": "电子", "industry_l2": "白酒Ⅱ"})
    patched = merge_metadata_patch(current, {"broker": "东吴证券"})
    assert patched["needs_review"] is True


def test_patch_rejects_forged_audit_source():
    from finance_rag.src.rag.ingestion.metadata_extractor import merge_metadata_patch

    with pytest.raises(ValueError):
        merge_metadata_patch({"report_type": "宏观"}, {"public_metadata_audit": {}, "needs_review": False})


def test_pdf_title_broker_is_not_publisher_but_last_page_copyright_is(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.set_metadata({"title": "中信证券-宏观报告"})
        document.new_page().insert_text((60, 60), "Research report")
        document.new_page().insert_text((60, 100), "本报告版权归东吴证券所有。", fontname="china-s")
        document.save(path)
    result = _resolve(pdf_path=path)
    assert result.broker == "东吴证券"
    field = result.audit["fields"]["broker"]
    assert field["status"] == "known"
    assert field["evidence"][0]["page"] == 2


def test_pdf_last_page_citation_does_not_define_publisher(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page().insert_text((60, 60), "Research report")
        document.new_page().insert_text((60, 100), "引用中信证券报告，本报告版权归东吴证券所有。", fontname="china-s")
        document.save(path)
    result = _resolve(pdf_path=path)
    assert result.broker == "东吴证券"


def test_pdf_last_page_recipient_is_not_publisher(tmp_path):
    import pymupdf

    path = tmp_path / "opaque.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.new_page().insert_text((60, 100), "本报告仅供中信证券使用", fontname="china-s")
        document.save(path)
    result = _resolve(pdf_path=path)
    assert result.broker == ""
    assert result.audit["fields"]["broker"]["status"] == "unknown"


@pytest.mark.parametrize("value", ["2026-02-31", "2026-09", "2026", "2026-9-24"])
def test_manual_patch_rejects_invalid_or_incomplete_date(value):
    from finance_rag.src.rag.ingestion.metadata_extractor import merge_metadata_patch

    with pytest.raises(ValueError, match="date"):
        merge_metadata_patch({"report_type": "宏观"}, {"date": value})
