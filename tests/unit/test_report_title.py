import fitz

from finance_rag.src.rag.ingestion import metadata_extractor


def test_pdf_title_is_used_with_filename_fallback(tmp_path):
    path = tmp_path / "opaque.pdf"
    with fitz.open() as document:
        document.new_page()
        document.set_metadata({"title": "通信行业：6G技术白皮书"})
        document.save(path)
    assert metadata_extractor.resolve_report_title(path, "opaque.pdf") == "通信行业：6G技术白皮书"
    assert metadata_extractor.resolve_report_title(tmp_path / "missing.pdf", "fallback.pdf") == "fallback"

