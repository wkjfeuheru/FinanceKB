"""严格元数据契约：类型必填、异类字段隔离与原子 PATCH。"""

import copy
import threading

import pytest

from finance_rag.src.infrastructure.vector_store.milvus_kb import KnowledgeBase, _report_scalars
from finance_rag.src.rag.ingestion import metadata_extractor as extractor
from finance_rag.src.schemas.document import DocumentMetadataPatch
from finance_rag.src.services.document_service import DocumentManager


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"report_type": "个股", "security_name": "贵州茅台", "industry_l1": "食品饮料"},
         {"report_type": "个股", "security_name": "贵州茅台"}),
        ({"report_type": "行业", "industry_l1": "电子", "security_name": "贵州茅台"},
         {"report_type": "行业", "industry_l1": "电子"}),
        ({"report_type": "宏观", "security_name": "贵州茅台", "industry_l1": "电子"},
         {"report_type": "宏观"}),
    ],
)
def test_extracted_payload_contains_only_applicable_fields(values, expected):
    payload = extractor.ExtractedMetadata(**values).as_metadata()
    assert payload == {**expected, "broker": "", "meta_source": "regex", "needs_review": False}


@pytest.mark.parametrize(
    ("values", "field"),
    [({}, "report_type"), ({"report_type": "策略"}, "report_type"),
     ({"report_type": "个股"}, "security_name"),
     ({"report_type": "个股", "security_name": "  "}, "security_name"),
     ({"report_type": "行业"}, "industry_l1"),
     ({"report_type": "行业", "industry_l1": "白酒"}, "industry_l1")],
)
def test_extracted_payload_rejects_missing_or_invalid_required_field(values, field):
    with pytest.raises(ValueError, match=field):
        extractor.ExtractedMetadata(**values).as_metadata()


def test_industry_parent_is_inferred_from_second_level():
    payload = extractor.ExtractedMetadata(report_type="行业", industry_l2="半导体").as_metadata()
    assert payload["industry_l1"] == "电子"
    assert payload["industry_l2"] == "半导体"


def test_invalid_second_level_is_removed_and_requires_review():
    payload = extractor.ExtractedMetadata(
        report_type="行业", industry_l1="食品饮料", industry_l2="半导体"
    ).as_metadata()
    assert "industry_l2" not in payload
    assert payload["needs_review"] is True


def test_apply_metadata_cleans_previous_type_and_preserves_public_metadata():
    original = {"category": "investment_research", "date": "2026-09-01", "security_name": "茅台"}
    merged = extractor.apply_metadata(original, extractor.ExtractedMetadata(report_type="宏观"))
    assert merged == {"category": "investment_research", "date": "2026-09-01",
                      "report_type": "宏观", "broker": "", "meta_source": "regex", "needs_review": False}
    assert original["security_name"] == "茅台"


def test_extraction_rejects_document_with_missing_industry(monkeypatch):
    monkeypatch.setattr(extractor, "_llm_extract", lambda **_: {"report_type": "行业"})
    with pytest.raises(ValueError, match="industry_l1"):
        extractor.extract_metadata("行业报告.pdf", enable_llm=True)


def test_milvus_adapter_blanks_inapplicable_fixed_columns():
    meta = {"report_type": "个股", "security_name": "贵州茅台", "industry_l1": "食品饮料"}
    before = dict(meta)
    scalars = _report_scalars(meta, {"block_type": "table", "start_page": 3})
    assert scalars["industry_l1"] == scalars["industry_l2"] == ""
    assert scalars["security_name"] == "贵州茅台"
    assert scalars["block_type"] == "table"
    assert scalars["start_page"] == 3
    assert meta == before


@pytest.mark.parametrize("metadata", [{}, {"report_type": "个股"}, {"report_type": "行业"}])
def test_milvus_adapter_rejects_incomplete_metadata(metadata):
    with pytest.raises(ValueError):
        _report_scalars(metadata, {})


@pytest.fixture(autouse=True)
def isolated_fingerprints(tmp_path, monkeypatch):
    from finance_rag.src.infrastructure.vector_store.fingerprint_store import FingerprintStore
    store = FingerprintStore(str(tmp_path / "fp.json"))
    monkeypatch.setattr(KnowledgeBase, "_get_fingerprint_store", lambda self: store)


class _RowsClient:
    """外部 Milvus 的内存替身；验证最终持久数据，不替换被测 PATCH 逻辑。"""

    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def query(self, **kwargs):
        return copy.deepcopy(self.rows)

    def upsert(self, *, collection_name, data):
        self.rows = copy.deepcopy(data)

    def flush(self, name):
        pass


def _kb(monkeypatch, values=None):
    row = {"id": "a", "report_type": "个股", "security_name": "贵州茅台",
           "industry_l1": "", "industry_l2": "", "broker": "中信证券",
           "category": "investment_research", "date": "2026-09-01", "meta_source": "llm",
           "needs_review": False, "content": "原文", "dense_vector": [0.1, 0.2], "start_page": 4}
    client = _RowsClient([{**row, **(values or {})}])
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection_name = "test"
    kb._write_lock = threading.RLock()
    monkeypatch.setattr(kb, "_get_client", lambda: client)
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_collection_has_field", lambda *args: True)
    manager = DocumentManager.__new__(DocumentManager)
    manager._kb = kb
    return manager, client


def test_patch_merges_current_values_without_erasing_security_name(monkeypatch):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {"broker": "华泰证券"})
    assert client.rows[0]["security_name"] == "贵州茅台"
    assert client.rows[0]["broker"] == "华泰证券"
    assert client.rows[0]["dense_vector"] == [0.1, 0.2]
    assert client.rows[0]["start_page"] == 4
    assert result["metadata"]["security_name"] == "贵州茅台"
    assert result["metadata"]["meta_source"] == "manual"


@pytest.mark.parametrize("patch", [{"security_name": ""}, {"report_type": "行业"},
                                  {"industry_l1": "电子"}, {"industry_l2": ""}])
def test_invalid_patch_leaves_all_persisted_values_unchanged(monkeypatch, patch):
    manager, client = _kb(monkeypatch)
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError):
        manager.update_document_metadata("研报.pdf", patch)
    assert client.rows == before


def test_patch_type_switch_cleans_old_type_fields(monkeypatch):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {"report_type": "行业", "industry_l2": "半导体"})
    assert client.rows[0]["security_name"] == ""
    assert client.rows[0]["industry_l1"] == "电子"
    assert client.rows[0]["industry_l2"] == "半导体"
    assert "security_name" not in result["metadata"]
    assert result["metadata"]["category"] == "investment_research"
    assert result["metadata"]["date"] == "2026-09-01"


def test_patch_invalid_second_level_is_removed_and_requires_review(monkeypatch):
    manager, client = _kb(monkeypatch, {"report_type": "行业", "security_name": "", "industry_l1": "电子"})
    result = manager.update_document_metadata("研报.pdf", {"industry_l2": "白酒Ⅱ"})
    assert client.rows[0]["industry_l2"] == ""
    assert client.rows[0]["needs_review"] is True
    assert "industry_l2" not in result["metadata"]


def test_patch_schema_rejects_unknown_field():
    with pytest.raises(ValueError):
        DocumentMetadataPatch(unknown_dimension="值")


def test_chat_source_schema_uses_only_supported_public_dimensions():
    from finance_rag.src.schemas.chat import SourceInfo

    source = SourceInfo(index=1, title="报告", source="研报.pdf", score=1.0, preview="正文")
    assert set(source.model_dump()) == {
        "index", "title", "source", "category", "collection", "chunk", "score", "rerank_score",
        "preview", "date", "block_type", "start_page", "end_page", "image_key", "truncated",
        "industry_l1", "report_type",
        "id", "chunk_id", "child_id", "chunk_key", "parent_id", "parent_key", "covered_child_ids",
        "security_name", "industry_l2", "broker", "table_id", "document_sha256", "document_version",
    }


def test_supported_metadata_filter_dimensions():
    assert set(extractor.metadata_field_validators()) == {
        "security_name", "industry_l1", "industry_l2", "report_type", "broker"
    }


def test_apply_preserves_explicit_document_version():
    merged = extractor.apply_metadata({"version": "3"}, extractor.ExtractedMetadata(report_type="宏观"))
    assert merged["version"] == "3"


def test_invalid_patch_in_one_row_prevents_other_rows_being_updated(monkeypatch):
    manager, client = _kb(monkeypatch)
    client.rows.append({**client.rows[0], "id": "b", "security_name": ""})
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError, match="security_name"):
        manager.update_document_metadata("研报.pdf", {"broker": "华泰证券"})
    assert client.rows == before


@pytest.mark.parametrize("field", ["category", "date"])
def test_canonical_public_varchar_normalizes_explicit_null_to_empty_string(field):
    payload = extractor.build_canonical_metadata({"report_type": "宏观", field: None})
    assert payload[field] == ""


@pytest.mark.parametrize(("field", "limit"), [("category", 64), ("date", 32)])
def test_canonical_public_varchar_rejects_overlong_values(field, limit):
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata({"report_type": "宏观", field: "a" * (limit + 1)})


@pytest.mark.parametrize("field", ["category", "date"])
def test_patch_public_varchar_null_clears_persisted_value(monkeypatch, field):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {field: None})
    assert client.rows[0][field] == ""
    assert result["metadata"][field] == ""


@pytest.mark.parametrize(("field", "limit"), [("category", 64), ("date", 32)])
def test_patch_public_varchar_overlong_value_does_not_write(monkeypatch, field, limit):
    manager, client = _kb(monkeypatch)
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError, match=field):
        manager.update_document_metadata("研报.pdf", {field: "a" * (limit + 1)})
    assert client.rows == before


@pytest.mark.parametrize(("field", "limit"), [("category", 64), ("date", 32)])
def test_public_varchar_rejects_utf8_payload_exceeding_storage_limit(field, limit):
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata({"report_type": "宏观", field: "中" * (limit // 3 + 1)})


@pytest.mark.parametrize(("field", "limit"), [("category", 64)])
def test_public_varchar_accepts_exact_storage_limit(field, limit):
    payload = extractor.build_canonical_metadata({"report_type": "宏观", field: "a" * limit})
    assert payload[field] == "a" * limit


def test_canonical_publication_date_requires_real_calendar_day():
    payload = extractor.build_canonical_metadata({"report_type": "宏观", "date": "2026-09-24"})
    assert payload["date"] == "2026-09-24"
    with pytest.raises(ValueError, match="date"):
        extractor.build_canonical_metadata({"report_type": "宏观", "date": "2026-02-31"})
@pytest.mark.parametrize("field", ["security_name", "broker"])
def test_public_names_reject_overlong_utf8_before_storage(field):
    values = {"report_type": "个股", "security_name": "杰瑞股份", field: "中" * 22}
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata(values)
