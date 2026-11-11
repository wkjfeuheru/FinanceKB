"""严格元数据契约：类型必填、异类字段隔离与原子 PATCH。"""

import copy
import threading

import pytest

from src.index.milvus_kb import KnowledgeBase, _report_scalars
from src.chunking import metadata_extractor as extractor
from src.api.schemas.document import DocumentMetadataPatch
from src.ingest.documents import DocumentManager


@pytest.mark.parametrize("report_type", ["个股", "行业", "宏观", "策略"])
def test_extracted_payload_contains_only_contract_fields(report_type):
    payload = extractor.ExtractedMetadata(report_type=report_type).as_metadata()
    # version 透传（缺省空串），不允许残留任何旧维度
    assert payload == {"category": "", "date": "", "version": "",
                       "report_type": report_type, "broker": ""}


@pytest.mark.parametrize("values", [{}, {"report_type": "点评"}])
def test_extracted_payload_rejects_missing_or_invalid_report_type(values):
    with pytest.raises(ValueError, match="report_type"):
        extractor.ExtractedMetadata(**values).as_metadata()


def test_apply_metadata_output_only_contains_contract_fields():
    original = {"date": "2026-09-01"}
    merged = extractor.apply_metadata(original, extractor.ExtractedMetadata(report_type="宏观"))
    # 抽取结果只补空缺：已解析的 date 不被空串覆盖；空 version 不写入；category 恒为空
    assert merged == {"category": "", "date": "2026-09-01",
                      "report_type": "宏观", "broker": ""}
    assert original == {"date": "2026-09-01"}  # 不污染入参


def test_milvus_adapter_builds_scalar_columns_from_contract_fields():
    meta = {"report_type": "个股", "broker": "中信证券"}
    before = dict(meta)
    scalars = _report_scalars(meta, {"block_type": "table", "start_page": 3})
    assert scalars == {
        "report_type": "个股", "broker": "中信证券",
        "block_type": "table", "start_page": 3, "end_page": 0,
        "image_key": "",
    }
    assert meta == before  # 不改动调用方传入的字典


def test_milvus_adapter_rejects_incomplete_metadata():
    with pytest.raises(ValueError):
        _report_scalars({}, {})


@pytest.fixture(autouse=True)
def isolated_fingerprints(tmp_path, monkeypatch):
    from src.index.fingerprint_store import FingerprintStore
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
    row = {"id": "a", "report_type": "个股", "broker": "中信证券",
           "category": "investment_research", "date": "2026-09-01",
           "content": "原文", "dense_vector": [0.1, 0.2], "start_page": 4}
    client = _RowsClient([{**row, **(values or {})}])
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection_name = "test"  # 非默认集合：PATCH 不触发 manifest 事务
    kb._write_lock = threading.RLock()
    monkeypatch.setattr(kb, "_get_client", lambda: client)
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_collection_has_field", lambda *args: True)
    manager = DocumentManager.__new__(DocumentManager)
    manager._kb = kb
    return manager, client


def test_patch_merges_current_values_without_erasing_report_type(monkeypatch):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {"broker": "华泰证券"})
    assert client.rows[0]["report_type"] == "个股"
    assert client.rows[0]["broker"] == "华泰证券"
    assert client.rows[0]["dense_vector"] == [0.1, 0.2]   # 向量不动
    assert client.rows[0]["start_page"] == 4              # 块级字段不动
    assert result["metadata"]["report_type"] == "个股"
    assert result["metadata"]["date"] == "2026-09-01"


@pytest.mark.parametrize("patch", [
    {"security_name": ""},      # 已删除字段：按未知字段拒绝，不能借补丁回流
    {"report_type": "点评"},     # 非法类型值
    {"date": "2026-02-31"},     # 非法日历日
])
def test_invalid_patch_leaves_all_persisted_values_unchanged(monkeypatch, patch):
    manager, client = _kb(monkeypatch)
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError):
        manager.update_document_metadata("研报.pdf", patch)
    assert client.rows == before


def test_patch_report_type_switch_keeps_other_fields(monkeypatch):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {"report_type": "行业"})
    assert client.rows[0]["report_type"] == "行业"
    assert client.rows[0]["broker"] == "中信证券"  # broker 不随类型切换清空
    assert result["metadata"]["category"] == ""
    assert result["metadata"]["date"] == "2026-09-01"


def test_patch_schema_rejects_unknown_field():
    with pytest.raises(ValueError):
        DocumentMetadataPatch(unknown_dimension="值")


def test_chat_source_schema_uses_only_supported_public_dimensions():
    from src.api.schemas.chat import SourceInfo

    source = SourceInfo(index=1, title="报告", source="研报.pdf", score=1.0, preview="正文")
    assert set(source.model_dump()) == {
        "index", "title", "source", "category", "collection", "chunk", "score", "rerank_score",
        "preview", "date", "block_type", "start_page", "end_page", "image_key", "truncated",
        "report_type", "broker",
        "id", "chunk_id", "child_id", "chunk_key", "parent_id", "parent_key", "covered_child_ids",
        "table_id", "document_sha256", "document_version",
    }


def test_supported_metadata_filter_dimensions():
    assert set(extractor.metadata_field_validators()) == {
        "date", "report_type", "broker"
    }


def test_canonical_and_patch_pass_through_explicit_version():
    """version 原样透传、不参与校验（入库链路在合并后才回填 version）。"""
    merged = extractor.ExtractedMetadata(version="3", report_type="宏观").as_metadata()
    assert merged["version"] == "3"
    patched = extractor.merge_metadata_patch(
        {"report_type": "宏观", "version": "3"}, {"broker": ""}
    )
    assert patched["version"] == "3"


def test_invalid_patch_in_one_row_prevents_other_rows_being_updated(monkeypatch):
    manager, client = _kb(monkeypatch)
    client.rows.append({**client.rows[0], "id": "b", "report_type": "错误"})
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError, match="report_type"):
        manager.update_document_metadata("研报.pdf", {"broker": "华泰证券"})
    assert client.rows == before  # 任一行非法则整批不写


@pytest.mark.parametrize("field", ["date"])
def test_canonical_public_varchar_normalizes_explicit_null_to_empty_string(field):
    payload = extractor.build_canonical_metadata({"report_type": "宏观", field: None})
    assert payload[field] == ""


@pytest.mark.parametrize(("field", "limit"), [("date", 32)])
def test_canonical_public_varchar_rejects_overlong_values(field, limit):
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata({"report_type": "宏观", field: "a" * (limit + 1)})


@pytest.mark.parametrize("field", ["date"])
def test_patch_public_varchar_null_clears_persisted_value(monkeypatch, field):
    manager, client = _kb(monkeypatch)
    result = manager.update_document_metadata("研报.pdf", {field: None})
    assert client.rows[0][field] == ""
    assert result["metadata"][field] == ""


@pytest.mark.parametrize(("field", "limit"), [("date", 32)])
def test_patch_public_varchar_overlong_value_does_not_write(monkeypatch, field, limit):
    manager, client = _kb(monkeypatch)
    before = copy.deepcopy(client.rows)
    with pytest.raises(ValueError, match=field):
        manager.update_document_metadata("研报.pdf", {field: "a" * (limit + 1)})
    assert client.rows == before


@pytest.mark.parametrize(("field", "limit"), [("date", 32)])
def test_public_varchar_rejects_utf8_payload_exceeding_storage_limit(field, limit):
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata({"report_type": "宏观", field: "中" * (limit // 3 + 1)})


def test_canonical_publication_date_requires_real_calendar_day():
    payload = extractor.build_canonical_metadata({"report_type": "宏观", "date": "2026-09-24"})
    assert payload["date"] == "2026-09-24"
    with pytest.raises(ValueError, match="date"):
        extractor.build_canonical_metadata({"report_type": "宏观", "date": "2026-02-31"})


@pytest.mark.parametrize("field", ["broker"])
def test_public_names_reject_overlong_utf8_before_storage(field):
    values = {"report_type": "个股", field: "中" * 22}  # 66 字节，超过 64 字节上限
    with pytest.raises(ValueError, match=field):
        extractor.build_canonical_metadata(values)
