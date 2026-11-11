"""研报元数据在 Milvus 侧的落地契约。

Milvus 集合是 ``enable_dynamic_field=False`` 的固定字段表，且标量**不接受 None**、
VARCHAR 超长会直接让整篇文档入库失败。因此这里锁住三件事：
字段齐备、默认值非空、超长值被截断而不是抛错。
"""

import pytest
from pymilvus import DataType

from src.index.milvus_kb import (
    _REPORT_FIELD_SPECS,
    _RETIRED_SCALAR_FIELDS,
    _SCALAR_INDEX_FIELDS,
    _TITLE_MAX_BYTES,
    _UNSUPPORTED,
    _add_report_fields,
    _add_scalar_indexes,
    _clip,
    _fill_missing_scalars,
    _report_scalars,
    _scalar_default,
    _varchar_max_length,
)


class _SchemaStub:
    def __init__(self) -> None:
        self.fields: list[tuple[str, int | None]] = []

    def add_field(self, name, dtype, max_length=None, **kwargs):  # noqa: ANN001
        self.fields.append((name, max_length))


def test_report_fields_cover_metadata_and_evidence_columns():
    schema = _SchemaStub()

    _add_report_fields(schema)

    names = [name for name, _ in schema.fields]
    assert names == [
        "report_type",
        "broker",
        "block_type",
        "start_page",
        "end_page",
        "image_key",
    ]
    assert _RETIRED_SCALAR_FIELDS.isdisjoint(names)
    limits = dict(schema.fields)
    assert limits["report_type"] == 16
    assert limits["broker"] == 64
    assert limits["block_type"] == 16
    assert limits["image_key"] == 256
    assert limits["start_page"] is None  # INT64 无长度
    assert limits["end_page"] is None


def test_report_field_specs_declare_varchar_lengths():
    varchar = {
        name: length for name, dtype, length in _REPORT_FIELD_SPECS if length is not None
    }

    assert varchar["report_type"] == 16
    assert varchar["broker"] == 64
    assert varchar["image_key"] == 256
    assert _RETIRED_SCALAR_FIELDS.isdisjoint(varchar)
    # 无长度的字段必须是 INT64 页码，不允许混入未声明长度的 VARCHAR
    assert {name for name, dtype, length in _REPORT_FIELD_SPECS if length is None} == {
        "start_page",
        "end_page",
    }


def test_scalars_default_to_non_null_values():
    scalars = _report_scalars({"report_type": "宏观"}, {})

    assert scalars["report_type"] == "宏观"
    assert scalars["broker"] == ""
    assert scalars["block_type"] == "text"      # 缺省视为普通文本块
    assert scalars["start_page"] == 0           # 页码缺失就是 0，不猜
    assert scalars["end_page"] == 0
    assert _RETIRED_SCALAR_FIELDS.isdisjoint(scalars)
    assert all(value is not None for value in scalars.values())


def test_scalars_read_document_and_chunk_level_sources():
    scalars = _report_scalars(
        {
            "report_type": "行业",
            "broker": "中信证券",
            "security_name": "贵州茅台",
            "industry_l1": "食品饮料",
            "industry_l2": "白酒Ⅱ",
            "meta_source": "regex",
            "needs_review": True,
        },
        {"block_type": "table", "start_page": 4, "end_page": 5, "image_key": "images/a/4-0.png"},
    )

    assert scalars["report_type"] == "行业"
    assert scalars["broker"] == "中信证券"
    assert scalars["block_type"] == "table"
    assert _RETIRED_SCALAR_FIELDS.isdisjoint(scalars)
    assert (scalars["start_page"], scalars["end_page"]) == (4, 5)
    assert scalars["image_key"] == "images/a/4-0.png"


def test_scalars_ignore_retired_fields_in_raw_metadata():
    """证券名和行业已退役，原始元数据里的值不能再写进标量。"""
    scalars = _report_scalars(
        {"report_type": "个股"},
        {},
        raw_meta={"security_name": "杰瑞股份", "industry_l1": "机械设备"},
    )

    assert _RETIRED_SCALAR_FIELDS.isdisjoint(scalars)
    assert scalars["report_type"] == "个股"


def test_scalars_clip_chunk_fields_but_require_valid_document_metadata():
    scalars = _report_scalars(
        {"report_type": "个股", "broker": "东吴证券"},
        {"block_type": "t" * 100, "image_key": "k" * 500},
    )

    assert scalars["report_type"] == "个股"
    assert scalars["broker"] == "东吴证券"
    assert len(scalars["block_type"]) == 16
    assert len(scalars["image_key"]) == 256

    # 文档级 report_type 走完整校验：非法值直接拒绝，绝不截断或伪造
    with pytest.raises(ValueError):
        _report_scalars({"report_type": "点评"}, {})


def test_scalars_tolerate_none_values():
    scalars = _report_scalars(
        {"report_type": "宏观", "needs_review": None},
        {"start_page": None, "block_type": None},
    )

    assert scalars["broker"] == ""
    assert scalars["block_type"] == "text"
    assert scalars["start_page"] == 0
    assert scalars["end_page"] == 0
    assert "needs_review" not in scalars
    assert all(value is not None for value in scalars.values())


def test_fill_missing_scalars_covers_newer_collection_columns():
    """旧集合多出来的列（含已退役列）必须补默认值，而不是让整批插入失败。"""
    rows = [{"id": "c1"}, {"id": "c2"}]

    filled = _fill_missing_scalars(
        rows,
        {
            "id": "",
            "security_name": "",
            "industry_l1": "",
            "needs_review": False,
        },
    )

    assert filled == ["industry_l1", "needs_review", "security_name"]
    assert rows == [
        {"id": "c1", "security_name": "", "industry_l1": "", "needs_review": False},
        {"id": "c2", "security_name": "", "industry_l1": "", "needs_review": False},
    ]


def test_fill_missing_scalars_is_noop_when_all_columns_present():
    rows = [{"id": "c1", "security_name": "贵州茅台"}]

    assert _fill_missing_scalars(rows, {"id": "", "security_name": ""}) == []
    assert rows == [{"id": "c1", "security_name": "贵州茅台"}]


def test_collection_scalar_defaults_skip_primary_vectors_and_function_outputs():
    """默认值只覆盖可写标量：主键、向量与 BM25 输出字段不能被补值。"""
    from src.index.milvus_kb import KnowledgeBase

    class _Client:
        def describe_collection(self, name):
            return {
                "fields": [
                    {"name": "id", "type": DataType.VARCHAR, "is_primary": True},
                    {"name": "security_name", "type": DataType.VARCHAR},
                    {"name": "needs_review", "type": DataType.BOOL},
                    {"name": "dense_vector", "type": DataType.FLOAT_VECTOR},
                    {"name": "sparse_vector", "type": DataType.SPARSE_FLOAT_VECTOR},
                ],
                "functions": [{"output_field_names": ["sparse_vector"]}],
            }

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection_name = "test"
    kb._scalar_defaults = None

    assert kb._collection_scalar_defaults(_Client()) == {
        "security_name": "",
        "needs_review": False,
    }


def test_scalar_defaults_cover_supported_milvus_types():
    assert _scalar_default(DataType.VARCHAR) == ""
    assert _scalar_default(DataType.INT64) == 0
    assert _scalar_default(DataType.BOOL) is False
    assert _scalar_default(DataType.FLOAT) == 0.0
    # 向量类型由写入路径显式提供或由 Function 生成，不能补默认值
    assert _scalar_default(DataType.FLOAT_VECTOR) is _UNSUPPORTED
    assert _scalar_default(DataType.SPARSE_FLOAT_VECTOR) is _UNSUPPORTED


def test_migration_defaults_cover_every_report_field():
    """旧集合搬运时，新集合要求的每一列都必须有默认值，否则整批插入失败。"""
    from scripts.migrate_kb_schema import REPORT_FIELDS, _derive_missing

    derived = _derive_missing({"content": "正文", "parent_id": "p1", "report_type": "宏观"})

    for field in REPORT_FIELDS:
        assert field in derived, field
        assert derived[field] is not None, field


def test_migration_only_copies_columns_present_in_backup():
    """备份集合缺少新列时必须跳过，否则 query 会因「字段不存在」直接失败。"""
    from scripts.migrate_kb_schema import _copy_fields_for

    class _Client:
        def describe_collection(self, name):
            return {"fields": [{"name": "id"}, {"name": "content"}, {"name": "category"}]}

    fields = _copy_fields_for(_Client(), "kb_backup")

    assert "category" in fields
    assert "id" in fields
    assert "security_name" not in fields
    assert "block_type" not in fields


def test_storage_clip_respects_utf8_byte_budget():
    value = _clip("中文标题" * 30, 128)
    assert len(value.encode("utf-8")) <= 128
    assert value == ("中文标题" * 30)[:42]


def test_migration_report_fields_match_schema_specs():
    from scripts.migrate_kb_schema import REPORT_FIELDS

    assert REPORT_FIELDS == tuple(name for name, _, _ in _REPORT_FIELD_SPECS)


def test_scalar_indexes_cover_filter_fields():
    class _Params(list):
        def add_index(self, field_name, index_type="", **kwargs):  # noqa: ANN001
            self.append((field_name, index_type))

    params = _Params()
    _add_scalar_indexes(params)

    assert params == [(name, "INVERTED") for name in _SCALAR_INDEX_FIELDS]
    assert _TITLE_MAX_BYTES == 512


class _IndexParams(list):
    def add_index(self, field_name, index_type="", **kwargs):  # noqa: ANN001
        self.append((field_name, index_type))


def _kb():
    from src.index.milvus_kb import KnowledgeBase

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.collection_name = "finance_kb"
    kb._schema_fields = None
    kb._title_max_bytes = None
    kb._noted_retired_fields = set()
    return kb


class _IndexClient:
    def __init__(self, fields, indexed, *, fail_fields=()):
        self.fields = list(fields)
        self.indexed = set(indexed)
        self.fail_fields = set(fail_fields)
        self.created: list[str] = []

    def describe_collection(self, name, timeout=None):  # noqa: ANN001
        return {"fields": [{"name": field} for field in self.fields]}

    def list_indexes(self, name, field_name="", timeout=None):  # noqa: ANN001
        if field_name:
            if field_name not in self.indexed:
                return []
            # source 用空索引名，确认空名字也不会被当成「没有索引」
            return [""] if field_name == "source" else [f"{field_name}_idx"]
        return [f"{field}_idx" for field in self.indexed]

    def prepare_index_params(self):
        return _IndexParams()

    def create_index(self, name, params, timeout=None):  # noqa: ANN001
        field = params[0][0]
        if field in self.fail_fields:
            raise RuntimeError(f"cannot index {field}")
        self.created.append(field)


def test_ensure_scalar_indexes_creates_only_missing_fields():
    client = _IndexClient(
        fields=_SCALAR_INDEX_FIELDS,
        indexed={"is_current", "source", "dense_vector"},
    )

    _kb()._ensure_scalar_indexes(client)

    assert client.created == ["category", "date", "report_type", "broker"]


def test_ensure_scalar_indexes_skips_when_all_present():
    client = _IndexClient(fields=_SCALAR_INDEX_FIELDS, indexed=set(_SCALAR_INDEX_FIELDS))

    _kb()._ensure_scalar_indexes(client)

    assert client.created == []


def test_ensure_scalar_indexes_skips_fields_absent_from_old_schema():
    client = _IndexClient(fields=["source", "is_current"], indexed=set())

    _kb()._ensure_scalar_indexes(client)

    assert client.created == ["is_current", "source"]


def test_ensure_scalar_indexes_continues_after_one_field_fails():
    client = _IndexClient(
        fields=["category", "broker"],
        indexed=set(),
        fail_fields={"category"},
    )

    _kb()._ensure_scalar_indexes(client)

    assert client.created == ["broker"]


def test_ensure_scalar_indexes_swallows_lookup_failure():
    class _Broken:
        def describe_collection(self, name, timeout=None):  # noqa: ANN001
            raise RuntimeError("milvus down")

    _kb()._ensure_scalar_indexes(_Broken())


class _TitleClient:
    def __init__(self, length, *, fail=False):
        self.length = length
        self.fail = fail
        self.altered = None

    def describe_collection(self, name, timeout=None):  # noqa: ANN001
        return {"fields": [{"name": "title", "params": {"max_length": str(self.length)}}]}

    def alter_collection_field(self, name, field, params, timeout=None):  # noqa: ANN001
        if self.fail:
            raise RuntimeError("alter rejected")
        self.altered = (field, params)
        self.length = params["max_length"]


def test_ensure_title_capacity_widens_short_field():
    client = _TitleClient(128)
    kb = _kb()

    kb._ensure_title_capacity(client)

    assert client.altered == ("title", {"max_length": 512})
    assert kb._title_byte_limit(client) == 512


def test_ensure_title_capacity_leaves_long_enough_field():
    client = _TitleClient(512)
    kb = _kb()

    kb._ensure_title_capacity(client)

    assert client.altered is None
    assert kb._title_max_bytes == 512


def test_ensure_title_capacity_keeps_old_limit_when_alter_fails():
    client = _TitleClient(128, fail=True)
    kb = _kb()

    kb._ensure_title_capacity(client)

    assert kb._title_max_bytes == 128


def test_title_clip_follows_collection_limit():
    text = "茅" * 80  # 240 字节，超过旧上限、短于新上限
    assert len(text.encode("utf-8")) == 240
    short = _kb()
    short._title_max_bytes = None
    clipped = _clip(text, short._title_byte_limit(_TitleClient(128)))
    assert len(clipped.encode("utf-8")) <= 128
    assert clipped != text

    wide = _kb()
    assert _clip(text, wide._title_byte_limit(_TitleClient(512))) == text


def test_varchar_max_length_reads_params_or_top_level():
    fields = [
        {"name": "title", "params": {"max_length": "128"}},
        {"name": "source", "max_length": 512},
    ]

    assert _varchar_max_length(fields, "title") == 128
    assert _varchar_max_length(fields, "source") == 512
    assert _varchar_max_length(fields, "missing") is None
