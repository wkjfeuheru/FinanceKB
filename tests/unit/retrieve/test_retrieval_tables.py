"""表格检索链路的回归测试：紧凑索引块贯穿检索，输出期才整表展开。

背景：召回与 BGE 重排全程基于「表头+摘要」紧凑块（信噪比高、成本低），
完整表格在拼入 LLM 上下文前才按 table_id 展开（chat_service.expand_table_docs）。
因此这里同时锁住两端：

* 检索器侧——父块扩展跳过表格/图片块（它们的 parent_id 指向各自存储），
  且不再持有任何整表展开路径；
* 输出期——整表替换紧凑摘要、8000 字符截断并显式标注、
  各类失败一律退化为保留摘要（链路不断）。
"""

from src.retrieve.hybrid_retriever import HybridRetriever
from src import engine as chat_service
from src.engine import (
    TABLE_MAX_CHARS,
    TABLE_TRUNCATION_NOTE,
    expand_table_docs,
)


class _FakeTableStore:
    def __init__(self, rows: dict[str, dict]):
        self._rows = rows
        self.calls: list[list[str]] = []

    def get_batch(self, table_ids: list[str]) -> list[dict]:
        self.calls.append(list(table_ids))
        return [self._rows[tid] for tid in table_ids if tid in self._rows]


class _FakeParentStore:
    def __init__(self, rows: dict[str, dict]):
        self._rows = rows

    def get_batch(self, parent_ids: list[str]) -> list[dict]:
        return [self._rows[pid] for pid in parent_ids if pid in self._rows]


def _retriever(*, parents=None, schema_fields=None) -> HybridRetriever:
    client = type(
        "_Client",
        (),
        {
            "describe_collection": lambda self, name: {
                "fields": [{"name": f} for f in (schema_fields or [])]
            }
        },
    )()
    return HybridRetriever(
        client=client,
        collection_name="finance_kb",
        embed_query_fn=lambda text: [0.0],
        parent_store=_FakeParentStore(parents or {}),
    )


def _table_match(table_id: str = "t1") -> dict:
    return {
        "id": "c-table",
        "parent_id": table_id,
        "block_type": "table",
        "content": "表格：盈利预测\n表头：年份 | 营收",
        "source": "s.pdf",
        "date": "2026-08-15",
        "score": 1.0,
    }


def _image_match() -> dict:
    return {
        "id": "c-image",
        "parent_id": "t1",
        "block_type": "image",
        "content": "燃机订单结构图",
        "source": "s.pdf",
        "date": "2026-08-15",
        "score": 0.9,
    }


def _table(table_id: str = "t1", rows: int = 400) -> dict:
    body = "\n".join(f"| 2026E | 第{i}行数据 | 明细{i} |" for i in range(rows))
    return {
        "id": table_id,
        "markdown": f"| 年份 | 项目 | 明细 |\n| --- | --- | --- |\n{body}",
        "payload": [["年份", "项目", "明细"]],
        "row_count": rows,
        "source": "s.pdf",
        "start_page": 4,
    }


def _use_store(monkeypatch, store) -> None:
    monkeypatch.setattr(chat_service, "_table_store_for", lambda collection: store)


# ---------------------------------------------------------------------------
# 检索器侧：父块扩展与字段探测
# ---------------------------------------------------------------------------

def test_parent_expansion_skips_table_and_image_rows():
    """表格/图片块的 parent_id 指向各自的存储，不能被父块逻辑覆盖。"""
    retriever = _retriever(parents={"t1": {"id": "t1", "content": "父块" * 100}})
    matches = [_table_match(), _image_match()]

    expanded = retriever._expand_to_parents(matches, k=2)

    assert expanded[0]["content"] == matches[0]["content"]
    assert expanded[1]["content"] == matches[1]["content"]


def test_parent_expansion_still_works_for_text_rows():
    parent_text = "子块" + "父块全文内容" * 20  # 包含子块，长度落在 [50, 3000]
    retriever = _retriever(parents={"p1": {"id": "p1", "content": parent_text}})
    matches = [
        {"id": "c1", "parent_id": "p1", "block_type": "text", "content": "子块", "score": 1.0}
    ]

    expanded = retriever._expand_to_parents(matches, k=1)

    assert expanded[0]["content"] == parent_text


def test_output_fields_include_report_fields_when_schema_has_them():
    retriever = _retriever(
        schema_fields=["id", "content", "block_type", "start_page", "broker"]
    )

    fields = retriever._output_fields()

    assert "block_type" in fields
    assert "broker" in fields
    # 已删除的行业/标的字段不能被请求：集合没有的字段会让整个检索报错
    assert "industry_l1" not in fields
    assert "security_name" not in fields
    assert "content" in fields


def test_output_fields_fall_back_to_base_when_schema_probe_fails():
    """旧集合（或探测失败）不能让整个检索挂掉，只能少取几个字段。"""

    class _Boom:
        def describe_collection(self, name):
            raise RuntimeError("集合不存在")

    retriever = HybridRetriever(
        client=_Boom(),
        collection_name="finance_kb",
        embed_query_fn=lambda text: [0.0],
        parent_store=None,
    )

    fields = retriever._output_fields()

    assert "id" in fields and "content" in fields


def test_retriever_no_longer_expands_tables():
    """检索器不再做整表展开：紧凑块贯穿召回/重排，展开只在输出期发生。"""
    assert not hasattr(HybridRetriever, "_expand_tables")
    assert not hasattr(HybridRetriever, "TABLE_MAX_CHARS")


# ---------------------------------------------------------------------------
# 输出期：expand_table_docs
# ---------------------------------------------------------------------------

def test_table_docs_expand_to_full_table_before_context(monkeypatch):
    """表格块在拼上下文前按 table_id 展开为整表——不受父块 3000 上限约束。"""
    store = _FakeTableStore({"t1": _table(rows=200)})
    _use_store(monkeypatch, store)
    match = _table_match()

    expanded = expand_table_docs([match])

    full_markdown = _table(rows=200)["markdown"]
    assert len(full_markdown) > 3000          # 前提：确实超过父块上限
    assert expanded[0]["content"] == full_markdown   # 一行都没丢
    assert "第199行数据" in expanded[0]["content"]
    assert expanded[0]["truncated"] is False
    assert expanded[0]["table_page"] == 4
    assert expanded[0]["table_row_count"] == 200


def test_oversized_table_is_truncated_and_explicitly_marked(monkeypatch):
    """超过 8000 字符的表按行截断，并带上提示语——不静默删数据。"""
    store = _FakeTableStore({"t1": _table(rows=1000)})
    _use_store(monkeypatch, store)

    expanded = expand_table_docs([_table_match()])
    content = expanded[0]["content"]

    assert expanded[0]["truncated"] is True
    assert content.endswith(TABLE_TRUNCATION_NOTE)
    assert len(content) <= TABLE_MAX_CHARS
    assert "| 年份 | 项目 | 明细 |" in content        # 表头保留
    assert all(line.startswith("|") for line in content.splitlines()[:-3])


def test_missing_row_keeps_compact_summary(monkeypatch):
    """整表缺失时保留紧凑摘要（表头+摘要本身已含语义），绝不把内容清空。"""
    store = _FakeTableStore({})
    _use_store(monkeypatch, store)
    match = _table_match()

    expanded = expand_table_docs([match])

    assert expanded[0]["content"] == match["content"]
    assert expanded[0]["truncated"] is False


def test_store_unavailable_keeps_compact_summary(monkeypatch):
    _use_store(monkeypatch, None)
    match = _table_match()

    expanded = expand_table_docs([match])

    assert expanded[0]["content"] == match["content"]


def test_get_batch_failure_keeps_compact_summary(monkeypatch):
    """PG 查询失败只降级为摘要，不阻断回答链路。"""

    class _Boom:
        def get_batch(self, table_ids):
            raise RuntimeError("PG 不可达")

    _use_store(monkeypatch, _Boom())
    match = _table_match()

    expanded = expand_table_docs([match])

    assert expanded[0]["content"] == match["content"]


def test_start_page_falls_back_to_table_page(monkeypatch):
    """切块期页码归属失败（0）时，用整表存储里的起始页补位。"""
    store = _FakeTableStore({"t1": _table(rows=5)})
    _use_store(monkeypatch, store)
    match = _table_match()
    match["start_page"] = 0

    expanded = expand_table_docs([match])

    assert expanded[0]["start_page"] == 4


def test_non_table_docs_are_untouched_and_trigger_no_query(monkeypatch):
    """图片/文本块不触发整表查询；没有表格块时完全不查 PG。"""
    store = _FakeTableStore({})
    _use_store(monkeypatch, store)
    docs = [
        _image_match(),
        {"id": "c1", "content": "正文", "block_type": "text", "score": 1.0},
    ]

    expanded = expand_table_docs(docs)

    assert expanded[0]["content"] == "燃机订单结构图"
    assert expanded[1]["content"] == "正文"
    assert store.calls == []


def test_tables_route_to_their_own_collection_store(monkeypatch):
    """多知识库：doc 自带的 collection 决定查哪个整表存储。"""
    stores: dict[str, _FakeTableStore] = {}

    def _store_for(collection: str):
        store = _FakeTableStore({"t1": _table(rows=3)})
        stores[collection] = store
        return store

    monkeypatch.setattr(chat_service, "_table_store_for", _store_for)

    doc_a = _table_match()
    doc_a["collection"] = "kb_a"
    doc_b = _table_match()
    doc_b["collection"] = "kb_b"

    expand_table_docs([doc_a, doc_b])

    assert set(stores) == {"kb_a", "kb_b"}
    assert all(store.calls == [["t1"]] for store in stores.values())
