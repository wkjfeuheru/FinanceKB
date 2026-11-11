"""跨存储发布协议回归：覆盖失败不丢旧数据（RAG-04）、判重不用有损摘要（RAG-05）。

``docs/rag-review.md`` 的两条验收主线在这里逐点注入故障验证：

* 父块写入、表格写入、向量 insert、激活/可见性切换、指纹更新各点失败时，
  **旧版本始终保持可检索**，且不会留下「指纹说已入库、实际没有」的假状态；
* 失败后重传同一新版本不产生重复块；进程在暂存阶段退出可由
  ``recover_pending_publishes`` 收敛；
* 判重使用完整正文与表体：相同表头、相同行标签但金额不同的两份报告都能入库。

Milvus 用内存替身（``_FakeMilvusClient``，支持 insert/upsert/delete/query），
PostgreSQL 派生存储与指纹存储用可注入故障的替身 / 真实实现，因此发布协议
（暂存 → 激活 → 提交指纹）在离线环境被完整执行。
"""

from __future__ import annotations

import copy
import hashlib
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.index.fingerprint_store import FingerprintStore
from src.index.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
    VectorStoreError,
)
from src.chunking.chunker import DoclingChunks, ParentChunk, TableChunk
from tests.unit.ingest.test_ingestion_pipeline import _FakeMilvusClient, _workdir

SOURCE = "研报.pdf"


# ---------------------------------------------------------------------------
# 测试替身与工厂
# ---------------------------------------------------------------------------


class _RecordingStore:
    """父块 / 整表存储替身：记录批次与回收 keep 集合，支持一次性故障注入。"""

    def __init__(self, label: str = "store") -> None:
        self.label = label
        self.batches: list[list[dict]] = []
        self.gc_keep: list[set[str]] = []
        self.fail_store = 0
        self.fail_gc = 0

    def store_batch(self, rows: list[dict]) -> None:
        if self.fail_store:
            self.fail_store -= 1
            raise RuntimeError(f"{self.label} 写入失败")
        self.batches.append(copy.deepcopy(rows))

    def delete_by_source(self, source: str) -> int:
        return 0

    def delete_by_source_except(self, source: str, keep_ids: set[str]) -> int:
        if self.fail_gc:
            self.fail_gc -= 1
            raise RuntimeError(f"{self.label} 回收失败")
        self.gc_keep.append(set(keep_ids))
        return 0


def _kb(monkeypatch) -> SimpleNamespace:
    """构造真实 KB（Milvus 走内存替身，派生/指纹存储可观测可注入故障）。

    嵌入必须替换为确定性替身：发布协议与向量取值无关，而真实 ONNX 模型会读取
    本地缓存甚至访问模型仓库，单测不允许联网。
    """
    client = _FakeMilvusClient()
    parents = _RecordingStore("parent_chunks")
    tables = _RecordingStore("table_chunks")
    fingerprint_path = _workdir() / f"publish-{uuid.uuid4().hex}.json"
    store = FingerprintStore(str(fingerprint_path))
    kb = KnowledgeBase(collection_name="publish_kb")
    monkeypatch.setattr(kb, "_get_client", lambda: client)
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_collection_has_version_fields", lambda c: True)
    monkeypatch.setattr(kb, "_collection_has_field", lambda c, f: True)
    monkeypatch.setattr(kb, "_get_parent_store", lambda: parents)
    monkeypatch.setattr(kb, "_get_table_store", lambda: tables)
    monkeypatch.setattr(kb, "_get_fingerprint_store", lambda: store)
    monkeypatch.setattr(
        kb, "_embed_documents", lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    )
    return SimpleNamespace(
        kb=kb,
        client=client,
        parents=parents,
        tables=tables,
        store=store,
        fingerprint_path=fingerprint_path,
    )


def _document(
    source: str,
    texts: list[str],
    *,
    table_markdown: str = "",
    dedup_body: str | None = None,
) -> ParsedDocument:
    """构造已切块文档；可带整表（表体与索引摘要分离，复现 RAG-05 场景）。"""
    docs = []
    parents = []
    for index, text in enumerate(texts):
        # 块主键跟随正文，和线上一致：正文一变主键就变，覆盖时必须删掉旧主键。
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(f"{source}:{digest}".encode("utf-8")).hexdigest()[:32]
        docs.append(Document(page_content=text, metadata={
            "id": chunk_id, "chunk_key": chunk_id,
            "content_hash": digest, "source": source, "title": source,
            "chunk": index, "parent_id": f"{source}::p{index}", "heading_path": "",
        }))
        parents.append(ParentChunk(id=f"{source}::p{index}", heading="",
                                  content=text, source=source, title=source))
    tables = []
    if table_markdown:
        tables.append(TableChunk(
            id=f"{source}::t0", parent_id=f"{source}::p0", markdown=table_markdown,
            payload=[["2026E", "100"]], row_count=1, source=source,
        ))
    # dedup_body 用来直接指定「完整正文」：markdown 路径下它代表解析正文
    return ParsedDocument(
        path=Path(source), source=source, title=source,
        chunks=DoclingChunks(
            chunks=docs, markdown=dedup_body, parents=parents, tables=tables,
        ),
        metadata={"category": "", "report_type": "宏观"},
    )


def _visible(client: _FakeMilvusClient) -> list[dict]:
    """当前可见实体（模拟检索侧 ``is_current == true`` 过滤）。"""
    return [row for row in client.entities if row.get("is_current") is True]


def _visible_ids(client: _FakeMilvusClient) -> set[str]:
    return {str(row["id"]) for row in _visible(client)}


def _assert_old_version_intact(client: _FakeMilvusClient, expected_ids: set[str]) -> None:
    """旧版本必须完整可检索。"""
    assert _visible_ids(client) == expected_ids


def _assert_no_pending(env: SimpleNamespace) -> None:
    assert env.store.pending_sources() == [], "失败后不得残留未收敛的发布登记"


# ---------------------------------------------------------------------------
# RAG-04：各点故障注入，旧版始终可检索
# ---------------------------------------------------------------------------


def test_parent_store_failure_keeps_old_version_visible(monkeypatch):
    """父块写入失败：不进入向量发布，旧版本原样可检索，指纹不变。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")
    old_ids = _visible_ids(env.client)
    old_fingerprint = env.store.get(SOURCE)

    env.parents.fail_store = 1
    with pytest.raises(RuntimeError, match="parent_chunks 写入失败"):
        env.kb.add_parsed_document(_document(SOURCE, ["新版本正文"]), content_hash="h2")

    _assert_old_version_intact(env.client, old_ids)
    assert env.store.get(SOURCE) == old_fingerprint, "失败不得改动指纹"
    _assert_no_pending(env)
    assert env.store.is_unchanged(SOURCE, "h1") is True, "旧版本仍完整可见"


def test_table_store_failure_keeps_old_version_visible(monkeypatch):
    """整表写入失败：同样不进入向量发布，旧版本可检索。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(
        _document(SOURCE, ["旧版本正文"], table_markdown="| 年份 | 营收 |\n| --- | --- |\n| 2025 | 10 |"),
        content_hash="h1",
    )
    old_ids = _visible_ids(env.client)

    env.tables.fail_store = 1
    with pytest.raises(RuntimeError, match="table_chunks 写入失败"):
        env.kb.add_parsed_document(
            _document(SOURCE, ["新版本正文"], table_markdown="| 年份 | 营收 |\n| --- | --- |\n| 2026 | 20 |"),
            content_hash="h2",
        )

    _assert_old_version_intact(env.client, old_ids)
    _assert_no_pending(env)


def test_vector_insert_failure_keeps_old_version_and_cleans_staged_rows(monkeypatch):
    """向量 insert 失败：暂存行被清理，旧版本可检索，且不残留发布登记。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")
    old_ids = _visible_ids(env.client)
    entities_before = len(env.client.entities)

    env.client.fail_insert = 1
    with pytest.raises(Exception, match="向量库写入失败"):
        env.kb.add_parsed_document(_document(SOURCE, ["新版本正文"]), content_hash="h2")

    _assert_old_version_intact(env.client, old_ids)
    assert len(env.client.entities) == entities_before, "暂存残留必须清理"
    _assert_no_pending(env)
    assert env.store.is_unchanged(SOURCE, "h1") is True


def test_activation_failure_rolls_back_and_keeps_old_version(monkeypatch):
    """可见性切换（激活）失败：回滚本次新增行，旧版本可检索，可安全重试。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")
    old_ids = _visible_ids(env.client)

    env.client.fail_upsert = 1
    with pytest.raises(Exception, match="激活|发布"):
        env.kb.add_parsed_document(_document(SOURCE, ["新版本正文"]), content_hash="h2")

    _assert_old_version_intact(env.client, old_ids)
    _assert_no_pending(env)

    # 重试同一新版本：不得产生重复块
    env.client.fail_upsert = 0
    env.kb.add_parsed_document(_document(SOURCE, ["新版本正文"]), content_hash="h2")
    visible = _visible(env.client)
    assert len(visible) == 1, visible
    assert visible[0]["content"] == "新版本正文"
    assert env.store.is_unchanged(SOURCE, "h2") is True


def test_fingerprint_commit_failure_blocks_skip_and_recovers_forward(monkeypatch):
    """指纹提交失败：登记保留 → 重传旧文件不会被误跳过；恢复时向前滚动。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")

    real_commit = env.store.commit_publish

    def flaky_commit(*args, **kwargs):
        raise RuntimeError("fingerprint disk full")

    monkeypatch.setattr(env.store, "commit_publish", flaky_commit)
    with pytest.raises(RuntimeError, match="fingerprint disk full"):
        env.kb.add_parsed_document(_document(SOURCE, ["新版本正文"]), content_hash="h2")

    # 新版本已激活，但指纹尚未提交：发布登记必须存在
    assert env.store.pending_sources() == [SOURCE]
    # 因此「重传旧文件」不会被错误地判为未变更而跳过
    assert env.store.is_unchanged(SOURCE, "h1") is False

    monkeypatch.setattr(env.store, "commit_publish", real_commit)
    outcomes = env.kb.recover_pending_publishes(SOURCE)

    assert outcomes == [{"source": SOURCE, "action": "rolled_forward", "chunks": 1}]
    assert env.store.pending_sources() == []
    assert env.store.is_unchanged(SOURCE, "h2") is True
    assert [row["content"] for row in _visible(env.client)] == ["新版本正文"]


def test_crash_after_staging_recovers_by_rollback(monkeypatch):
    """进程在暂存后退出（尚未激活）：恢复删除暂存行，旧版本保持可检索。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")
    old_ids = _visible_ids(env.client)

    # 手工重放「暂存 + 登记」两步（模拟进程在此之后被杀）。
    # 新主键必须和旧版本不同：真实 Milvus 不允许同一主键存两行。
    new_doc = _document(SOURCE, ["新版本正文"])
    new_id = new_doc.chunks.chunks[0].metadata["id"]
    assert new_id not in old_ids
    staged = [{"id": new_id, "source": SOURCE, "is_current": False,
               "content": "新版本正文", "dense_vector": [0.1]}]
    env.kb._insert_rows(env.client, staged)
    env.store.begin_publish(
        SOURCE, sha256="h2", chunk_ids=[new_id], previous_ids=sorted(old_ids),
        chunk_count=1, record=env.store.get(SOURCE), staged_ids=[new_id],
    )
    assert env.store.pending_sources() == [SOURCE]
    assert len(env.client.entities) == len(old_ids) + 1

    outcomes = env.kb.recover_pending_publishes(SOURCE)

    assert outcomes == [{"source": SOURCE, "action": "rolled_back", "chunks": 1}]
    _assert_old_version_intact(env.client, old_ids)
    assert env.store.pending_sources() == []
    assert env.store.get(SOURCE)["sha256"] == "h1"
    del new_doc


def test_same_primary_key_republish_upserts_without_second_insert(monkeypatch):
    """主键未变时不再 insert（真实 Milvus 会主键冲突），只 upsert 覆盖原行。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["同一段正文"]), content_hash="h1")
    inserted_after_first = len(env.client.inserted)
    env.client.fail_insert = 1

    env.kb.add_parsed_document(_document(SOURCE, ["同一段正文"]), content_hash="h2")

    visible = _visible(env.client)
    assert len(env.client.inserted) == inserted_after_first
    assert len(visible) == 1
    assert len(env.client.entities) == 1
    assert visible[0]["content"] == "同一段正文"
    assert env.store.is_unchanged(SOURCE, "h2") is True


def test_successful_publish_switches_all_stores_to_one_version(monkeypatch):
    """成功发布后：向量、父块、整表、指纹都对应同一个可见版本。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(
        _document(SOURCE, ["旧版本正文"], table_markdown="| 年份 |\n| --- |\n| 2025 |"),
        content_hash="h1",
    )
    result = env.kb.add_parsed_document(
        _document(SOURCE, ["新版本正文"], table_markdown="| 年份 |\n| --- |\n| 2026 |"),
        content_hash="h2",
    )

    visible = _visible(env.client)
    assert [row["content"] for row in visible] == ["新版本正文"]
    assert len(env.client.entities) == len(visible), "旧版本行应被回收，不留不可见残留"
    # 父块 / 整表回收使用各自主键作为 keep 集合（每次发布后都执行一次）
    assert env.parents.gc_keep[-1] == {f"{SOURCE}::p0"}
    assert env.tables.gc_keep[-1] == {f"{SOURCE}::t0"}
    record = env.store.get(SOURCE)
    assert record["sha256"] == "h2"
    assert record["content_sha256"], "指纹必须记录完整正文哈希（判重依据）"
    assert env.store.pending_sources() == []
    assert result["duplicate_review"] == []


def test_retry_after_failed_publish_does_not_duplicate_chunks(monkeypatch):
    """失败后重试同一新版本：可见实体数量与切块数量一致，无重复块。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document(SOURCE, ["旧版本正文"]), content_hash="h1")

    env.client.fail_insert = 1
    with pytest.raises(VectorStoreError, match="向量库写入失败"):
        env.kb.add_parsed_document(_document(SOURCE, ["新版本一", "新版本二"]), content_hash="h2")

    env.client.fail_insert = 0
    env.kb.add_parsed_document(_document(SOURCE, ["新版本一", "新版本二"]), content_hash="h2")

    visible = _visible(env.client)
    assert sorted(row["content"] for row in visible) == ["新版本一", "新版本二"]
    assert len(env.client.entities) == 2


# ---------------------------------------------------------------------------
# RAG-05：判重使用完整正文与表体
# ---------------------------------------------------------------------------


_TABLE_INDEX_TEXT = "| 年份 | 营业收入 |\n| --- | --- |\n| 2026E | 100 |"


def test_same_table_shape_different_numbers_both_ingest(monkeypatch):
    """相同表头、相同行标签但金额不同的两份报告都必须能入库。

    修复前：判重输入是子块的表格索引文本（省略表体数值），两份报告的
    SimHash 完全相同，第二份会被当成重复文档拒绝。
    """
    env = _kb(monkeypatch)
    first = _document(
        "报告A.pdf", [_TABLE_INDEX_TEXT],
        table_markdown="| 年份 | 营业收入 |\n| --- | --- |\n| 2026E | 100 |",
        dedup_body="营业收入 100 亿元",
    )
    second = _document(
        "报告B.pdf", [_TABLE_INDEX_TEXT],
        table_markdown="| 年份 | 营业收入 |\n| --- | --- |\n| 2026E | 999 |",
        dedup_body="营业收入 999 亿元",
    )
    # 索引文本（检索用）逐字相同——正是旧实现的判重输入
    assert first.chunks.chunks[0].page_content == second.chunks.chunks[0].page_content

    env.kb.add_parsed_document(first, content_hash="h-a")
    result = env.kb.add_parsed_document(second, content_hash="h-b")

    assert result["chunk_count"] == 1
    assert {"报告A.pdf", "报告B.pdf"} <= {row["source"] for row in _visible(env.client)}


def test_exact_duplicate_content_is_rejected_with_reason(monkeypatch):
    """完全相同内容仍按明确的重复策略拒绝，并说明匹配对象。"""
    env = _kb(monkeypatch)
    env.kb.add_parsed_document(_document("原件.pdf", ["同一份正文"]), content_hash="h1")

    with pytest.raises(ValueError, match="完全相同的文档已入库"):
        env.kb.add_parsed_document(_document("副本.pdf", ["同一份正文"]), content_hash="h2")


def test_near_duplicate_is_reviewed_not_blocked(monkeypatch):
    """近似文档（SimHash 接近但内容不同）只记复核信号，不拒绝入库。"""
    env = _kb(monkeypatch)
    base = "\n".join(f"第{i}段：2026 年营业收入与净利润分析。" for i in range(60))
    almost = "\n".join(
        f"第{i}段：2026 年营业收入与净利润分析。" + ("补充口径说明。" if i in {5, 40} else "")
        for i in range(60)
    )
    from src.ingest.cleaner import simhash

    distance = bin(simhash(base) ^ simhash(almost)).count("1")
    assert distance <= 3, f"用例构造的近似样本距离 {distance}，需落在复核阈值内"

    env.kb.add_parsed_document(_document("近似A.pdf", [base]), content_hash="h1")
    result = env.kb.add_parsed_document(_document("近似B.pdf", [almost]), content_hash="h2")

    assert result["duplicate_review"] == [
        {"source": "近似A.pdf", "hamming_distance": distance}
    ]
    assert "近似B.pdf" in {row["source"] for row in _visible(env.client)}
