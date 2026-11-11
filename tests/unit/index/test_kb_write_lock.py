"""集合级写锁单元测试。

回归点：向量库写入的「同 collection 串行」此前由**调用方**各自持锁保证——
入库流水线用 ``asyncio`` 写锁、``document_service`` 用模块级 ``threading`` 锁，
两把锁互不排斥。于是对象存储事件回调（``process_object_event`` →
``_store_parsed_document``）与流水线写入并发时会**同时进入**
``add_parsed_document``，破坏：

* 文档间 SimHash 去重（"先查再插"，并发会让近似重复文档同时通过检查）；
* 同 source 覆盖的原子性（并发"删旧行 + 插新行"会互相删除）。

现改为 ``KnowledgeBase`` 在**资源侧**自持集合级写锁，所有调用方自动受保护。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from src.index.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
    _collection_write_lock,
)
from src.chunking.chunker import DoclingChunks


def _parsed(source: str) -> ParsedDocument:
    """构造最小 ParsedDocument（本文件不触发真实嵌入/写入）。"""
    return ParsedDocument(
        path=Path(source),
        source=source,
        title=Path(source).stem,
        chunks=DoclingChunks(markdown="## 标题\n内容"),
        metadata={"category": ""},
    )


def _run_concurrently(targets: list) -> list[BaseException]:
    """并发执行若干无参可调用对象，返回线程内抛出的异常（便于断言失败原因）。"""
    errors: list[BaseException] = []

    def _wrap(fn):
        def _inner() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - 测试需要收集线程内异常
                errors.append(exc)

        return _inner

    threads = [threading.Thread(target=_wrap(fn)) for fn in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    return errors


# ---------------------------------------------------------------------------
# 锁的粒度与共享
# ---------------------------------------------------------------------------


def test_write_lock_is_shared_per_collection():
    """同一 collection 名共享一把锁；不同 collection 各自独立。"""
    assert _collection_write_lock("lock_c1") is _collection_write_lock("lock_c1")
    assert _collection_write_lock("lock_c1") is not _collection_write_lock("lock_c2")


def test_knowledge_base_instances_share_lock_by_collection():
    """同名 collection 的不同 KnowledgeBase 实例指向同一把写锁。"""
    kb_a = KnowledgeBase(collection_name="lock_shared_kb")
    kb_b = KnowledgeBase(collection_name="lock_shared_kb")
    kb_c = KnowledgeBase(collection_name="lock_other_kb")
    assert kb_a._write_lock is kb_b._write_lock
    assert kb_a._write_lock is not kb_c._write_lock


# ---------------------------------------------------------------------------
# 同 collection 串行
# ---------------------------------------------------------------------------


def test_add_parsed_document_serializes_same_collection(monkeypatch):
    """同 collection 的并发入库必须串行（任一时刻只有一个进入临界区）。"""
    kb = KnowledgeBase(collection_name="lock_serial_kb")
    state = {"running": 0, "max_running": 0}

    def fake_locked(parsed, path=None, content_hash=None, **kwargs):
        state["running"] += 1
        state["max_running"] = max(state["max_running"], state["running"])
        try:
            time.sleep(0.05)
            return {
                "source": parsed.source,
                "title": "",
                "chunk_count": 0,
                "parent_count": 0,
            }
        finally:
            state["running"] -= 1

    monkeypatch.setattr(kb, "_add_parsed_document_locked", fake_locked)

    errors = _run_concurrently(
        [
            (lambda index=index: kb.add_parsed_document(_parsed(f"d{index}.md")))
            for index in range(3)
        ]
    )

    assert not errors, errors
    assert state["max_running"] == 1, "同一 collection 的写入必须串行"


def test_add_parsed_document_does_not_serialize_across_collections(monkeypatch):
    """不同 collection 之间不应互相阻塞（锁粒度是集合级而非全局）。"""
    kb_x = KnowledgeBase(collection_name="lock_gran_x")
    kb_y = KnowledgeBase(collection_name="lock_gran_y")
    # 两个线程必须**同时**进入临界区才能通过 barrier；若被串行化会超时并报错
    barrier = threading.Barrier(2, timeout=5)

    def fake_locked(parsed, path=None, content_hash=None, **kwargs):
        barrier.wait()
        return {"source": parsed.source, "title": "", "chunk_count": 0, "parent_count": 0}

    monkeypatch.setattr(kb_x, "_add_parsed_document_locked", fake_locked)
    monkeypatch.setattr(kb_y, "_add_parsed_document_locked", fake_locked)

    errors = _run_concurrently(
        [
            lambda: kb_x.add_parsed_document(_parsed("x.md")),
            lambda: kb_y.add_parsed_document(_parsed("y.md")),
        ]
    )

    assert not errors, errors


def test_delete_path_shares_lock_with_ingest_path(monkeypatch):
    """删除路径与入库路径共用同一把集合锁 —— 这正是本次修复的漏洞。

    修复前：流水线的 ``asyncio`` 写锁与 ``_upload_lock``（``threading``）
    互不排斥，删除/事件回调可绕过流水线的串行点。
    """
    kb = KnowledgeBase(collection_name="lock_unify_kb")
    entered = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def slow_add(parsed, path=None, content_hash=None, **kwargs):
        order.append("add-enter")
        entered.set()
        release.wait(timeout=5)  # 持锁不放，模拟耗时的向量库写入
        order.append("add-exit")
        return {"source": parsed.source, "title": "", "chunk_count": 0, "parent_count": 0}

    def fake_delete(*args, **kwargs):
        order.append("remove-enter")
        return 0

    class _FakeClient:
        def flush(self, collection_name: str) -> None:
            return None

    monkeypatch.setattr(kb, "_add_parsed_document_locked", slow_add)
    monkeypatch.setattr(kb, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(kb, "ensure_collection", lambda: None)
    monkeypatch.setattr(kb, "_delete_by_source", fake_delete)
    monkeypatch.setattr(kb, "_count_by_source", lambda *args, **kwargs: 1)

    adder = threading.Thread(target=kb.add_parsed_document, args=(_parsed("u.md"),))
    adder.start()
    assert entered.wait(timeout=5), "入库未进入临界区"

    remover = threading.Thread(target=kb.remove_document, args=("u.md",))
    remover.start()
    time.sleep(0.2)
    # 入库仍持锁期间，删除不得进入临界区
    assert "remove-enter" not in order, "删除路径未与入库路径共用同一把锁"

    release.set()
    adder.join(timeout=5)
    remover.join(timeout=5)

    assert order == ["add-enter", "add-exit", "remove-enter"], order
