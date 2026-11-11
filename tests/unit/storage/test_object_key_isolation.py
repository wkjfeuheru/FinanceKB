"""对象键隔离与版本删除回归（RAG-06 / RAG-07）。

``docs/rag-review.md`` 的验收主线：

* **RAG-06**：同集合的 ``a.pdf`` 与 ``a.md`` 两份原文件字节均不变；不同集合上传
  同名不同内容文件时预览/图片/下载各自正确；历史键有明确的兼容读取方案，
  路径组成部分经过合法化。
* **RAG-07**：删除历史版本后当前版本仍可检索、下载与跳转页码；索引删除失败时
  不提前删除仍被使用的原文件；重复删除幂等，回收失败可重试且不影响其他文档。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.storage.objects import object_keys
from src.ingest.pdf_assets import ImageAsset, image_object_key
from src.ingest import documents as ds
from src.ingest.pipeline import prepare_chunked_document


class _ByteStorage:
    """内存对象存储替身：支持 upload / download / exists / delete_sync / list_keys_sync。"""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail_delete = False

    async def upload(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def download(self, key: str) -> bytes:
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def delete_sync(self, key: str) -> None:
        if self.fail_delete:
            raise RuntimeError("storage delete failed")
        self.deleted.append(key)
        self.objects.pop(key, None)

    async def list_keys(self, prefix: str) -> list[str]:
        return self.list_keys_sync(prefix)

    def list_keys_sync(self, prefix: str) -> list[str]:
        return sorted(key for key in self.objects if key.startswith(prefix))


class _FakeKB:
    """KB 替身：记录删除调用并可控地失败，用于验证「先删索引再回收资产」。"""

    collection_name = "kb_alpha"

    def __init__(self) -> None:
        self.removed: list[tuple[str, str | None]] = []
        self.fail_remove = False
        self.fingerprint: dict[str, dict[str, Any]] = {}

    def get_fingerprint_store(self):
        class _Store:
            def __init__(self, records):
                self._records = records

            def get(self, source):
                return self._records.get(source)

        return _Store(self.fingerprint)

    def remove_document(self, source: str, version: str | None = None) -> dict[str, Any]:
        if self.fail_remove:
            raise RuntimeError("向量删除失败")
        self.removed.append((source, version))
        return {"source": source, "version": version or "", "deleted_count": 3}


def _manager(storage: _ByteStorage, kb: _FakeKB) -> ds.DocumentManager:
    manager = ds.DocumentManager.__new__(ds.DocumentManager)
    manager._kb = kb
    manager._collection_name = "kb_alpha"
    manager._storage = storage
    manager._parse_executor = None
    manager._bg_tasks = set()
    manager._pipeline_enabled = False
    return manager


def _parsed(source: str, *, markdown: str | None = "# 正文\n\n内容", version: str = "",
            images: list[ImageAsset] | None = None, body: bytes | None = None):
    from src.index.milvus_kb import ParsedDocument
    from src.chunking.chunker import DoclingChunks

    path = Path(f"tests/unit/.tmp/object-keys/{source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body if body is not None else b"%PDF-1.4 fake " + source.encode("utf-8"))
    metadata = {"category": "", "report_type": "宏观"}
    if version:
        metadata["version"] = version
    return ParsedDocument(
        path=path,
        source=source,
        title=source,
        chunks=DoclingChunks(markdown=markdown, images=list(images or [])),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# RAG-06：键隔离
# ---------------------------------------------------------------------------


async def test_same_stem_different_extension_originals_do_not_collide():
    """同集合上传 ``a.pdf`` 与 ``a.md``：两份原文件字节均不变。"""
    storage = _ByteStorage()
    manager = _manager(storage, _FakeKB())

    pdf = _parsed("a.pdf", markdown="PDF 解析正文")
    md = _parsed("a.md", markdown=None)

    await manager._prepare_parsed_document(pdf, "hash-pdf")
    await manager._prepare_parsed_document(md, "hash-md")

    pdf_original = object_keys.original_key("kb_alpha", "a.pdf", filename="a.pdf")
    md_original = object_keys.original_key("kb_alpha", "a.md", filename="a.md")
    pdf_parsed = object_keys.parsed_key("kb_alpha", "a.pdf")

    # 三份对象互不覆盖：a.pdf 原文、a.md 原文、a.pdf 的解析产物
    assert storage.objects[pdf_original] == pdf.path.read_bytes()
    assert storage.objects[md_original] == md.path.read_bytes()
    assert storage.objects[pdf_parsed] == "PDF 解析正文".encode("utf-8")
    assert len({pdf_original, md_original, pdf_parsed}) == 3


async def test_same_filename_in_different_collections_is_isolated():
    """不同集合上传同名、不同内容文件：各自的原文/图片互不覆盖。"""
    storage = _ByteStorage()
    asset = ImageAsset(key_hint="1-0", page=1, data=b"\x89PNG-first", ext="png")
    other_asset = ImageAsset(key_hint="1-0", page=1, data=b"\x89PNG-second", ext="png")

    manager_a = _manager(storage, _FakeKB())
    await manager_a._prepare_parsed_document(
        _parsed("同名.pdf", markdown="A 集合正文", images=[asset], body=b"%PDF alpha"),
        "hash-a",
    )

    manager_b = _manager(storage, _FakeKB())
    manager_b._collection_name = "kb_beta"
    await manager_b._prepare_parsed_document(
        _parsed("同名.pdf", markdown="B 集合正文", images=[other_asset], body=b"%PDF beta"),
        "hash-b",
    )

    key_a = object_keys.original_key("kb_alpha", "同名.pdf", filename="同名.pdf")
    key_b = object_keys.original_key("kb_beta", "同名.pdf", filename="同名.pdf")
    assert key_a != key_b
    assert storage.objects[key_a] == b"%PDF alpha"
    assert storage.objects[key_b] == b"%PDF beta"
    # 图片同样按集合隔离
    assert storage.objects[image_object_key("同名.pdf", asset, collection="kb_alpha")] == asset.data
    assert storage.objects[image_object_key("同名.pdf", other_asset, collection="kb_beta")] == other_asset.data


def test_image_key_matches_between_chunk_metadata_and_upload():
    """图片键必须与切块阶段写入向量库的 ``image_key`` 完全一致。

    若两处算法不一致，前端「查看原图」会取不到对象——这条链路横跨切块与写库，
    必须由同一套键计算保证。
    """
    asset = ImageAsset(key_hint="4-0", page=4, data=b"\x89PNG-image", ext="png")
    upload_key = image_object_key("研报.pdf", asset, collection="kb_alpha", version="")

    chunk_key = image_object_key("研报.pdf", asset, collection="kb_alpha", version="")

    assert upload_key == chunk_key
    assert upload_key == object_keys.image_key("kb_alpha", "研报.pdf", asset, version="")
    # 与集合、内容绑定：换集合或换图片内容都会得到不同的键
    assert image_object_key("研报.pdf", asset, collection="kb_beta") != upload_key
    other = ImageAsset(key_hint="4-0", page=4, data=b"\x89PNG-other", ext="png")
    assert image_object_key("研报.pdf", other, collection="kb_alpha") != upload_key


async def test_prepare_chunked_document_threads_collection_into_image_key(monkeypatch):
    """准备流程必须把集合名透传到图片键，否则图片键与上传键不一致。"""
    from src.ingest import image_captioner
    from src.index.milvus_kb import ParsedDocument
    from src.chunking.chunker import DoclingChunks

    asset = ImageAsset(key_hint="2-0", page=2, data=b"\x89PNG-caption", ext="png")

    parsed = ParsedDocument(
        path=Path("tests/unit/.tmp/object-keys/宏观研报.pdf"),
        source="宏观研报.pdf",
        title="宏观研报",
        chunks=DoclingChunks(markdown="# 宏观展望\n\n正文", images=[asset], blocks=[]),
        metadata={"category": "", "report_type": "宏观"},
    )
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "结构示意图")

    prepared = await prepare_chunked_document(parsed, collection="kb_alpha")

    image_chunk = [c for c in prepared.chunks.chunks
                   if c.metadata.get("block_type") == "image"][0]
    assert image_chunk.metadata["image_key"] == image_object_key(
        "宏观研报.pdf", asset, collection="kb_alpha", version=""
    )


def test_key_components_are_sanitized():
    """路径组成部分必须合法化：不允许目录逃逸/控制字符，且长度受限。"""
    escaped = object_keys.sanitize_component("../../etc/passwd")
    assert "/" not in escaped and ".." not in escaped

    long_name = "超长文件名" * 60
    assert len(object_keys.sanitize_component(long_name)) <= 96

    key = object_keys.original_key(
        "kb/../alpha", "../恶意.pdf", filename="../../x.pdf", version="v/1"
    )
    assert ".." not in key
    assert key.count("/") == 5  # tenant/collection/doc/version/partition/file


def test_legacy_keys_are_candidates_for_reads():
    """历史键必须有明确读取兼容：候选列表包含旧协议键且保序。"""
    candidates = object_keys.original_key_candidates(
        "kb_alpha", "a.pdf", version="2026中报"
    )
    assert candidates[0] == object_keys.original_key(
        "kb_alpha", "a.pdf", version="2026中报", filename="a.pdf"
    )
    assert "docs/a.pdf" in candidates
    # 解析产物候选同样覆盖旧键 docs/a.md
    assert "docs/a.md" in object_keys.parsed_key_candidates("kb_alpha", "a.pdf")


# ---------------------------------------------------------------------------
# RAG-07：删除版本
# ---------------------------------------------------------------------------


async def test_deleting_history_version_keeps_current_assets():
    """删除历史版本：只回收该版本命名空间的资产，当前版本仍可预览。"""
    storage = _ByteStorage()
    kb = _FakeKB()
    manager = _manager(storage, kb)

    current = _parsed("研报.pdf", markdown="当前版本", version="2026中报")
    history = _parsed("研报.pdf", markdown="历史版本", version="2025年报")
    await manager._prepare_parsed_document(current, "h-current")
    await manager._prepare_parsed_document(history, "h-history")

    current_key = object_keys.original_key(
        "kb_alpha", "研报.pdf", version="2026中报", filename="研报.pdf"
    )
    history_key = object_keys.original_key(
        "kb_alpha", "研报.pdf", version="2025年报", filename="研报.pdf"
    )
    assert current_key in storage.objects and history_key in storage.objects

    # 指纹记录当前可见版本为 2026中报，删除历史版本 2025年报
    kb.fingerprint["研报.pdf"] = {"version": "2026中报", "sha256": "h-current"}
    manager.delete_document("研报.pdf", version="2025年报")

    assert history_key not in storage.objects, "被删版本的资产应回收"
    assert current_key in storage.objects, "当前版本资产必须保留"
    assert storage.objects[current_key] == current.path.read_bytes()
    # 当前版本仍可预览：候选键第一个即可命中
    assert await storage.download(manager.original_key_candidates("研报.pdf")[0])


def test_index_delete_failure_does_not_touch_assets():
    """索引删除失败时不得提前删除原文件（顺序：先删索引再回收资产）。"""
    storage = _ByteStorage()
    kb = _FakeKB()
    manager = _manager(storage, kb)

    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        manager._prepare_parsed_document(_parsed("研报.pdf", markdown="正文"), "h1")
    )
    key = object_keys.original_key("kb_alpha", "研报.pdf", filename="研报.pdf")
    assert key in storage.objects

    kb.fail_remove = True
    with pytest.raises(RuntimeError, match="向量删除失败"):
        manager.delete_document("研报.pdf")

    assert key in storage.objects, "索引删除失败时原文件必须保留"
    assert storage.deleted == []


def test_repeated_delete_is_idempotent_and_retryable():
    """重复删除幂等；资产回收失败可重试，不影响其它文档与集合。"""
    storage = _ByteStorage()
    kb = _FakeKB()
    manager = _manager(storage, kb)

    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.run_until_complete(manager._prepare_parsed_document(_parsed("研报.pdf"), "h1"))
    loop.run_until_complete(
        manager._prepare_parsed_document(_parsed("其它.pdf", markdown="别动我"), "h2")
    )
    other_key = object_keys.original_key("kb_alpha", "其它.pdf", filename="其它.pdf")

    storage.fail_delete = True
    first = manager.delete_document("研报.pdf")
    assert first["deleted_count"] == 3
    assert other_key in storage.objects, "回收失败不得影响其它文档"

    storage.fail_delete = False
    manager.delete_document("研报.pdf")
    second = manager.delete_document("研报.pdf")
    assert second["source"] == "研报.pdf"
    assert other_key in storage.objects
    assert object_keys.original_key("kb_alpha", "研报.pdf", filename="研报.pdf") not in storage.objects


def test_full_delete_purges_all_versions_and_legacy_keys():
    """删除整篇文档：新命名空间按前缀回收，旧协议键一并清理。"""
    storage = _ByteStorage()
    kb = _FakeKB()
    manager = _manager(storage, kb)

    prefix = object_keys.document_root_prefix("kb_alpha", "研报.pdf")
    storage.objects[f"{prefix}/v-2026中报/original/研报.pdf"] = b"v1"
    storage.objects[f"{prefix}/v-2025年报/original/研报.pdf"] = b"v2"
    storage.objects[f"{prefix}/images/abc.png"] = b"img"
    storage.objects["docs/研报.pdf"] = b"legacy"
    storage.objects["docs/研报.md"] = b"legacy-parsed"

    manager.delete_document("研报.pdf")

    assert not [key for key in storage.objects if key.startswith(prefix)]
    assert "docs/研报.pdf" not in storage.objects
    assert "docs/研报.md" not in storage.objects
