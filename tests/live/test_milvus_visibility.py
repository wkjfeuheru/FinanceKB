"""真实 Milvus 上的暂存可见性、检索过滤与覆盖回收。

只用一次性集合。嵌入是固定向量，不加载 ONNX。指纹与派生存储不写生产库。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.core.config import EMBEDDING_DIM
from src.index.fingerprint_store import FingerprintStore
from src.index.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
)
from src.chunking.chunker import DoclingChunks

pytestmark = pytest.mark.live

OLD = "旧版本ALPHA正文"
NEW = "新版本BETA正文"
SOURCE = "live-report.pdf"


class _SilentStore:
    def store_batch(self, rows):
        return None

    def delete_by_source(self, source):
        return 0

    def delete_by_source_except(self, source, keep_ids):
        return 0


def _parsed(text: str, chunk_id: str) -> ParsedDocument:
    return ParsedDocument(
        path=Path(SOURCE),
        source=SOURCE,
        title=SOURCE,
        chunks=DoclingChunks(
            chunks=[
                Document(
                    page_content=text,
                    metadata={
                        "id": chunk_id,
                        "chunk_key": chunk_id,
                        "content_hash": chunk_id,
                        "source": SOURCE,
                        "title": SOURCE,
                        "chunk": 0,
                        "parent_id": f"p-{chunk_id}",
                        "heading_path": "",
                    },
                )
            ],
            markdown=text,
            parents=[],
        ),
        metadata={"category": "", "report_type": "宏观"},
    )


def _contents(payload) -> set[str]:
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            content = node.get("content")
            if isinstance(content, str):
                found.add(content)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _query(client, collection: str, expr: str) -> list[dict]:
    return client.query(
        collection_name=collection,
        filter=expr,
        output_fields=["id", "content", "is_current"],
        limit=20,
    )


def test_staging_stays_invisible_until_activation_and_old_ids_are_removed(tmp_path, monkeypatch):
    """暂存行不出现在 is_current == true 的检索里；激活后只剩新版本。"""
    collection = f"livekb_{uuid.uuid4().hex[:8]}"
    kb = KnowledgeBase(collection_name=collection)
    fingerprint = FingerprintStore(str(tmp_path / "fingerprints.json"))
    silent = _SilentStore()
    monkeypatch.setattr(kb, "_get_fingerprint_store", lambda: fingerprint)
    monkeypatch.setattr(kb, "_get_parent_store", lambda: silent)
    monkeypatch.setattr(kb, "_get_table_store", lambda: silent)
    vector = [0.02] * EMBEDDING_DIM
    old_id = uuid.uuid4().hex
    new_id = uuid.uuid4().hex
    observed: dict = {}
    original_upsert = kb._upsert_rows

    def spy_upsert(client, rows):
        if rows and rows[0].get("content") == NEW:
            expr = f'source == "{SOURCE}"'
            observed["visible"] = _query(client, collection, f"{expr} and is_current == true")
            observed["staged"] = _query(client, collection, f"{expr} and is_current == false")
            observed["hits"] = client.search(
                collection_name=collection,
                data=[vector],
                anns_field="dense_vector",
                search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
                limit=5,
                filter=f'is_current == true and source == "{SOURCE}"',
                output_fields=["id", "content", "is_current"],
            )
        return original_upsert(client, rows)

    monkeypatch.setattr(kb, "_upsert_rows", spy_upsert)
    client = None
    try:
        kb.add_parsed_document(_parsed(OLD, old_id), content_hash="h-old", dense_vectors=[vector])
        kb.add_parsed_document(_parsed(NEW, new_id), content_hash="h-new", dense_vectors=[vector])

        visible_while_staged = {row["content"] for row in observed["visible"]}
        staged_while_staged = {row["content"] for row in observed["staged"]}
        assert visible_while_staged == {OLD}
        assert staged_while_staged == {NEW}
        assert NEW not in _contents(observed["hits"])
        assert OLD in _contents(observed["hits"])

        client = kb._get_client()
        current = _query(client, collection, f'source == "{SOURCE}" and is_current == true')
        everything = _query(client, collection, f'source == "{SOURCE}"')
        assert {row["content"] for row in current} == {NEW}
        assert {row["id"] for row in everything} == {new_id}
        assert all(row["is_current"] is True for row in everything)
        after = client.search(
            collection_name=collection,
            data=[vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 8}},
            limit=5,
            filter=f'is_current == true and source == "{SOURCE}"',
            output_fields=["content"],
        )
        assert _contents(after) == {NEW}
    finally:
        try:
            client = client or kb._get_client()
            if client.has_collection(collection):
                client.drop_collection(collection)
        except Exception as exc:  # noqa: BLE001 - 清理失败不能盖住用例本身的错误
            print(f"清理集合 {collection} 失败：{exc}", file=sys.stderr)
