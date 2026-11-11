"""真实 Milvus 上的 schema 漂移回归：集合多出未声明标量字段时仍可入库。

线上事故：集合由更新版本代码创建，多出 ``security_name`` 等字段；旧写入
路径只提供六个研报字段，pymilvus 直接拒绝整批插入::

    DataNotMatchException: Insert missed an field `security_name` ...

本用例用一次性集合模拟「集合有当前代码不认识的标量字段」，锁住
「按真实 schema 补默认值」这一修复。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pymilvus import DataType, Function, FunctionType, MilvusClient

from src.chunking.chunker import DoclingChunks
from src.core.config import EMBEDDING_DIM
from src.index.fingerprint_store import FingerprintStore
from src.index.milvus_kb import KnowledgeBase, ParsedDocument, _add_report_fields

pytestmark = pytest.mark.live

SOURCE = "schema-drift-live.pdf"


class _SilentStore:
    def store_batch(self, rows):
        return None

    def delete_by_source(self, source):
        return 0

    def delete_by_source_except(self, source, keep_ids):
        return 0


def _parsed(chunk_id: str) -> ParsedDocument:
    return ParsedDocument(
        path=Path(SOURCE),
        source=SOURCE,
        title=SOURCE,
        chunks=DoclingChunks(
            chunks=[
                Document(
                    page_content="国产替代加速，EDA 产业生态重构。",
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
            markdown="国产替代加速，EDA 产业生态重构。",
            parents=[],
        ),
        metadata={"category": "investment_research", "report_type": "行业", "broker": "测试证券"},
    )


def _create_drifted_collection(client: MilvusClient, collection: str) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("content", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("title", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk", DataType.INT64)
    schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
    schema.add_field("chunk_key", DataType.VARCHAR, max_length=64)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("heading_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("date", DataType.VARCHAR, max_length=32)
    schema.add_field("version", DataType.VARCHAR, max_length=32)
    schema.add_field("ingested_at", DataType.INT64)
    schema.add_field("is_current", DataType.BOOL)
    schema.add_field("is_deleted", DataType.BOOL)
    _add_report_fields(schema)
    # 模拟旧集合仍保留的退役列，以及更新版本加上、当前写入路径不认识的字段。
    schema.add_field("security_name", DataType.VARCHAR, max_length=64)
    schema.add_field("industry_l1", DataType.VARCHAR, max_length=32)
    schema.add_field("industry_l2", DataType.VARCHAR, max_length=32)
    schema.add_field("meta_source", DataType.VARCHAR, max_length=16)
    schema.add_field("needs_review", DataType.BOOL)
    schema.add_field("future_metric", DataType.INT64)
    schema.add_field("future_flag", DataType.BOOL)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25_fn",
        function_type=FunctionType.BM25,
        input_field_names=["content"],
        output_field_names=["sparse_vector"],
    ))
    params = client.prepare_index_params()
    params.add_index(
        field_name="dense_vector",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=params,
        consistency_level="Strong",
    )
    client.load_collection(collection)


def test_ingest_into_collection_with_unknown_scalar_fields(tmp_path):
    collection = f"driftkb_{uuid.uuid4().hex[:8]}"
    client = MilvusClient(uri="http://localhost:19530")
    kb = KnowledgeBase(collection_name=collection)
    fingerprint = FingerprintStore(str(tmp_path / "fingerprints.json"))
    silent = _SilentStore()
    kb._get_fingerprint_store = lambda: fingerprint  # type: ignore[method-assign]
    kb._get_parent_store = lambda: silent  # type: ignore[method-assign]
    kb._get_table_store = lambda: silent  # type: ignore[method-assign]
    chunk_id = uuid.uuid4().hex
    try:
        _create_drifted_collection(client, collection)
        result = kb.add_parsed_document(
            _parsed(chunk_id),
            content_hash="schema-drift-content",
            dense_vectors=[[0.01] * EMBEDDING_DIM],
        )
        assert result["chunk_count"] == 1

        rows = client.query(
            collection_name=collection,
            filter=f'id == "{chunk_id}"',
            output_fields=[
                "security_name", "industry_l1", "meta_source",
                "needs_review", "future_metric", "future_flag", "is_current",
            ],
            limit=4,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["security_name"] == ""
        assert row["industry_l1"] == ""
        assert row["meta_source"] == ""
        assert row["needs_review"] is False
        assert row["future_metric"] == 0
        assert row["future_flag"] is False
        assert row["is_current"] is True
    finally:
        try:
            if client.has_collection(collection):
                client.drop_collection(collection)
        except Exception as exc:  # noqa: BLE001 - 清理失败不能盖住用例错误
            print(f"清理集合 {collection} 失败：{exc}", file=sys.stderr)
