"""RAG 父块持久化存储（PostgreSQL）。"""

from __future__ import annotations

from typing import Any

from src.core.config import KB_COLLECTION_NAME
from src.storage.relational_db.parent_store import ParentStoreRepository


class ParentStore:
    """父块存储：``parent_id -> 父块全文``，持久化到 PostgreSQL，按集合隔离。

    对外接口保持稳定（``store / store_batch / get / get_batch / delete_by_source``），
    供 :class:`~src.index.milvus_kb.KnowledgeBase`
    与 :class:`~src.retrieve.hybrid_retriever.HybridRetriever` 使用。
    """

    def __init__(self, collection: str = KB_COLLECTION_NAME) -> None:
        self._collection = collection
        self._repo = ParentStoreRepository()

    def store(
        self,
        parent_id: str,
        content: str,
        heading: str = "",
        source: str = "",
        heading_path: str = "",
    ) -> None:
        self._repo.upsert_many(
            self._collection,
            [
                {
                    "id": parent_id,
                    "content": content,
                    "heading": heading,
                    "heading_path": heading_path,
                    "source": source,
                }
            ],
        )

    def store_batch(self, items: list[dict[str, Any]]) -> None:
        self._repo.upsert_many(self._collection, items)

    def get(self, parent_id: str) -> dict[str, str] | None:
        row = self._repo.get_one(self._collection, parent_id)
        return self._to_dict(row) if row is not None else None

    def get_batch(self, parent_ids: list[str]) -> list[dict[str, str]]:
        rows = self._repo.get_many(self._collection, parent_ids)
        return [self._to_dict(row) for row in rows]

    def delete_by_source(self, source: str) -> int:
        return self._repo.delete_by_source(self._collection, source)

    def delete_by_source_except(self, source: str, keep_ids: set[str]) -> int:
        """回收该来源下不再被可见版本引用的父块行（发布成功后调用）。"""
        return self._repo.delete_by_source_except(self._collection, source, set(keep_ids))

    @staticmethod
    def _to_dict(row: dict[str, Any]) -> dict[str, str]:
        return {
            "id": row["id"],
            "content": row["content"],
            "heading": row.get("heading", ""),
            "heading_path": row.get("heading_path", ""),
            "source": row.get("source", ""),
        }
