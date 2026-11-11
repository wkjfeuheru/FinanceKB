"""真实 PostgreSQL 上的父块补列与 delete_by_source_except。

在同一实例上建一次性库，不改现有 finance_rag 里的 parent_chunks。
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url

from src.core.config import DATABASE_URL
from src.storage.relational_db.parent_store import ParentStoreRepository
from src.storage.relational_db.table_store import TableStoreRepository

pytestmark = pytest.mark.live


def _admin_engine():
    admin_url = make_url(DATABASE_URL).set(database="postgres")
    return create_engine(admin_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)


def test_heading_path_expansion_and_source_recycle():
    """旧表补上 heading_path；回收只删指定集合、指定来源里未被保留的行。"""
    name = f"finance_rag_live_{uuid.uuid4().hex[:8]}"
    admin = _admin_engine()
    scratch_url = make_url(DATABASE_URL).set(database=name)
    parent = table = None
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))

        legacy = create_engine(scratch_url, pool_pre_ping=True)
        with legacy.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE parent_chunks ("
                    "collection VARCHAR(64) NOT NULL, "
                    "id VARCHAR(64) NOT NULL, "
                    "content TEXT NOT NULL, "
                    "heading VARCHAR(512) NOT NULL DEFAULT '', "
                    "source VARCHAR(512) NOT NULL DEFAULT '', "
                    "PRIMARY KEY (collection, id))"
                )
            )
        legacy.dispose()

        parent = ParentStoreRepository(scratch_url)
        parent.upsert_many(
            "kb_a",
            [
                {
                    "id": "keep",
                    "content": "保留",
                    "heading": "标题",
                    "heading_path": "第一章 > 第一节",
                    "source": "doc.pdf",
                },
                {
                    "id": "drop",
                    "content": "过期",
                    "heading": "",
                    "heading_path": "旧章节",
                    "source": "doc.pdf",
                },
                {
                    "id": "other-source",
                    "content": "另一篇",
                    "heading": "",
                    "heading_path": "别的",
                    "source": "other.pdf",
                },
            ],
        )
        parent.upsert_many(
            "kb_b",
            [
                {
                    "id": "keep",
                    "content": "别的集合",
                    "heading": "",
                    "heading_path": "其他库",
                    "source": "doc.pdf",
                }
            ],
        )

        columns = {column["name"] for column in inspect(parent._get_engine()).get_columns("parent_chunks")}
        assert "heading_path" in columns
        assert parent.get_one("kb_a", "keep")["heading_path"] == "第一章 > 第一节"

        removed = parent.delete_by_source_except("kb_a", "doc.pdf", {"keep"})
        assert removed == 1
        assert parent.get_one("kb_a", "keep")["content"] == "保留"
        assert parent.get_one("kb_a", "drop") is None
        assert parent.get_one("kb_a", "other-source")["content"] == "另一篇"
        assert parent.get_one("kb_b", "keep")["content"] == "别的集合"

        table = TableStoreRepository(scratch_url)
        table_row = {
            "parent_id": "p",
            "payload": [["2026", "100"]],
            "markdown": "| 2026 | 100 |",
            "row_count": 1,
            "heading_path": "",
            "start_page": 1,
        }
        table.upsert_many(
            "kb_a",
            [
                {**table_row, "id": "t-keep", "source": "doc.pdf"},
                {**table_row, "id": "t-drop", "source": "doc.pdf"},
                {**table_row, "id": "t-other", "source": "other.pdf"},
            ],
        )
        table.upsert_many("kb_b", [{**table_row, "id": "t-keep", "source": "doc.pdf"}])
        assert table.delete_by_source_except("kb_a", "doc.pdf", {"t-keep"}) == 1
        assert table.get_one("kb_a", "t-keep")["markdown"] == "| 2026 | 100 |"
        assert table.get_one("kb_a", "t-drop") is None
        assert table.get_one("kb_a", "t-other") is not None
        assert table.get_one("kb_b", "t-keep") is not None
    finally:
        for repository in (parent, table):
            engine = getattr(repository, "_engine", None) if repository is not None else None
            if engine is not None:
                engine.dispose()
        with admin.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()
