"""Document/version/chunk manifest persistence for event-driven indexing."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from typing import Any

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, insert, inspect, select, update

from finance_rag.src.core.config import DATABASE_URL

metadata = MetaData()
documents = Table(
    "documents", metadata,
    Column("document_id", String(128), primary_key=True),
    Column("tenant_id", String(64), nullable=False),
    Column("source", String(512), nullable=False),
    Column("title", String(256), nullable=False),
    Column("current_version_id", String(128), nullable=True),
    Column("status", String(32), nullable=False, default="processing"),
    Column("category", String(64), nullable=False, default=""),
    Column("date", String(32), nullable=False, default=""),
    Column("object_key", String(1024), nullable=False, default=""),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)
document_versions = Table(
    "document_versions", metadata,
    Column("version_id", String(128), primary_key=True),
    Column("document_id", String(128), nullable=False),
    Column("object_version_id", String(256), nullable=False, default=""),
    Column("etag", String(256), nullable=False, default=""),
    Column("file_sha256", String(64), nullable=False),
    Column("status", String(32), nullable=False, default="processing"),
    Column("parser_version", String(128), nullable=False, default="mineru"),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("error", Text, nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
document_chunks = Table(
    "document_chunks", metadata,
    Column("chunk_id", String(128), primary_key=True),
    Column("document_id", String(128), nullable=False),
    Column("version_id", String(128), nullable=False),
    Column("chunk_key", String(128), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("parent_id", String(128), nullable=False),
    Column("position", Integer, nullable=False),
    Column("state", String(32), nullable=False, default="active"),
)


class DocumentManifestRepository:
    """Small repository used by the incremental indexer and object events."""

    def __init__(self, database_url: str = DATABASE_URL) -> None:
        self._engine = create_engine(database_url, pool_pre_ping=True)

    def ensure_schema(self) -> None:
        metadata.create_all(self._engine, tables=[documents, document_versions, document_chunks])

    @contextmanager
    def metadata_date_transaction(self, tenant_id: str, source: str, date: str | None, *, expected: dict | None = None):
        """仅已有 manifest，锁定精确来源；调用者失败时回滚，不建表或插入。"""
        with self._engine.begin() as connection:
            if not inspect(connection).has_table("documents"):
                if expected is not None and expected != {"status": "skipped_missing_table", "rows": []}:
                    raise ValueError("manifest table drift")
                yield {"status": "skipped_missing_table", "before": []}
                return
            condition = (documents.c.tenant_id == tenant_id) & (documents.c.source == source)
            before = [dict(row) for row in connection.execute(
                select(documents).where(condition).with_for_update()).mappings()]
            actual = {"status": "present" if before else "skipped_missing_document", "rows": before}
            if expected is not None and json.dumps(actual, sort_keys=True, default=str) != json.dumps(expected, sort_keys=True, default=str):
                raise ValueError("manifest before rows drift")
            result = {"status": "updated" if before and date is not None else
                      "unchanged" if before else "skipped_missing_document", "before": before}
            if before and date is not None:
                connection.execute(update(documents).where(condition).values(date=date))
            yield result

    def restore_metadata_dates(self, tenant_id: str, source: str, before: list[dict], written_date: str) -> str:
        """精确补偿日期，遇到第三方新值拒绝覆盖；同样不建表。"""
        with self._engine.begin() as connection:
            if not inspect(connection).has_table("documents"):
                if before:
                    raise ValueError("manifest disappeared during compensation")
                return "skipped_missing_table"
            condition = (documents.c.tenant_id == tenant_id) & (documents.c.source == source)
            current = [dict(row) for row in connection.execute(select(documents).where(condition).with_for_update()).mappings()]
            old = {row["document_id"]: row for row in before}
            if {row["document_id"] for row in current} != set(old):
                raise ValueError("manifest rows changed during compensation")
            for row in current:
                if row["date"] not in {old[row["document_id"]]["date"], written_date}:
                    raise ValueError("manifest date changed during compensation")
            for row in current:
                connection.execute(update(documents).where(condition & (documents.c.document_id == row["document_id"])).values(date=old[row["document_id"]]["date"]))
            return "restored" if before else "skipped_missing_document"

    def read_metadata_dates(self, tenant_id: str, source: str) -> dict[str, Any]:
        """只读诊断，不以连接错误冒充缺表。"""
        with self._engine.connect() as connection:
            if not inspect(connection).has_table("documents"):
                return {"status": "skipped_missing_table", "rows": []}
            rows = [dict(row) for row in connection.execute(select(documents).where(
                (documents.c.tenant_id == tenant_id) & (documents.c.source == source))).mappings()]
            return {"status": "present" if rows else "skipped_missing_document", "rows": rows}

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._engine.connect() as connection:
            row = connection.execute(select(documents).where(documents.c.document_id == document_id)).mappings().first()
        return dict(row) if row else None

    def upsert_document(self, values: dict[str, Any]) -> None:
        self.ensure_schema()
        now = datetime.now(timezone.utc)
        payload = {"updated_at": now, **values}
        with self._engine.begin() as connection:
            existing = connection.execute(select(documents.c.document_id).where(documents.c.document_id == values["document_id"])).first()
            if existing:
                connection.execute(update(documents).where(documents.c.document_id == values["document_id"]).values(payload))
            else:
                connection.execute(insert(documents).values(payload))

    def create_version(self, values: dict[str, Any]) -> None:
        self.ensure_schema()
        with self._engine.begin() as connection:
            connection.execute(insert(document_versions).values(created_at=datetime.now(timezone.utc), **values))

    def set_version_status(self, version_id: str, status: str, *, error: str = "") -> None:
        self.ensure_schema()
        with self._engine.begin() as connection:
            connection.execute(update(document_versions).where(document_versions.c.version_id == version_id).values(status=status, error=error))

    def replace_chunks(self, rows: list[dict[str, Any]]) -> None:
        self.ensure_schema()
        if not rows:
            return
        version_id = rows[0]["version_id"]
        with self._engine.begin() as connection:
            connection.execute(document_chunks.delete().where(document_chunks.c.version_id == version_id))
            connection.execute(insert(document_chunks), rows)

    def list_documents(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        self.ensure_schema()
        statement = select(documents).order_by(documents.c.updated_at.desc())
        if not include_deleted:
            statement = statement.where(documents.c.status != "deleted")
        with self._engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]


__all__ = ["DocumentManifestRepository"]
