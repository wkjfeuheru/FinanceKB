from sqlalchemy import create_engine, text

from finance_rag.src.infrastructure.relational_db.parent_store import ParentStoreRepository
from finance_rag.src.rag.retrieval.parent_store import ParentStore


def test_legacy_table_gets_heading_path_column_added(tmp_path):
    """旧表（无 heading_path 列）应被自动补列，升级后写入不再报「列不存在」。"""
    database_url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(database_url)
    with engine.begin() as connection:
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

    repository = ParentStoreRepository(database_url)
    repository.upsert_many(
        "kb_a",
        [
            {
                "id": "p1",
                "content": "父块",
                "heading": "第一条 目的",
                "heading_path": "第一章 总则 > 第一条 目的",
                "source": "doc.md",
            }
        ],
    )

    row = repository.get_one("kb_a", "p1")
    assert row["heading_path"] == "第一章 总则 > 第一条 目的"


def test_parent_store_repository_round_trip(tmp_path):
    repository = ParentStoreRepository(f"sqlite:///{tmp_path / 'parent.db'}")

    repository.upsert_many(
        "kb_a",
        [
            {"id": "p1", "content": "父块一", "heading": "## 标题", "source": "doc.md"},
            {"id": "p2", "content": "父块二", "heading": "", "source": "doc.md"},
        ],
    )

    rows = repository.get_many("kb_a", ["p1", "p2"])
    assert len(rows) == 2
    assert rows[0]["content"] == "父块一"

    # 覆盖写入（upsert 语义）
    repository.upsert_many("kb_a", [{"id": "p1", "content": "更新", "heading": "", "source": "doc.md"}])
    assert repository.get_one("kb_a", "p1")["content"] == "更新"

    # 集合隔离：同一 id 在不同集合独立存储
    repository.upsert_many("kb_b", [{"id": "p1", "content": "kb_b", "heading": "", "source": "doc.md"}])
    assert repository.get_one("kb_a", "p1")["content"] == "更新"
    assert repository.get_one("kb_b", "p1")["content"] == "kb_b"

    # delete_by_source 只删当前集合
    assert repository.delete_by_source("kb_a", "doc.md") == 2
    assert repository.get_many("kb_a", ["p1", "p2"]) == []
    assert repository.get_one("kb_b", "p1")["content"] == "kb_b"


def test_parent_store_to_dict_strips_collection():
    row = {"collection": "kb_a", "id": "p1", "content": "c", "heading": "h", "source": "s"}
    assert ParentStore._to_dict(row) == {
        "id": "p1",
        "content": "c",
        "heading": "h",
        "heading_path": "",
        "source": "s",
    }
