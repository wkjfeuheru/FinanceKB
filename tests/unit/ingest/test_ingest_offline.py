"""离线入库脚本：文件发现，以及单文件按解析、切块、写入的顺序调用。"""

import hashlib
from pathlib import Path

import pytest

from scripts.ingest_offline import discover_files, ingest_file


def test_discover_files_walks_directories_and_filters_extensions(tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"a")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.md").write_text("正文", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"x")

    found = discover_files([tmp_path], extensions={".pdf", ".md"})

    assert [item.name for item in found] == ["a.pdf", "b.md"]


def test_discover_files_only_is_a_union_and_limit_applies_after_filter(tmp_path: Path):
    for name in ("alpha-0026.pdf", "beta-0027.pdf", "gamma.pdf"):
        (tmp_path / name).write_bytes(b"x")

    found = discover_files(
        [tmp_path],
        extensions={".pdf"},
        only=["0026", "0027"],
        limit=1,
    )

    assert [item.name for item in found] == ["alpha-0026.pdf"]


def test_discover_files_rejects_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_files([tmp_path / "missing"], extensions={".pdf"})


def test_ingest_file_runs_parse_chunk_then_store(tmp_path: Path, monkeypatch):
    path = tmp_path / "报告.pdf"
    path.write_bytes(b"pdf-bytes")
    calls: list[str] = []

    class _Fingerprints:
        def is_unchanged(self, source, file_hash):  # noqa: ANN001
            calls.append(f"hash:{source}")
            return False

    class _KB:
        def get_fingerprint_store(self):
            return _Fingerprints()

    class _Manager:
        kb = _KB()
        _collection = "finance_kb"

        def _parse_file(self, file_path, filename, ext):  # noqa: ANN001
            calls.append(f"parse:{filename}:{ext}")
            return {"path": file_path, "metadata": {}}

        async def _persist_parsed_document(self, parsed, content_hash):  # noqa: ANN001
            calls.append(f"store:{parsed.chunks}:{content_hash[:8]}")
            return {"source": "报告.pdf", "title": "报告", "chunk_count": 3, "parent_count": 1}

    class _Chunked:
        chunks = 3
        metadata = {
            "date": "2025-03-01",
            "report_type": "个股",
            "broker": "中信证券",
            "version": "2",
        }

    def _chunk(parsed, collection):  # noqa: ANN001
        calls.append(f"chunk:{collection}")
        return _Chunked()

    monkeypatch.setattr("scripts.ingest_offline.build_chunked", _chunk)
    result = ingest_file(_Manager(), path)

    assert calls == [
        "hash:报告.pdf",
        "parse:报告.pdf:.pdf",
        "chunk:finance_kb",
        f"store:3:{hashlib.sha256(b'pdf-bytes').hexdigest()[:8]}",
    ]
    assert result["status"] == "completed"
    assert result["chunk_count"] == 3
    assert result["broker"] == "中信证券"
    assert result["version"] == "2"


def test_ingest_file_skips_unchanged_content(tmp_path: Path):
    path = tmp_path / "报告.pdf"
    path.write_bytes(b"same")

    class _Fingerprints:
        def is_unchanged(self, source, file_hash):  # noqa: ANN001
            return True

    class _KB:
        def get_fingerprint_store(self):
            return _Fingerprints()

    class _Manager:
        kb = _KB()

        def _parse_file(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("未变更的文件不应解析")

    result = ingest_file(_Manager(), path)

    assert result == {
        "status": "skipped",
        "source": "报告.pdf",
        "chunk_count": 0,
        "reason": "内容未变更",
    }
