"""离线入库：解析、元数据抽取、切块、向量化、写入。

不启动 Web 服务，也不把文件丢进上传任务队列。每个文件按生产入库的同一组函数
串行走完：

1. 解析（MinerU / 文本直读）
2. 元数据抽取（发布日期、机构、报告类型）
3. 切块（父子块、表格、图片）
4. 向量化并写入 Milvus，同时把原文、解析稿和图片写入对象存储

用法::

    python scripts/ingest_offline.py files/docs
    python scripts/ingest_offline.py a.pdf b.md
    python scripts/ingest_offline.py files/docs --limit 2 --force
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_files(
    paths: list[Path],
    *,
    extensions: set[str],
    only: list[str] | None = None,
    limit: int = 0,
) -> list[Path]:
    """展开文件和目录，只保留支持的扩展名，按路径排序。"""
    found: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在：{path}")
        if path.is_dir():
            found.extend(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in extensions
            )
        elif path.suffix.lower() in extensions:
            found.append(path)
    unique = sorted(set(found))
    if only:
        unique = [item for item in unique if any(piece in item.name for piece in only)]
    if limit > 0:
        unique = unique[:limit]
    return unique


def build_chunked(parsed: Any, collection: str) -> Any:
    """切块并抽取元数据。同步函数，不能在事件循环线程里调用。"""
    from src.ingest.pipeline import prepare_chunked_document_sync

    return prepare_chunked_document_sync(parsed, collection=collection)


def ingest_file(
    manager: Any,
    path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """对单个文件跑完解析、切块、向量化和入库。"""
    filename = path.name
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if not force and manager.kb.get_fingerprint_store().is_unchanged(filename, content_hash):
        return {
            "status": "skipped",
            "source": filename,
            "chunk_count": 0,
            "reason": "内容未变更",
        }

    print(f"  解析 {filename}", flush=True)
    parsed = manager._parse_file(path, filename, path.suffix.lower())
    print("  抽取元数据并切块", flush=True)
    chunked = build_chunked(parsed, manager._collection)
    print("  向量化并写入", flush=True)
    stored = asyncio.run(manager._persist_parsed_document(chunked, content_hash))
    metadata = dict(chunked.metadata or {})
    return {
        "status": "completed",
        "source": stored.get("source", filename),
        "title": stored.get("title", ""),
        "chunk_count": int(stored.get("chunk_count") or 0),
        "parent_count": int(stored.get("parent_count") or 0),
        "date": metadata.get("date", ""),
        "report_type": metadata.get("report_type", ""),
        "broker": metadata.get("broker", ""),
        "version": metadata.get("version", ""),
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("-" * 72)
    print(f"{'文件':<36}{'状态':<10}{'块':>6}  日期 / 类型 / 机构")
    for item in results:
        detail = ""
        if item["status"] == "completed":
            detail = " / ".join(
                part for part in (
                    str(item.get("date") or ""),
                    str(item.get("report_type") or ""),
                    str(item.get("broker") or ""),
                ) if part
            )
        elif item.get("reason"):
            detail = str(item["reason"])
        elif item.get("error"):
            detail = str(item["error"])
        print(
            f"{str(item.get('source', ''))[:35]:<36}"
            f"{item['status']:<10}{int(item.get('chunk_count') or 0):>6}  {detail}"
        )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - 非常规 stdout
            pass

    from src.core.config import (
        KB_COLLECTION_NAME,
        MAX_UPLOAD_SIZE_MB,
        MINERU_SUPPORTED_EXTENSIONS,
    )
    from src.ingest.documents import DocumentManager

    parser = argparse.ArgumentParser(
        description="离线入库：解析、元数据抽取、切块、向量化、写入",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=Path, help="文件或目录，目录会递归扫描")
    parser.add_argument("--collection", default="", help="Milvus 集合名，默认取 KB_COLLECTION_NAME")
    parser.add_argument("--only", action="append", default=[], help="文件名片段，可重复，多个为并集")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 个文件，0 表示不限制")
    parser.add_argument("--force", action="store_true", help="内容未变也重新入库")
    args = parser.parse_args(argv)

    extensions = {item.lower() if item.startswith(".") else f".{item.lower()}"
                  for item in MINERU_SUPPORTED_EXTENSIONS}
    try:
        files = discover_files(args.paths, extensions=extensions, only=args.only, limit=args.limit)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    if not files:
        print("没有可入库的文件")
        return 1

    limit_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    oversized = [item for item in files if item.stat().st_size > limit_bytes]
    if oversized:
        print(f"以下文件超过 {MAX_UPLOAD_SIZE_MB} MB，已跳过：")
        for item in oversized:
            print(f"  {item}")
        files = [item for item in files if item not in oversized]
    if not files:
        return 1

    collection = args.collection or KB_COLLECTION_NAME
    manager = DocumentManager(collection_name=collection)
    print(f"集合：{collection}")
    print(f"文件：{len(files)}")

    results: list[dict[str, Any]] = []
    try:
        for index, path in enumerate(files, start=1):
            print(f"[{index}/{len(files)}] {path}")
            try:
                results.append(ingest_file(
                    manager, path, force=args.force,
                ))
            except Exception as exc:  # noqa: BLE001 - 单篇失败继续处理其余文件
                print(f"  失败：{type(exc).__name__}: {exc}", flush=True)
                results.append({
                    "status": "failed",
                    "source": path.name,
                    "chunk_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        manager._parse_executor.shutdown(wait=False, cancel_futures=True)

    _print_summary(results)
    failed = [item for item in results if item["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
