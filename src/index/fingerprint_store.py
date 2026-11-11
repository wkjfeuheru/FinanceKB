"""文件哈希指纹存储，用于增量构建：跳过未变更文件的重复解析 + 嵌入。

以 JSON 文件持久化到 ``MILVUS_FINGERPRINT_PATH``，支持按 source 记录 SHA256、
SimHash、mtime、size、version 等元数据。写入采用「临时文件 + 原子替换」，
并加锁保护，避免并发更新互相覆盖或进程崩溃截断 JSON。

除指纹记录外，本存储还承载**跨存储发布日志**（``__pending_publish__``）：
向量库 / 关系库 / 对象存储的写入无法组成一个原子事务，因此发布分
「暂存 → 激活 → 提交指纹」三段，中间态登记在此。只要存在未完成的发布登记，
``is_unchanged`` 一律回答「未变更不成立」，避免把未完成的版本误判为已入库。
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from src.core.config import KB_COLLECTION_NAME, MILVUS_FINGERPRINT_PATH

# 发布日志在 JSON 中的保留键（不会与文件名冲突：source 是文件名）
PENDING_PUBLISH_KEY = "__pending_publish__"


def fingerprint_path_for(
    collection: str, *, default_collection: str | None = None
) -> str:
    """解析集合对应的指纹文件路径。

    默认集合沿用 ``MILVUS_FINGERPRINT_PATH``（继承既有指纹），其余集合追加
    ``_<sanitized-collection>`` 后缀，避免多集合互相跳过或误判已入库。
    迁移历史集合时必须直接使用基础文件路径，本函数会把它视为非默认集合。
    """
    default = default_collection or KB_COLLECTION_NAME
    if collection == default:
        return MILVUS_FINGERPRINT_PATH
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection)
    return f"{MILVUS_FINGERPRINT_PATH}_{safe}"


class FingerprintStore:
    """文件哈希指纹存储。"""

    def __init__(self, store_path: str | None = None):
        self._path = Path(store_path) if store_path else Path(MILVUS_FINGERPRINT_PATH)
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), "utf-8"
        )
        os.replace(tmp_path, self._path)

    def get(self, source: str) -> dict[str, Any] | None:
        """获取指定 source 的指纹记录，不存在返回 None。"""
        with self._lock:
            return copy.deepcopy(self._data.get(source))

    # ------------------------------------------------------------------
    # 发布日志（跨存储发布中间态）
    # ------------------------------------------------------------------

    def _pending_map(self) -> dict[str, dict[str, Any]]:
        value = self._data.get(PENDING_PUBLISH_KEY)
        return value if isinstance(value, dict) else {}

    def begin_publish(
        self,
        source: str,
        *,
        sha256: str,
        chunk_ids: list[str],
        previous_ids: list[str],
        version: str = "",
        simhash: int | None = None,
        chunk_count: int = 0,
        file_path: str = "",
        metadata: dict[str, Any] | None = None,
        record: dict[str, Any] | None = None,
        content_sha256: str = "",
        staged_ids: list[str] | None = None,
    ) -> None:
        """登记一次进行中的发布（暂存行已写入、尚未切换可见性）。

        ``chunk_ids`` 为新版本主键，``previous_ids`` 为发布前可见版本的旧主键。
        ``staged_ids`` 是本次实际插入的新主键（已存在的主键不会再 insert）。
        恢复不需要重放向量。
        """
        entry: dict[str, Any] = {
            "sha256": sha256,
            "version": version,
            "chunk_ids": [str(item) for item in chunk_ids],
            "previous_ids": [str(item) for item in previous_ids],
            "staged_ids": [str(item) for item in (staged_ids or [])],
            "activated": False,
            "chunk_count": int(chunk_count),
            "file_path": file_path,
            "content_sha256": content_sha256,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if simhash is not None:
            entry["simhash"] = simhash
        if metadata is not None:
            entry["metadata"] = copy.deepcopy(metadata)
        if record is not None:
            # 发布前该 source 的指纹记录，供回滚时逐源精确恢复
            entry["previous_record"] = copy.deepcopy(record)
        with self._lock:
            pending = self._pending_map()
            pending[source] = entry
            self._data[PENDING_PUBLISH_KEY] = pending
            self._save()

    def mark_publish_activated(self, source: str) -> None:
        """标记可见性切换已完成、指纹尚未提交。

        主键全部与旧版本重合时没有新的暂存行，不能靠「暂存行是否还在」判断
        激活有没有发生。这个标志补上那一种情况。
        """
        with self._lock:
            pending = self._pending_map()
            entry = pending.get(source)
            if not entry:
                return
            entry["activated"] = True
            pending[source] = entry
            self._data[PENDING_PUBLISH_KEY] = pending
            self._save()

    def pending_publish(self, source: str) -> dict[str, Any] | None:
        """返回某来源未完成的发布登记（无则 None）。"""
        with self._lock:
            entry = self._pending_map().get(source)
            return copy.deepcopy(entry) if entry else None

    def pending_sources(self) -> list[str]:
        """列出全部存在未完成发布的来源。"""
        with self._lock:
            return sorted(self._pending_map())

    def has_pending_publish(self, source: str) -> bool:
        with self._lock:
            return source in self._pending_map()

    def commit_publish(
        self,
        source: str,
        *,
        sha256: str,
        file_path: str = "",
        chunk_count: int = 0,
        simhash: int | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_sha256: str | None = None,
    ) -> None:
        """发布成功：指纹正式生效并清除发布登记（同一把锁、同一次落盘）。"""
        record: dict[str, Any] = {
            "sha256": sha256,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if simhash is not None:
            record["simhash"] = simhash
        if version is not None:
            record["version"] = version
        if content_sha256:
            record["content_sha256"] = content_sha256
        with self._lock:
            merged = {**self._data.get(source, {}), **record}
            if metadata is not None:
                for key in ("public_metadata_audit", "type_metadata_audit"):
                    merged.pop(key, None)
                    if key in metadata:
                        merged[key] = copy.deepcopy(metadata[key])
            self._data[source] = merged
            pending = self._pending_map()
            pending.pop(source, None)
            if pending:
                self._data[PENDING_PUBLISH_KEY] = pending
            else:
                self._data.pop(PENDING_PUBLISH_KEY, None)
            self._save()

    def abort_publish(
        self, source: str, *, restore_record: dict[str, Any] | None = None
    ) -> None:
        """放弃发布登记（回滚）；可按需把指纹恢复为发布前的记录。"""
        with self._lock:
            if restore_record is not None:
                self._data[source] = copy.deepcopy(restore_record)
            pending = self._pending_map()
            pending.pop(source, None)
            if pending:
                self._data[PENDING_PUBLISH_KEY] = pending
            else:
                self._data.pop(PENDING_PUBLISH_KEY, None)
            self._save()

    def is_unchanged(self, source: str, file_hash: str) -> bool:
        """检查文件是否未变更（哈希一致）。

        存在未完成的发布登记时一律返回 False：此时集合内容与指纹都不代表一个
        完整可见版本，不能据此跳过入库（否则未完成版本会被当成已入库）。
        """
        with self._lock:
            if source in self._pending_map():
                return False
            record = self._data.get(source)
            return record is not None and record.get("sha256") == file_hash

    def find_exact_content(
        self, content_sha256: str, *, exclude_source: str = ""
    ) -> str | None:
        """按「完整正文 + 表体」的内容哈希查找已入库来源（精确重复判定）。"""
        if not content_sha256:
            return None
        with self._lock:
            for source, record in self._data.items():
                if source in {PENDING_PUBLISH_KEY, exclude_source}:
                    continue
                if isinstance(record, dict) and record.get("content_sha256") == content_sha256:
                    return source
        return None

    def is_changed_by_hash(self, source: str, content_hash: str) -> bool:
        """基于内容哈希判断文件是否变更。

        Returns:
            True 表示文件已变更或首次入库，False 表示未变更。
        """
        record = self.get(source)
        if record is None:
            return True  # 首次入库
        stored_hash = record.get("sha256", "")
        return stored_hash != content_hash

    def update(
        self,
        source: str,
        sha256: str,
        file_path: str,
        chunk_count: int,
        *,
        simhash: int | None = None,
        mtime: float | None = None,
        size: int | None = None,
        version: str | None = None,
        metadata: dict[str, Any] | None = None,
        content_sha256: str | None = None,
    ) -> None:
        """更新（或创建）指纹记录；可选记录元数据检测与去重所需的扩展字段。"""
        record: dict[str, Any] = {
            "sha256": sha256,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        if simhash is not None:
            record["simhash"] = simhash
        if mtime is not None:
            record["mtime"] = mtime
        if size is not None:
            record["size"] = size
        if version is not None:
            record["version"] = version
        if content_sha256:
            record["content_sha256"] = content_sha256
        with self._lock:
            record = {**self._data.get(source, {}), **record}
            if metadata is not None:
                for key in ("public_metadata_audit", "type_metadata_audit"):
                    record.pop(key, None)
                    if key in metadata:
                        record[key] = copy.deepcopy(metadata[key])
            self._data[source] = record
            self._save()

    def update_metadata(self, source: str, metadata: dict[str, Any]) -> None:
        """只更新审核载荷，保持 SHA、内容指纹及文件属性原样。"""
        with self._lock:
            record = copy.deepcopy(self._data.get(source, {}))
            for key in ("public_metadata_audit", "type_metadata_audit"):
                record.pop(key, None)
                if key in metadata:
                    record[key] = copy.deepcopy(metadata[key])
            self._data[source] = record
            self._save()

    def restore_record(self, source: str, record: dict[str, Any] | None) -> None:
        """逐源精确补偿；不重建记录，不改变其他来源。"""
        with self._lock:
            if record is None:
                self._data.pop(source, None)
            else:
                self._data[source] = copy.deepcopy(record)
            self._save()

    def get_all_simhashes(self) -> dict[str, int]:
        """返回 {source: simhash}，仅包含已记录 simhash 的条目（跳过发布日志）。"""
        return {
            source: record["simhash"]
            for source, record in self._data.items()
            if source != PENDING_PUBLISH_KEY and isinstance(record.get("simhash"), int)
        }

    def remove(self, source: str) -> None:
        """删除指纹记录（连同该来源的发布登记）。"""
        with self._lock:
            self._data.pop(source, None)
            pending = self._pending_map()
            if source in pending:
                pending.pop(source, None)
                if pending:
                    self._data[PENDING_PUBLISH_KEY] = pending
                else:
                    self._data.pop(PENDING_PUBLISH_KEY, None)
            self._save()

    def clear(self) -> int:
        """清空全部指纹，返回被清除的条数。

        集合重建（``rebuild_collection``）后必须调用：数据没了而指纹还记着
        「已入库」时，后续上传会被静默跳过，知识库永远是空的。
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            self._save()
            return count

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """计算文件 SHA256 哈希。"""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


__all__ = ["FingerprintStore", "fingerprint_path_for"]
