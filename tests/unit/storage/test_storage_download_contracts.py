"""存储后端下载契约：共享实现、异常映射与同步/异步协议。

D-01 / D-02 合并后，本地与 OSS 后端的 ``download_to_path`` /
``download_to_path_sync``、``download`` / ``download_sync`` 各自共用同一
内部实现，仅入口协议（异步调度 vs 同步直调）不同。这里钉住：

* 共享实现的可见行为在同步、异步两个入口完全一致；
* OSS 异常映射：404 -> FileNotFoundError，其余 -> StorageError；
* 本地后端：源缺失 -> FileNotFoundError；同路径直接返回源，不做复制。
"""

from __future__ import annotations

import asyncio

import pytest

from src.core.exceptions import StorageError
from src.storage.objects.local import LocalStorageBackend
from src.storage.objects.oss import OSSStorageBackend


# ---------------------------------------------------------------------------
# LocalStorageBackend
# ---------------------------------------------------------------------------

def test_local_download_to_path_copies_in_both_protocols(tmp_path):
    backend = LocalStorageBackend(root_dir=str(tmp_path))
    asyncio.run(backend.upload("docs/a.pdf", b"payload"))

    async_target = tmp_path / "out-async" / "a.pdf"
    result = asyncio.run(backend.download_to_path("docs/a.pdf", async_target))
    assert result == async_target
    assert async_target.read_bytes() == b"payload"

    sync_target = tmp_path / "out-sync" / "a.pdf"
    assert backend.download_to_path_sync("docs/a.pdf", sync_target) == sync_target
    assert sync_target.read_bytes() == b"payload"


def test_local_download_to_path_same_path_returns_source_without_copy(tmp_path):
    backend = LocalStorageBackend(root_dir=str(tmp_path))
    asyncio.run(backend.upload("docs/a.pdf", b"payload"))
    source = tmp_path / "docs" / "a.pdf"

    assert asyncio.run(backend.download_to_path("docs/a.pdf", source)) == source
    assert backend.download_to_path_sync("docs/a.pdf", source) == source


def test_local_download_to_path_missing_source_raises(tmp_path):
    backend = LocalStorageBackend(root_dir=str(tmp_path))
    target = tmp_path / "never" / "x.pdf"

    with pytest.raises(FileNotFoundError):
        asyncio.run(backend.download_to_path("docs/missing.pdf", target))
    with pytest.raises(FileNotFoundError):
        backend.download_to_path_sync("docs/missing.pdf", target)
    assert not target.exists()


# ---------------------------------------------------------------------------
# OSSStorageBackend
# ---------------------------------------------------------------------------

class _FakeObject:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeOSSStatus(Exception):
    """带 HTTP 状态码的伪 SDK 异常（对齐 oss2 异常的 status 属性）。"""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class _FakeBucket:
    def __init__(self, payload: bytes = b"", error: Exception | None = None):
        self._payload = payload
        self._error = error
        self.requested: list[str] = []

    def get_object(self, key: str) -> _FakeObject:
        self.requested.append(key)
        if self._error is not None:
            raise self._error
        return _FakeObject(self._payload)


def _oss_backend(bucket: _FakeBucket) -> OSSStorageBackend:
    backend = OSSStorageBackend()
    backend._client = bucket  # 绕过惰性初始化，注入离线替身
    return backend


def test_oss_download_maps_404_to_file_not_found_in_both_protocols():
    backend = _oss_backend(_FakeBucket(error=_FakeOSSStatus(404)))

    with pytest.raises(FileNotFoundError):
        asyncio.run(backend.download("docs/x.pdf"))
    with pytest.raises(FileNotFoundError):
        backend.download_sync("docs/x.pdf")


def test_oss_download_maps_other_failures_to_storage_error_in_both_protocols():
    for error in (_FakeOSSStatus(500), RuntimeError("boom")):
        backend = _oss_backend(_FakeBucket(error=error))

        with pytest.raises(StorageError):
            asyncio.run(backend.download("docs/x.pdf"))
        with pytest.raises(StorageError):
            backend.download_sync("docs/x.pdf")


def test_oss_download_and_to_path_share_payload_in_both_protocols(tmp_path):
    backend = _oss_backend(_FakeBucket(payload=b"payload"))

    assert asyncio.run(backend.download("docs/x.pdf")) == b"payload"
    assert backend.download_sync("docs/x.pdf") == b"payload"

    target = tmp_path / "x.pdf"
    assert asyncio.run(backend.download_to_path("docs/x.pdf", target)) == target
    assert target.read_bytes() == b"payload"

    sync_target = tmp_path / "y.pdf"
    assert backend.download_to_path_sync("docs/x.pdf", sync_target) == sync_target
    assert sync_target.read_bytes() == b"payload"
