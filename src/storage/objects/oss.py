"""阿里云 OSS 对象存储后端。

依赖 oss2，通过环境变量配置连接参数；存储桶必须由外部预先创建。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.core.config import (
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET,
    OSS_BUCKET,
    OSS_ENDPOINT,
    OSS_REGION,
)
from src.core.exceptions import StorageError
from .base import StorageBackend

logger = logging.getLogger(__name__)


class OSSStorageBackend(StorageBackend):
    """阿里云 OSS 存储后端，客户端惰性初始化且不会自动创建存储桶。"""

    def __init__(self) -> None:
        self._bucket = OSS_BUCKET
        self._endpoint = OSS_ENDPOINT
        self._region = OSS_REGION
        self._access_key_id = OSS_ACCESS_KEY_ID
        self._access_key_secret = OSS_ACCESS_KEY_SECRET
        self._client = None

    @property
    def backend_name(self) -> str:
        return "oss"

    @property
    def client(self):
        """惰性初始化 OSS Bucket 客户端，不检查或创建存储桶。"""
        if self._client is None:
            import oss2

            try:
                auth = oss2.Auth(self._access_key_id, self._access_key_secret)
                self._client = oss2.Bucket(
                    auth, self._endpoint, self._bucket, region=self._region
                )
            except Exception as exc:
                raise StorageError(f"OSS 客户端初始化失败：{exc}") from exc
        return self._client

    async def upload(self, key: str, data: bytes) -> None:
        try:
            result = self.client.put_object(key, data)
            result.status
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"OSS 上传失败（{key}）：{exc}") from exc
        logger.debug("OSS 上传：%s (%d bytes)", key, len(data))

    def _fetch_object(self, key: str) -> bytes:
        """共享下载实现：404 映射为 FileNotFoundError，其余映射为 StorageError。"""
        try:
            return self.client.get_object(key).read()
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                raise FileNotFoundError(f"OSS 对象不存在：{key}") from exc
            raise StorageError(f"OSS 下载失败（{key}）：{exc}") from exc

    async def download(self, key: str) -> bytes:
        # 同步 SDK 的网络 IO 经线程池调度，避免阻塞事件循环。
        return await asyncio.to_thread(self._fetch_object, key)

    async def download_text(self, key: str, encoding: str = "utf-8") -> str:
        return (await self.download(key)).decode(encoding)

    async def delete(self, key: str) -> None:
        try:
            self.client.delete_object(key)
        except Exception as exc:
            raise StorageError(f"OSS 删除失败（{key}）：{exc}") from exc
        logger.debug("OSS 删除：%s", key)

    async def exists(self, key: str) -> bool:
        try:
            return self.client.object_exists(key)
        except Exception as exc:
            raise StorageError(f"OSS 检查对象失败（{key}）：{exc}") from exc

    async def list_keys(self, prefix: str) -> list[str]:
        """列出某前缀下的全部对象键（资产回收用）。"""
        try:
            return await asyncio.to_thread(self.list_keys_sync, prefix)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"OSS 列举对象失败（prefix={prefix}）：{exc}") from exc

    def list_keys_sync(self, prefix: str) -> list[str]:
        """同步列举（OSS SDK 本身为同步调用）。"""
        import oss2

        try:
            keys = [obj.key for obj in oss2.ObjectIterator(self.client, prefix=prefix)]
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"OSS 列举对象失败（prefix={prefix}）：{exc}") from exc
        return sorted(keys)

    async def download_to_path(self, key: str, local_path: Path) -> Path:
        data = await self.download(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        logger.debug("OSS -> 本地临时文件：%s -> %s", key, local_path)
        return local_path

    def upload_sync(self, key: str, data: bytes) -> None:
        try:
            result = self.client.put_object(key, data)
            result.status
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"OSS 上传失败（{key}）：{exc}") from exc

    def download_sync(self, key: str) -> bytes:
        return self._fetch_object(key)

    def download_text_sync(self, key: str, encoding: str = "utf-8") -> str:
        return self.download_sync(key).decode(encoding)

    def exists_sync(self, key: str) -> bool:
        try:
            return self.client.object_exists(key)
        except Exception as exc:
            raise StorageError(f"OSS 检查对象失败（{key}）：{exc}") from exc

    def delete_sync(self, key: str) -> None:
        try:
            self.client.delete_object(key)
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                return
            raise StorageError(f"OSS 删除失败（{key}）：{exc}") from exc

    def download_to_path_sync(self, key: str, local_path: Path) -> Path:
        data = self.download_sync(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        return local_path
