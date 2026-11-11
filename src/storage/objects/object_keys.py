"""对象存储键协议：租户 / 集合 / 文档身份 / 版本隔离。

背景（``docs/rag-review.md`` RAG-06）
------------------------------------
旧键是 ``docs/{filename}``、``docs/{stem}.md``、``images/{stem}/{hint}.{ext}``：

* ``a.pdf`` 的解析产物写成 ``docs/a.md``，会**覆盖**上传的原文 ``a.md``；
* 键里没有集合，两个知识库上传同名文件会互相覆盖；
* 键里没有版本，历史版本与当前版本的资产无法区分，删除任一版本都会殃及另一方。

新键把四段身份全部写进路径，并保持旧键可读：

.. code-block:: text

    {tenant}/{collection}/{document_id}/{version_segment}/original/{filename}
    {tenant}/{collection}/{document_id}/{version_segment}/parsed/document.md
    {tenant}/{collection}/{document_id}/images/{sha256[:16]}.{ext}

* ``document_id``：由 ``source`` 派生的稳定身份（哈希，避免文件名中的非法字符）；
* ``version_segment``：``v-<版本标签>``；无版本标签（覆盖模式）时为 ``current``；
* 图片按**内容**寻址：同一张图跨版本共享同一键，重复入库幂等且不会无限增长；
  删除单个版本时不动图片，避免删掉当前版本仍在引用的资产。

读取端一律通过 :func:`original_key_candidates` / :func:`parsed_key_candidates`
逐级回退（新键 → 当前键 → 旧键），因此历史对象无需迁移即可继续预览。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from src.core.config import TENANT_ID

#: 无版本标签（覆盖模式）时使用的版本段
CURRENT_VERSION_SEGMENT = "current"

#: 路径段最大长度：避免超长文件名把 key 推到对象存储的限制之外
_MAX_SEGMENT_LENGTH = 96

_UNSAFE_SEGMENT_RE = re.compile(r"[\\/\x00-\x1f\x7f]+")
_CONTROL_OR_SPACE_RE = re.compile(r"[\s]+")


def sanitize_component(value: Any, *, fallback: str = "unnamed") -> str:
    """把任意字符串合法化为单个路径段。

    去掉路径分隔符与控制字符（防止 ``../`` 逃逸或对象键越权），压缩空白，
    并限制长度；结果为空时返回 ``fallback``。中文与常见符号原样保留，
    便于在对象存储控制台人工排查。
    """
    text = "" if value is None else str(value)
    text = _UNSAFE_SEGMENT_RE.sub("_", text)
    text = _CONTROL_OR_SPACE_RE.sub("_", text).strip("._")
    text = text.replace("..", "_")
    if not text:
        return fallback
    return text[:_MAX_SEGMENT_LENGTH]


def document_id(source: str) -> str:
    """逻辑文档的稳定身份：与文件名解耦的短哈希（同名文件固定映射同一目录）。"""
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return digest[:16]


def version_segment(version: str = "") -> str:
    """版本段：有版本标签用 ``v-<标签>``，否则用 ``current``（覆盖模式的可见版本）。"""
    label = sanitize_component(version, fallback="") if str(version or "").strip() else ""
    return f"v-{label}" if label else CURRENT_VERSION_SEGMENT


def document_prefix(collection: str, source: str, version: str = "") -> str:
    """某文档（可指定版本段）的对象前缀，供列表 / 回收使用。"""
    return (
        f"{sanitize_component(TENANT_ID)}/{sanitize_component(collection, fallback='default')}"
        f"/{document_id(source)}/{version_segment(version)}"
    )


def document_root_prefix(collection: str, source: str) -> str:
    """该文档在对象存储中的根前缀（含全部版本），供整篇删除与 GC。"""
    return (
        f"{sanitize_component(TENANT_ID)}/{sanitize_component(collection, fallback='default')}"
        f"/{document_id(source)}"
    )


def original_key(
    collection: str, source: str, *, version: str = "", filename: str | None = None
) -> str:
    """原文件键：``{prefix}/original/{filename}``（保留扩展名，a.pdf 与 a.md 不同键）。"""
    name = sanitize_component(filename or Path(source).name, fallback="source")
    return f"{document_prefix(collection, source, version)}/original/{name}"


def parsed_key(collection: str, source: str, *, version: str = "") -> str:
    """解析产物键：``{prefix}/parsed/document.md``（与原文分区，永不互相覆盖）。"""
    return f"{document_prefix(collection, source, version)}/parsed/document.md"


def image_key(
    collection: str, source: str, asset: Any, *, version: str = ""
) -> str:
    """图片键：按图片内容寻址，跨版本共享。

    内容寻址的好处：重复入库幂等、同一张图不会在多个版本下重复存储，
    且删除单个版本时无需引用计数即可安全跳过图片。
    """
    data = getattr(asset, "data", b"") or b""
    digest = hashlib.sha256(data).hexdigest()[:16] if data else sanitize_component(
        getattr(asset, "key_hint", ""), fallback="image"
    )
    ext = sanitize_component(getattr(asset, "ext", "") or "png", fallback="png")
    return f"{document_root_prefix(collection, source)}/images/{digest}.{ext}"


# ---------------------------------------------------------------------------
# 历史键（旧协议）：仅用于兼容读取与清理，不再写入
# ---------------------------------------------------------------------------


def legacy_original_key(source: str) -> str:
    """旧原文件键：``docs/{filename}``。"""
    return f"docs/{Path(source).name}"


def legacy_parsed_key(source: str) -> str:
    """旧解析产物键：``docs/{stem}.md``。"""
    return f"docs/{Path(source).stem}.md"


def legacy_image_key(source: str, asset: Any) -> str:
    """旧图片键：``images/{stem}/{hint}.{ext}``。"""
    ext = str(getattr(asset, "ext", "") or "png").lower()
    hint = sanitize_component(getattr(asset, "key_hint", ""), fallback="image")
    return f"images/{sanitize_component(Path(source).stem, fallback='source')}/{hint}.{ext}"


def original_key_candidates(
    collection: str, source: str, *, version: str = ""
) -> list[str]:
    """原文件读取候选键：当前版本 → 覆盖模式键 → 旧键（去重且保序）。"""
    return _dedupe([
        original_key(collection, source, version=version),
        original_key(collection, source, version=""),
        legacy_original_key(source),
    ])


def parsed_key_candidates(collection: str, source: str, *, version: str = "") -> list[str]:
    """解析产物读取候选键：当前版本 → 覆盖模式键 → 旧键。"""
    return _dedupe([
        parsed_key(collection, source, version=version),
        parsed_key(collection, source, version=""),
        legacy_parsed_key(source),
    ])


def _dedupe(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


__all__ = [
    "CURRENT_VERSION_SEGMENT",
    "document_id",
    "document_prefix",
    "document_root_prefix",
    "image_key",
    "legacy_image_key",
    "legacy_original_key",
    "legacy_parsed_key",
    "original_key",
    "original_key_candidates",
    "parsed_key",
    "parsed_key_candidates",
    "sanitize_component",
    "version_segment",
]
