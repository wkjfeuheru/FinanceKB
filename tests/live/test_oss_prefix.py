"""真实 OSS 上的前缀列举，以及新旧键共存时的读取回退。"""

from __future__ import annotations

import uuid

import pytest

from src.storage.objects.object_keys import (
    document_root_prefix,
    legacy_original_key,
    original_key,
    original_key_candidates,
    parsed_key,
)
from src.storage.objects.oss import OSSStorageBackend

pytestmark = pytest.mark.live

COLLECTION = "livecheck"


def _read_first(storage: OSSStorageBackend, keys: list[str]) -> tuple[str, bytes]:
    """按候选顺序读到第一个存在的对象。"""
    for key in keys:
        try:
            return key, storage.download_sync(key)
        except FileNotFoundError:
            continue
    raise AssertionError(f"候选键都不存在：{keys}")


def test_list_keys_is_prefix_scoped_and_legacy_key_is_only_a_fallback():
    """前缀列举不含前缀外的对象；新键存在时读新键，只有旧键时才落到旧键。"""
    token = uuid.uuid4().hex
    source = f"livecheck-{token}.pdf"
    storage = OSSStorageBackend()
    new_original = original_key(COLLECTION, source, filename=source)
    new_parsed = parsed_key(COLLECTION, source)
    legacy = legacy_original_key(source)
    unrelated = f"livecheck-unrelated/{token}.bin"
    created = [new_original, new_parsed, legacy, unrelated]
    try:
        storage.upload_sync(legacy, b"legacy-bytes")
        key, payload = _read_first(storage, original_key_candidates(COLLECTION, source))
        assert key == legacy
        assert payload == b"legacy-bytes"

        storage.upload_sync(new_original, b"current-bytes")
        storage.upload_sync(new_parsed, b"parsed")
        storage.upload_sync(unrelated, b"other")
        key, payload = _read_first(storage, original_key_candidates(COLLECTION, source))
        assert key == new_original
        assert payload == b"current-bytes"

        listed = storage.list_keys_sync(document_root_prefix(COLLECTION, source))
        assert listed == sorted(listed)
        assert listed == sorted([new_original, new_parsed])
        assert legacy not in listed
        assert unrelated not in listed
    finally:
        for key in created:
            try:
                storage.delete_sync(key)
            except Exception:
                pass
