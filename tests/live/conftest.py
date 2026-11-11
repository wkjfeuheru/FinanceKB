"""隔离环境用例。

默认测试收集会跳过这些用例，避免 ``pytest tests`` 连上真实
Milvus / PostgreSQL / OSS。显式执行::

    pytest tests/live -m live
"""

from __future__ import annotations

import re

import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("-m") or ""
    if re.search(r"\blive\b", markexpr):
        return
    skip = pytest.mark.skip(reason="隔离环境用例，请使用 pytest tests/live -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
