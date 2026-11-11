"""合规模块删除后的契约。

删除的目的是「不再维护」，不是「仓库里没有这个词」——因此这里同时锁住
**该没的没了** 与 **该留的还在**，避免清理时误伤通用能力。
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.api.routes.compliance",
        "src.compliance_service",
        "src.compliance_rules",
        "src.api.schemas.compliance",
    ],
)
def test_compliance_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_generic_citation_validation_is_kept():
    """研报同样需要通用引用校验，不能随合规模块一起删掉。"""
    from src.answer.citations import (
        CitationValidator,
        apply_refusal_policy,
        run_citation_validation,
    )

    assert hasattr(CitationValidator, "validate")
    assert callable(run_citation_validation)
    assert callable(apply_refusal_policy)


def test_compliance_specific_validators_are_gone():
    from src.answer import citations as citation_validator

    assert not hasattr(citation_validator, "validate_clause_citations")
    assert not hasattr(citation_validator, "validate_finding_evidence")
    assert not hasattr(citation_validator, "extract_clause_references")


def test_compliance_config_keys_are_gone():
    from src.core import config

    for key in (
        "COMPLIANCE_CATEGORIES",
        "COMPLIANCE_REQUIRE_CLAUSE",
        "COMPLIANCE_CLAUSE_MIN_SCORE",
        "ENABLE_METADATA_FILTER",
    ):
        assert not hasattr(config, key), key


def test_iter_active_kbs_is_explicit_single_collection(monkeypatch):
    """平台只有研报一个检索集合；这里恒为默认集合一项。"""
    from src.core.config import KB_COLLECTION_NAME
    from src import engine as chat_service

    calls: list[tuple] = []

    def _fake_get_kb(*args, **kwargs):
        calls.append((args, kwargs))
        return "kb-default"

    monkeypatch.setattr(chat_service, "get_knowledge_base", _fake_get_kb)

    kbs = chat_service.iter_active_kbs()

    assert kbs == [(KB_COLLECTION_NAME, "kb-default")]
    assert calls == [((), {})]


def test_app_route_table_has_report_routes_and_no_compliance():
    """路由表级验证：删干净了，且研报新增的两个接口确实挂上了。

    用 ``openapi()["paths"]`` 而不是遍历 ``app.routes``：新版 FastAPI 会把
    ``include_router`` 的结果包成 ``_IncludedRouter``（没有 ``.path``），
    直接遍历会漏掉全部业务路由。
    """
    from src.main import app

    paths = set(app.openapi()["paths"])

    assert "/api/documents/{source}/page/{page}" in paths
    assert "/api/documents/{source}/metadata" in paths
    assert "/api/chat/stream" in paths
    assert not [path for path in paths if "compliance" in path]
    assert not [path for path in paths if "knowledge-bases" in path]
