"""公开证据身份、元数据及回答状态的兼容契约。"""

import pytest
from pydantic import ValidationError

from src.agent.tools.retrieval import _normalize_evidence
from src.api.schemas.chat import ChatResponse, SourceInfo


def test_normalize_preserves_evidence_identity_and_metadata():
    # security_name / industry_l1 / industry_l2 已从证据契约移除
    evidence = _normalize_evidence({
        "id": 12, "child_id": 7, "chunk_key": "chunk-7", "parent_key": "parent-2",
        "covered_child_ids": [7, 8], "collection": "research_reports",
        "report_type": "个股", "broker": "测试证券", "table_id": "table-1",
        "document_sha256": "a" * 64, "document_version": "2", "content": "证据全文",
    }, 0)

    assert evidence["id"] == 12
    assert evidence["child_id"] == 7
    assert evidence["chunk_key"] == "chunk-7"
    assert evidence["parent_key"] == "parent-2"
    assert evidence["covered_child_ids"] == [7, 8]
    assert evidence["collection"] == "research_reports"
    assert evidence["report_type"] == "个股"
    assert evidence["broker"] == "测试证券"
    assert evidence["table_id"] == "table-1"
    assert evidence["document_sha256"] == "a" * 64
    assert evidence["document_version"] == "2"
    assert evidence["content"] == "证据全文"


def test_normalize_copies_covered_child_ids():
    child_ids = ["child-1"]
    evidence = _normalize_evidence({"covered_child_ids": child_ids}, 0)
    evidence["covered_child_ids"].append("child-2")
    assert child_ids == ["child-1"]


def test_normalize_falls_back_to_real_child_and_parent_identity():
    evidence = _normalize_evidence({"id":"c1", "parent_id":"p1"}, 0)
    assert evidence["child_id"] == "c1"
    assert evidence["chunk_key"] == "c1"
    assert evidence["parent_key"] == "p1"


@pytest.mark.parametrize("item", [{}, {"document_sha256": "", "document_version": ""}])
def test_normalize_legacy_evidence_keeps_unknown_document_identity(item):
    evidence = _normalize_evidence(item, 3)
    assert evidence["id"] == "evidence-3"
    assert evidence["document_sha256"] is None
    assert evidence["document_version"] is None
    assert evidence["covered_child_ids"] == []


def test_source_serialization_exposes_identity_without_full_content():
    # security_name / industry_l1 / industry_l2 已从 SourceInfo 移除
    source = SourceInfo(
        index=1, title="测试研报", source="report.pdf", score=0.9, preview="证据预览",
        id=12, child_id=7, chunk_id="chunk-legacy", chunk_key="chunk-7",
        parent_id="parent-legacy", parent_key="parent-2", covered_child_ids=[7, 8],
        collection="research_reports", report_type="个股", broker="测试证券",
        table_id="table-1",
        document_sha256="a" * 64, document_version="2", content="不得公开的全文",
    ).model_dump()

    assert source["id"] == 12
    assert source["child_id"] == 7
    assert source["chunk_id"] == "chunk-legacy"
    assert source["chunk_key"] == "chunk-7"
    assert source["parent_id"] == "parent-legacy"
    assert source["parent_key"] == "parent-2"
    assert source["covered_child_ids"] == [7, 8]
    assert source["report_type"] == "个股"
    assert source["broker"] == "测试证券"
    assert source["table_id"] == "table-1"
    assert source["document_sha256"] == "a" * 64
    assert source["document_version"] == "2"
    assert "content" not in source


def test_source_legacy_payload_has_unknown_document_identity():
    source = SourceInfo(index=1, title="旧研报", source="old.pdf", score=0.5, preview="旧证据")
    assert source.document_sha256 is None
    assert source.document_version is None
    assert source.covered_child_ids == []


@pytest.mark.parametrize("status", ["complete", "partial", "refused", "unverified", "clarification"])
def test_chat_response_serializes_supported_answer_status(status):
    response = ChatResponse(
        answer="回答", rewritten_query="问题", answer_status=status, missing_items=["毛利率"],
    ).model_dump()
    assert response["answer_status"] == status
    assert response["missing_items"] == ["毛利率"]


def test_chat_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ChatResponse(answer="回答", rewritten_query="问题", answer_status="invented")


def test_chat_response_legacy_payload_defaults_to_unverified():
    response = ChatResponse(answer="回答", rewritten_query="问题")
    other = ChatResponse(answer="另一回答", rewritten_query="另一问题")
    assert response.answer_status == "unverified"
    assert response.missing_items == []
    response.missing_items.append("营业收入")
    assert other.missing_items == []
