"""入库明细契约：文档级元数据在写库前必须通过完整校验。

公共元数据是检索过滤与回答归因的依据，入库时写错一行，
后续所有维度都会静默失效——这里逐条断言而不是抽查。
"""

import pytest

from src.chunking.metadata_extractor import (
    ExtractedMetadata,
    apply_metadata,
    build_canonical_metadata,
    extract_metadata,
    merge_metadata_patch,
    validate_metadata_field,
)


def test_extract_derives_report_fields_from_filename():
    """研报文件名高度规范：能确定性拿到的字段绝不交给 LLM。"""
    extracted = extract_metadata("券商行业深度报告-中信证券-20261001.pdf", enable_llm=False)

    assert extracted.report_type == "行业"
    assert extracted.broker == "中信证券"


def test_canonical_metadata_contains_only_core_fields():
    """输出字段契约：核心字段 + 研报条件字段；security_name 等已删除字段不得回流。"""
    canonical = build_canonical_metadata(
        {"category": "research", "date": "2026-10-01", "version": "v1",
         "report_type": "行业", "broker": "中信证券"}
    )

    assert canonical == {"category": "", "date": "2026-10-01",
                         "report_type": "行业", "broker": "中信证券", "version": "v1"}
    assert "security_name" not in canonical
    assert "industry_l1" not in canonical
    assert "industry_l2" not in canonical
    assert "meta_source" not in canonical
    assert "needs_review" not in canonical


def test_explicit_report_fields_are_kept_verbatim():
    """显式提交值是人工判断，必须原样保留，不得被推断覆盖。"""
    canonical = build_canonical_metadata({"report_type": "个股", "broker": "中信证券"})

    assert canonical["report_type"] == "个股"
    assert canonical["broker"] == "中信证券"


def test_invalid_report_type_is_rejected_not_silently_blanked():
    """非法报告类型写入后过滤会静默失效，必须在入口拒绝。"""
    with pytest.raises(ValueError, match="report_type"):
        build_canonical_metadata({"report_type": "点评"})


def test_broker_field_length_is_capped():
    """过滤走精确匹配，超长机构名写不进 schema，必须入口拒绝。"""
    with pytest.raises(ValueError, match="broker"):
        build_canonical_metadata({"report_type": "行业", "broker": "券" * 30})


def test_merge_patch_only_accepts_known_fields():
    """已删除字段（industry_l1 等）不允许再通过修正接口回流。"""
    with pytest.raises(ValueError, match="未知元数据字段"):
        merge_metadata_patch({"report_type": "行业"}, {"industry_l1": "非银金融"})

    with pytest.raises(ValueError, match="未知元数据字段"):
        merge_metadata_patch({"report_type": "行业"}, {"security_name": "贵州茅台"})

    with pytest.raises(ValueError, match="未知元数据字段"):
        merge_metadata_patch({"report_type": "行业"}, {"category": "investment_research"})


def test_merge_patch_empty_string_clears_field():
    """空串是人工「清空」语义，合并后必须仍然通过完整校验。"""
    merged = merge_metadata_patch({"report_type": "行业", "broker": "中信证券"}, {"broker": ""})

    assert merged["broker"] == ""
    assert merged["report_type"] == "行业"


def test_merge_patch_invalid_date_is_rejected():
    """日期必须是完整 ISO 日历日；正文抓到的干扰日期不允许入库。"""
    with pytest.raises(ValueError, match="date"):
        merge_metadata_patch({"report_type": "行业"}, {"date": "2026-13-01"})


def test_validate_field_returns_empty_for_illegal_value():
    """单字段校验返回空串而不是抛异常：空值不参与过滤，不会误收窄召回。"""
    assert validate_metadata_field("report_type", "点评") == ""
    assert validate_metadata_field("report_type", None) == ""
    assert validate_metadata_field("report_type", "宏观") == "宏观"


def test_apply_metadata_merges_extracted_result():
    """抽取结果并入文档级 metadata，且不改动调用方传入的字典。"""
    original = {"date": "2026-10-01"}
    merged = apply_metadata(original, ExtractedMetadata(report_type="行业"))

    assert merged["report_type"] == "行业"
    assert merged["broker"] == ""
    # 不改动调用方传入的字典
    assert "report_type" not in original
    assert original["date"] == "2026-10-01"
