"""自动元数据过滤的白名单与值校验。

现状：``_parse_filters_output`` 的白名单是**防注入设计**（注释明写「只白名单
category/date 两个键」）。扩字段必须同时扩校验，否则等于开了一条注入通道。

另一条更隐蔽的风险：**误抽比漏抽危险**。过滤走精确匹配，一次错误的条件
会让检索静默落空或只覆盖一部分，而答案会基于错误的证据范围生成。
"""

import json

from finance_rag.src.services.chat_service import (
    _parse_filters_output,
    normalize_metadata_filters,
)


def _payload(**fields) -> str:
    return json.dumps({"metadata": fields}, ensure_ascii=False)


def test_unknown_stock_identifier_is_dropped():
    assert _parse_filters_output(_payload(stock_identifier="600519")) == {}


def test_industry_must_be_in_sw_taxonomy():
    assert _parse_filters_output(_payload(industry_l1="食品饮料")) == {
        "industry_l1": ["食品饮料"]
    }
    assert _parse_filters_output(_payload(industry_l1="白酒")) == {}


def test_industry_l2_must_belong_to_selected_l1():
    keep = _parse_filters_output(
        _payload(industry_l1="食品饮料", industry_l2="白酒Ⅱ")
    )
    assert keep["industry_l2"] == ["白酒Ⅱ"]

    drop = _parse_filters_output(
        _payload(industry_l1="食品饮料", industry_l2="半导体")
    )
    assert "industry_l2" not in drop
    assert drop["industry_l1"] == ["食品饮料"]


def test_industry_l2_without_l1_is_checked_against_whole_taxonomy():
    assert _parse_filters_output(_payload(industry_l2="半导体")) == {
        "industry_l2": ["半导体"]
    }
    assert _parse_filters_output(_payload(industry_l2="不存在的行业")) == {}


def test_report_type_is_an_enum():
    assert _parse_filters_output(_payload(report_type="行业")) == {"report_type": ["行业"]}
    assert _parse_filters_output(_payload(report_type="策略")) == {}


def test_broker_is_free_text_but_bounded():
    assert _parse_filters_output(_payload(broker="中信证券")) == {"broker": ["中信证券"]}
    assert _parse_filters_output(_payload(broker="   ")) == {}
    assert _parse_filters_output(_payload(broker="券" * 200)) == {}


def test_unknown_field_is_dropped():
    """防注入：任意键不能被透传成 Milvus 过滤表达式。"""
    assert _parse_filters_output(_payload(tenant_id="other")) == {}
    assert _parse_filters_output(_payload(source='x" or id != ""')) == {}


def test_existing_category_and_date_behaviour_is_preserved():
    parsed = _parse_filters_output(
        json.dumps(
            {"category": ["投研类"], "date": {"gte": "2024"}, "industry_l1": "电子"},
            ensure_ascii=False,
        )
    )

    assert parsed["category"] == ["investment_research"]
    assert parsed["date"] == {"gte": "2024-01-01"}
    assert parsed["industry_l1"] == ["电子"]


def test_normalize_metadata_filters_revalidates_api_input():
    """API 直接传入的 filters 也要过同一套校验（不能只信 LLM 那条路）。

    统一返回列表形式：Milvus 侧 `in [...]` 对单值字段同样成立，
    两条路径形状一致可以少一类分支。
    """
    assert normalize_metadata_filters({"stock_identifier": "600519"}) == {}
    assert normalize_metadata_filters({"industry_l1": "白酒"}) == {}
    assert normalize_metadata_filters({"tenant_id": "other"}) == {}
    assert normalize_metadata_filters({"report_type": "个股"}) == {
        "report_type": ["个股"]
    }
    assert normalize_metadata_filters({"date": {"gte": "2024"}}) == {
        "date": {"gte": "2024-01-01"}
    }


def test_exact_publication_days_use_scalar_in_condition():
    from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever

    assert HybridRetriever._build_filter({"date": ["2026-09-24", "2026-09-27"]}) == (
        'date in ["2026-09-24", "2026-09-27"]'
    )
