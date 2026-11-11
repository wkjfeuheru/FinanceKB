"""自动元数据过滤的白名单与值校验。

现状：``_parse_filters_output`` 的白名单是**防注入设计**——只有
date / report_type / broker 三个键能通过，任意其他键一律丢弃。
扩字段必须同时扩校验，否则等于开了一条注入通道。

另一条更隐蔽的风险：**误抽比漏抽危险**。过滤走精确匹配，一次错误的条件
会让检索静默落空或只覆盖一部分，而答案会基于错误的证据范围生成。
"""

import json

from src.engine import (
    _parse_filters_output,
    normalize_metadata_filters,
)


def _payload(**fields) -> str:
    return json.dumps({"metadata": fields}, ensure_ascii=False)


def test_unknown_stock_identifier_is_dropped():
    assert _parse_filters_output(_payload(stock_identifier="600519")) == {}


def test_deleted_industry_and_security_fields_are_dropped():
    """已删除的行业/标的字段不得再进入过滤表达式（防止旧输出回流）。"""
    assert _parse_filters_output(_payload(industry_l1="食品饮料")) == {}
    assert _parse_filters_output(_payload(industry_l2="半导体")) == {}
    assert _parse_filters_output(_payload(security_name="贵州茅台")) == {}


def test_report_type_is_an_enum():
    assert _parse_filters_output(_payload(report_type="行业")) == {"report_type": ["行业"]}
    assert _parse_filters_output(_payload(report_type="策略")) == {"report_type": ["策略"]}
    assert _parse_filters_output(_payload(report_type="点评")) == {}


def test_broker_is_free_text_but_bounded():
    assert _parse_filters_output(_payload(broker="中信证券")) == {"broker": ["中信证券"]}
    assert _parse_filters_output(_payload(broker="   ")) == {}
    assert _parse_filters_output(_payload(broker="券" * 200)) == {}


def test_unknown_field_is_dropped():
    """防注入：任意键不能被透传成 Milvus 过滤表达式。"""
    assert _parse_filters_output(_payload(tenant_id="other")) == {}
    assert _parse_filters_output(_payload(source='x" or id != ""')) == {}


def test_retired_category_is_dropped_and_date_is_preserved():
    parsed = _parse_filters_output(
        json.dumps(
            {"category": ["投研类"], "date": {"gte": "2024"}, "industry_l1": "电子"},
            ensure_ascii=False,
        )
    )

    assert "category" not in parsed
    assert parsed["date"] == {"gte": "2024-01-01"}
    assert "industry_l1" not in parsed


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
    from src.retrieve.hybrid_retriever import HybridRetriever

    assert HybridRetriever._build_filter({"date": ["2026-09-24", "2026-09-27"]}) == (
        'date in ["2026-09-24", "2026-09-27"]'
    )
