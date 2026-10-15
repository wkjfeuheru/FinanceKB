"""研报元数据抽取的回归测试。

这一层的失败模式是**静默漏召**而非报错：过滤走精确匹配，一旦把行业写成
「白酒」而词表里是「白酒Ⅱ」，`industry_l1 == "食品饮料"` 的过滤就会悄悄
少召回一批文档，而分析师不会知道。因此这里重点锁住词表校验行为。

所有用例都显式传 ``enable_llm``，避免测试意外触发真实模型调用。
"""

import pytest

from finance_rag.src.rag.ingestion import metadata_extractor
from finance_rag.src.rag.ingestion.metadata_extractor import (
    apply_metadata,
    extract_metadata,
    load_industry_taxonomy,
)


def _llm(payload: dict):
    """替换 LLM 抽取，让校验逻辑可以被独立测试。"""
    return lambda **_: dict(payload)


def test_filename_regex_extracts_name_and_broker():
    meta = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券-20260815.pdf",
        "贵州茅台2026中报点评",
        "",
        enable_llm=False,
    )

    assert meta.security_name == "贵州茅台"
    assert meta.broker == "中信证券"
    assert meta.report_type == "个股"
    assert meta.meta_source == "regex"


def test_industry_outside_taxonomy_is_dropped(monkeypatch):
    """「白酒」是三级叫法，申万二级是「白酒Ⅱ」——非法值必须丢弃并标记待确认。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"industry_l1": "白酒", "industry_l2": "白酒"}),
    )

    with pytest.raises(ValueError, match="report_type"):
        extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)


def test_industry_l2_must_belong_to_selected_l1(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"report_type": "行业", "industry_l1": "食品饮料", "industry_l2": "半导体"}),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert meta.industry_l1 == "食品饮料"   # 一级合法，保留
    assert meta.industry_l2 == ""            # 二级不属于该一级，丢弃
    assert meta.needs_review is True


def test_valid_industry_passes_and_marks_llm_source(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm(
            {
                "industry_l1": "食品饮料",
                "industry_l2": "白酒Ⅱ",
                "report_type": "行业",
            }
        ),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert (meta.industry_l1, meta.industry_l2) == ("食品饮料", "白酒Ⅱ")
    assert meta.meta_source == "llm"
    assert meta.needs_review is False


def test_regex_values_are_kept_when_llm_returns_nothing(monkeypatch):
    monkeypatch.setattr(metadata_extractor, "_llm_extract", _llm({}))

    meta = extract_metadata(
        "宁德时代(300750)深度报告-华泰证券-20260901.pdf", "", "", enable_llm=True
    )

    assert meta.security_name == "宁德时代"
    assert meta.broker == "华泰证券"
    assert meta.needs_review is False   # 个股不要求行业字段


def test_regex_wins_over_llm_for_deterministic_fields(monkeypatch):
    """文件名能确定性拿到的字段，不允许被模型改写。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"security_name": "另一家公司", "broker": "别的券商"}),
    )

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.security_name == "贵州茅台"
    assert meta.broker == "中信证券"


def test_invalid_report_type_is_rejected(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"report_type": "策略"}),
    )

    with pytest.raises(ValueError, match="report_type"):
        extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)


def test_macro_report_type_comes_from_filename():
    meta = extract_metadata(
        "2026年宏观展望-中金公司-20261101.pdf", "", "", enable_llm=False
    )

    assert meta.report_type == "宏观"
    assert "security_name" not in meta.as_metadata()


def test_industry_report_type_comes_from_filename():
    with pytest.raises(ValueError, match="industry_l1"):
        extract_metadata("白酒行业深度报告-招商证券-20260901.pdf", "", "", enable_llm=False)


def test_disabled_llm_allows_complete_single_stock_metadata():
    """个股只要求简称，不要求行业。"""
    meta = extract_metadata("贵州茅台(600519)点评.pdf", "", "", enable_llm=False)

    assert meta.industry_l1 == ""
    assert meta.meta_source == "regex"
    assert meta.needs_review is False


def test_taxonomy_ships_full_sw2021_enum():
    taxonomy = load_industry_taxonomy()

    assert len(taxonomy) == 31
    assert sum(len(v) for v in taxonomy.values()) == 134
    assert "白酒Ⅱ" in taxonomy["食品饮料"]
    assert "半导体" in taxonomy["电子"]


def test_llm_extraction_failure_degrades_to_regex(monkeypatch):
    def _boom(**_):
        raise RuntimeError("模型不可用")

    monkeypatch.setattr(metadata_extractor, "_llm_extract", _boom)

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.security_name == "贵州茅台"
    assert meta.industry_l1 == ""
    assert meta.meta_source == "regex"


# --- 真实模型返回形状：langchain 的 AIMessage，不是字符串 ----------------------


class _FakeAIMessage:
    """模拟 langchain ``AIMessage``：``str()`` 会带上元数据，``.content`` 才是正文。"""

    def __init__(self, content) -> None:
        self.content = content
        self.additional_kwargs = {"refusal": None}
        self.response_metadata = {"model_name": "qwen-turbo", "finish_reason": "stop"}

    def __str__(self) -> str:  # pragma: no cover - 仅用于复现真实 str() 行为
        return (
            f"content={self.content!r} additional_kwargs={self.additional_kwargs} "
            f"response_metadata={self.response_metadata}"
        )


def test_parse_json_payload_unwraps_langchain_message():
    """实测踩过的坑：把 AIMessage 直接塞进解析器，首尾花括号切出来不是合法 JSON。"""
    message = _FakeAIMessage(
        '{"security_name": "国机汽车", '
        '"report_type": "个股", "broker": "太平洋证券", '
        '"industry_l1": "汽车", "industry_l2": "汽车零部件"}'
    )

    payload = metadata_extractor._parse_json_payload(message)

    assert payload["security_name"] == "国机汽车"
    assert payload["industry_l2"] == "汽车零部件"


def test_parse_json_payload_handles_segmented_content():
    message = _FakeAIMessage([{"type": "text", "text": '{"report_type": "宏观"}'}])

    assert metadata_extractor._parse_json_payload(message) == {"report_type": "宏观"}


def test_llm_extraction_end_to_end_from_message(monkeypatch):
    """整条链路：模型返回消息对象 → 校验后落进元数据。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        lambda **_: metadata_extractor._parse_json_payload(
            _FakeAIMessage(
                '{"security_name": "国机汽车", '
                '"report_type": "个股", "industry_l1": "汽车"}'
            )
        ),
    )

    meta = extract_metadata("H3_AP202609261829925702_1.pdf", "", "首页正文", enable_llm=True)

    assert meta.security_name == "国机汽车"
    assert "industry_l1" not in meta.as_metadata()
    assert meta.meta_source == "llm"
    assert meta.needs_review is False


def test_apply_metadata_merges_without_mutating_input():
    original = {"category": "investment_research", "date": "2026-08-15"}
    extracted = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券.pdf", "", "", enable_llm=False
    )

    merged = apply_metadata(original, extracted)

    assert merged["category"] == "investment_research"   # 原有字段保留
    assert merged["security_name"] == "贵州茅台"
    assert merged["meta_source"] == "regex"
    assert "security_name" not in original               # 不污染入参
def test_llm_receives_pdf_title_when_filename_is_opaque(monkeypatch):
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return {"report_type": "行业", "industry_l1": "通信"}

    monkeypatch.setattr("finance_rag.src.core.config.get_rewrite_model", lambda: Model())
    title = "通信行业：Auto-RAN，意图驱动的6G无线自智网络"
    result = metadata_extractor.extract_metadata("opaque.pdf", title, "空白封面")
    assert title in prompts[0]
    assert result.industry_l1 == "通信"
