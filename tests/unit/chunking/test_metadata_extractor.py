"""研报元数据抽取的回归测试。

这一层的失败模式是**静默漏召**而非报错：过滤走精确匹配，一旦把
report_type / broker 写成检索侧不认的值，过滤就会悄悄少召回一批文档，
而分析师不会知道。因此这里重点锁住字段取值契约，以及
「正则优先、LLM 只补空缺」的确定性次序。

所有用例都显式传 ``enable_llm`` 或替换 ``_llm_extract``，
避免测试意外触发真实模型调用。
"""

import pytest

from src.chunking import metadata_extractor
from src.chunking.metadata_extractor import (
    apply_metadata,
    extract_metadata,
)


def _llm(payload: dict):
    """替换 LLM 抽取，让校验逻辑可以被独立测试。"""
    return lambda **_: dict(payload)


def test_filename_regex_extracts_broker_and_report_type():
    meta = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券-20260815.pdf",
        "贵州茅台2026中报点评",
        "",
        enable_llm=False,
    )

    assert meta.broker == "中信证券"
    assert meta.report_type == "个股"
    assert meta.date == ""  # 发布日期由公共元数据解析负责，文件名日期不作数


def test_llm_fills_fields_that_regex_cannot_determine(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"report_type": "行业", "broker": "国金证券"}),
    )

    meta = extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)

    assert meta.report_type == "行业"
    assert meta.broker == "国金证券"


def test_regex_values_are_kept_when_llm_returns_nothing(monkeypatch):
    monkeypatch.setattr(metadata_extractor, "_llm_extract", _llm({}))

    meta = extract_metadata(
        "宁德时代(300750)深度报告-华泰证券-20260901.pdf", "", "", enable_llm=True
    )

    assert meta.report_type == "个股"
    assert meta.broker == "华泰证券"


def test_regex_wins_over_llm_for_deterministic_fields(monkeypatch):
    """文件名能确定性拿到的字段，不允许被模型改写。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"report_type": "行业", "broker": "别的券商"}),
    )

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.report_type == "个股"
    assert meta.broker == "中信证券"


def test_invalid_report_type_is_rejected(monkeypatch):
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        _llm({"report_type": "点评"}),
    )

    with pytest.raises(ValueError, match="report_type"):
        extract_metadata("研报.pdf", "研报", "正文", enable_llm=True)


def test_macro_report_type_comes_from_filename():
    meta = extract_metadata(
        "2026年宏观展望-中金公司-20261101.pdf", "", "", enable_llm=False
    )

    assert meta.report_type == "宏观"
    # 结果只含契约字段，不允许残留任何旧维度（version 透传，缺省为空串）
    assert set(meta.as_metadata()) == {"category", "date", "version", "report_type", "broker"}


def test_industry_report_type_comes_from_filename():
    """行业研报不再依赖行业词表，类型来自文件名即可入库。"""
    meta = extract_metadata("白酒行业深度报告-招商证券-20260901.pdf", "", "", enable_llm=False)

    assert meta.report_type == "行业"
    assert meta.broker == "招商证券"


def test_strategy_report_type_comes_from_filename():
    meta = extract_metadata("A股投资策略周报-中信证券-20260901.pdf", "", "", enable_llm=False)

    assert meta.report_type == "策略"
    assert meta.broker == "中信证券"


def test_security_code_wins_over_strategy_keyword():
    """证券代码是个股信号，文件名里的「策略」不能把公司点评改成策略报告。"""
    meta = extract_metadata(
        "贵州茅台(600519)投资策略点评-中信证券.pdf", "", "", enable_llm=False
    )

    assert meta.report_type == "个股"


def test_disabled_llm_allows_single_stock_without_broker():
    """broker 允许为空：文件名没有券商名时不得伪造。"""
    meta = extract_metadata("贵州茅台(600519)点评.pdf", "", "", enable_llm=False)

    assert meta.report_type == "个股"
    assert meta.broker == ""


def test_llm_extraction_failure_degrades_to_regex(monkeypatch):
    def _boom(**_):
        raise RuntimeError("模型不可用")

    monkeypatch.setattr(metadata_extractor, "_llm_extract", _boom)

    meta = extract_metadata(
        "贵州茅台(600519)点评-中信证券.pdf", "", "", enable_llm=True
    )

    assert meta.report_type == "个股"
    assert meta.broker == "中信证券"


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
    message = _FakeAIMessage('{"report_type": "个股", "broker": "太平洋证券"}')

    payload = metadata_extractor._parse_json_payload(message)

    assert payload["report_type"] == "个股"
    assert payload["broker"] == "太平洋证券"


def test_parse_json_payload_handles_segmented_content():
    message = _FakeAIMessage([{"type": "text", "text": '{"report_type": "宏观"}'}])

    assert metadata_extractor._parse_json_payload(message) == {"report_type": "宏观"}


def test_llm_extraction_end_to_end_from_message(monkeypatch):
    """整条链路：模型返回消息对象 → 校验后落进元数据。"""
    monkeypatch.setattr(
        metadata_extractor,
        "_llm_extract",
        lambda **_: metadata_extractor._parse_json_payload(
            _FakeAIMessage('{"report_type": "个股", "broker": "太平洋证券"}')
        ),
    )

    meta = extract_metadata("H3_AP202609261829925702_1.pdf", "", "首页正文", enable_llm=True)

    assert meta.report_type == "个股"
    assert meta.broker == "太平洋证券"


def test_apply_metadata_merges_without_mutating_input():
    original = {"date": "2026-08-15"}
    extracted = extract_metadata(
        "贵州茅台(600519)2026中报点评-中信证券.pdf", "", "", enable_llm=False
    )

    merged = apply_metadata(original, extracted)

    # 抽取结果只补空缺：date 由公共解析负责，抽取未产出的空值不得覆盖它
    assert merged == {"category": "", "date": "2026-08-15",
                      "report_type": "个股", "broker": "中信证券"}
    assert original == {"date": "2026-08-15"}  # 不污染入参


def test_llm_receives_pdf_title_when_filename_is_opaque(monkeypatch):
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return {"report_type": "行业", "broker": "中信证券"}

    monkeypatch.setattr("src.core.config.get_rewrite_model", lambda: Model())
    title = "通信行业：Auto-RAN，意图驱动的6G无线自智网络"
    result = metadata_extractor.extract_metadata("opaque.pdf", title, "空白封面")
    assert title in prompts[0]
    assert result.report_type == "行业"
    assert result.broker == "中信证券"
