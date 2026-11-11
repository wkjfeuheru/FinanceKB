"""补检改写的范围守卫回归（RAG-09）。

``docs/rag-review.md`` 的失败模式：``item.entity`` / ``item.period`` 为空时，
「贵州茅台2025年营业收入」被改写成「营业收入」也会通过；期间正则又只禁止新增，
所以「2025年 → 2026年」以外的期间删除同样漏过。硬过滤未覆盖时会跨主体、
跨期间召回。

验收标准：
* 结构化实体、期间为空时，上述示例仍不能退化为「营业收入」；
* 删除或更换公司、年份、季度、指标的改写均被拒绝或回退；
* 合法同义词改写仍可执行（受控别名映射），原请求范围不变。
"""

from __future__ import annotations

import pytest

from src.agent.planning import Subproblem, required_scope, rewrite_missing


class _RewriteModel:
    """模型替身：把探针改写原样作为 JSON 返回，隔离范围守卫本身。"""

    def __init__(self, proposed: str) -> None:
        self.proposed = proposed

    def invoke(self, prompt: str) -> str:
        del prompt
        return '{"query":%s}' % _json(self.proposed)


def _json(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _rewrite(item_query: str, proposed: str, *, entity=None, period=None, metric=None) -> str:
    item = Subproblem(
        id="q1", query=item_query, entity=entity, period=period, metric=metric
    )
    return rewrite_missing(item, [], model=_RewriteModel(proposed))


# ---------------------------------------------------------------------------
# 结构化字段为空：范围约束仍必须成立
# ---------------------------------------------------------------------------


def test_structured_fields_empty_still_rejects_scope_reduction():
    """结构化字段为空时不得退化为「营业收入」（本次修复的核心案例）。"""
    assert _rewrite("贵州茅台2025年营业收入", "营业收入") == "贵州茅台2025年营业收入"


@pytest.mark.parametrize(
    "proposed",
    [
        "营业收入",                  # 丢弃主体与期间
        "贵州茅台营业收入",            # 丢弃年份
        "贵州茅台2026年营业收入",       # 更换年份
        "五粮液2025年营业收入",         # 更换主体
        "贵州茅台2025年净利润",         # 更换指标
        "贵州茅台2025年营业收入与净利润",  # 新增指标
    ],
)
def test_scope_changes_fall_back_to_original(proposed):
    assert _rewrite("贵州茅台2025年营业收入", proposed) == "贵州茅台2025年营业收入"


def test_quarter_swap_within_same_year_is_rejected():
    """同年内的季度偷换（Q1 → Q3）必须回退：期间正则只查年份是不够的。"""
    assert _rewrite("贵州茅台2026Q1营业收入", "贵州茅台2026Q3营业收入") == "贵州茅台2026Q1营业收入"
    assert _rewrite("贵州茅台2026H1毛利率", "贵州茅台2026H2毛利率") == "贵州茅台2026H1毛利率"


# ---------------------------------------------------------------------------
# 合法同义词改写仍可执行（受控别名）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item_query,proposed",
    [
        ("贵州茅台2025年营业收入", "贵州茅台2025年营收"),
        ("贵州茅台2025年营业收入", "贵州茅台2025年营业总收入 财报"),
        ("贵州茅台2025年营业收入", "贵州茅台2025年度营业收入是多少"),
        ("宁德时代2024年净利润", "宁德时代2024年归母净利润 业绩"),
        ("贵州茅台2026H1毛利率", "贵州茅台2026年中报销售毛利率"),
        ("贵州茅台2026Q1评级", "贵州茅台2026年第一季度投资评级"),
        ("协和电子2024年营业收入", "协和电子2024年营业收入是多少"),  # 公司名含「和」
    ],
)
def test_controlled_alias_rewrites_are_accepted(item_query, proposed):
    """受控别名映射内的改写必须能通过，否则补检退化为「永远用原查询」。"""
    assert _rewrite(item_query, proposed) == proposed


def test_subject_token_with_join_word_is_not_split():
    """「协和电子」里的「和」不是连接词：主体约束必须按整名保留。"""
    assert _rewrite("协和电子2024年营业收入", "电子2024年营业收入") == "协和电子2024年营业收入"


def test_structured_fields_are_added_to_text_constraints():
    """结构化字段即使与文本不同，也必须进入约束集合（并集语义）。"""
    constraints = dict(required_scope(
        Subproblem(id="q1", query="该公司2025年营业收入", entity="贵州茅台", period="2025年")
    ))
    assert "贵州茅台" in constraints
    assert "营业收入" in constraints
    assert "2025" in constraints
    # 结构化主体被丢弃 → 回退
    assert _rewrite(
        "该公司2025年营业收入", "该公司2025年营业收入 详细", entity="贵州茅台"
    ) == "该公司2025年营业收入"


def test_model_without_rewrite_returns_original_item_query():
    """无模型时不改写（保持既有契约）。"""
    item = Subproblem(id="q1", query="贵州茅台2025年营业收入")
    assert rewrite_missing(item, [], model=None) == item.query


def test_unparsable_model_output_falls_back():
    class Broken:
        def invoke(self, prompt):
            return "不是 JSON"

    item = Subproblem(id="q1", query="贵州茅台2025年营业收入")
    assert rewrite_missing(item, [], model=Broken()) == item.query
