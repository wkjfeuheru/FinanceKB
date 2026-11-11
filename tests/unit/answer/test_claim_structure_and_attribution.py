"""列表结构解析（RAG-12）与来源归属核验（RAG-13）回归。

RAG-12：``1. 营业收入100亿元[1]。`` 曾被切成无引用声明 ``1`` 与正常事实，
核验必然失败，重试仍用编号列表时最终拒答。
RAG-13：Prompt 要求跨报告结论注明报告日期，但确定性数值检查要求日期出现在
正文引文里；日期只存在于元数据时，正确回答被误拒。
"""

from __future__ import annotations

from src.answer import citations as validator


def _claims(answer: str) -> list[tuple[str, list[int]]]:
    return [(item["claim"], item["citation_indices"]) for item in validator._answer_claims(answer)]


# ---------------------------------------------------------------------------
# RAG-12：列表标记不是事实
# ---------------------------------------------------------------------------


def test_ordered_list_marker_is_not_a_claim():
    """行首编号不再是事实声明（修复前会多出一条无引用声明 ``1``）。"""
    assert _claims("1. 营业收入100亿元[1]。") == [("营业收入100亿元", [1])]


def test_paragraph_and_list_forms_verify_consistently():
    """同一事实在普通段落、有序列表、无序列表中的核验输入必须一致。"""
    paragraph = _claims("营业收入100亿元，同比增长12%[1]。")
    ordered = _claims("1. 营业收入100亿元，同比增长12%[1]。")
    unordered = _claims("- 营业收入100亿元，同比增长12%[1]。")

    assert paragraph == ordered == unordered


def test_multiple_list_items_keep_their_own_citations():
    """多行列表逐条解析，各自绑定引用编号。"""
    assert _claims("1. 营业收入100亿元[1]。\n2. 净利润50亿元[2]。\n") == [
        ("营业收入100亿元", [1]),
        ("净利润50亿元", [2]),
    ]


def test_nested_and_chinese_markers_are_stripped():
    """嵌套缩进与中文顿号编号同样是列表标记，不是事实。"""
    assert _claims("  1. 营业收入100亿元[1]。") == [("营业收入100亿元", [1])]
    assert _claims("1、营业收入100亿元[1]。") == [("营业收入100亿元", [1])]
    assert _claims("· 营业收入100亿元[1]。") == [("营业收入100亿元", [1])]


def test_year_and_amount_leading_facts_are_not_damaged():
    """``2025年…``、``1.5亿元…`` 不因清理列表前缀而丢失事实数字。"""
    assert _claims("2025年营业收入100亿元[1]。") == [("2025年营业收入100亿元", [1])]
    assert _claims("1.5亿元营业收入[1]。") == [("1.5亿元营业收入", [1])]
    assert _claims("2026H1 营业收入100亿元[1]。") == [("2026H1 营业收入100亿元", [1])]


def test_uncited_fact_after_list_marker_still_fails():
    """真正无引用的列表事实仍不能通过门禁。"""
    claims = _claims("1. 营业收入100亿元。")

    assert claims == [("营业收入100亿元", [])]


def test_list_items_inherit_citation_from_leading_sentence():
    """引导句已引用时，紧随的无编号列表项沿用该编号。"""
    answer = "横店东磁主要业务包括磁性材料、光伏和锂电[1]。\n- 磁性材料\n- 光伏\n- 锂电"

    assert _claims(answer) == [
        ("横店东磁主要业务包括磁性材料、光伏和锂电", [1]),
        ("磁性材料", [1]),
        ("光伏", [1]),
        ("锂电", [1]),
    ]


def test_plain_sentence_does_not_inherit_previous_citation():
    """普通句子不因上一条已引用而免检。"""
    answer = "横店东磁主要业务包括磁性材料[1]。\n- 磁性材料\n公司还布局了其他方向。"

    assert _claims(answer) == [
        ("横店东磁主要业务包括磁性材料", [1]),
        ("磁性材料", [1]),
        ("公司还布局了其他方向", []),
    ]


def test_blank_line_blocks_list_citation_inheritance():
    """空行断开列表继承，后续列表项必须自带引用。"""
    answer = "主要包括磁性材料[1]。\n\n- 光伏"

    assert _claims(answer) == [
        ("主要包括磁性材料", [1]),
        ("光伏", []),
    ]


def test_colon_lead_in_inherits_citations_from_following_list():
    """引用写在列表项上时，冒号引导句沿用这些编号，空行不影响。"""
    answer = (
        "横店东磁当前的主要业务覆盖以下四大板块：\n\n"
        "- **磁材**：永磁铁氧体 [1][2]。\n\n"
        "- **光伏**：高效电池 [1]。"
    )

    assert _claims(answer) == [
        ("横店东磁当前的主要业务覆盖以下四大板块：", [1, 2]),
        ("**磁材**：永磁铁氧体", [1, 2]),
        ("**光伏**：高效电池", [1]),
    ]


def test_list_item_clauses_share_the_items_citations():
    """同一列表项里，分号前的分句沿用该项末尾的引用。"""
    answer = (
        "业务覆盖如下：\n\n"
        "- 磁材以铁氧体为核心；主要应用于家电和汽车 [1][2]。"
    )

    assert _claims(answer) == [
        ("业务覆盖如下：", [1, 2]),
        ("磁材以铁氧体为核心", [1, 2]),
        ("主要应用于家电和汽车", [1, 2]),
    ]


def test_nested_list_item_shares_citation_written_on_its_last_subitem():
    """编号项的标题和子项没有编号时，沿用该编号项最后一条子项上的引用。"""
    answer = (
        "横店东磁当前主要业务覆盖以下四大板块，具体如下：\n\n"
        "1. **磁材业务**\n"
        "   - 核心产品：永磁铁氧体\n"
        "   - 主要应用/市场：家电 [1][2]\n"
        "2. **器件业务**\n"
        "   - 核心产品：电感\n"
        "   - 主要应用：服务器 [1][3]\n"
    )

    assert _claims(answer) == [
        ("横店东磁当前主要业务覆盖以下四大板块，具体如下：", [1, 2, 3]),
        ("**磁材业务**", [1, 2]),
        ("核心产品：永磁铁氧体", [1, 2]),
        ("主要应用/市场：家电", [1, 2]),
        ("**器件业务**", [1, 3]),
        ("核心产品：电感", [1, 3]),
        ("主要应用：服务器", [1, 3]),
    ]


def test_flat_list_items_do_not_take_a_later_items_citation():
    """同一缩进的平级列表项不能借用下一条的引用。"""
    assert _claims("- 营业收入100亿元\n- 净利润50亿元[1]。") == [
        ("营业收入100亿元", []),
        ("净利润50亿元", [1]),
    ]


def test_quote_accepts_table_cells_joined_by_newline_and_ellipsis():
    """换行拼起来的单元格、以及省略号两侧的原文，仍算逐字引文。"""
    table = "<td>磁材</td><td>永磁铁氧体、软磁铁氧体、塑磁、金属软磁粉 芯</td><td>家电、汽车、新能源</td>"
    prose = "公司拥有约 23GW电池片及21GW组件产能，2026H1收入 73.7亿元；全黑、防眩光组件具备相对溢价"

    assert validator._quote_in_source(
        1,
        "磁材：永磁铁氧体、软磁铁氧体、塑磁、金属软磁粉 芯\n家电、汽车、新能源",
        [{"content": table}],
    )
    assert validator._quote_in_source(
        1,
        "公司拥有约 23GW电池片及21GW组件产能……全黑、防眩光组件具备相对溢价",
        [{"content": prose}],
    )
    assert (
        validator._quote_in_source(
            1,
            "公司聚焦18650小圆柱电池……并逐步向户储及工商业储能拓展",
            [{"content": "公司聚焦18650小圆柱电池，2026H1实现收入17.5亿元。"}],
        )
        is False
    )


def test_factual_sentence_before_list_does_not_inherit_forward():
    """不以冒号结尾的事实句不能借后面列表的引用。"""
    answer = "营业收入100亿元。\n\n- 净利润50亿元[1]。"

    assert _claims(answer) == [
        ("营业收入100亿元", []),
        ("净利润50亿元", [1]),
    ]


# ---------------------------------------------------------------------------
# RAG-13：正文事实与来源归属分开核验
# ---------------------------------------------------------------------------


def _doc(**overrides) -> dict:
    # security_name 已从元数据体系移除，主体一致性只靠正文引文本身核验
    doc = {
        "content": "贵州茅台2026H1营业收入100亿元，同比增长12%。",
        "date": "2026-08-15",
        "start_page": 4,
        "end_page": 5,
    }
    doc.update(overrides)
    return doc


_FULL_QUOTE = "贵州茅台2026H1营业收入100亿元，同比增长12%"


def _check(claim: str, indices=None, quote: str | None = None, item_id: str | None = None) -> validator.ClaimCheck:
    # 生产链路里模型回显的 claim 不含引用标记（``_answer_claims`` 已剥离），
    # 这里保持同样的形态，避免把 ``[1]`` 里的数字混进数值检查。
    return validator.ClaimCheck(
        claim=claim,
        citation_indices=indices or [1],
        status="supported",
        supporting_quote=quote if quote is not None else _FULL_QUOTE,
        reason="原文支持",
        item_id=item_id,
    )


def test_report_date_only_in_metadata_passes():
    """元数据日期正确、正文不含日期的合规引用必须通过（修复前被误拒）。"""
    check = _check("贵州茅台2026H1营业收入100亿元，同比增长12%（2026-08-15）。")

    assert validator._check_quotes(check, [_doc()]) is True


def test_wrong_or_fabricated_report_date_fails():
    """换成另一份报告的日期或编造日期必须失败；显式「报告日期：」同样受约束。"""
    assert validator._check_quotes(
        _check("营业收入100亿元（2025-04-28）。", quote="营业收入100亿元"), [_doc()]
    ) is False
    assert validator._check_quotes(
        _check("营业收入100亿元（2026-08-16）。", quote="营业收入100亿元"), [_doc()]
    ) is False
    assert validator._check_quotes(
        _check("营业收入100亿元，报告日期：2026-08-15。", quote="营业收入100亿元"), [_doc()]
    ) is True
    assert validator._check_quotes(
        _check("营业收入100亿元，报告日期：2026-01-01。", quote="营业收入100亿元"), [_doc()]
    ) is False


def test_page_attribution_is_checked_against_source_metadata():
    """页码必须落在被引用来源的页码区间内；错误页码或凭空页码都失败。"""
    assert validator._check_quotes(
        _check("营业收入100亿元，详见第5页。", quote="营业收入100亿元"), [_doc()]
    ) is True
    assert validator._check_quotes(
        _check("营业收入100亿元，详见第9页。", quote="营业收入100亿元"), [_doc()]
    ) is False
    # 来源没有页码元数据时不得声称具体页码
    assert validator._check_quotes(
        _check("营业收入100亿元，详见第4页。", quote="营业收入100亿元"),
        [_doc(start_page=0, end_page=0)],
    ) is False


def test_date_support_does_not_loosen_amount_checks():
    """增加日期支持不得放宽金额、单位、期间与增长方向检查。"""
    # 金额不同 → 失败
    assert validator._check_quotes(
        _check("营业收入120亿元（2026-08-15）。", quote="营业收入100亿元"), [_doc()]
    ) is False
    # 期间不同 → 失败（2025 年在引文中不存在）
    assert validator._check_quotes(
        _check("贵州茅台2025年营业收入100亿元（2026-08-15）。"), [_doc()]
    ) is False
    # 增长方向相反 → 失败
    assert validator._check_quotes(
        _check("营业收入100亿元，同比下降12%（2026-08-15）。", quote=_FULL_QUOTE), [_doc()]
    ) is False
    # 主体不同 → 失败
    assert validator._check_quotes(
        _check("五粮液公司营业收入100亿元（2026-08-15）。", quote="营业收入100亿元"), [_doc()]
    ) is False


def test_bare_date_in_body_stays_a_fact_not_attribution():
    """正文里的裸日期是期间事实，不能当归属信息放行。"""
    fact, attributions = validator._split_attributions(
        "2026年8月15日发布的中报显示营业收入100亿元[1]。"
    )

    assert attributions == []
    assert "2026年8月15日" in fact


def test_attribution_dates_do_not_pollute_scope_periods():
    """归属日期不参与「已支撑事项」的作用域判断（否则合法答案被判越界）。"""
    check = _check(
        "贵州茅台2025年营业收入100亿元（2026-08-15）。",
        quote="贵州茅台2025年营业收入100亿元",
        item_id="q1",
    )
    allowed = [{"id": "q1", "query": "贵州茅台2025年营业收入", "entity": "贵州茅台",
                "metric": "营业收入", "period": "2025年"}]

    issue = validator._restriction_issue(check, allowed, [])

    assert issue is None
