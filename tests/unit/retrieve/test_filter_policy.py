"""可信过滤计划的用户可见边界。"""

import pytest

def build_filter_plan(*args, **kwargs):
    from src.retrieve.filter_policy import build_filter_plan as actual
    return actual(*args, **kwargs)


def validate_explicit_filters(*args, **kwargs):
    from src.retrieve.filter_policy import validate_explicit_filters as actual
    return actual(*args, **kwargs)


def test_financial_period_is_not_publication_filter():
    plan = build_filter_plan(
        "杰瑞股份2026H1营业收入是多少？",
        inferred_filters={"date": {"gte": "2026-01-01", "lte": "2026-06-30"}},
    )
    assert "date" not in plan.hard_filters
    assert "date" not in plan.tentative_filters
    assert plan.discarded


@pytest.mark.parametrize("filters", [
    {"date": {"gte": "2026-02-31"}},
    {"date": {"gte": "2026-06-01", "lte": "2026-01-01"}},
    {"date": {"gt": "2026-01-01"}},
    {"tenant_id": "other"},
    {"report_type": "点评"},
    {"report_type": "宏观", "security_name": "贵州茅台"},
    {"industry_l1": "食品饮料", "industry_l2": "半导体"},
    ["report_type"],
])
def test_invalid_api_filter_is_rejected(filters):
    with pytest.raises(ValueError):
        validate_explicit_filters(filters)


def test_explicit_none_and_empty_dict_are_distinct():
    assert validate_explicit_filters(None) is None
    assert validate_explicit_filters({}) == {}
    assert build_filter_plan("问题", explicit_filters={}).origin == "explicit"


def test_explicit_range_is_never_relaxed_by_candidate():
    plan = build_filter_plan(
        "查2026年10月3日发布的报告",
        explicit_filters={"date": {"gte": "2026-10-03"}},
        inferred_filters={"date": {"gte": "2026-01-01"}},
    )
    assert plan.hard_filters == {"date": {"gte": "2026-10-03"}}
    assert plan.attempts()[0]["filters"] == plan.hard_filters


def test_unconfirmed_hint_has_hard_only_attempt_even_when_nonempty():
    # 未确认的候选（比较句中的报告类型）进 tentative：先试合并范围，再退回纯硬范围
    plan = build_filter_plan("比较贵州茅台与五粮液的个股研报", inferred_filters={"report_type": "个股"})
    assert plan.tentative_filters == {"report_type": ["个股"]}
    assert plan.attempts() == [
        {
            "reason": "tentative_priority",
            "filters": {"report_type": ["个股"]},
            "dense_only": False,
        },
        {"reason": "hard_only", "filters": {}, "dense_only": False},
    ]


def test_direct_publication_day_and_institution_are_hard():
    plan = build_filter_plan(
        "中信证券2026年10月3日发布的报告",
        inferred_filters={"broker": "中信证券", "date": {"gte": "2026-10-03", "lte": "2026-10-03"}},
    )
    assert plan.hard_filters == {"broker": ["中信证券"], "date": {"gte": "2026-10-03", "lte": "2026-10-03"}}


def test_cited_data_provider_is_not_report_broker():
    plan = build_filter_plan("引用IDC数据的报告", inferred_filters={"broker": "IDC"})
    assert "broker" not in plan.hard_filters


@pytest.mark.parametrize("query,day,broker", [
    ("太平洋证券2026年9月24日国机汽车H1业绩点评中，国机汽车2026H1营业收入是多少？", "2026-09-24", "太平洋证券"),
    ("东吴证券2026年9月27日中国通号点评中，2026年7月至8月中标项目有哪些？", "2026-09-27", "东吴证券"),
    ("南华期货2026年9月12日宏观周报对美国通胀的判断是什么？", "2026-09-12", "南华期货"),
])
def test_institution_day_report_reference_is_hard_even_with_financial_period(query, day, broker):
    plan = build_filter_plan(query, inferred_filters={"date": {"gte": day, "lte": day}, "broker": broker})
    assert plan.hard_filters["date"] == {"gte": day, "lte": day}
    assert plan.hard_filters["broker"] == [broker]


def test_partial_candidate_cannot_exclude_second_report_date_or_publisher():
    plan = build_filter_plan(
        "比较东吴证券2026年9月27日点评与太平洋证券2026年9月24日点评",
        inferred_filters={"broker": "东吴证券", "date": {"gte": "2026-09-27", "lte": "2026-09-27"}},
    )
    assert plan.hard_filters["broker"] == ["东吴证券", "太平洋证券"]
    assert plan.hard_filters["date"] == ["2026-09-24", "2026-09-27"]
    assert plan.tentative_filters["broker"] == ["东吴证券"]
    assert plan.tentative_filters["date"] == {"gte": "2026-09-27", "lte": "2026-09-27"}
    assert plan.attempts()[-1]["filters"] == plan.hard_filters


def test_report_publication_day_is_distinct_from_financial_as_of_day():
    plan = build_filter_plan(
        "东吴证券2026年9月27日点评中，截至2026年6月30日的收入是多少？",
        inferred_filters={"date": {"gte": "2026-09-27", "lte": "2026-09-27"}},
    )
    assert plan.hard_filters["date"] == {"gte": "2026-09-27", "lte": "2026-09-27"}


def test_mixed_publication_and_financial_boundary_is_discarded_while_publication_day_stays_hard():
    plan = build_filter_plan(
        "中信证券2026年10月3日发布的报告给出的截至2026年6月30日营业收入是多少？",
        inferred_filters={"date": {"gte": "2026-06-30", "lte": "2026-10-03"}},
    )
    assert plan.hard_filters["date"] == {"gte": "2026-10-03", "lte": "2026-10-03"}
    assert "date" not in plan.tentative_filters
    assert any(item["field"] == "date" for item in plan.discarded)


def test_missing_candidate_does_not_drop_unambiguous_publication_day():
    plan = build_filter_plan("中信证券2026年10月3日发布的报告", inferred_filters={})
    assert plan.hard_filters["date"] == {"gte": "2026-10-03", "lte": "2026-10-03"}
    assert plan.hard_filters["broker"] == ["中信证券"]


@pytest.mark.parametrize("date_filter", [
    ["2026-02-31"],
    ["2026-09-24", "2026-02-31"],
    ["2026-09-24", 20260927],
])
def test_explicit_exact_date_set_rejects_invalid_members(date_filter):
    with pytest.raises(ValueError):
        validate_explicit_filters({"date": date_filter})


def test_explicit_exact_date_set_is_canonical_and_distinct_from_range():
    assert validate_explicit_filters({"date": ["2026-09-27", "2026-09-24", "2026-09-27"]}) == {
        "date": ["2026-09-27", "2026-09-24"]
    }
    assert validate_explicit_filters({"date": {"gte": "2026-09-24", "lte": "2026-09-27"}}) == {
        "date": {"gte": "2026-09-24", "lte": "2026-09-27"}
    }
