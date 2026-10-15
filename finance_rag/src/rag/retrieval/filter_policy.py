"""将显式范围和自动候选分开处理，生成可审计的有界检索计划。"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from finance_rag.src.rag.ingestion.metadata_extractor import (
    load_industry_taxonomy,
    metadata_field_validators,
)
from finance_rag.src.rag.ingestion.public_metadata import _ALIASES as _PUBLIC_BROKER_ALIASES
from finance_rag.src.rag.ingestion.public_metadata import _BROKERS as _KNOWN_BROKERS
from finance_rag.src.rag.models.document_category import DOCUMENT_CATEGORIES


_FIELDS = frozenset({"category", "date", "security_name", "industry_l1", "industry_l2", "report_type", "broker"})
_CATEGORY_LABELS = {label: value for value, label in DOCUMENT_CATEGORIES.items()}
_CATEGORY_ALIASES = {
    "investment_research": ("投研", "研报", "研究报告"),
    "compliance_risk": ("合规", "风控", "监管政策"),
    "business_operations": ("业务运营",),
    "management": ("管理类",),
}
_BROKER_ALIASES = _PUBLIC_BROKER_ALIASES
_DATE_TEXT = re.compile(r"(?<!\d)(20\d{2})[年/._-](\d{1,2})[月/._-](\d{1,2})日?(?!\d)")
_PUBLICATION_WORD = re.compile(r"(?:发布|发表|出具|刊发|出版|报告日期|研报日期|发布时间)")
_REPORT_WORD = re.compile(r"(?:报告|研报|研究报告|点评|周报)")
_COMPARISON = re.compile(r"(?:比较|对比|与|和|及|、|两家|多家|分别|各自)")
_COMPANY_JOIN = re.compile(r"(?:[、，,]|与|和|及|对比)")
_FINANCIAL_METRIC_START = re.compile(
    r"^(?:营业收入|营收|收入|归母净利润|扣非净利润|净利润|利润总额|营业利润|利润|"
    r"同比变化|同比增长|同比|环比|毛利率|净利率|资产负债率|总资产|净资产|"
    r"经营现金流|现金流|增长率|增速|每股收益|市盈率|市净率|销量|订单|成本|费用|评级|目标价)"
)
_FINANCIAL_PERIOD_START = re.compile(r"^(?:20\d{2}(?:年|H[12]|Q[1-4])|上半年|下半年|去年|本年|一季度|二季度|三季度|四季度)")
_CATEGORY_TOPIC_BOUNDARY = re.compile(r"(文件|文档|报告|研报)(?:中(?:的|关于|有关|涉及|提及)|关于|有关|涉及|提及)")
# “涉及/提及”属于主题边界，词内的“及”不能先被并列分句拆开。
_CATEGORY_SCOPE_JOIN = re.compile(r"(?:与|和|(?<![涉提])及|[、，,])")
_CATEGORY_DOCUMENT_TARGET = re.compile(r"(?:文件|文档|报告|研报)(?:中|里|内|$)")
_PUBLICATION_AFTER_DATE = re.compile(r"^(?:期间)?(?:发布|发表|出具|刊发|出版)的?(?:报告|研报|研究报告)")
_DATE_INTERVAL_LINK = re.compile(r"^\s*(?:至|到|—|－|~|～)\s*$")


def _date_value(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        raise ValueError("date 必须是完整的 YYYY-MM-DD 日期")
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        raise ValueError("date 不是有效日历日期") from exc


def _normalized_values(field_name: str, value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list) and value:
        raw = value
    else:
        raise ValueError(f"{field_name} 必须是非空字符串或非空字符串数组")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise ValueError(f"{field_name} 包含非法值")
    validator = metadata_field_validators().get(field_name)
    normalized = []
    for item in raw:
        candidate = item.strip()
        if field_name == "category":
            candidate = _CATEGORY_LABELS.get(candidate, candidate)
            valid = candidate in DOCUMENT_CATEGORIES
        else:
            valid = bool(validator and validator(candidate))
        if not valid:
            raise ValueError(f"{field_name} 包含非法值：{candidate}")
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def validate_explicit_filters(filters: dict | None) -> dict | None:
    """严格校验 API 显式过滤；任何非法键或值均不能静默丢失。"""
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ValueError("filters 必须是对象")
    if unknown := set(filters) - _FIELDS:
        raise ValueError(f"未知过滤字段：{', '.join(sorted(map(str, unknown)))}")
    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if key != "date":
            normalized[key] = _normalized_values(key, value)
            continue
        if isinstance(value, list):
            if not value:
                raise ValueError("date 精确日期集合不能为空")
            normalized[key] = list(dict.fromkeys(_date_value(item) for item in value))
            continue
        if not isinstance(value, dict) or not value or set(value) - {"gte", "lte"}:
            raise ValueError("date 必须包含 gte 或 lte")
        dates = {op: _date_value(part) for op, part in value.items()}
        if dates.get("gte") and dates.get("lte") and dates["gte"] > dates["lte"]:
            raise ValueError("date 的 gte 不能晚于 lte")
        normalized[key] = dates
    types = set(normalized.get("report_type", []))
    if types:
        if "security_name" in normalized and "个股" not in types:
            raise ValueError("security_name 只能用于个股报告")
        if ("industry_l1" in normalized or "industry_l2" in normalized) and "行业" not in types:
            raise ValueError("行业字段只能用于行业报告")
    if "security_name" in normalized and any(key in normalized for key in ("industry_l1", "industry_l2")) and not {"个股", "行业"} <= types:
        raise ValueError("跨类型过滤必须同时指定个股和行业报告")
    if "industry_l1" in normalized and "industry_l2" in normalized:
        taxonomy = load_industry_taxonomy()
        allowed = {child for parent in normalized["industry_l1"] for child in taxonomy.get(parent, ())}
        if any(child not in allowed for child in normalized["industry_l2"]):
            raise ValueError("industry_l2 不属于所选 industry_l1")
    return normalized


@dataclass
class FilterPlan:
    origin: str
    hard_filters: dict[str, Any] = field(default_factory=dict)
    tentative_filters: dict[str, Any] = field(default_factory=dict)
    discarded: list[dict[str, Any]] = field(default_factory=list)
    explicit_filters: dict[str, Any] | None = None
    inferred_filters: dict[str, Any] | None = None

    def attempts(self) -> list[dict[str, Any]]:
        if self.tentative_filters:
            return [
                {"reason": "tentative_priority", "filters": {**copy.deepcopy(self.hard_filters), **copy.deepcopy(self.tentative_filters)}, "dense_only": False},
                {"reason": "hard_only", "filters": copy.deepcopy(self.hard_filters), "dense_only": False},
            ]
        return [{"reason": "hard_only", "filters": copy.deepcopy(self.hard_filters), "dense_only": False}]

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin, "explicit": copy.deepcopy(self.explicit_filters),
            "inferred": copy.deepcopy(self.inferred_filters),
            "hard": copy.deepcopy(self.hard_filters),
            "tentative": copy.deepcopy(self.tentative_filters),
            "discarded": copy.deepcopy(self.discarded),
        }


def _publication_dates(query: str) -> set[str]:
    grounded_dates = []
    for match in _DATE_TEXT.finditer(query):
        try:
            parsed = date(int(match[1]), int(match[2]), int(match[3])).isoformat()
        except ValueError:
            continue
        # 完整日历日紧邻报告称谓，或与发布动词相连，才是报告日期。
        # 财务 H1 / 季度和正文里的其他时间不会因同句出现“报告”而升级。
        before = query[max(0, match.start() - 16):match.start()]
        after = re.split(r"[，,。；;？?]", query[match.end():match.end() + 30], maxsplit=1)[0]
        if re.search(r"(?:截至|截止|报告期末|期末)\s*$", before):
            continue
        publication_before = re.search(r"(?:发布于?|发表于?|出具于?|报告日期|研报日期|发布时间)\s*[:：]?\s*$", before)
        report_reference = re.search(r"(?:点评|周报|研报|研究报告)", after)
        institution_report = re.search(r"(?:证券|期货|研究院|研究所|公司).{0,12}$", before) and re.search(r"^(?:的)?报告", after)
        if publication_before or _PUBLICATION_WORD.search(after[:8]) or report_reference or institution_report:
            grounded_dates.append(parsed)
    return set(grounded_dates)


def _publication_range(query: str) -> dict[str, str] | None:
    """只识别日期与报告发布动作明确相连的上界或双端区间。"""
    dates = list(_DATE_TEXT.finditer(query))
    intervals: list[dict[str, str]] = []
    for first, second in zip(dates, dates[1:]):
        between = query[first.end():second.start()]
        after = query[second.end():second.end() + 20]
        if _DATE_INTERVAL_LINK.fullmatch(between) and _PUBLICATION_AFTER_DATE.match(after):
            try:
                lower = date(int(first[1]), int(first[2]), int(first[3])).isoformat()
                upper = date(int(second[1]), int(second[2]), int(second[3])).isoformat()
            except ValueError:
                continue
            if lower <= upper:
                intervals.append({"gte": lower, "lte": upper})
    grounded = _publication_dates(query)
    if len(intervals) == 1 and grounded <= {intervals[0]["lte"]}:
        return intervals[0]

    deadlines = []
    for match in dates:
        before = query[max(0, match.start() - 4):match.start()]
        after = query[match.end():match.end() + 20]
        if re.search(r"(?:截至|截止)\s*$", before) and _PUBLICATION_AFTER_DATE.match(after):
            try:
                deadlines.append(date(int(match[1]), int(match[2]), int(match[3])).isoformat())
            except ValueError:
                continue
    if len(deadlines) == 1 and not grounded:
        return {"lte": deadlines[0]}
    return None


def _date_scope(query: str, bounds: dict[str, str] | list[str]) -> str:
    publication_range = _publication_range(query)
    if publication_range is not None:
        return "hard" if bounds == publication_range else "discard"
    grounded = _publication_dates(query)
    if not grounded:
        return "discard"
    if isinstance(bounds, list):
        chosen = set(bounds)
        return "hard" if chosen == grounded else "tentative" if chosen <= grounded else "discard"
    if not set(bounds.values()) <= grounded:
        return "discard"
    lower, upper = bounds.get("gte"), bounds.get("lte")
    if not any((lower is None or lower <= day) and (upper is None or day <= upper) for day in grounded):
        return "discard"
    if all((lower is None or lower <= day) and (upper is None or day <= upper) for day in grounded):
        return "hard"
    return "tentative"


def _mentioned_publishers(query: str) -> set[str]:
    names = set()
    for name in sorted(_KNOWN_BROKERS, key=len, reverse=True):
        start = query.find(name)
        if start < 0:
            continue
        before = query[max(0, start - 8):start]
        if re.search(r"(?:引用|参考|采用|根据)\s*$", before):
            continue
        after = re.split(r"[，,。；;？?]", query[start + len(name):start + len(name) + 35], maxsplit=1)[0]
        if _REPORT_WORD.search(after) or _PUBLICATION_WORD.search(after):
            names.add(_BROKER_ALIASES.get(name, name))
    return names


def _mentioned_categories(query: str) -> list[str]:
    scoped_parts = []
    topic_seen = False
    # 各并列片段独立取文档目标；前一片段的主题不能吞掉后续明确的文件范围。
    for clause in _CATEGORY_SCOPE_JOIN.split(query):
        topic_boundary = _CATEGORY_TOPIC_BOUNDARY.search(clause)
        if topic_boundary:
            scoped_parts.append(clause[:topic_boundary.start() + len(topic_boundary[1])])
            topic_seen = True
        elif not topic_seen or _CATEGORY_DOCUMENT_TARGET.search(clause):
            scoped_parts.append(clause)
    query = " ".join(scoped_parts)
    return [value for value, label in DOCUMENT_CATEGORIES.items()
            if value in query or label in query or any(alias in query for alias in _CATEGORY_ALIASES[value])]


def _has_unconfirmed_second_company(query: str, names: list[str]) -> bool:
    """只把连接词后无法确认为同一公司指标/期间的对象视为多实体。"""
    if len(names) == 1 and re.search(r"两家|多家", query):
        return True
    for name in names:
        for occurrence in re.finditer(re.escape(name), query):
            for join in _COMPANY_JOIN.finditer(query, occurrence.end()):
                following = query[join.end():].lstrip()
                if not following:
                    continue
                if any(following.startswith(known) for known in names):
                    continue
                if _FINANCIAL_METRIC_START.match(following) or _FINANCIAL_PERIOD_START.match(following):
                    continue
                return True
            prefix = query[:occurrence.start()]
            joins = list(_COMPANY_JOIN.finditer(prefix))
            if joins:
                left = prefix[:joins[-1].start()].split("，")[-1].split(",")[-1].split("、")[-1]
                if left and not any(known in left for known in names) and not _FINANCIAL_METRIC_START.search(left):
                    return True
    return False


def _scope_for_candidate(query: str, key: str, values: Any) -> str:
    """返回 hard/tentative/discard：不从候选本身推造用户范围。"""
    if key == "date":
        return _date_scope(query, values)
    if key == "broker":
        names = values
        mentioned = all(name in query or any(alias in query for alias, canonical in _BROKER_ALIASES.items() if canonical == name) for name in names)
        if not mentioned:
            return "discard"
        publishers = _mentioned_publishers(query)
        if not set(names) <= publishers:
            return "discard" if not publishers else "tentative"
        return "hard" if publishers <= set(names) else "tentative"
    if key in {"security_name", "industry_l1", "industry_l2"}:
        if not all(value in query for value in values):
            return "discard"
        if key == "security_name":
            if _has_unconfirmed_second_company(query, values) or "行业" in query:
                return "tentative"
            return "hard"
        if _COMPARISON.search(query):
            return "tentative"
        if key.startswith("industry_") and ("公司" in query or "个股" in query):
            return "tentative"
        return "hard"
    if key == "report_type":
        if not all(value in query or (value == "个股" and "公司" in query) for value in values):
            return "discard"
        return "tentative" if _COMPARISON.search(query) or ("行业" in query and "公司" in query) else "hard"
    if key == "category":
        mentioned = set(_mentioned_categories(query))
        if not set(values) <= mentioned:
            return "discard"
        return "hard" if set(values) == mentioned else "tentative"
    return "discard"


def build_filter_plan(
    query: str, *, explicit_filters: dict | None = None, inferred_filters: dict | None = None,
) -> FilterPlan:
    """从原问句审查自动候选；显式 filters 总是最终硬约束。"""
    explicit = validate_explicit_filters(explicit_filters)
    if explicit is not None:
        return FilterPlan("explicit", copy.deepcopy(explicit), explicit_filters=explicit,
                          inferred_filters=copy.deepcopy(inferred_filters))
    plan = FilterPlan("inferred" if inferred_filters is not None else "none",
                      inferred_filters=copy.deepcopy(inferred_filters))
    if inferred_filters is None:
        return plan
    if not isinstance(inferred_filters, dict):
        plan.discarded.append({"field": "*", "value": inferred_filters, "reason": "invalid_object"})
        return plan
    for key, value in inferred_filters.items():
        try:
            normalized = validate_explicit_filters({key: value})
        except ValueError as exc:
            plan.discarded.append({"field": key, "value": value, "reason": str(exc)})
            continue
        scope = _scope_for_candidate(query, key, normalized[key])
        if scope == "discard":
            plan.discarded.append({"field": key, "value": value, "reason": "not_grounded_in_query"})
        elif scope == "hard":
            plan.hard_filters[key] = normalized[key]
        else:
            plan.tentative_filters[key] = normalized[key]
    # 候选字段各自合法仍可能组合非法；此时只保留已确定的共同硬范围。
    try:
        validate_explicit_filters({**plan.hard_filters, **plan.tentative_filters})
    except ValueError:
        for key, value in list(plan.tentative_filters.items()):
            plan.discarded.append({"field": key, "value": value, "reason": "incompatible_combination"})
        plan.tentative_filters.clear()
    # 明确的报告发布时间与发布机构可直接从原问句确定；LLM 漏列或混入财务日期不应放宽此范围。
    publication_range = _publication_range(query)
    publication_days = _publication_dates(query)
    if publication_range is not None:
        plan.tentative_filters.pop("date", None)
        plan.hard_filters["date"] = publication_range
    elif len(publication_days) == 1:
        day = next(iter(publication_days))
        plan.tentative_filters.pop("date", None)
        plan.hard_filters["date"] = {"gte": day, "lte": day}
    elif len(publication_days) > 1:
        plan.hard_filters["date"] = sorted(publication_days)
    publishers = _mentioned_publishers(query)
    if len(publishers) == 1:
        publisher = next(iter(publishers))
        plan.tentative_filters.pop("broker", None)
        plan.hard_filters["broker"] = [publisher]
    elif len(publishers) > 1:
        plan.hard_filters["broker"] = sorted(publishers)
    categories = _mentioned_categories(query)
    if categories:
        plan.hard_filters["category"] = categories
    return plan


__all__ = ["FilterPlan", "build_filter_plan", "validate_explicit_filters"]
