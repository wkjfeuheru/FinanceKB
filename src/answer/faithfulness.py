"""仅核对声明对应证据的数字、指标、期间和方向，不调用模型。"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

NUMBER = (
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:%|个百分点|基点|"
    r"(?:万亿|千亿|百亿|十亿|亿|千万|百万|十万|万|千|百)?"
    r"\s*(?:人民币|美元|港元|元|吨|股|人|家)|年|倍|万|亿|GW|MW|kW)?"
)
METRICS = r"营业收入|净利润|毛利率|现金流|营收增速|利润增速|收入|营收|利润"


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    text = re.sub(r"\[\d+\]", "", text)
    for marker, label in (
        ("H1", "上半年"),
        ("H2", "下半年"),
        ("Q1", "第一季度"),
        ("Q2", "第二季度"),
        ("Q3", "第三季度"),
        ("Q4", "第四季度"),
    ):
        text = re.sub(rf"((?:19|20)\d{{2}})\s*{marker}", rf"\1年{label}", text, flags=re.I)
    return text


def _normalize_number(token: str) -> str:
    token = re.sub(r"\s|,", "", token)
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(.*)", token)
    value = format(Decimal(match[1]), "f")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value + match[2]


def _metric_values(text: str) -> set[tuple[str, str]]:
    period = r"(?<![\d.])(?:19|20)\d{2}\s*年(?:度)?"
    pairs = re.findall(rf"({METRICS})[^\d。；;,，]*?({NUMBER})", re.sub(period, "", text))
    aliases = {"营收": "收入", "营业收入": "收入"}
    return {(aliases.get(metric, metric), _normalize_number(value)) for metric, value in pairs}


def _period_values(text: str) -> set[tuple[str, str, str]]:
    """把明确期间绑定到同句及其后续分句的指标值，阻止跨年份调换。"""
    text = re.sub(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", lambda m: m[0].replace(",", ""), text)
    current = ""
    values = set()
    for part in re.split(r"[，,；;。]", text):
        periods = re.findall(
            r"(?:19|20)\d{2}\s*年(?:度)?(?:\s*(?:上半年|下半年|第[一二三四]季度))?", part
        )
        # 多个期间同处一个分句时交由语义核验，不能猜测数字的期间归属。
        if len(periods) > 1:
            current = ""
            continue
        if periods:
            current = re.sub(r"\s|年度", lambda m: "年" if m[0] == "年度" else "", periods[0])
        if current:
            values.update((current, metric, value) for metric, value in _metric_values(part))
    return values


def check_numbers(claim: str, quote: str) -> dict:
    """核对对应引文，保留单位，不换算或推导；重复表述不消耗证据次数。"""
    claim, quote = _normalize_text(claim), _normalize_text(quote)
    wanted = list(dict.fromkeys(_normalize_number(t) for t in re.findall(NUMBER, claim)))
    known = {_normalize_number(t) for t in re.findall(NUMBER, quote)}
    unsupported = [t for t in wanted if t not in known and not _year_token_covered(t, quote)]
    issues = ["声明数值或单位不在对应引文中"] if unsupported else []
    if _subperiod_mismatch(claim, quote):
        issues.append("声明期间与对应引文不一致")
    requested, supported = _metric_values(claim), _metric_values(quote)
    if requested - supported:
        issues.append("声明指标与数值缺少对应关系")
    period_values = _period_values(quote)
    if period_values and _period_values(claim) - period_values:
        issues.append("声明期间、指标与数值对应关系不一致")
    return {
        "status": "failed" if issues else "verified",
        "valid": not issues,
        "supported": [t for t in wanted if t not in unsupported],
        "unsupported": unsupported,
        "issues": issues,
    }


def _detail_mismatch(claim: str, quote: str) -> bool:
    """保守的显式数值、单位、年份与方向检查，语义核验仍由模型负责。"""
    claim, quote = _normalize_text(claim), _normalize_text(quote)
    if not check_numbers(claim, quote)["valid"]:
        return True
    directions = [
        (
            r"增长|上升|增加|提高|上涨|提升|increase|grow|rise",
            r"下降|减少|降低|下跌|下滑|decrease|decline|fall",
        ),
        (r"盈利|利润为正|profit", r"亏损|利润为负|loss"),
    ]
    for positive, negative in directions:
        if re.search(positive, claim, re.I) and not re.search(positive, quote, re.I):
            return True
        if re.search(negative, claim, re.I) and not re.search(negative, quote, re.I):
            return True
    if bool(re.search(r"未|不|没有|并非|not\b", claim, re.I)) != bool(
        re.search(r"未|不|没有|并非|not\b", quote, re.I)
    ):
        return True
    return False


def _year_token_covered(token: str, quote: str) -> bool:
    """``2026年`` 可以由引文中的 ``2026年``、``2026H1`` 或 ``2026Q1`` 覆盖。"""
    match = re.fullmatch(r"((?:19|20)\d{2})年", re.sub(r"\s", "", token))
    if not match:
        return False
    year = match.group(1)
    return re.search(rf"{year}\s*(?:年|H[12]|Q[1-4])", quote, re.I) is not None


def _subperiod_mismatch(claim: str, quote: str) -> bool:
    """半年、季度必须对上；全年声明不能只靠半年或季度引文。"""
    pairs = (
        (r"上半年", r"上半年|H1"),
        (r"下半年", r"下半年|H2"),
        (r"(?:第一|一)季度|Q1", r"(?:第一|一)季度|Q1"),
        (r"(?:第二|二)季度|Q2", r"(?:第二|二)季度|Q2"),
        (r"(?:第三|三)季度|Q3", r"(?:第三|三)季度|Q3"),
        (r"(?:第四|四)季度|Q4", r"(?:第四|四)季度|Q4"),
    )
    for claim_pattern, quote_pattern in pairs:
        if re.search(claim_pattern, claim, re.I) and not re.search(quote_pattern, quote, re.I):
            return True
    full_year = re.search(r"(?:19|20)\d{2}\s*年", claim) and not re.search(
        r"上半年|下半年|季度|H[12]|Q[1-4]", claim, re.I
    )
    quote_subperiod = re.search(
        r"(?:19|20)\d{2}\s*年?\s*(?:H[12]|Q[1-4]|上半年|下半年|第[一二三四]季度)", quote, re.I
    )
    quote_year = re.search(
        r"(?:19|20)\d{2}\s*年(?!\s*(?:上半年|下半年|第[一二三四]季度|[一二三四]季度))", quote
    )
    return bool(full_year and quote_subperiod and not quote_year)
