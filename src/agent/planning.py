"""按需规划金融问题；简单问题保留原查询，复杂问题显式分解。"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class Subproblem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    query: str = Field(min_length=1)
    entity: str | None = None
    metric: str | None = None
    period: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    needs_decomposition: bool
    subproblems: list[Subproblem]
    clarification: str | None = None


def _as_plan_id(value: Any) -> Any:
    """模型常把子问题 id 写成数字；校验要求字符串，这里只转换编号。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return value


def parse_json(raw: Any) -> dict:
    content = getattr(raw, "content", raw)
    if not isinstance(content, str):
        raise ValueError("模型输出不是文本JSON")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出不是JSON对象")
    subproblems = value.get("subproblems")
    if isinstance(subproblems, list):
        for item in subproblems:
            if not isinstance(item, dict):
                continue
            if "id" in item:
                item["id"] = _as_plan_id(item.get("id"))
            depends_on = item.get("depends_on")
            if isinstance(depends_on, list):
                item["depends_on"] = [_as_plan_id(dep) for dep in depends_on]
    return value


def needs_decomposition(query: str) -> bool:
    if re.search(r"比较|对比|分别|各自|两家|多家|变化原因|先.+再|[、；;]", query):
        return True
    periods = set(re.findall(r"20\d{2}(?:年|H[12]|Q[1-4])?", query))
    metrics = set(re.findall(r"营业收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率", query))
    # 公司名称内的“和”不是连接词，例如协和电子、泰和新材。
    prefix = re.split(r"20\d{2}|营业收入|营收|净利润|毛利率|评级|目标价|的", query, maxsplit=1)[0]
    joined = bool(re.search(r"与|以及|[甲乙丙丁A-Z]和[甲乙丙丁A-Z]|[\u4e00-\u9fffA-Za-z]{2,}\s*和\s*[\u4e00-\u9fffA-Za-z]{2,}", prefix))
    return len(periods) > 1 or len(metrics) > 1 or joined


_ANAPHORA_RE = re.compile(r"它|该公司|这家公司|上述公司|该报告|这份报告")
_ANAPHORA_CONTENT_RE = re.compile(
    r"20\d{2}(?:年|H[12]|Q[1-4])?|营业收入|营业总收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率"
)
_ANAPHORA_FILLER_RE = re.compile(r"[的是多少如何怎样什么哪些吗呢请请问了在从由将被这那及等，,。？?\s]")


def _self_contained_query(query: str) -> bool:
    """问句里已经有另一分句或具体对象时，指代不再阻断检索。"""
    clauses = [part.strip() for part in re.split(r"[；;]", query) if part.strip()]
    if len(clauses) >= 2 and any(not _ANAPHORA_RE.search(part) and len(part) >= 8 for part in clauses):
        return True
    text = _ANAPHORA_RE.sub("", query)
    text = _ANAPHORA_CONTENT_RE.sub("", text)
    text = _ANAPHORA_FILLER_RE.sub("", text)
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z]{2,}", text))


def resolve_reference(query: str, history: list[dict] | None, model: Any) -> tuple[str, str | None]:
    """仅解析历史中已有的实体；问句本身已可检索时保留原句。"""
    if not _ANAPHORA_RE.search(query):
        return query, None
    if _self_contained_query(query):
        return query, None
    if not history or model is None:
        raise ValueError("指代不明确，请说明公司或报告")
    history_text = "\n".join(m.get("content", "") for m in history if m.get("role") == "user")
    prompt = ("仅把用户问题中的指代替换为历史用户已明确的实体，不改变指标、期间，不回答事实。"
              "不明确时返回空entity。只输出JSON：{query:str,entity:str}。\n"
              + json.dumps({"query": query, "history": history[-10:]}, ensure_ascii=False))
    for _ in range(2):
        try:
            resolved = parse_json(model.invoke(prompt))
            entity, text = resolved.get("entity"), resolved.get("query")
            if not isinstance(entity, str) or not entity or entity not in history_text or not isinstance(text, str):
                raise ValueError("历史中没有明确实体")
            expected = re.sub(r"它|该公司|这家公司|上述公司|该报告|这份报告", entity, query)
            if re.sub(r"\s+", "", expected) != re.sub(r"\s+", "", text):
                raise ValueError("指代解析改变了问题范围")
            return text, entity
        except Exception:
            continue
    raise ValueError("无法可靠解析历史指代，请明确实体")


_ENTITY_PARTICLE_RE = re.compile(r"[的向为了在从由将被其这那及等]")
_ENTITY_CLAUSE_RE = re.compile(
    r"趋势|环境|产业|行业|投资|制造|升级|机遇|增速|说明|综合|之间|带来|同比|下降|上升|能力|智能|通信|设备"
)


def explicit_entities(query: str) -> set[str]:
    """只识别显式枚举的短机构名或公司名，不把分句当成研究对象。"""
    parts = re.split(r"[、，,；;]|以及|与|和", query)
    entities = []
    for part in parts:
        prefix = re.split(r"营业收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率|的|20\d{2}", part, maxsplit=1)[0]
        prefix = re.sub(r"^(?:请|分别|比较|对比|分析|查询|查看|研究|报告|给出|说明)+", "", prefix)
        prefix = re.sub(r"(?:各自|分别)$", "", prefix.strip())
        if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,8}", prefix):
            continue
        if _ENTITY_PARTICLE_RE.search(prefix) or _ENTITY_CLAUSE_RE.search(prefix):
            continue
        entities.append(prefix)
    if len(entities) < 2:
        return set()
    return set(entities)


def validate_combinations(plan: QueryPlan, query: str, entities: set[str]) -> None:
    """保留共同后缀的组合要求，也保留各实体明确指定的不同口径。"""
    if not entities:
        return
    ordered = sorted(entities, key=query.index)
    metric_pattern = r"营业收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率"
    period_pattern = r"20\d{2}(?:年|H[12]|Q[1-4])?"
    last_start = query.index(ordered[-1])
    # 只有全部出现在最后实体之后的维度才是共同后缀；局部限定不做笛卡尔扩展。
    shared_metrics = re.findall(metric_pattern, query[last_start:]) if not re.search(metric_pattern, query[:last_start]) else []
    shared_periods = re.findall(period_pattern, query[last_start:]) if not re.search(period_pattern, query[:last_start]) else []
    for index, entity in enumerate(ordered):
        start = query.index(entity) + len(entity)
        end = query.index(ordered[index + 1]) if index + 1 < len(ordered) else len(query)
        segment = query[start:end]
        metrics = shared_metrics or re.findall(metric_pattern, segment)
        periods = shared_periods or re.findall(period_pattern, segment)
        for metric in metrics or [None]:
            aliases = ("营业收入", "营收", "收入") if metric in {"营业收入", "营收"} else (metric,)
            for period in periods or [None]:
                if not any(entity in item.query
                           and (metric is None or any(alias in item.query for alias in aliases))
                           and (period is None or period in item.query)
                           for item in plan.subproblems):
                    raise ValueError("规划遗漏了实体、指标或期间的必要组合")


def validate_plan(plan: QueryPlan, query: str, max_subproblems: int) -> QueryPlan:
    if len(plan.subproblems) > max_subproblems:
        raise ValueError(f"必要事项超过{max_subproblems}项，请缩小比较范围")
    if not plan.subproblems and not plan.clarification:
        raise ValueError("规划未包含必要事项")
    if plan.clarification:
        return plan
    entities = explicit_entities(query)
    if len(entities) > max_subproblems:
        raise ValueError(f"显式研究对象超过{max_subproblems}个，请缩小范围")
    combined = "\n".join(item.query for item in plan.subproblems)
    if any(entity not in combined for entity in entities):
        raise ValueError("规划遗漏了用户明确的研究对象")
    validate_combinations(plan, query, entities)
    metrics = set(re.findall(r"营业收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率", query))
    for metric in metrics:
        aliases = ("营业收入", "营收", "收入") if metric in {"营业收入", "营收"} else (metric,)
        if not any(value in combined for value in aliases):
            raise ValueError("规划遗漏了用户明确的指标")
    periods = set(re.findall(r"20\d{2}(?:年|H[12]|Q[1-4])?", query))
    if any(re.sub(r"\s+", "", period) not in re.sub(r"\s+", "", combined) for period in periods):
        raise ValueError("规划遗漏了用户明确的期间")
    ids = [item.id for item in plan.subproblems]
    if len(set(ids)) != len(ids):
        raise ValueError("子问题ID重复")
    for item in plan.subproblems:
        if item.entity and item.entity not in query:
            raise ValueError("规划新增了用户未指定的实体")
        if item.period and item.period not in query:
            raise ValueError("规划新增了用户未指定的期间")
        if any(dep not in ids for dep in item.depends_on):
            raise ValueError("规划引用未知依赖")
    visiting, visited = set(), set()
    by_id = {item.id: item for item in plan.subproblems}
    def visit(item_id):
        if item_id in visiting:
            raise ValueError("子问题依赖成环")
        if item_id in visited:
            return
        visiting.add(item_id)
        for dep in by_id[item_id].depends_on:
            visit(dep)
        visiting.remove(item_id)
        visited.add(item_id)
    for item_id in ids:
        visit(item_id)
    if not plan.needs_decomposition and len(ids) > 1:
        raise ValueError("简单规划不能包含多个子问题")
    return plan


def plan_query(query: str, history: list[dict] | None = None, *, model: Any = None,
               max_subproblems: int = 3) -> QueryPlan:
    # 规划只需要理解指代，不承担全部历史：按预算保留最新若干轮（RAG-15）。
    # 否则复杂问题会把整段对话塞进规划 prompt，长会话直接超窗。
    from src.core.config import PLANNING_MAX_HISTORY_TOKENS
    from src.core.token_budget import fit_history

    history, history_report = fit_history(history, PLANNING_MAX_HISTORY_TOKENS)
    if history_report.get("dropped"):
        logger.info(
            "规划阶段按预算裁剪历史：保留 %d 条、丢弃 %d 条（约 %d token）",
            history_report["kept"], history_report["dropped"], history_report["tokens"],
        )
    try:
        query, entity = resolve_reference(query, history, model)
    except ValueError as exc:
        return QueryPlan(needs_decomposition=False, subproblems=[], clarification=str(exc))
    if not needs_decomposition(query):
        return QueryPlan(needs_decomposition=False, subproblems=[Subproblem(id="q1", query=query, entity=entity)])
    if len(explicit_entities(query)) > max_subproblems:
        return QueryPlan(needs_decomposition=True, subproblems=[], clarification=f"研究对象超过{max_subproblems}个，请缩小范围。")
    if model is None:
        return QueryPlan(needs_decomposition=True, subproblems=[], clarification="无法可靠分解当前问题，请明确一个实体、指标及期间，或缩小比较范围。")
    prompt = ("你是金融问题规划器。只输出JSON，不回答事实。仅复杂问题按必要实体/维度/期间拆分，"
              f"最多{max_subproblems}项；超出时返回clarification且subproblems为空，不可截掉事项。"
              "不得新增原问题没有的实体和期间，未知字段为null。依赖必须无环。"
              "结构：{needs_decomposition:bool,subproblems:[{id,query,entity,metric,period,depends_on:[]}],clarification:null}。"
              "历史仅用于理解指代，不能引入新研究对象。\n"
              + json.dumps({"query": query, "history": history or []}, ensure_ascii=False))
    error = "无法生成可靠规划"
    for _ in range(2):
        try:
            result = QueryPlan.model_validate(parse_json(model.invoke(prompt)))
            return validate_plan(result, query, max_subproblems)
        except Exception as exc:
            error = str(exc)
    return QueryPlan(needs_decomposition=True, subproblems=[], clarification=f"请明确或缩小问题范围：{error}")


# ---------------------------------------------------------------------------
# 补检改写的范围约束（RAG-09）
# ---------------------------------------------------------------------------
#
# 补检改写只允许「换检索措辞」，不允许改变问题的适用范围。结构化字段
# （entity/period/metric）经常为空，因此约束必须**同时从问题文本**建立：
# 否则「贵州茅台2025年营业收入」被改写成「营业收入」也能通过，硬过滤未覆盖
# 时会跨主体、跨期间召回。
#
# 术语等价只走受控映射（下方常量），不做自由推断；范围无法确认时回退原查询。

_METRIC_SCOPE_RE = re.compile(
    r"营业收入|营业总收入|营收|收入|归母净利润|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率"
)
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营业总收入", "营收", "收入"),
    "营业总收入": ("营业总收入", "营业收入", "营收", "收入"),
    "营收": ("营业收入", "营业总收入", "营收", "收入"),
    "收入": ("营业收入", "营业总收入", "营收", "收入"),
    "归母净利润": ("归母净利润", "净利润", "归母净利"),
    "净利润": ("归母净利润", "净利润", "归母净利"),
    "毛利率": ("毛利率", "销售毛利率"),
    "净利率": ("净利率", "销售净利率"),
    "评级": ("评级", "投资评级"),
    "目标价": ("目标价", "目标价格"),
    "现金流": ("现金流", "经营性现金流", "经营活动现金流"),
    "资产负债率": ("资产负债率",),
}

_PERIOD_RE = re.compile(r"20\d{2}(?:年|H[12]|Q[1-4])?")
_YEAR_RE = re.compile(r"20\d{2}")
_QUARTER_LABELS = ("一", "二", "三", "四")
# 粒度标记：只用于「不得引入新粒度」，粒度等价由受控映射给出
_GRANULARITY_RE = re.compile(
    r"20\d{2}\s*(?:H[12]|Q[1-4])|第[一二三四]季度|[一二三四]季报|半年报|中报|年报|上半年|下半年|全年"
)

# 问题措辞（疑问、指代、指令、泛化名词）：只在**整段**相等时忽略，
# 绝不做子串替换——「协和电子」里的「和」不是连接词。
_SCOPE_STOPWORDS = frozenset({
    "请", "请问", "查询", "查看", "分析", "分别", "比较", "对比", "给出", "说明",
    "研究", "报告", "是多少", "多少", "如何", "怎样", "什么", "哪些", "是否",
    "吗", "呢", "介绍", "摘要", "主要", "相关", "数据", "指标", "表现", "变化",
    "原因", "情况", "表述", "内容", "总结", "概述", "阐述", "提及", "涉及",
    "以及", "及其", "等等", "多少倍", "怎么样", "的情况", "的内容", "的表述",
})
_SCOPE_PREFIX_RE = re.compile(r"^(?:关于|对于|请问|请|查询|查看|分析|比较|对比)+")
_SCOPE_SUFFIX_RE = re.compile(r"(?:的表述|的内容|的情况|的说明|的摘要|数据|情况|表述|内容|摘要|介绍|说明)+$")


def _year_variants(year: str) -> tuple[str, ...]:
    """年份级写法：仅允许同一年份的等价表达。"""
    return (year, f"{year}年", f"{year}年度", f"{year}全年")


def granularity_code(token: str) -> str:
    """粒度编码：quarter(n) / h1 / h2 / fy / year（无法识别按 year）。"""
    quarter = re.search(r"Q([1-4])", token) or re.search(r"第([一二三四])季度", token) or re.search(
        r"([一二三四])季报", token
    )
    if quarter:
        raw = quarter.group(1)
        index = _QUARTER_LABELS.index(raw) + 1 if raw in _QUARTER_LABELS else int(raw)
        return f"quarter{index}"
    if "H1" in token or "上半年" in token or "中报" in token or "半年报" in token:
        return "h1"
    if "H2" in token or "下半年" in token:
        return "h2"
    if "年报" in token or "全年" in token:
        return "fy"
    return "year"


def _granularity_variants(token: str) -> tuple[str, ...]:
    """粒度级写法：只允许**同一粒度**的等价表达（季度不得换成另一个季度）。"""
    year_match = _YEAR_RE.match(token)
    if year_match is None:
        return ()
    year = year_match.group(0)
    code = granularity_code(token)
    if code == "year":
        return ()
    if code == "h1":
        return (f"{year}H1", f"{year}年上半年", f"{year}年中报", f"{year}年半年报", f"{year}中报")
    if code == "h2":
        return (f"{year}H2", f"{year}年下半年", f"{year}年H2")
    if code == "fy":
        return (f"{year}年报", f"{year}全年", f"{year}年度报告", f"{year}年度")
    quarter = code.removeprefix("quarter")
    label = _QUARTER_LABELS[int(quarter) - 1]
    return (
        f"{year}Q{quarter}", f"{year}年Q{quarter}", f"{year}年第{label}季度",
        f"{year}年{label}季报", f"{year}年{label}季度",
    )


def _name_like_tokens(text: str) -> set[str]:
    """文本中的「名称型」片段（主体候选）。

    先去掉期间与指标，再剥离疑问/泛化前后缀，剩下的中文或字母串即视为主体名
    （如「贵州茅台」「宁德时代」「GDP增速」）。整段命中停用词表才忽略，
    因此「协和电子」不会被「和」拆坏。
    """
    residue = _PERIOD_RE.sub(" ", text or "")
    residue = _METRIC_SCOPE_RE.sub(" ", residue)
    tokens: set[str] = set()
    for raw in re.findall(r"[\u4e00-\u9fffA-Za-z]{2,}", residue):
        token = _SCOPE_SUFFIX_RE.sub("", _SCOPE_PREFIX_RE.sub("", raw)).strip()
        if len(token) < 2 or token in _SCOPE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def required_scope(item: Subproblem) -> list[tuple[str, tuple[str, ...]]]:
    """建立补检改写必须保留的 ``(约束说明, 允许写法)`` 列表。

    期间与指标取「结构化字段 ∪ 问题文本」，主体取「结构化 entity ∪ 文本名称片段」；
    结构化字段为空时约束依然成立。期间拆成**年份**与**粒度**两条独立约束，
    因此「2026Q1 → 2026Q3」这类同年的粒度偷换也会被拒绝。
    """
    text = item.query or ""
    constraints: list[tuple[str, tuple[str, ...]]] = []

    periods = set(_PERIOD_RE.findall(text))
    if item.period:
        periods.add(item.period)
    for token in sorted(periods):
        year_match = _YEAR_RE.match(token)
        if year_match is None:
            constraints.append((token, (token,)))
            continue
        year = year_match.group(0)
        constraints.append((year, _year_variants(year)))
        granularity = _granularity_variants(token)
        if granularity:
            constraints.append((token, granularity))

    metrics = set(_METRIC_SCOPE_RE.findall(text)) | set(_METRIC_SCOPE_RE.findall(item.metric or ""))
    constraints.extend(
        (token, _METRIC_ALIASES.get(token, (token,))) for token in sorted(metrics)
    )

    subjects = _name_like_tokens(text)
    if item.entity:
        subjects.add(item.entity)
    constraints.extend((token, (token,)) for token in sorted(subjects))

    return constraints


def _scope_drift(item: Subproblem, candidate: str) -> str:
    """返回改写相对原问题的范围漂移说明（空串表示没有漂移）。"""
    original_years = set(_YEAR_RE.findall(item.query)) | set(_YEAR_RE.findall(item.period or ""))
    if not set(_YEAR_RE.findall(candidate)) <= original_years:
        return "新增年份"
    original_codes = {
        granularity_code(token) for token in _GRANULARITY_RE.findall(item.query)
    }
    candidate_codes = {
        granularity_code(token) for token in _GRANULARITY_RE.findall(candidate)
    }
    if not candidate_codes <= original_codes:
        return "新增粒度"
    named = r"[\u4e00-\u9fffA-Za-z]{1,20}?(?:公司|集团|股份|银行|证券|电子|科技|汽车)"
    if any(name not in item.query for name in re.findall(named, candidate)):
        return "新增主体"
    original_metrics = set(_METRIC_SCOPE_RE.findall(item.query)) | set(
        _METRIC_SCOPE_RE.findall(item.metric or "")
    )
    allowed_metrics: set[str] = set()
    for metric in original_metrics:
        allowed_metrics.update(_METRIC_ALIASES.get(metric, (metric,)))
    if not set(_METRIC_SCOPE_RE.findall(candidate)) <= allowed_metrics:
        return "新增指标"
    return ""


def rewrite_missing(item: Subproblem, attempted: list[str], *, model: Any = None) -> str:
    if model is None:
        return item.query
    prompt = ("仅改写以下缺失事项的检索查询，不回答，不新增实体/期间/指标。避开已检索查询。"
              "只输出JSON：{\"query\":\"新的查询\"}。\n"
              + json.dumps({"item": item.model_dump(), "attempted": attempted}, ensure_ascii=False))
    try:
        query = str(parse_json(model.invoke(prompt))["query"]).strip()
        if not query:
            return item.query
        # 1) 必须保留原问题的全部范围约束（结构化字段为空也要校验）
        try:
            missing = [
                token for token, variants in required_scope(item)
                if not any(variant in query for variant in variants)
            ]
        except Exception:  # noqa: BLE001 - 约束无法确认时回退原查询
            return item.query
        if missing:
            logger.info("补检改写缺少必要范围（%s），回退原查询", "、".join(missing))
            return item.query
        # 2) 不得放宽范围：新增年份/粒度/主体/指标一律回退
        drift = _scope_drift(item, query)
        if drift:
            logger.info("补检改写范围漂移（%s），回退原查询", drift)
            return item.query
        return query
    except Exception:
        return item.query
