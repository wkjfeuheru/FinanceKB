"""按需规划金融问题；简单问题保留原查询，复杂问题显式分解。"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


def resolve_reference(query: str, history: list[dict] | None, model: Any) -> tuple[str, str | None]:
    """仅解析历史中已有的实体；解析失败由规划器返回澄清。"""
    if not re.search(r"它|该公司|这家公司|上述公司|该报告|这份报告", query):
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


def explicit_entities(query: str) -> set[str]:
    """只识别显式枚举的研究对象，不推断不存在的公司。"""
    parts = re.split(r"[、，,；;]|以及|与|和", query)
    entities = []
    for part in parts:
        prefix = re.split(r"营业收入|营收|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率|的|20\d{2}", part, maxsplit=1)[0]
        prefix = re.sub(r"^(?:请|分别|比较|对比|分析|查询|查看|研究|报告|给出|说明)+", "", prefix)
        prefix = re.sub(r"(?:各自|分别)$", "", prefix.strip())
        if prefix:
            entities.append(prefix)
    if len(entities) < 2:
        return set()
    return {item for item in entities if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]{1,20}", item)}


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


def rewrite_missing(item: Subproblem, attempted: list[str], *, model: Any = None) -> str:
    if model is None:
        return item.query
    prompt = ("仅改写以下缺失事项的检索查询，不回答，不新增实体/期间/指标。避开已检索查询。"
              "只输出JSON：{\"query\":\"新的查询\"}。\n"
              + json.dumps({"item": item.model_dump(), "attempted": attempted}, ensure_ascii=False))
    try:
        query = str(parse_json(model.invoke(prompt))["query"]).strip()
        if not query or (item.entity and item.entity not in query) or (item.period and item.period not in query):
            return item.query
        aliases = {"营业收入": ("营业收入", "营收", "收入"), "营收": ("营业收入", "营收", "收入"),
                   "净利润": ("净利润", "归母净利润"), "收入": ("营业收入", "营收", "收入")}
        metrics = re.findall(r"营业收入|营收|收入|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率", item.metric or item.query)
        if any(not any(alias in query for alias in aliases.get(metric, (metric,))) for metric in metrics):
            return item.query
        def canonical(value):
            return "收入" if value in {"营业收入", "营收", "收入"} else value
        candidate_metrics = re.findall(r"营业收入|营收|收入|净利润|毛利率|净利率|评级|目标价|现金流|资产负债率", query)
        if not {canonical(value) for value in candidate_metrics} <= {canonical(value) for value in metrics}:
            return item.query
        named = r"[\u4e00-\u9fffA-Za-z]{1,20}?(?:公司|集团|股份|银行|证券|电子|科技|汽车)"
        if any(name not in item.query for name in re.findall(named, query)):
            return item.query
        original_periods = set(re.findall(r"20\d{2}(?:年|H[12]|Q[1-4])?", item.query))
        if not set(re.findall(r"20\d{2}(?:年|H[12]|Q[1-4])?", query)) <= original_periods:
            return item.query
        return query
    except Exception:
        return item.query
