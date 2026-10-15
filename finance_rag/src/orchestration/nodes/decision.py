"""受业务范围约束的 Agent 决策和检索工具协议。"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finance_rag.src.agent.planning import Subproblem


class RetrieveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    subproblem_id: str


def decide_intent(query: str, history: list[dict] | None = None) -> str:
    if re.fullmatch(r"\s*(你好|您好|嗨|hello|hi|谢谢|感谢|再见)[！!。,.，\s]*", query, re.IGNORECASE):
        return "direct"
    if re.fullmatch(r"\s*(这个系统|系统|你)(怎么使用|能做什么|有什么功能)[？?\s]*", query):
        return "direct"
    if re.search(r"它|这家公司|该公司|上述公司|这份报告|该报告", query) and not history:
        return "clarify"
    return "retrieve"


def request_tool_calls(items: list[Subproblem], queries: list[str], model: Any) -> dict:
    name = "retrieve_finance_evidence"
    if model is None or not callable(getattr(model, "bind_tools", None)):
        return {"error": "当前模型未支持检索工具调用，请保持Agent流程未启用", "calls": [], "policy_forced": False}
    expected = {item.id: query for item, query in zip(items, queries, strict=True)}
    tool = {"type": "function", "function": {"name": name,
            "description": "检索金融研报证据；金融事实必须检索，范围由服务端控制。",
            "parameters": RetrieveArgs.model_json_schema()}}
    prompt = "必须为每个必要事项调用检索工具。不得添加filters/collection等参数，不回答事实。\n" + json.dumps(expected, ensure_ascii=False)
    try:
        response = model.bind_tools([tool]).invoke(prompt)
        calls = list(getattr(response, "tool_calls", []) or [])
        if getattr(response, "invalid_tool_calls", None):
            raise ValueError("模型工具参数无法解析")
        result, used_ids, used_items = [], set(), set()
        for entry in calls:
            if entry.get("name") != name:
                raise ValueError("模型调用未知工具")
            args = RetrieveArgs.model_validate(entry.get("args"))
            call_id = entry.get("id")
            if not call_id or call_id in used_ids:
                raise ValueError("工具调用ID缺失或重复")
            if args.subproblem_id not in expected or args.subproblem_id in used_items:
                raise ValueError("工具引用未知或重复事项")
            if args.query.strip() != expected[args.subproblem_id].strip():
                raise ValueError("工具查询偏离已审查计划")
            used_ids.add(call_id)
            used_items.add(args.subproblem_id)
            result.append({"id": call_id, "name": name, "args": args.model_dump()})
        forced = False
        for item_id, query in expected.items():
            if item_id not in used_items:
                forced = True
                result.append({"id": "policy-" + uuid.uuid4().hex, "name": name,
                               "args": {"query": query, "subproblem_id": item_id}})
        return {"error": None, "calls": result, "policy_forced": forced}
    except Exception as exc:
        return {"error": f"检索工具协议不兼容：{exc}", "calls": [], "policy_forced": False}
