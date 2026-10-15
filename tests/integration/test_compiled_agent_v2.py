import json

import pytest

from finance_rag.src.agent.planning import QueryPlan, Subproblem
from finance_rag.src.orchestration import graph
from finance_rag.src.services.retrieval_service import RetrievalDependencies, RetrievalPolicy, prepare_context


class OfflineModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        if "必须为每个必要事项" in prompt:
            return type("Response", (), {"tool_calls": []})()
        if "supplied_claims" in prompt:
            data = json.loads(prompt.split("\n", 1)[1])
            return json.dumps({"checks":[{**claim,"status":"supported",
                "supporting_quote":"甲公司营收10亿元", "reason":"原文直接支持"}
                for claim in data["supplied_claims"]]}, ensure_ascii=False)
        return "甲公司营收10亿元[1]"


@pytest.mark.asyncio
async def test_compiled_graph_runs_two_retries_and_validated_answer(monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.services import citation_validator
    model = OfflineModel()
    monkeypatch.setattr(graph,"get_model",lambda:model)
    monkeypatch.setattr(chat_service,"get_model",lambda:model)
    monkeypatch.setattr(citation_validator,"ENABLE_CITATION_VALIDATION",True)
    calls=[]
    async def search(request, queries, keywords, filters):
        calls.append(queries)
        return {"docs":[{"id":f"c{len(calls)}","collection":"kb","source":"a.pdf",
                         "title":"甲报告","content":"甲公司营收10亿元","score":1}], "diagnostics":{}}
    def grade(plan, docs):
        supported = len(calls)==3
        return {"sufficient":supported,"unknown":False,
                "items":[{"item_id":"q1","status":"supported" if supported else "missing",
                          "evidence_ids":["kb:c3"] if supported else []}],
                "supported_items":["q1"] if supported else [],"missing_items":[] if supported else ["q1"]}
    deps=RetrievalDependencies(search=search,planner=lambda *a:QueryPlan(needs_decomposition=False,
        subproblems=[Subproblem(id="q1",query="甲公司营收")]),grader=grade,
        rewriter=lambda item, missing, attempted: f"甲公司营收 补检{len(attempted)}",prepare_context=prepare_context)
    result=await graph.build_graph().ainvoke(graph.initial_state("甲公司营收", request_filters={},
        policy=RetrievalPolicy(mode="agentic",rewrite=True,reflect=True,use_tools=True),dependencies=deps))
    assert len(calls)==3
    assert result["retry_count"]==2
    assert result["answer_result"]["answer_status"]=="complete"
    assert result["answer_result"]["citation_validation"]["status"]=="verified"
    diagnostic=result["pipeline"]["diagnostics"]
    assert {c["id"] for c in diagnostic["tool_calls"]} == {c["tool_call_id"] for c in diagnostic["tool_results"]}


@pytest.mark.asyncio
async def test_greeting_never_calls_models_or_tools(monkeypatch):
    monkeypatch.setattr(graph,"get_model",lambda:pytest.fail("不应调用模型"))
    result=await graph.build_graph().ainvoke(graph.initial_state("你好"))
    assert result["answer_result"]["citation_validation"]["status"]=="not_required"
