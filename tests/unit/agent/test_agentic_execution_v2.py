"""Agentic v2 执行边界；所有依赖均为离线替身。"""
from copy import deepcopy

import pytest

from src.agent.planning import QueryPlan, Subproblem, plan_query, rewrite_missing
from src.agent.nodes.grading import grade_coverage
from src.retrieve.pipeline import (
    RetrievalDependencies, RetrievalPolicy, RetrievalRequest, execute_retrieval,
)


def plan(query="甲公司营收"):
    return QueryPlan(needs_decomposition=False, subproblems=[Subproblem(id="q1", query=query)])


def coverage(status="missing"):
    return {"items": [{"item_id": "q1", "status": status,
                       "evidence_ids": ["kb:c1"] if status == "supported" else [], "reason": ""}],
            "sufficient": status == "supported", "supported_items": ["q1"] if status == "supported" else [],
            "missing_items": [] if status == "supported" else ["q1"], "unknown": False}


@pytest.mark.asyncio
async def test_last_allowed_retry_executes_and_keeps_prior_evidence():
    calls = []
    async def search(request, queries, keywords, filter_plan):
        calls.append(list(queries))
        return {"docs": [{"id": f"c{len(calls)}", "collection": "kb", "content": queries[0], "score": 1}],
                "diagnostics": {}, "filters": filter_plan.hard_filters}
    def rewrite(item, missing, attempted):
        return f"甲公司营收 补检{len(attempted)}"
    deps = RetrievalDependencies(search=search, planner=lambda *a: plan(),
                                 grader=lambda *a: coverage(), rewriter=rewrite,
                                 prepare_context=lambda docs: {"context_docs": deepcopy(docs), "sources": [], "context_text": ""})
    result = await execute_retrieval(RetrievalRequest("甲公司营收", filters={}),
                                    RetrievalPolicy(mode="agentic", reflect=True), deps)
    assert len(calls) == 3
    assert result["retry_count"] == 2
    assert len(result["evidence_pool"]) == 3
    assert result["stop_reason"] == "retry_limit"


@pytest.mark.asyncio
async def test_repeated_rewrite_stops_without_repeating_search():
    calls = []
    async def search(request, queries, keywords, filter_plan):
        calls.extend(queries)
        return {"docs": [], "diagnostics": {}}
    deps = RetrievalDependencies(search=search, planner=lambda *a: plan(),
                                 grader=lambda *a: coverage(), rewriter=lambda item, *a: item.query,
                                 prepare_context=lambda docs: {"context_docs": docs, "sources": [], "context_text": ""})
    result = await execute_retrieval(RetrievalRequest("甲公司营收", filters={}),
                                    RetrievalPolicy(mode="agentic", reflect=True), deps)
    assert calls == ["甲公司营收"]
    assert result["stop_reason"] == "repeated_query"


@pytest.mark.asyncio
async def test_infrastructure_error_is_not_empty_evidence():
    async def search(*a):
        return {"docs": [], "retrieval_error": "数据库不可用", "diagnostics": {}}
    deps = RetrievalDependencies(search=search, planner=lambda *a: plan(),
                                 prepare_context=lambda docs: {"context_docs": docs, "sources": [], "context_text": ""})
    result = await execute_retrieval(RetrievalRequest("甲公司营收", filters={}), RetrievalPolicy(), deps)
    assert result["decision"] == "error"
    assert result["retrieval_error"] == "数据库不可用"


def test_simple_question_does_not_call_planner_model():
    class NoModel:
        def invoke(self, *a):
            pytest.fail("简单问题不应分解")
    result = plan_query("甲公司营业收入是多少？", model=NoModel())
    assert not result.needs_decomposition
    assert len(result.subproblems) == 1


def test_complex_plan_does_not_silently_drop_fourth_item():
    class Model:
        def invoke(self, *a):
            return '{"needs_decomposition":true,"subproblems":[{"id":"q1","query":"甲营收"},{"id":"q2","query":"乙营收"},{"id":"q3","query":"丙营收"},{"id":"q4","query":"丁营收"}]}'
    result = plan_query("比较甲、乙、丙、丁的营收", model=Model(), max_subproblems=3)
    assert result.clarification
    assert not result.subproblems


def test_high_score_is_not_sufficient_for_two_entities():
    p = QueryPlan(needs_decomposition=True, subproblems=[Subproblem(id="q1", query="甲营收"), Subproblem(id="q2", query="乙营收")])
    class Model:
        def invoke(self, *a):
            return '{"items":[{"item_id":"q1","status":"supported","evidence_ids":["kb:c1"],"reason":""},{"item_id":"q2","status":"missing","evidence_ids":[],"reason":"没有乙"}]}'
    result = grade_coverage(p, [{"id":"c1","collection":"kb","content":"甲营收10亿元","rerank_score":0.99}], model=Model())
    assert not result["sufficient"]
    assert result["missing_items"] == ["q2"]


def test_unknown_evidence_id_cannot_pass_coverage():
    class Model:
        def invoke(self, *a):
            return '{"items":[{"item_id":"q1","status":"supported","evidence_ids":["invented"],"reason":""}]}'
    result = grade_coverage(plan(), [{"id":"c1","collection":"kb","content":"甲"}], model=Model())
    assert result["unknown"]
    assert not result["sufficient"]


def test_retry_configuration_cannot_exceed_two():
    with pytest.raises(ValueError):
        RetrievalPolicy(max_retries=3)


@pytest.mark.asyncio
async def test_new_missing_evidence_is_not_starved_by_old_high_scores():
    calls=[]
    async def search(*a):
        calls.append(1)
        return {"docs": [{"id":"old","collection":"kb","content":"无关模板","score":1}]
                if len(calls)==1 else [{"id":"new","collection":"kb","content":"甲营收10亿元","score":0.1}]}
    def grade(p, docs):
        return coverage("supported" if docs and docs[0]["id"]=="new" else "missing")
    deps=RetrievalDependencies(search=search,planner=lambda *a:plan(),grader=grade,
        rewriter=lambda *a:"甲营收新的查询",prepare_context=lambda docs:{"context_docs":docs,"sources":[],"context_text":""})
    result=await execute_retrieval(RetrievalRequest("甲公司营收",filters={},rerank_top_n=1),
        RetrievalPolicy(mode="agentic",reflect=True),deps)
    assert result["docs"][0]["id"]=="new"
    assert result["decision"]=="generate"


def test_company_name_containing_join_word_is_simple():
    class NoModel:
        def invoke(self, *a): pytest.fail("公司名称不能当并列结构")
    result=plan_query("协和电子2024年营业收入是多少？",model=NoModel())
    assert not result.needs_decomposition


def test_history_reference_is_resolved_without_decomposing():
    class Model:
        def invoke(self, prompt):
            return '{"query":"宁德时代2024年营业收入是多少？","entity":"宁德时代"}'
    result=plan_query("它2024年营业收入是多少？",history=[{"role":"user","content":"请查看宁德时代研报"}],model=Model())
    assert not result.needs_decomposition
    assert result.subproblems[0].entity=="宁德时代"
    assert "它" not in result.subproblems[0].query


def test_rewrite_cannot_change_metric_or_add_entity():
    class Model:
        def invoke(self, prompt): return '{"query":"甲公司2024年净利润，乙公司收入"}'
    item=Subproblem(id="q1",query="甲公司2024年营业收入",entity="甲公司",period="2024年",metric="营业收入")
    assert rewrite_missing(item,[],model=Model())==item.query


def test_incomplete_evidence_does_not_reserve_context_slot():
    from src.retrieve.pipeline import select_docs
    old={"id":"old","collection":"kb","content":"不完整","score":1}
    new={"id":"new","collection":"kb","content":"完整","score":0.1}
    c={"items":[{"item_id":"q1","status":"missing","evidence_ids":["kb:old"]}],"missing_items":["q1"]}
    assert select_docs([old,new],c,1,[new])[0]["id"]=="new"


def test_new_evidence_gets_trial_even_when_supported_slots_are_full():
    from src.retrieve.pipeline import select_docs
    old={"id":"old","collection":"kb","content":"只有甲","score":1}
    new={"id":"new","collection":"kb","content":"同时有甲乙","score":0.1}
    c={"items":[{"item_id":"q1","status":"supported","evidence_ids":["kb:old"]},
                {"item_id":"q2","status":"missing","evidence_ids":[]}],"missing_items":["q2"]}
    assert select_docs([old,new],c,1,[new])[0]["id"]=="new"


def test_model_cannot_hide_four_entities_in_three_subproblems():
    class Model:
        def invoke(self, prompt):
            return '{"needs_decomposition":true,"subproblems":[{"id":"q1","query":"甲公司营收"},{"id":"q2","query":"乙公司营收"},{"id":"q3","query":"丙公司营收"}]}'
    result=plan_query("比较甲公司、乙公司、丙公司、丁公司营收",model=Model())
    assert result.clarification
    assert not result.subproblems


def test_model_cannot_omit_second_metric():
    class Model:
        def invoke(self, prompt):
            return '{"needs_decomposition":true,"subproblems":[{"id":"q1","query":"甲公司营收"}]}'
    result=plan_query("甲公司营收和净利润分别是多少",model=Model())
    assert result.clarification


def test_model_cannot_omit_entity_metric_combinations():
    class Model:
        def invoke(self,prompt):
            return '{"needs_decomposition":true,"subproblems":[{"id":"q1","query":"甲公司2024年营业收入"},{"id":"q2","query":"乙公司2024年净利润"}]}'
    result=plan_query("比较甲公司和乙公司2024年的营业收入和净利润",model=Model())
    assert result.clarification


def test_self_contained_anaphora_is_not_blocked():
    class Model:
        def invoke(self, prompt):
            return ('{"needs_decomposition":true,"subproblems":['
                    '{"id":"q1","query":"3D数据中心万卡算力"},'
                    '{"id":"q2","query":"燃气轮机订单"}]}')
    query = "一份关于3D数据中心的研究称其IT独立层可支撑万卡；另一份点评称该公司燃气轮机订单超预期。请说明共同变化。"
    result = plan_query(query, model=Model())
    assert not result.clarification
    assert len(result.subproblems) == 2


def test_bare_anaphora_without_history_still_clarifies():
    class Model:
        def invoke(self, *a):
            pytest.fail("没有历史的纯指代不应调用规划模型")
    result = plan_query("它的营收是多少？", model=Model())
    assert result.clarification
    assert not result.subproblems


def test_long_clause_is_not_counted_as_many_entities():
    class Model:
        def invoke(self, prompt):
            return ('{"needs_decomposition":true,"subproblems":['
                    '{"id":"q1","query":"通信设备企业产品重构"},'
                    '{"id":"q2","query":"制造业投资增速"}]}')
    query = "移动通信产业向智能能力升级，为通信设备企业带来产品重构等机遇；而8月制造业投资累计同比增速下降。请综合说明行业趋势与宏观投资环境之间的张力。"
    result = plan_query(query, model=Model())
    assert not result.clarification


def test_numeric_subproblem_id_is_accepted():
    class Model:
        def invoke(self, prompt):
            return ('{"needs_decomposition":true,"subproblems":['
                    '{"id":1,"query":"甲公司2024年营业收入","depends_on":[]},'
                    '{"id":2,"query":"乙公司2025年净利润","depends_on":[1]}]}')
    result = plan_query("比较甲公司2024年营业收入与乙公司2025年净利润", model=Model())
    assert not result.clarification
    assert [item.id for item in result.subproblems] == ["1", "2"]
    assert result.subproblems[1].depends_on == ["1"]


@pytest.mark.asyncio
async def test_invalid_plan_still_retrieves_original_query():
    seen = []

    async def search(request, queries, keywords, filter_plan):
        seen.append(list(queries))
        return {"docs": [{"id": "c1", "collection": "kb", "content": "证据", "score": 1}], "diagnostics": {}}

    deps = RetrievalDependencies(
        search=search,
        planner=lambda *a: QueryPlan(needs_decomposition=True, subproblems=[], clarification="研究对象超过3个，请缩小范围。"),
        prepare_context=lambda docs: {"context_docs": docs, "sources": [{"id": "c1"}], "context_text": ""},
    )
    result = await execute_retrieval(
        RetrievalRequest("比较甲、乙、丙、丁的营收", filters={}),
        RetrievalPolicy(grade=False, reflect=False),
        deps,
    )
    assert seen and seen[0] == ["比较甲、乙、丙、丁的营收"]
    assert result["docs"]
    assert result["decision"] == "clarification"
    assert "超过3个" in result["clarification"]


@pytest.mark.asyncio
async def test_original_query_is_searched_with_subqueries():
    seen = []

    async def search(request, queries, keywords, filter_plan):
        seen.append(list(queries))
        return {"docs": [{"id": "c1", "collection": "kb", "content": "证据", "score": 1}], "diagnostics": {}}

    planned = QueryPlan(needs_decomposition=True, subproblems=[
        Subproblem(id="q1", query="甲公司营收"),
        Subproblem(id="q2", query="乙公司净利润"),
    ])
    deps = RetrievalDependencies(
        search=search,
        planner=lambda *a: planned,
        grader=lambda *a: coverage("supported"),
        prepare_context=lambda docs: {"context_docs": docs, "sources": [], "context_text": ""},
    )
    await execute_retrieval(
        RetrievalRequest("比较甲公司营收和乙公司净利润", filters={}),
        RetrievalPolicy(mode="agentic", rewrite=True, reflect=False),
        deps,
    )
    assert seen[0][0] == "比较甲公司营收和乙公司净利润"
    assert "甲公司营收" in seen[0]
    assert "乙公司净利润" in seen[0]


def test_explicit_different_entity_metrics_do_not_require_cross_product():
    class Model:
        def invoke(self,prompt):
            return '{"needs_decomposition":true,"subproblems":[{"id":"q1","query":"甲公司2024年营业收入"},{"id":"q2","query":"乙公司2025年净利润"}]}'
    result=plan_query("比较甲公司2024年营业收入与乙公司2025年净利润",model=Model())
    assert not result.clarification


def test_bare_evidence_id_is_accepted_when_unique():
    class Model:
        def invoke(self, prompt):
            assert "不要拆成额外 item_id" in prompt
            assert '"q1"' in prompt
            return '{"items":[{"item_id":"q1","status":"supported","evidence_ids":["c1"],"reason":"有"}]}'
    result = grade_coverage(plan(), [{"id": "c1", "collection": "kb", "content": "甲营收10亿元"}], model=Model())
    assert result["sufficient"]
    assert result["items"][0]["evidence_ids"] == ["kb:c1"]
    assert result["call_failed"] is False


def test_ambiguous_bare_evidence_id_cannot_pass_coverage():
    class Model:
        def invoke(self, *a):
            return '{"items":[{"item_id":"q1","status":"supported","evidence_ids":["c1"],"reason":""}]}'
    docs = [
        {"id": "c1", "collection": "kb", "content": "甲"},
        {"id": "c1", "collection": "other", "content": "乙"},
    ]
    result = grade_coverage(plan(), docs, model=Model())
    assert result["unknown"]
    assert result["call_failed"] is False


def test_model_call_failure_is_marked_for_fallback():
    class Model:
        def invoke(self, *a):
            raise RuntimeError("limit_requests")
    result = grade_coverage(plan(), [{"id": "c1", "collection": "kb", "content": "甲"}], model=Model())
    assert result["unknown"]
    assert result["call_failed"] is True
    assert result["rate_limited"] is False
    assert result["items"][0]["reason"] == "覆盖校验模型调用失败"


def test_rate_limit_stops_without_a_second_call():
    class RateLimitError(Exception):
        status_code = 429

    calls = {"n": 0}

    class Model:
        def invoke(self, prompt):
            calls["n"] += 1
            assert "rerank_score" not in prompt
            assert "甲营收" in prompt
            raise RateLimitError("rate limit")

    result = grade_coverage(
        plan(),
        [{"id": "c1", "collection": "kb", "content": "甲营收10亿元", "rerank_score": 0.99, "title": "甲公司"}],
        model=Model(),
    )
    assert calls["n"] == 1
    assert result["call_failed"] is True
    assert result["rate_limited"] is True
    assert result["unknown"] is True
