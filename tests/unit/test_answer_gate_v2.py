import pytest

from finance_rag.src.services.answer_service import answer_pipeline


def pipeline():
    return {"decision":"generate", "coverage":{"supported_items":["q1"],"missing_items":[],"unknown":False},
            "plan":{"needs_decomposition":False,"subproblems":[{"id":"q1","query":"甲公司营收"}]},
            "context_text":"[1] 甲公司营收10亿元", "context_docs":[{"content":"甲公司营收10亿元","id":"c1"}],
            "sources":[{"index":1,"title":"报告","source":"a.pdf","score":1,"preview":"甲公司营收"}]}


class Model:
    def invoke(self, prompt):
        return "甲公司营收10亿元[1]"


@pytest.mark.asyncio
async def test_unknown_validation_never_returns_draft():
    async def validator(*a, **kw):
        return {"status":"unknown","score":None,"issues":["不可用"]}
    result = await answer_pipeline("甲公司营收", pipeline(), model=Model(), validator=validator)
    assert result["answer_status"] == "unverified"
    assert "10亿元" not in result["answer"]
    assert result["sources"]


@pytest.mark.asyncio
async def test_partial_answer_lists_gaps_after_validating_facts():
    p = pipeline()
    p["decision"] = "partial"
    p["coverage"]["missing_items"] = ["q2"]
    p["plan"]["subproblems"].append({"id":"q2","query":"乙公司营收"})
    async def validator(answer, docs, **kw):
        assert "缺口" not in answer
        return {"status":"verified","score":1,"issues":[]}
    result = await answer_pipeline("甲乙营收", p, model=Model(), validator=validator)
    assert result["answer_status"] == "partial"
    assert result["missing_items"] == ["乙公司营收"]
    assert "乙公司营收" in result["answer"]


@pytest.mark.asyncio
async def test_failed_validation_regenerates_once_then_refuses():
    calls=[]
    class Draft:
        def invoke(self, prompt):
            calls.append(prompt)
            return "错误结论[1]"
    async def validator(*a, **kw):
        return {"status":"failed","score":0,"issues":[],"checks":[]}
    result = await answer_pipeline("甲公司营收", pipeline(), model=Draft(), validator=validator)
    assert len(calls)==2
    assert result["answer_status"] == "refused"
    assert "错误结论" not in result["answer"]


@pytest.mark.asyncio
async def test_retrieval_error_raises_instead_of_success_answer():
    with pytest.raises(RuntimeError, match="数据库"):
        await answer_pipeline("甲营收", {"retrieval_error":"数据库不可用"}, model=Model())


@pytest.mark.asyncio
async def test_legacy_generation_cannot_bypass_missing_coverage():
    p=pipeline()
    p.pop("coverage")
    async def validator(*a, **kw):
        return {"status":"verified", "score":1, "issues":[]}
    result=await answer_pipeline("甲公司营收",p,model=Model(),validator=validator)
    assert result["answer_status"]=="unverified"
    assert "10亿元" not in result["answer"]


@pytest.mark.asyncio
async def test_partial_passes_allowed_and_conflicting_items_to_claim_gate():
    p=pipeline()
    p["decision"]="partial"
    p["coverage"]["missing_items"]=["q2"]
    p["plan"]["subproblems"].append({"id":"q2","query":"甲公司净利润","metric":"净利润"})
    class Draft:
        def invoke(self,prompt): return "甲公司营收10亿元[1]，净利润5亿元[1]"
    async def validator(*a, **kw):
        if not kw.get("blocked_items"):
            return {"status":"verified","score":1,"issues":[]}
        assert [item["id"] for item in kw["allowed_items"]]==["q1"]
        assert [item["id"] for item in kw["blocked_items"]]==["q2"]
        return {"status":"failed","score":0,"issues":["冲突事项不得发布"],"checks":[]}
    result=await answer_pipeline("甲公司营收和利润",p,model=Draft(),validator=validator)
    assert result["answer_status"]=="refused"
    assert "5亿元" not in result["answer"]


@pytest.mark.asyncio
async def test_sse_does_not_expose_private_supporting_quotes():
    from finance_rag.src.services.answer_service import answer_events
    result={"answer":"甲营收10亿元[1]","sources":[], "citation_validation":{"status":"verified","checks":[
        {"claim":"甲营收10亿元", "supporting_quote":"private-full-context","supporting_quotes":[{"quote":"private-full-context"}]}]}}
    events=[e async for e in answer_events(result)]
    assert "private-full-context" not in str(events)
