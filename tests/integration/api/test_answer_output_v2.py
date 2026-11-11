import pytest

@pytest.fixture
def chat_service():
    # 避免收集本文件时提前加载.env，干扰鉴权测试的导入前环境设置。
    from src import engine as service
    return service


@pytest.mark.asyncio
async def test_http_and_sse_share_result_and_do_not_revalidate(monkeypatch, chat_service):
    from src.answer import generator as answer_service
    from src.retrieve import pipeline as retrieval_service
    calls=[]
    async def retrieval(*a, **kw):
        return {"docs":[],"decision":"unverified"}
    async def answer(*a, **kw):
        calls.append("answer")
        return {"answer":"无法验证", "sources":[],"rewritten_query":"甲营收","filters":{},
                "citation_validation":{"status":"unknown","score":None,"issues":[]},
                "answer_rejected":False,"low_confidence":True,"answer_status":"unverified","missing_items":[]}
    monkeypatch.setattr(retrieval_service,"execute_retrieval",retrieval)
    monkeypatch.setattr(answer_service,"answer_pipeline",answer)
    monkeypatch.setattr(chat_service,"ENABLE_LANGGRAPH",False)
    monkeypatch.setattr(chat_service,"get_model",lambda:object())
    events=[e async for e in chat_service.chat_stream("甲营收")]
    assert [e["content"] for e in events if e["type"]=="token"] == ["无法验证"]
    result=await chat_service.chat("甲营收")
    assert result["answer_status"] == "unverified"
    assert result["citation_validation"]["score"] is None
    assert calls == ["answer","answer"]


@pytest.mark.asyncio
async def test_non_stream_error_is_not_empty_success(monkeypatch, chat_service):
    async def stream(*a, **kw):
        yield {"type":"error","message":"数据库不可用"}
    monkeypatch.setattr(chat_service,"chat_stream",stream)
    with pytest.raises(RuntimeError,match="数据库不可用"):
        await chat_service.chat("甲营收")


@pytest.mark.asyncio
@pytest.mark.parametrize("amount,expected", [("10.0", "complete"), ("20", "refused")])
async def test_real_answer_gate_reports_survive_standard_chat_and_sse(
    monkeypatch, chat_service, amount, expected,
):
    import json

    from src.answer import citations
    from src.retrieve import pipeline as retrieval

    claim = f"甲公司营收{amount}亿元"

    class OfflineModel:
        def invoke(self, prompt):
            if "supplied_claims" in prompt:
                return json.dumps({"checks": [{
                    "claim": claim, "citation_indices": [1], "status": "supported",
                    "supporting_quote": "甲公司营收10亿元", "reason": "原文支持",
                }]}, ensure_ascii=False)
            return claim + "[1]"

    async def search(*args, **kwargs):
        return {
            "docs": [{"id": "c1", "content": "甲公司营收10亿元", "source": "a.pdf", "score": 1}],
            "decision": "generate", "coverage": {"supported_items": ["q1"], "missing_items": []},
        }

    monkeypatch.setattr(chat_service, "ENABLE_LANGGRAPH", False)
    monkeypatch.setattr(chat_service, "get_answer_model", lambda: OfflineModel())
    monkeypatch.setattr(citations, "ENABLE_CITATION_VALIDATION", True)
    monkeypatch.setattr(retrieval, "execute_retrieval", search)
    events = [event async for event in chat_service.chat_stream("甲公司营收是多少")]
    result = await chat_service.chat("甲公司营收是多少")
    assert events[-1]["answer_status"] == result["answer_status"] == expected
    assert result["citation_validation"]["citations"]["used"] == [1]
    assert events[-1]["citation_validation"] == result["citation_validation"]
    assert result["citation_validation"]["numbers"]["status"] == (
        "verified" if expected == "complete" else "failed"
    )
    assert "supporting_quote" not in str(result)
    if expected == "refused":
        assert claim not in result["answer"]
