import pytest

@pytest.fixture
def chat_service():
    # 避免收集本文件时提前加载.env，干扰鉴权测试的导入前环境设置。
    from finance_rag.src.services import chat_service as service
    return service


@pytest.mark.asyncio
async def test_http_and_sse_share_result_and_do_not_revalidate(monkeypatch, chat_service):
    from finance_rag.src.services import answer_service, retrieval_service
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
