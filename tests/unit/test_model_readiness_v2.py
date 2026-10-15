import json

import pytest

from scripts.check_agentic_model import check_model


@pytest.mark.asyncio
async def test_missing_tool_calls_do_not_claim_endpoint_compatible():
    class Model:
        def bind_tools(self, tools): return self
        def invoke(self, prompt): return type("Response", (), {"tool_calls":[]})()
    result=await check_model(Model())
    assert result["tool_protocol"]=="failed"
    assert not result["ready"]


@pytest.mark.asyncio
async def test_actual_tool_protocol_and_claim_json_are_required():
    class Model:
        def bind_tools(self, tools): return self
        def invoke(self, prompt):
            if "必须为每个必要事项" in prompt:
                return type("Response", (), {"tool_calls":[{"id":"call1","name":"retrieve_finance_evidence",
                    "args":{"query":"测试公司营业收入","subproblem_id":"q1"}}]})()
            data=json.loads(prompt.split("\n",1)[1])
            return json.dumps({"checks":[{**claim,"status":"supported",
                "supporting_quote":"测试公司营业收入10亿元","reason":"原文支持"}
                for claim in data["supplied_claims"]]},ensure_ascii=False)
    result=await check_model(Model())
    assert result["ready"]
    assert result["claim_protocol"]=="verified"
