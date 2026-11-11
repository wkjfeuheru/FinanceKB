import pytest

from src.agent.planning import Subproblem
from src.agent.nodes.decision import decide_intent, request_tool_calls


def test_greeting_and_financial_fact_are_distinct():
    assert decide_intent("你好", []) == "direct"
    assert decide_intent("你好，甲公司营收是多少？", []) == "retrieve"
    assert decide_intent("它的营收呢？", []) == "clarify"


def test_fact_tool_call_is_forced_when_model_attempts_direct_answer():
    class Model:
        def bind_tools(self, tools):
            assert tools[0]["function"]["parameters"]["additionalProperties"] is False
            return self
        def invoke(self, prompt):
            return type("Response", (), {"tool_calls": []})()
    result = request_tool_calls([Subproblem(id="q1", query="甲营收")], ["甲营收"], Model())
    assert result["policy_forced"]
    assert result["calls"][0]["args"] == {"query": "甲营收", "subproblem_id": "q1"}


@pytest.mark.parametrize("args", [{"query":"甲营收","subproblem_id":"q1","filters":{}},
                                  {"query":"甲营收","subproblem_id":"invented"}])
def test_invalid_tool_arguments_never_execute(args):
    class Model:
        def bind_tools(self, tools): return self
        def invoke(self, prompt):
            return type("Response", (), {"tool_calls": [{"id":"call1","name":"retrieve_finance_evidence","args":args}]})()
    assert request_tool_calls([Subproblem(id="q1", query="甲营收")], ["甲营收"], Model())["error"]
