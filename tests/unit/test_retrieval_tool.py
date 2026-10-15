from finance_rag.src.agent.tools.retrieval import AgentRetrievalTool, RetrievalTool


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [{
            "chunk_id": "chunk-1",
            "content": "金融监管规则",
            "score": "0.85",
            "source": "rules.pdf",
            "category": "policy",
        }]


def test_empty_query_does_not_call_retriever():
    retriever = FakeRetriever()

    result = RetrievalTool(retriever).invoke("  ")

    assert result == {"query": "  ", "evidence": [], "metadata": {"count": 0}}
    assert retriever.calls == []


def test_retrieval_tool_normalizes_evidence_and_forwards_options():
    retriever = FakeRetriever()

    result = RetrievalTool(retriever).invoke(
        "  监管规则  ", k=2, filters={"category": "policy"}, use_rerank=True
    )

    assert result["query"] == "  监管规则  "
    assert result["metadata"] == {"count": 1, "filters": {"category": "policy"}}
    expected = {
        "id": "chunk-1",
        "content": "金融监管规则",
        "score": 0.85,
        "source": "rules.pdf",
        "title": "",
        "chunk": None,
        "parent_id": "",
        "category": "policy",
        "date": "",
        "heading": "",
        "heading_path": "",
        "rerank_score": None,
        "block_type": "text",
        "start_page": 0,
        "end_page": 0,
        "image_key": "",
    }
    assert len(result["evidence"]) == 1
    assert {key: result["evidence"][0][key] for key in expected} == expected
    assert retriever.calls[0][0] == "监管规则"
    assert retriever.calls[0][1]["k"] == 2
    assert retriever.calls[0][1]["use_rerank"] is True


def test_normalize_evidence_passes_through_table_and_image_fields():
    """block_type/页码/image_key 必须透传：输出期整表展开与上下文预算靠它们。"""
    from finance_rag.src.agent.tools.retrieval import _normalize_evidence

    evidence = _normalize_evidence({
        "id": "c-table", "content": "表头：年份 | 营收", "parent_id": "t1",
        "block_type": "table", "start_page": 4, "end_page": 4,
        "score": 0.5,
    }, 0)

    assert evidence["block_type"] == "table"
    assert evidence["start_page"] == 4
    assert evidence["end_page"] == 4
    assert evidence["parent_id"] == "t1"


def test_agent_tool_exposes_stable_name_and_description():
    assert AgentRetrievalTool.name == "retrieve_finance_evidence"
    assert AgentRetrievalTool.description
