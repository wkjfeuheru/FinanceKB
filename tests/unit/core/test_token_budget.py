"""模型窗口级总预算回归（RAG-15）。

``docs/rag-review.md`` 的问题：问题、历史内容与条数缺少上限，复杂规划发送全部
历史；证据只做单块字符截断，没有模型窗口级总预算；超长输入可能表现为异常澄清
或不可控超时。

验收标准：
* 超长问题及历史得到明确、可理解的限制反馈；
* 所有模型调用的输入与预留输出之和不超过配置窗口；
* 裁剪后的事实核验只使用实际送入生成模型的证据，来源编号无悬空、错位；
* 截断标记准确，表格不在金额、行或标签中间粗暴切断。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core import config, token_budget
from src.api.schemas.chat import ChatRequest
from src.engine import build_context


def _doc(doc_id: str, content: str, *, block_type: str = "text", **extra) -> dict:
    doc = {
        "id": doc_id,
        "collection": "kb",
        "source": f"{doc_id}.pdf",
        "title": f"{doc_id} 标题",
        "content": content,
        "block_type": block_type,
        "date": "2026-08-15",
        "start_page": 1,
        "end_page": 2,
    }
    doc.update(extra)
    return doc


# ---------------------------------------------------------------------------
# 请求入口限制：明确反馈，而不是静默截断或异常澄清
# ---------------------------------------------------------------------------


def test_overlong_query_is_rejected_with_actionable_message():
    with pytest.raises(ValidationError) as excinfo:
        ChatRequest(query="问" * (config.CHAT_MAX_QUERY_CHARS + 1))

    message = str(excinfo.value)
    assert "query" in message
    assert str(config.CHAT_MAX_QUERY_CHARS) in message


def test_too_many_history_messages_are_rejected():
    history = [{"role": "user", "content": f"第{i}问"} for i in range(config.CHAT_MAX_HISTORY_TURNS + 1)]

    with pytest.raises(ValidationError) as excinfo:
        ChatRequest(query="当前问题", history=history)

    assert "history" in str(excinfo.value)


def test_overlong_history_content_is_rejected_with_reason():
    """条数合规但总长度超预算时同样要有明确反馈（只限条数挡不住超长消息）。"""
    content = "长" * config.CHAT_MAX_HISTORY_MESSAGE_CHARS
    history = [{"role": "user", "content": content}
               for _ in range(config.CHAT_MAX_HISTORY_TURNS)]

    with pytest.raises(ValidationError) as excinfo:
        ChatRequest(query="当前问题", history=history)

    assert "历史对话" in str(excinfo.value)


def test_normal_request_passes_limits():
    request = ChatRequest(query="贵州茅台2026H1营业收入是多少？",
                          history=[{"role": "user", "content": "介绍一下贵州茅台"}])

    assert request.query
    assert len(request.history) == 1


# ---------------------------------------------------------------------------
# Token 估算与窗口校验
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_conservative():
    """中文按最坏密度保守估算：1 个汉字不少于 0.707 token。"""
    assert token_budget.estimate_tokens("中" * 100) >= 70
    assert token_budget.estimate_tokens("") == 0


def test_input_plus_reserved_output_never_exceeds_window(monkeypatch):
    monkeypatch.setattr(config, "MODEL_CONTEXT_WINDOW", 4096)
    monkeypatch.setattr(config, "MODEL_RESERVED_OUTPUT_TOKENS", 1024)
    monkeypatch.setattr(config, "MODEL_SAFETY_MARGIN_TOKENS", 256)

    available = token_budget.available_input_tokens()
    assert available == 4096 - 1024 - 256
    # 可用预算本身留了安全余量，因此不报错
    token_budget.require_within_window(available, kind="生成")
    # 输入 + 预留输出一旦超过窗口就必须报错
    with pytest.raises(ValueError, match="超过模型窗口"):
        token_budget.require_within_window(4096 - 1024 + 1, kind="生成")


def test_require_within_window_message_is_actionable(monkeypatch):
    monkeypatch.setattr(config, "MODEL_CONTEXT_WINDOW", 2048)
    with pytest.raises(ValueError) as excinfo:
        token_budget.require_within_window(4096, kind="规划")

    message = str(excinfo.value)
    assert "规划" in message and "缩短问题或历史" in message


# ---------------------------------------------------------------------------
# 历史滑窗：保留最新轮次，超长消息按边界裁剪
# ---------------------------------------------------------------------------


def test_history_keeps_newest_turns_within_budget():
    history = [{"role": "user", "content": f"第{i}轮问题" * 20} for i in range(10)]

    kept, report = token_budget.fit_history(history, max_tokens=200)

    assert kept, "至少要保留最新一轮"
    assert kept[-1]["content"].startswith("第9轮问题")
    assert report["dropped"] > 0
    assert report["tokens"] <= 200
    # 最新一轮永远优先于更早的轮次
    assert report["kept"] < report["input_messages"]


def test_history_respects_turn_cap():
    history = [{"role": "user", "content": "短问题"} for _ in range(50)]

    kept, report = token_budget.fit_history(history, max_tokens=100000, max_turns=5)

    assert len(kept) == 5
    assert report["dropped"] == 45


def test_overlong_history_message_is_clipped_with_marker():
    history = [{"role": "user", "content": "很长的历史内容。" * 500}]

    kept, report = token_budget.fit_history(history, max_tokens=120, max_turns=5)

    assert report["truncated_messages"] == 1
    assert kept[0]["content"].endswith(token_budget.HISTORY_TRUNCATION_NOTE)


# ---------------------------------------------------------------------------
# 证据总预算：编号连续、表格按行裁剪
# ---------------------------------------------------------------------------


def test_evidence_budget_drops_blocks_beyond_budget(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MAX_CONTEXT_TOKENS", 400)
    docs = [_doc(f"c{i}", "营业收入100亿元。" * 10) for i in range(20)]

    context, sources = build_context(docs)

    assert 0 < len(sources) < 20, "超出总预算的块必须被丢弃"
    # 来源编号与上下文编号一一对应，无悬空或错位
    assert [src["index"] for src in sources] == list(range(1, len(sources) + 1))
    for index in range(1, len(sources) + 1):
        assert f"[{index}] " in context
    assert f"[{len(sources) + 1}]" not in context


def test_context_docs_match_sources_and_carry_clipped_content():
    """送入生成模型的 context_docs 必须与 sources 等长同序（核验只用实际证据）。"""
    docs = [_doc(f"c{i}", "证据正文" * 50) for i in range(12)]
    context_docs: list[dict] = []

    context, sources = build_context(docs, context_docs=context_docs)

    assert len(context_docs) == len(sources)
    assert [doc["id"] for doc in context_docs] == [src["id"] for src in sources]
    for doc in context_docs:
        assert doc["content"] in context


def test_budget_report_records_dropped_and_truncated_blocks(monkeypatch):
    monkeypatch.setattr(config, "CHAT_MAX_CONTEXT_TOKENS", 300)
    docs = [_doc(f"c{i}", "很长的证据正文。" * 40) for i in range(6)]
    report: dict = {}

    build_context(docs, budget_report=report)

    assert report["input_blocks"] == 6
    assert report["kept"] < 6
    assert report["dropped"] == 6 - report["kept"]
    assert report["tokens"] <= 300


def test_table_is_clipped_on_row_boundary(monkeypatch):
    """表格按行边界裁剪：不在金额、行或标签中间粗暴切断。"""
    monkeypatch.setattr(config, "CHAT_MAX_CONTEXT_TOKENS", 260)
    rows = "\n".join(f"| 2026E | 第{i}行金额 10000 万元 |" for i in range(60))
    table = f"| 年份 | 金额 |\n| --- | --- |\n{rows}"
    report: dict = {}

    context, sources = build_context([_doc("t1", table, block_type="table")], budget_report=report)

    body = context.split("\n", 1)[1]
    table_lines = [line for line in body.splitlines() if line.startswith("|")]
    # 每个保留的表格行都是完整行（列数与表头一致）
    assert table_lines, "表格至少要保留表头与一行"
    for line in table_lines:
        if "|" in line and "---" not in line:
            assert line.count("|") == 3
            assert line.rstrip().endswith("|")
    assert report["truncated_blocks"] == 1
    assert token_budget.TRUNCATION_NOTE in body


def test_text_block_clipping_prefers_sentence_boundary():
    text, truncated = token_budget.clip_to_budget("第一句。" + "第二句。" * 200, 60)

    assert truncated is True
    assert text.endswith(token_budget.TRUNCATION_NOTE)
    assert text.startswith("第一句。第二句。")


def test_empty_context_is_reported_not_faked():
    context, sources = build_context([])
    assert context == "（未检索到相关文档）"
    assert sources == []


# ---------------------------------------------------------------------------
# 所有模型调用都受窗口约束：核验与生成超限时明确降级，不发超窗请求
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_over_window_returns_unknown_without_calling_model(monkeypatch):
    from types import SimpleNamespace

    from src.answer import citations as validator

    monkeypatch.setattr(config, "MODEL_CONTEXT_WINDOW", 1024)
    monkeypatch.setattr(config, "MODEL_RESERVED_OUTPUT_TOKENS", 256)

    called: list[str] = []

    class Model:
        async def ainvoke(self, prompt):
            called.append(prompt)
            return SimpleNamespace(content='{"checks":[]}')

    result = await validator.validate_claims(
        "营业收入100亿元[1]。" + "补充说明[1]。" * 200,
        [{"content": "营业收入100亿元。" * 100}],
        model=Model(),
        enabled=True,
    )

    assert result["status"] == "unknown"
    assert any("超过模型窗口" in issue for issue in result["issues"])
    assert called == [], "超窗时不得发出核验请求"


@pytest.mark.asyncio
async def test_generation_over_window_returns_unverified_without_calling_model(monkeypatch):
    from src.answer.generator import (
        UNVERIFIED_ANSWER,
        answer_pipeline,
    )

    monkeypatch.setattr(config, "MODEL_CONTEXT_WINDOW", 1024)
    monkeypatch.setattr(config, "MODEL_RESERVED_OUTPUT_TOKENS", 256)

    called: list[str] = []

    class Model:
        async def ainvoke(self, prompt):
            called.append(prompt)
            return "答案[1]"

    pipeline = {
        "decision": "generate",
        "context_text": "证据正文。" * 2000,
        "context_docs": [{"content": "证据正文。" * 2000, "id": "c1"}],
        "sources": [],
        "coverage": {"supported_items": ["q1"], "missing_items": [], "sufficient": True, "unknown": False},
        "plan": {"subproblems": [{"id": "q1", "query": "贵州茅台营业收入", "depends_on": []}]},
    }

    result = await answer_pipeline("贵州茅台营业收入", pipeline, model=Model())

    assert result["answer"] == UNVERIFIED_ANSWER
    assert result["answer_status"] == "unverified"
    assert any("超过模型窗口" in issue for issue in result["citation_validation"]["issues"])
    assert called == [], "超窗时不得发出生成请求"
