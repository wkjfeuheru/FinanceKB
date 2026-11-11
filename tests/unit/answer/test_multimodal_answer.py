"""最终回答附带 OSS 原图，图片事实不因图注占位被提前拒绝。"""
from __future__ import annotations

import json

import pytest

from src.agent.planning import QueryPlan, Subproblem
from src.agent.nodes.grading import decide_from_coverage, grade_coverage
from src.answer.generator import answer_pipeline, collect_context_images
from src.answer.citations import validate_claims


class _Storage:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.keys: list[str] = []

    def download_sync(self, key: str) -> bytes:
        self.keys.append(key)
        if self.fail:
            raise FileNotFoundError(key)
        return b"\x89PNG\r\n\x1a\n"


class _Recorder:
    def __init__(self, text: str) -> None:
        self.text = text
        self.payloads: list = []

    def invoke(self, payload):
        self.payloads.append(payload)
        return self.text


def _image_doc() -> dict:
    return {
        "id": "c1",
        "collection": "kb",
        "content": "第3页图表（未能生成视觉描述）",
        "block_type": "image",
        "image_key": "default/finance_kb/doc/images/chart.png",
    }


def _pipeline(**extra) -> dict:
    base = {
        "decision": "generate",
        "coverage": {
            "supported_items": ["q1"],
            "visual_pending_items": [],
            "missing_items": [],
            "unknown": False,
            "sufficient": True,
        },
        "plan": {"needs_decomposition": False, "subproblems": [{"id": "q1", "query": "营业收入"}]},
        "context_text": "[1] 甲公司营业收入10亿元",
        "context_docs": [{"content": "甲公司营业收入10亿元", "id": "c1", "block_type": "text"}],
        "sources": [{"index": 1, "title": "报告", "source": "a.pdf", "score": 1, "preview": "甲"}],
    }
    base.update(extra)
    return base


def test_answer_model_factory_uses_vision_settings(monkeypatch):
    from src.core import config

    captured: dict = {}

    def init_chat_model(model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return "client"

    monkeypatch.setattr(config, "IMAGE_CAPTION_API_KEY", "test-key")
    monkeypatch.setattr(config, "IMAGE_CAPTION_MODEL", "qwen-vl-max")
    monkeypatch.setattr(config, "IMAGE_CAPTION_BASE_URL", "https://dashscope.example/v1")
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 120.0)
    monkeypatch.setattr("langchain.chat_models.init_chat_model", init_chat_model)
    assert config.create_answer_model() == "client"
    assert captured["model_id"] == "openai:qwen-vl-max"
    assert captured["base_url"] == "https://dashscope.example/v1"
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 120.0
    monkeypatch.setattr(config, "IMAGE_CAPTION_API_KEY", "")
    assert config.create_answer_model() is None


@pytest.mark.asyncio
async def test_text_only_answer_stays_plain_text():
    model = _Recorder("营业收入10亿元[1]。")

    async def validator(answer, docs, **kwargs):
        assert kwargs["images"] == []
        return {"status": "verified", "score": 1, "issues": [], "checks": []}

    result = await answer_pipeline("营业收入", _pipeline(), model=model, validator=validator)
    assert result["answer_status"] == "complete"
    assert isinstance(model.payloads[0], str)
    assert "data:image" not in model.payloads[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefer_text", [False, True])
async def test_image_block_attaches_oss_original_by_citation_index(monkeypatch, prefer_text):
    storage = _Storage()
    monkeypatch.setattr(
        "src.storage.objects.get_storage", lambda: storage
    )
    model = _Recorder("营业收入10亿元[1]。")
    seen: dict = {}

    async def validator(answer, docs, **kwargs):
        seen["images"] = kwargs["images"]
        return {"status": "verified", "score": 1, "issues": [], "checks": []}

    doc = _image_doc()
    result = await answer_pipeline(
        "营业收入",
        _pipeline(
            context_text="[1] 第3页图表",
            context_docs=[doc],
            coverage={
                "supported_items": [],
                "visual_pending_items": ["q1"],
                "missing_items": [],
                "unknown": False,
                "sufficient": False,
            },
            decision="refuse",
        ),
        model=model,
        validator=validator,
        prefer_text=prefer_text,
    )
    assert result["answer_status"] == "complete"
    assert storage.keys == [doc["image_key"]]
    message = model.payloads[0][0]
    parts = message.content
    assert parts[1]["text"] == "[1] 原图"
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["images"][0]["index"] == 1
    assert seen["images"][0]["data_url"] == parts[2]["image_url"]["url"]


@pytest.mark.asyncio
async def test_image_download_failure_still_answers_without_original(monkeypatch):
    monkeypatch.setattr(
        "src.storage.objects.get_storage", lambda: _Storage(fail=True)
    )
    model = _Recorder("图表未能生成视觉描述[1]")

    async def validator(answer, docs, **kwargs):
        assert kwargs["images"] == []
        return {"status": "verified", "score": 1, "issues": [], "checks": []}

    result = await answer_pipeline(
        "营业收入",
        _pipeline(context_text="[1] 第3页图表", context_docs=[_image_doc()]),
        model=model,
        validator=validator,
    )
    assert result["answer_status"] == "complete"
    assert isinstance(model.payloads[0], str)
    assert "data:image" not in model.payloads[0]


def test_duplicate_image_key_is_downloaded_once_and_cap_keeps_context_order(monkeypatch):
    storage = _Storage()
    monkeypatch.setattr(
        "src.storage.objects.get_storage", lambda: storage
    )
    monkeypatch.setattr("src.core.config.ANSWER_MAX_IMAGES", 1)
    docs = [
        {**_image_doc(), "image_key": "a.png"},
        {**_image_doc(), "id": "c2", "image_key": "a.png"},
        {**_image_doc(), "id": "c3", "image_key": "b.png"},
    ]
    images = collect_context_images(docs)
    assert storage.keys == ["a.png"]
    assert [item["index"] for item in images] == [1, 2]


@pytest.mark.asyncio
async def test_image_claim_can_publish_when_quote_is_not_in_caption():
    class Model:
        def invoke(self, payload):
            assert "data:image/png;base64," in str(payload)
            return json.dumps({
                "checks": [{
                    "claim": "营业收入10亿元",
                    "citation_indices": [1],
                    "status": "supported",
                    "supporting_quote": "营业收入10亿元",
                    "reason": "",
                }]
            }, ensure_ascii=False)

    result = await validate_claims(
        "营业收入10亿元[1]。",
        [_image_doc()],
        model=Model(),
        images=[{"index": 1, "image_key": _image_doc()["image_key"], "data_url": "data:image/png;base64,YQ=="}],
    )
    assert result["status"] == "verified"


@pytest.mark.asyncio
async def test_text_claim_still_rejects_quote_missing_from_source():
    class Model:
        def invoke(self, payload):
            return json.dumps({
                "checks": [{
                    "claim": "营业收入10亿元",
                    "citation_indices": [1],
                    "status": "supported",
                    "supporting_quote": "营业收入10亿元",
                    "reason": "",
                }]
            }, ensure_ascii=False)

    result = await validate_claims(
        "营业收入10亿元[1]。",
        [{"content": "无关文本", "block_type": "text"}],
        model=Model(),
    )
    assert result["status"] == "failed"


def test_placeholder_image_evidence_is_not_refused_before_generation():
    class Model:
        def invoke(self, prompt):
            return json.dumps({
                "items": [{
                    "item_id": "q1",
                    "status": "missing",
                    "evidence_ids": ["kb:c1"],
                    "reason": "图注没有数字",
                }]
            }, ensure_ascii=False)

    plan = QueryPlan(
        needs_decomposition=False,
        subproblems=[Subproblem(id="q1", query="营业收入", entity="甲公司", period="2024")],
    )
    covered = grade_coverage(plan, [_image_doc()], model=Model())
    assert covered["visual_pending_items"] == ["q1"]
    assert covered["supported_items"] == []
    assert "q1" not in covered["missing_items"]
    assert not covered["sufficient"]
    assert decide_from_coverage(covered) == "generate"

    class Supported:
        def invoke(self, prompt):
            return json.dumps({
                "items": [{
                    "item_id": "q1",
                    "status": "supported",
                    "evidence_ids": ["kb:c1"],
                    "reason": "",
                }]
            }, ensure_ascii=False)

    kept = grade_coverage(plan, [_image_doc()], model=Supported())
    assert kept["supported_items"] == ["q1"]
