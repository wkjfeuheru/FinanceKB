"""生成和核验共用的模型调用与多模态消息。"""

from __future__ import annotations

import asyncio
from typing import Any


def _message_text(response: Any) -> str:
    text = response if isinstance(response, str) else getattr(response, "content", response)
    if isinstance(text, list):
        text = " ".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in text
        )
    return str(text or "").strip()


def model_input(prompt: str, images: list[dict] | None = None) -> Any:
    """没有原图时仍传纯文本；有原图时按引用编号附上 data URL。"""
    if not images:
        return prompt
    from langchain_core.messages import HumanMessage

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "text", "text": f"[{image['index']}] 原图"})
        content.append({"type": "image_url", "image_url": {"url": image["data_url"]}})
    return [HumanMessage(content=content)]


async def invoke_model(model: Any, prompt: str, images: list[dict] | None = None) -> Any:
    if model is None:
        raise RuntimeError("LLM未配置")
    payload = model_input(prompt, images)
    if callable(getattr(model, "ainvoke", None)):
        return await model.ainvoke(payload)
    if callable(getattr(model, "invoke", None)):
        return await asyncio.to_thread(model.invoke, payload)
    raise RuntimeError("模型缺少调用接口")


async def invoke_text(model: Any, prompt: str, images: list[dict] | None = None) -> str:
    text = _message_text(await invoke_model(model, prompt, images))
    if not text:
        raise RuntimeError("生成模型未返回有效答案")
    return text
