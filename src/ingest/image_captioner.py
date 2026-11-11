"""研报嵌套图片的中文描述（视觉模型）。

**这是本项目里数据外发范围最大的一条链路**：入库时会把每一张图原样发往
外部视觉模型（默认 DashScope ``qwen3-vl-plus``）。因此：

- 过小/过短的图片（印章、分隔线、页眉 logo）在本地就被拦下，不产生外发；
- 每次真正外发都写一条 ``image_egress`` 审计，便于合规回溯；
- 任何失败都只告警并返回空描述，不阻断入库。

配置值在调用时从 :mod:`src.core.config` 读取（而不是导入期绑定），
这样测试可以直接替换阈值，也不需要为了测一个边界去改环境变量。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
import urllib.request
from typing import Any

from src.core import config
from src.utils import audit

logger = logging.getLogger(__name__)

# 未知尺寸时的宽容策略：不因「量不出来」就把图丢掉。
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data) < 24:
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if data[:6] not in (b"GIF87a", b"GIF89a") or len(data) < 10:
        return None
    return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    index, size = 2, len(data)
    while index + 9 < size:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker == 0xD9:
            return None
        segment_length = int.from_bytes(data[index + 2 : index + 4], "big")
        if marker in _JPEG_SOF_MARKERS:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            return width, height
        index += 2 + segment_length
    return None


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """从图片头部解析 ``(宽, 高)``；解析不出返回 None（调用方应宽容处理）。"""
    for parser in (_png_dimensions, _gif_dimensions, _jpeg_dimensions):
        try:
            dimensions = parser(data)
        except Exception:  # noqa: BLE001 - 头部残缺不应抛
            continue
        if dimensions:
            return dimensions
    return None


def _build_vision_payload(data: bytes, *, ext: str, prompt: str) -> dict[str, Any]:
    """构造 OpenAI 兼容的多模态请求体（图片以 data URL 形式内联）。"""
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "model": config.IMAGE_CAPTION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{encoded}"},
                    },
                ],
            }
        ],
    }


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    """最小 JSON POST（标准库实现，避免为一次调用引入新依赖）。"""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - 配置化端点
        return json.loads(response.read().decode("utf-8"))


def _extract_text(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        # 部分兼容端点返回分段内容
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        ).strip()
    return str(content or "").strip()


def _call_vision(data: bytes, *, ext: str, prompt: str, timeout: float) -> str:
    api_key = config.IMAGE_CAPTION_API_KEY
    payload = _build_vision_payload(data, ext=ext, prompt=prompt)
    url = f"{config.IMAGE_CAPTION_BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return _extract_text(_post_json(url, payload, headers, timeout))


def caption_image(data: bytes, *, ext: str = "png", source: str = "") -> str:
    """为一张图片生成中文描述；被跳过或失败时返回空串。"""
    if not data or len(data) < config.IMAGE_CAPTION_MIN_BYTES:
        return ""

    dimensions = _image_dimensions(data)
    if dimensions is not None:
        width, height = dimensions
        if min(width, height) < config.IMAGE_CAPTION_MIN_DIM_PX:
            logger.debug("图片过小（%sx%s），跳过描述：%s", width, height, source)
            return ""

    prompt = config.IMAGE_CAPTION_PROMPT
    timeout = config.IMAGE_CAPTION_TIMEOUT_SECONDS
    attempts = max(1, int(config.IMAGE_CAPTION_MAX_RETRIES) + 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            text = _call_vision(data, ext=ext, prompt=prompt, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            # 描述失败不影响入库，因此这里对超时/限流/5xx/解析错误一视同仁地重试
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 4))
            continue
        if text:
            # 成功外发才记审计：失败没有产生数据出境
            audit.log(
                "image_egress",
                resource=source,
                detail=f"{config.IMAGE_CAPTION_MODEL} 图片描述",
                model=config.IMAGE_CAPTION_MODEL,
                bytes=len(data),
            )
            return text
        last_error = None
        break

    if last_error is not None:
        logger.warning("图片描述失败（%s）：%s", source or "unknown", last_error)
    return ""


_IMAGE_PLACEHOLDER = "[图片] 第 {page} 页（未能生成描述，可点击查看原图）"
# 视觉描述缺失但有原文图注时的降级文本：图注本身就是可召回的语义
_FIGURE_ONLY_PLACEHOLDER = "第 {page} 页图表：{caption}（未能生成视觉描述，可点击查看原图）"


def _asset_area(asset: Any) -> float:
    """图片面积（外发预算的优先级依据）；尺寸未知返回 inf（视为最大）。

    尺寸「量不出来」不能成为把图排到最后的原因——宁可优先外发。
    """
    dimensions = _image_dimensions(getattr(asset, "data", b"") or b"")
    if dimensions is None:
        return float("inf")
    return float(dimensions[0]) * float(dimensions[1])


def _figure_caption_text(
    captions_by_page: dict[int, list[str]],
    page: int,
    position_on_page: int,
    images_on_page: int,
) -> str:
    """取某张图对应的原文图注（页级对齐）。

    页内图片数与图注数相等 → 按出现顺序 1:1 对应；不等（MinerU 漏抽了
    图注或图注挂在整页）→ 该页全部图注拼接赋给页内每张图——宁重复，不丢语义。
    """
    page_captions = captions_by_page.get(page) or []
    if not page_captions:
        return ""
    if len(page_captions) == images_on_page:
        return page_captions[position_on_page]
    return "；".join(page_captions)


def build_image_chunks(
    assets: list[Any],
    *,
    source: str,
    title: str = "",
    start_index: int = 0,
    max_captions: int | None = None,
    figure_captions: dict[int, list[str]] | None = None,
    collection: str = "",
    version: str = "",
) -> list[Any]:
    """为每张图片生成一个可检索的子块。

    块内容按证据强度降级：

    1. **视觉描述（+ 原文图注拼接）**——content_list 的 ``img_caption`` /
       ``chart_caption`` 与视觉模型描述互补，两者都有则拼接；
    2. **仅原文图注**——视觉描述失败、图片过小被拦或超外发上限的图，
       至少还能靠图注被语义召回，而不是只剩无语义的占位文本；
    3. **占位文本**——两种来源都缺失，图存在但不可语义召回，
       分析师仍能通过页码跳转看到原图。

    ``max_captions`` 限制**调用视觉模型**的图片数量（默认取
    ``IMAGE_CAPTION_MAX_IMAGES``）：研报动辄几十张图，逐张外发既慢又贵。
    预算按**图片面积降序**分配——小图（印章/图标）信息量低，大图（财务
    图表）才是召回价值所在；尺寸未知的图视为最大，不因「量不出来」被排后。
    超出上限的图片仍然建块（内容为图注或占位文本），不会从索引里消失。

    块 id 只由 ``source + 图片在文中的位置`` 决定，与描述内容、外发预算
    均无关，因此重复入库是覆盖而非新增；预算排序也不影响块身份。
    """
    from langchain_core.documents import Document

    from src.ingest.pdf_assets import image_object_key

    limit = max(0, int(config.IMAGE_CAPTION_MAX_IMAGES if max_captions is None else max_captions))
    captions_by_page = figure_captions or {}

    # 外发预算按面积降序分配；面积相同按原文顺序（确定性，便于复现）
    order_by_area = sorted(
        range(len(assets)), key=lambda i: (-_asset_area(assets[i]), i)
    )
    egress_set = set(order_by_area[:limit])
    if len(assets) > limit:
        logger.info(
            "图片数 %d 超过外发上限 %d，按面积优先选取 %d 张，其余仅用图注/占位文本：%s",
            len(assets), limit, limit, source,
        )

    # 页级对齐需要每页图片数与页内序号
    page_image_counts: dict[int, int] = {}
    for asset in assets:
        page_image_counts[asset.page] = page_image_counts.get(asset.page, 0) + 1
    page_seen: dict[int, int] = {}

    chunks: list[Any] = []
    for offset, asset in enumerate(assets):
        if offset in egress_set:
            caption = caption_image(asset.data, ext=asset.ext, source=source)
        else:
            caption = ""
        position_on_page = page_seen.get(asset.page, 0)
        page_seen[asset.page] = position_on_page + 1
        fig_text = _figure_caption_text(
            captions_by_page, asset.page, position_on_page,
            page_image_counts.get(asset.page, 0),
        )

        if caption and fig_text:
            text = f"{caption}\n原文图注：{fig_text}"
        elif caption:
            text = caption
        elif fig_text:
            text = _FIGURE_ONLY_PLACEHOLDER.format(page=asset.page, caption=fig_text)
        else:
            text = _IMAGE_PLACEHOLDER.format(page=asset.page)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_id = hashlib.sha256(
            f"{source}:image:{asset.key_hint}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "id": chunk_id,
                    "chunk_key": chunk_id,
                    "content_hash": digest,
                    "source": source,
                    "title": title or source,
                    "chunk": start_index + offset,
                    "parent_id": "",
                    "heading": "",
                    "heading_path": "",
                    "block_type": "image",
                    "start_page": asset.page,
                    "end_page": asset.page,
                    # 图片本体在对象存储的 key：前端「查看原图」与审计都用它。
                    # 与上传侧共用同一 key 计算（集合/版本隔离，见 object_keys）。
                    "image_key": image_object_key(
                        source, asset, collection=collection, version=version
                    )[:256],
                },
            )
        )
    return chunks


__all__ = ["build_image_chunks", "caption_image"]
