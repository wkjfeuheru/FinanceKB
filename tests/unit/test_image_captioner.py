"""研报图片描述（视觉模型）的回归测试。

这条链路会**把研报图片原样发往外部视觉模型**（DashScope qwen-vl-max），
是本项目里数据外发范围最大的一处，因此这里同时锁住「该跳过的不外发」
与「外发必须留审计」。所有网络调用都被替换，测试不联网。
"""

import pymupdf

from finance_rag.src.core import config
from finance_rag.src.rag.ingestion import image_captioner
from finance_rag.src.rag.ingestion import pdf_assets


def _png(width: int, height: int) -> bytes:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pixmap.clear_with(255)
    return pixmap.tobytes("png")


def _open_limits(monkeypatch, *, min_dim: int = 1, min_bytes: int = 1) -> None:
    monkeypatch.setattr(config, "IMAGE_CAPTION_MIN_DIM_PX", min_dim)
    monkeypatch.setattr(config, "IMAGE_CAPTION_MIN_BYTES", min_bytes)


def test_vision_payload_embeds_base64_image_and_prompt():
    payload = image_captioner._build_vision_payload(b"abc", ext="png", prompt="描述这张图")

    message = payload["messages"][0]
    assert message["role"] == "user"
    parts = message["content"]
    assert parts[0]["text"] == "描述这张图"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,YWJj"


def test_caption_skips_images_below_min_dimensions(monkeypatch):
    _open_limits(monkeypatch, min_dim=80, min_bytes=1)
    called: list[bytes] = []
    monkeypatch.setattr(image_captioner, "_post_json", lambda *a, **k: called.append(b"x") or {})

    assert image_captioner.caption_image(_png(4, 4), ext="png") == ""
    assert called == []          # 过小的图不产生任何外发


def test_caption_skips_images_below_min_bytes(monkeypatch):
    _open_limits(monkeypatch, min_dim=1, min_bytes=10**9)
    called: list[bytes] = []
    monkeypatch.setattr(image_captioner, "_post_json", lambda *a, **k: called.append(b"x") or {})

    assert image_captioner.caption_image(_png(40, 40), ext="png") == ""
    assert called == []


def test_caption_returns_model_text(monkeypatch):
    _open_limits(monkeypatch)
    monkeypatch.setattr(
        image_captioner,
        "_post_json",
        lambda *a, **k: {"choices": [{"message": {"content": "  燃机订单结构图  "}}]},
    )

    assert image_captioner.caption_image(_png(40, 40), ext="png") == "燃机订单结构图"


def test_caption_retries_then_gives_up(monkeypatch):
    _open_limits(monkeypatch)
    monkeypatch.setattr(config, "IMAGE_CAPTION_MAX_RETRIES", 2)
    monkeypatch.setattr(image_captioner.time, "sleep", lambda _seconds: None)
    attempts: list[int] = []

    def _boom(*a, **k):
        attempts.append(1)
        raise RuntimeError("限流")

    monkeypatch.setattr(image_captioner, "_post_json", _boom)

    assert image_captioner.caption_image(_png(40, 40), ext="png") == ""
    assert len(attempts) == 3      # 首次 + 2 次重试


def test_caption_logs_egress_audit(monkeypatch):
    _open_limits(monkeypatch)
    monkeypatch.setattr(
        image_captioner, "_post_json", lambda *a, **k: {"choices": [{"message": {"content": "图"}}]}
    )
    entries: list[dict] = []
    monkeypatch.setattr(
        image_captioner.audit,
        "log",
        lambda action, **kw: entries.append({"action": action, **kw}),
    )

    image_captioner.caption_image(_png(40, 40), ext="png", source="研报.pdf")

    assert entries and entries[0]["action"] == "image_egress"
    assert entries[0]["resource"] == "研报.pdf"


def test_caption_failure_is_not_audited_as_egress(monkeypatch):
    _open_limits(monkeypatch)
    monkeypatch.setattr(config, "IMAGE_CAPTION_MAX_RETRIES", 0)

    def _boom(*a, **k):
        raise RuntimeError("网络不可达")

    monkeypatch.setattr(image_captioner, "_post_json", _boom)
    entries: list[dict] = []
    monkeypatch.setattr(
        image_captioner.audit, "log", lambda action, **kw: entries.append({"action": action, **kw})
    )

    assert image_captioner.caption_image(_png(40, 40), ext="png") == ""
    assert [entry for entry in entries if entry["action"] == "image_egress"] == []


def test_unknown_dimensions_are_not_treated_as_too_small():
    """尺寸解析不出来时不能误判为「太小」——宁可外发，也不能静默丢图。"""
    truncated = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    assert image_captioner._image_dimensions(truncated) is None


def test_caption_calls_are_capped_but_all_images_still_get_chunks(monkeypatch):
    """上限只约束「外发次数」：超限的图仍建占位块，不会从索引里消失。"""
    _open_limits(monkeypatch)
    calls: list[bytes] = []

    def _fake_caption(data, *, ext="png", source=""):
        calls.append(data)
        return f"第 {len(calls)} 张图"

    monkeypatch.setattr(image_captioner, "caption_image", _fake_caption)
    assets = [
        pdf_assets.ImageAsset(key_hint=f"{page}-0", page=page, data=bytes([page]), ext="png")
        for page in range(1, 6)
    ]

    chunks = image_captioner.build_image_chunks(assets, source="研报.pdf", max_captions=2)

    assert len(calls) == 2                      # 只外发 2 次
    assert len(chunks) == 5                     # 5 张图都有块
    assert chunks[0].page_content == "第 1 张图"
    assert chunks[1].page_content == "第 2 张图"
    assert "第 3 页" in chunks[2].page_content   # 超限的是占位文本
    assert all(c.metadata["block_type"] == "image" for c in chunks)


def test_caption_cap_defaults_to_config(monkeypatch):
    _open_limits(monkeypatch)
    monkeypatch.setattr(config, "IMAGE_CAPTION_MAX_IMAGES", 1)
    calls: list[bytes] = []
    monkeypatch.setattr(
        image_captioner, "caption_image",
        lambda data, **kw: calls.append(data) or "图",
    )
    assets = [
        pdf_assets.ImageAsset(key_hint=f"{page}-0", page=page, data=bytes([page]), ext="png")
        for page in range(1, 4)
    ]

    image_captioner.build_image_chunks(assets, source="研报.pdf")

    assert len(calls) == 1


def test_png_and_jpeg_dimensions_are_parsed():
    png = _png(37, 19)
    assert image_captioner._image_dimensions(png) == (37, 19)

    # 最小 JPEG：SOI + SOF0（长度 17，精度 8，高 19，宽 37）
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + (19).to_bytes(2, "big")
        + (37).to_bytes(2, "big")
        + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
        + b"\xff\xd9"
    )
    assert image_captioner._image_dimensions(jpeg) == (37, 19)


# --- 原文图注（content_list）参与索引文本组装 --------------------------------

def test_image_chunk_combines_caption_with_figure_caption(monkeypatch):
    """视觉描述与原文图注拼接：两条语义来源互补，互为兜底。"""
    _open_limits(monkeypatch)
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "订单结构图")
    assets = [pdf_assets.ImageAsset(key_hint="4-0", page=4, data=b"\x89PNG", ext="png")]

    chunks = image_captioner.build_image_chunks(
        assets, source="研报.pdf", figure_captions={4: ["图12 燃机订单结构"]},
    )

    assert chunks[0].page_content == "订单结构图\n原文图注：图12 燃机订单结构"


def test_figure_caption_rescues_uncaptioned_image(monkeypatch):
    """视觉描述失败/超外发上限的图，至少还能靠原文图注被语义召回。"""
    _open_limits(monkeypatch)
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "")
    assets = [pdf_assets.ImageAsset(key_hint="7-0", page=7, data=b"\x89PNG", ext="png")]

    chunks = image_captioner.build_image_chunks(
        assets, source="研报.pdf", figure_captions={7: ["图13 盈利预测"]},
    )

    text = chunks[0].page_content
    assert "第 7 页图表" in text
    assert "图13 盈利预测" in text
    assert "未能生成视觉描述" in text


def test_page_level_caption_alignment_one_to_one(monkeypatch):
    """页内图片数与图注数相等 → 按序 1:1；不等 → 整页图注拼接（宁重复不丢）。"""
    _open_limits(monkeypatch)
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "")
    assets = [
        pdf_assets.ImageAsset(key_hint="3-0", page=3, data=b"a", ext="png"),
        pdf_assets.ImageAsset(key_hint="3-1", page=3, data=b"b", ext="png"),
        pdf_assets.ImageAsset(key_hint="5-0", page=5, data=b"c", ext="png"),
        pdf_assets.ImageAsset(key_hint="5-1", page=5, data=b"d", ext="png"),
    ]

    chunks = image_captioner.build_image_chunks(
        assets, source="研报.pdf",
        figure_captions={3: ["图A", "图B"], 5: ["只有一条图注"]},
    )

    assert "图A" in chunks[0].page_content and "图B" not in chunks[0].page_content
    assert "图B" in chunks[1].page_content and "图A" not in chunks[1].page_content
    # 页 5 有两张图但只有一条图注：整页图注赋给两张图，两张都能命中
    assert "只有一条图注" in chunks[2].page_content
    assert "只有一条图注" in chunks[3].page_content


def test_egress_budget_prefers_larger_images(monkeypatch):
    """外发预算按面积降序：大图优先得到视觉描述，小图靠图注/占位兜底。"""
    _open_limits(monkeypatch)
    calls: list[bytes] = []
    monkeypatch.setattr(
        image_captioner, "caption_image",
        lambda data, **kw: calls.append(data) or "描述",
    )
    small = _png(10, 10)
    large = _png(400, 300)
    # 顺序故意把小图放前面：面积优先必须越过它
    assets = [
        pdf_assets.ImageAsset(key_hint="1-0", page=1, data=small, ext="png"),
        pdf_assets.ImageAsset(key_hint="2-0", page=2, data=large, ext="png"),
    ]

    chunks = image_captioner.build_image_chunks(assets, source="研报.pdf", max_captions=1)

    assert calls == [large]                     # 只有面积大的那张外发
    assert chunks[0].page_content != "描述"      # 小图没得到描述（占位）
    assert chunks[1].page_content == "描述"


def test_image_chunk_ids_stable_under_area_priority(monkeypatch):
    """预算排序只影响外发，不影响块 id：重复入库仍是覆盖而非新增。"""
    _open_limits(monkeypatch)
    monkeypatch.setattr(image_captioner, "caption_image", lambda *a, **k: "图")
    small = pdf_assets.ImageAsset(key_hint="1-0", page=1, data=_png(10, 10), ext="png")
    large = pdf_assets.ImageAsset(key_hint="2-0", page=2, data=_png(400, 300), ext="png")

    first = image_captioner.build_image_chunks(
        [small, large], source="s.pdf", max_captions=1,
    )
    second = image_captioner.build_image_chunks(
        [large, small], source="s.pdf", max_captions=1,
    )

    # key_hint（图在文中的位置）决定 id，与 assets 顺序、预算分配均无关
    assert {c.metadata["id"] for c in first} == {c.metadata["id"] for c in second}
    assert len(first) == len(second) == 2
