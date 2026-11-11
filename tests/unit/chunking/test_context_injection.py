"""打分上下文注入与多样性配额的单元测试。

对应 gold20 复盘的三个检索失败模式（2026-10-01）：
* 短标题块（<120 字符）正文里没有章节词对齐信号 → 重排/嵌入输入需要块外上下文；
* 表格块关键词密度高 → 最终 top-5 需要非文本块配额；
* 单文档霸榜挤掉跨文档证据 → 最终 top-5 需要每文档配额。
"""

from __future__ import annotations

from src.chunking.context_injection import (
    context_header_for_chunk,
    embed_text_for_chunk,
    is_non_text_block,
    rerank_text_for_match,
)
from src.retrieve.hybrid_retriever import _apply_diversity


# ---------------------------------------------------------------------------
# 上下文注入
# ---------------------------------------------------------------------------


class TestContextHeader:
    def test_composes_heading_path(self):
        meta = {"heading_path": "第一章 总则 > 投资评级"}
        header = context_header_for_chunk(meta, {})
        assert header == "第一章 总则 > 投资评级"

    def test_both_empty_returns_empty(self):
        assert context_header_for_chunk({}, {}) == ""

    def test_broker_and_title_never_leak_in(self):
        """broker/title 抽取质量不可靠（实测有 'kysec'、张冠李戴），绝不注入。"""
        header = context_header_for_chunk(
            {"heading_path": "h", "title": "H3_APxxx_1"}, {"broker": "kysec"}
        )
        assert "kysec" not in header
        assert "H3_APxxx_1" not in header

    def test_header_capped_at_512_chars(self):
        header = context_header_for_chunk({"heading_path": "长" * 2000}, {})
        assert len(header) <= 512


class TestEmbedTextForChunk:
    class _Chunk:
        def __init__(self, content: str, metadata: dict):
            self.page_content = content
            self.metadata = metadata

    def test_content_unchanged_without_header(self):
        """无上下文时嵌入文本必须与原文逐字节一致（保 content_hash 语义）。"""
        chunk = self._Chunk("纯正文", {})
        assert embed_text_for_chunk(chunk, {}) == "纯正文"

    def test_header_prepended_with_newline(self):
        chunk = self._Chunk("投资评级：买入（首次）", {"heading_path": "公司治理"})
        text = embed_text_for_chunk(chunk, {})
        assert text == "公司治理\n投资评级：买入（首次）"

    def test_original_content_not_mutated(self):
        raw = "正文本体"
        chunk = self._Chunk(raw, {"heading_path": "第一节"})
        embed_text_for_chunk(chunk, {})
        assert chunk.page_content == raw


class TestRerankTextForMatch:
    def test_short_title_chunk_gets_heading_context(self):
        """gold20 的 rr-single-005：15 字标题块靠 heading_path 补上章节词对齐信号。"""
        match = {
            "content": "## 加速赋能 AI 新药研发",
            "heading_path": "加速赋能 AI 新药研发",
        }
        text = rerank_text_for_match(match)
        assert text.startswith("加速赋能 AI 新药研发\n")
        assert "## 加速赋能 AI 新药研发" in text

    def test_plain_match_unchanged_when_no_metadata(self):
        match = {"content": "只有正文"}
        assert rerank_text_for_match(match) == "只有正文"

    def test_total_capped_at_2200(self):
        match = {
            "content": "长" * 3000,
            "heading_path": "路" * 100,
        }
        assert len(rerank_text_for_match(match)) <= 2200


def test_is_non_text_block():
    assert is_non_text_block({"block_type": "table"})
    assert is_non_text_block({"block_type": "image"})
    assert not is_non_text_block({"block_type": "text"})
    assert not is_non_text_block({})  # 缺省按 text 处理


# ---------------------------------------------------------------------------
# 多样性配额
# ---------------------------------------------------------------------------


def _match(source: str, block_type: str = "text", rank: int = 0) -> dict:
    return {"source": source, "block_type": block_type, "rank": rank}


class TestApplyDiversity:
    def test_single_doc_cannot_fill_top5(self):
        """gold20 问题 3：单文档霸榜时第二篇文档的证据能进槽。"""
        pool = [_match("A", rank=i) for i in range(6)] + [
            _match("B", rank=6),
            _match("B", rank=7),
        ]
        picked = _apply_diversity(pool, top_n=5, max_per_source=2, max_non_text=0)
        sources = [m["source"] for m in picked]
        assert sources == ["A", "A", "B", "B", "A"]  # 槽位 5 由回填补齐
        assert picked[-1]["rank"] == 2  # 回填按原始顺位

    def test_table_quota(self):
        """gold20 问题 2：表格霸榜被压到配额内，正文块顺位上移。

        候选池里正文块足够时配额严格生效；池内正文不足时才允许回填表格。
        """
        pool = [
            _match("A", "table", rank=0),
            _match("A", "table", rank=1),
            _match("A", "table", rank=2),
            _match("B", "text", rank=3),
            _match("C", "text", rank=4),
            _match("D", "text", rank=5),
            _match("B", "table", rank=6),
        ]
        picked = _apply_diversity(pool, top_n=4, max_per_source=0, max_non_text=2)
        tables = [m for m in picked if m["block_type"] == "table"]
        assert len(tables) == 2
        assert [m["rank"] for m in picked] == [0, 1, 3, 4]

    def test_quota_starvation_backfills_in_rank_order(self):
        """配额把候选耗尽时必须回填，永远返回足量结果。"""
        pool = [_match("A", rank=i) for i in range(3)]
        picked = _apply_diversity(pool, top_n=5, max_per_source=1, max_non_text=0)
        assert len(picked) == 3  # 候选只有 3 条，全部返回
        assert [m["rank"] for m in picked] == [0, 1, 2]

    def test_zero_disables_quota(self):
        pool = [_match("A", rank=i) for i in range(5)]
        picked = _apply_diversity(pool, top_n=3, max_per_source=0, max_non_text=0)
        assert [m["rank"] for m in picked] == [0, 1, 2]

    def test_pool_not_larger_than_top_n_untouched(self):
        pool = [_match("A", rank=i) for i in range(4)]
        picked = _apply_diversity(pool, top_n=5, max_per_source=1, max_non_text=0)
        assert len(picked) == 4  # 无取舍空间，原样返回

    def test_does_not_mutate_input(self):
        pool = [_match("A", rank=i) for i in range(4)]
        snapshot = [dict(m) for m in pool]
        _apply_diversity(pool, top_n=2, max_per_source=1, max_non_text=0)
        assert pool == snapshot
