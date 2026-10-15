"""打分上下文注入：把「块文本之外」的元数据拼进嵌入/重排输入。

背景（gold20 复盘，2026-10-01）：研报的「评级 / 标题 / 主题」类证据常是
<120 字符的标题块，正文里根本没有公司名/券商名——查询词与块文本零交集，
稠密与 BM25 都给不出高分，块甚至进不了重排候选池。而 Milvus 标量字段里
security_name / heading_path 是齐的，所以在**打分输入层**拼上它们。

两条铁律：

* ``content`` 本体保持原文——块 id（content_hash）与 gold 证据引用不受影响，
  前端展示 / 引用拼装的仍是原文；
* 不拼 ``broker`` 与 ``title``——元数据抽取对这两项质量不可靠
  （实测出现 'kysec' 拼音、张冠李戴的券商名；title 兜底是文件名 stem），
  错误上下文会让 cross-encoder 主动压低正确块。

生效说明：嵌入侧拼接只影响新算的向量；已有向量按 content_hash 复用会跳过，
需要强制重嵌（rebuild_collection 或清空该 source 向量）才能全量生效。
"""

from __future__ import annotations

from typing import Any

# 上下文头的分隔符与长度上限：header 本身不长，但 heading_path 理论上可到
# 1024 字符，封顶防止长章节路径挤占重排输入里正文的位置。
_HEADER_SEPARATOR = "；"
_MAX_HEADER_CHARS = 512

# 重排输入总长：原实现只取 content[:2000]，拼上下文后放宽到 2200，
# 保证正文有效长度不因注入而明显缩水。
_MAX_RERANK_CHARS = 2200

# 表格/图片块的 block_type 集合（与 HybridRetriever._NON_PARENT_BLOCK_TYPES 一致）
_NON_TEXT_BLOCK_TYPES = frozenset({"table", "image"})


def _header_parts(values: tuple[str | None, ...]) -> list[str]:
    return [v.strip() for v in values if v and str(v).strip()]


def context_header_for_chunk(chunk_metadata: dict[str, Any], doc_metadata: dict[str, Any]) -> str:
    """入库侧：由块元数据 + 文档级元数据拼上下文头。

    文档级 ``security_name``（公司简称）来自研报元数据抽取；块级
    ``heading_path`` 来自切块时的标题栈。
    """
    parts = _header_parts(
        (
            str(doc_metadata.get("security_name") or ""),
            str(chunk_metadata.get("heading_path") or ""),
        )
    )
    header = _HEADER_SEPARATOR.join(parts)
    return header[:_MAX_HEADER_CHARS]


def embed_text_for_chunk(chunk: Any, doc_metadata: dict[str, Any]) -> str:
    """入库侧：该子块真正送进嵌入模型的文本（= 上下文头 + 原文）。"""
    header = context_header_for_chunk(getattr(chunk, "metadata", {}) or {}, doc_metadata)
    content = str(getattr(chunk, "page_content", "") or "")
    return f"{header}\n{content}" if header else content


def rerank_text_for_match(match: dict[str, Any]) -> str:
    """检索侧：送进 cross-encoder 的候选文本（= 上下文头 + 原文）。"""
    parts = _header_parts(
        (
            str(match.get("security_name") or ""),
            str(match.get("heading_path") or ""),
        )
    )
    header = _HEADER_SEPARATOR.join(parts)[:_MAX_HEADER_CHARS]
    content = str(match.get("content") or "")
    composed = f"{header}\n{content}" if header else content
    return composed[:_MAX_RERANK_CHARS]


def is_non_text_block(match_or_meta: dict[str, Any]) -> bool:
    """该候选是否为表格/图片块（多样性配额的作用对象）。"""
    return str(match_or_meta.get("block_type") or "text") in _NON_TEXT_BLOCK_TYPES
