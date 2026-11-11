"""递归字符切分器（纯标准库实现，行为与 langchain 版本等价）。

为什么需要这个模块
------------------
``langchain_text_splitters`` 的包级 ``__init__`` 会把**全部** splitter 子类都导入
一遍，其中包括 ``langchain_text_splitters.sentence_transformers``，从而把
``sentence_transformers`` → ``transformers`` → ``torch``（外加 ``datasets``、
``nltk``）拖进进程冷启动。实测 ``import langchain_text_splitters`` 约 **11 秒**
（约 5000 个模块），而本仓库只需要 ``RecursiveCharacterTextSplitter`` 一个类、
且只有 :mod:`src.chunking.chunker` 一处调用。

从子模块导入无法规避：Python 导入子模块必然先执行父包的 ``__init__``。

因此这里按 :mod:`langchain_text_splitters.character` /
:mod:`langchain_text_splitters.base` 的语义做了一份**逐行对齐**的纯标准库实现，
参数只保留本仓库真正使用的部分（``chunk_size`` / ``chunk_overlap`` /
``separators``，以及 langchain 的默认 ``keep_separator=True``、
``length_function=len``、``strip_whitespace=True``）。

等价性由 ``tests/unit/chunking/test_recursive_splitter.py`` 在真实语料上逐字符串比对守护；
改动本文件前请先跑该测试。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# 本仓库使用的分隔符（与 langchain 默认不同的中文标点增强版），
# 由调用方传入，这里只作为缺省值。
_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")


def _split_text_with_regex(text: str, separator: str, *, keep_separator: bool) -> list[str]:
    """按分隔符切分并保留分隔符（对齐 langchain 的同名函数）。

    ``keep_separator=True`` 时用 ``re.split(f"({separator})", text)`` 捕获分隔符，
    再按「分隔符拼到**前一段**尾部」的方式配对；奇数长度时把首项单独保留。
    最后剔除空串。
    """
    if separator:
        if keep_separator:
            # 括号让 re.split 把分隔符本身也保留在结果里
            splits_: list[str] = re.split(f"({separator})", text)
            splits = [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]
            if len(splits_) % 2 == 0:
                splits += splits_[-1:]
            splits = [splits_[0], *splits]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class RecursiveCharacterTextSplitter:
    """递归字符切分器（langchain ``RecursiveCharacterTextSplitter`` 的等价实现）。

    Args:
        chunk_size: 目标块长度上限（按 ``len`` 计）。
        chunk_overlap: 相邻块的目标重叠长度。
        separators: 分隔符候选，按优先级从粗到细排列；末位通常是 ``""``
            （逐字符兜底）。
        keep_separator: 是否把分隔符保留在块内（对齐 langchain 默认 ``True``）。
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        *,
        separators: list[str] | tuple[str, ...] | None = None,
        keep_separator: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能为负")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = list(separators) if separators else list(_DEFAULT_SEPARATORS)
        self._keep_separator = keep_separator

    def split_text(self, text: str) -> list[str]:
        """把文本切分为多个块。"""
        return self._split_text(text, self._separators)

    # ------------------------------------------------------------------
    # 内部实现（与 langchain 同名方法逐行对齐）
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """按当前分隔符切分，超长片段递归下钻到更细的分隔符。"""
        final_chunks: list[str] = []

        # 选出第一个在文本中出现的分隔符（空串表示逐字符兜底）
        separator = separators[-1]
        new_separators: list[str] = []
        for index, candidate in enumerate(separators):
            if not candidate:
                separator = candidate
                break
            if re.search(re.escape(candidate), text):
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        splits = _split_text_with_regex(
            text, re.escape(separator), keep_separator=self._keep_separator
        )

        # 合并小片段；超长片段交给下一级分隔符继续切
        good_splits: list[str] = []
        # keep_separator=True 时分隔符已并入片段，合并时不再重复插入
        merge_separator = "" if self._keep_separator else separator
        for piece in splits:
            if len(piece) < self._chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, merge_separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(piece)
                else:
                    final_chunks.extend(self._split_text(piece, new_separators))
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, merge_separator))
        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """把细粒度片段合并到接近 ``chunk_size``，并按 ``chunk_overlap`` 回退重叠。"""
        separator_len = len(separator)

        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for piece in splits:
            len_ = len(piece)
            if (
                total + len_ + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        "切块长度 %d 超过配置的 %d", total, self._chunk_size
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # 回退到不超过 chunk_overlap，或直到能装下当前片段
                    while total > self._chunk_overlap or (
                        total + len_ + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= len(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(piece)
            total += len_ + (separator_len if len(current_doc) > 1 else 0)

        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @staticmethod
    def _join_docs(docs: list[str], separator: str) -> str | None:
        """拼接片段并去除首尾空白；结果为空返回 None。"""
        text = separator.join(docs).strip()
        return text or None


__all__ = ["RecursiveCharacterTextSplitter"]
