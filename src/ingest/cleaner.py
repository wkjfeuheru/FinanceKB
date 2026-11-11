"""内容清洗与重复诊断：保守噪声过滤 + SimHash 近似重复诊断。

* ``TextCleaner``：正则匹配常见噪声模式（页码、版权声明、导航链接、OCR 乱码），
  直接过滤；规则保守，避免误删正文。
* ``simhash`` / ``hamming_distance``：纯标准库实现的 64 位 SimHash，
  文档内段落仅诊断，文档间入库拦截保持原有规则（无新增依赖）。

清洗、段落诊断与文档间去重均为固定处理逻辑，不提供配置开关；
仅 ``DUPLICATE_HAMMING_THRESHOLD`` 等算法参数以常量形式固化在本模块。
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningRemoval:
    text: str
    rule: str


@dataclass(frozen=True)
class CleaningResult:
    text: str
    removals: tuple[CleaningRemoval, ...]


# ---------------------------------------------------------------------------
# 噪声规则（保守匹配）
# ---------------------------------------------------------------------------

# 只有明确的“第X页”文字标记才可视为页码；独立数字/比值保留。
_PAGE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9一二三四五六七八九十百千]+\s*页(?:\s*[/，,共].*)?"
    r")\s*$"
)

# 版权 / 免责声明行（逐行匹配，避免贪婪误删正文）
_COPYRIGHT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"版权所有"
    r"|未经[^。\n]*许可"
    r"|免责声明\s*[:：]?"
    r"|本(?:报告|文|材料|文件)版权"
    r"|本(?:报告|文|材料|文件)[^。\n]*仅供参考"
    r"|市场有风险[，,]\s*投资需谨慎"
    r").*$"
)

# 纯 URL 行
_URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)

# Markdown 链接：去链接保留文字 [文字](http...) -> 文字
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:https?://|www\.)[^)]*\)")

# 目录点线行：正文 + 连续点线引导符 + 结尾页码（如 "第一章 总则 ....... 1"）
_TOC_DOT_LINE_RE = re.compile(r"^.{4,}?(?:[.·…]\s*){3,}\s*\d{0,4}\s*$")

# OCR / 文本层残留：替换字符与符号字体私有区字形（Wingdings 把项目符号搬成
# U+F0B7 / U+F06E 之类的码位，语义上不是正文），保守替换为单个空格。
_OCR_GARBAGE_RE = re.compile(r"[\ufffd\ue000-\uf8ff]+")


class TextCleaner:
    """规则噪声过滤器（固定流程，始终执行）。"""

    def clean(self, markdown: str) -> str:
        """过滤噪声、残留页眉页脚并归一化空白。"""
        return self.clean_with_audit(markdown).text

    def clean_with_audit(self, markdown: str) -> CleaningResult:
        """返回本次清洗结果和删除原文；审计不依赖共享可变状态。"""
        removals: list[CleaningRemoval] = []
        kept: list[str] = []
        rules = ((_PAGE_LINE_RE, 'page_marker'),
                 (_COPYRIGHT_LINE_RE, 'copyright'),
                 (_URL_LINE_RE, 'url'), (_TOC_DOT_LINE_RE, 'toc'))
        for line in markdown.split('\n'):
            rule = next((name for pattern, name in rules if pattern.match(line.strip())), None)
            if rule:
                removals.append(CleaningRemoval(line, rule))
                continue
            def replace_link(match):
                removals.append(CleaningRemoval('[', 'markdown_link_markup'))
                removals.append(CleaningRemoval(match.group(0)[len(match.group(1)) + 1:], 'markdown_link_markup'))
                return match.group(1)
            kept.append(_MD_LINK_RE.sub(replace_link, line))
        text = '\n'.join(kept)
        for match in _OCR_GARBAGE_RE.finditer(text):
            removals.append(CleaningRemoval(match.group(), 'ocr_garbage'))
        text = _OCR_GARBAGE_RE.sub(' ', text)
        for match in re.finditer(r'\n{3,}', text):
            removals.append(CleaningRemoval(match.group()[2:], 'excess_blank_lines'))
        text = re.sub(r'\n{3,}', '\n\n', text)
        if text != text.strip():
            leading = text[:len(text) - len(text.lstrip())]
            trailing = text[len(text.rstrip()):] if text.strip() else ''
            if leading:
                removals.append(CleaningRemoval(leading, 'edge_whitespace'))
            if trailing:
                removals.append(CleaningRemoval(trailing, 'edge_whitespace'))
        for removal in removals:
            logger.info('清洗删除 rule=%s text=%r', removal.rule, removal.text)
        return CleaningResult(text.strip(), tuple(removals))


# ---------------------------------------------------------------------------
# SimHash（纯标准库：字符 n-gram + md5 独立位投票 + 标点归一化）
# ---------------------------------------------------------------------------

_BIT_COUNT = 64
_NORMALIZE_RE = re.compile(r"[\W_]+", re.UNICODE)


def _normalize_for_hash(text: str) -> str:
    """哈希前归一化：去除标点/空白，保留中文与字母数字。

    使仅标点/空白差异的近似重复文本哈希一致，显著降低短文本噪声。
    """
    return _NORMALIZE_RE.sub("", text).lower()


def _token_hashes(text: str, n_gram: int = 3) -> list[int]:
    """提取字符 n-gram 并返回其 64 位哈希列表（md5 前 8 字节）。"""
    if len(text) < n_gram:
        grams = [text]
    else:
        grams = [text[i : i + n_gram] for i in range(len(text) - n_gram + 1)]
    return [
        int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "big")
        for g in grams
    ]


def simhash(text: str, n_gram: int = 3, normalize: bool = True) -> int:
    """计算 64 位 SimHash 指纹（字符 n-gram + md5 独立位投票）。

    ``normalize=True`` 时先去除标点/空白，使近似重复文本指纹更稳定。
    """
    source = _normalize_for_hash(text) if normalize else text
    if not source:
        return 0

    weights: Counter[int] = Counter()
    for h in _token_hashes(source, n_gram):
        for bit in range(_BIT_COUNT):
            if (h >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1

    value = 0
    for bit in range(_BIT_COUNT):
        if weights[bit] > 0:
            value |= 1 << bit
    return value


def hamming_distance(a: int, b: int) -> int:
    """两个 64 位指纹的海明距离。"""
    return (a ^ b).bit_count()


# 文档间 SimHash 去重：海明距离阈值（固化常量，非配置开关）。
# 海明距离 <= 该值视为疑似重复文档，拦截入库。
DUPLICATE_HAMMING_THRESHOLD = 3


def deduplicate_paragraphs(
    markdown: str,
    threshold: int = 3,
    min_length: int = 40,
) -> str:
    """兼容旧接口：诊断疑似重复段落，原文一字不删。

    * 仅对长度 >= ``min_length`` 的段落做去重（短行噪声差异大，误伤风险高）；
    * Markdown 表格行与标题行（以 ``#`` 开头）不参与去重；
    * 段落原文与顺序全部保留。
    """
    paragraphs = markdown.split("\n\n")
    seen: list[tuple[int, int]] = []  # (simhash, 原文长度)
    kept: list[str] = []
    for para in paragraphs:
        text = para.strip()
        if not text:
            kept.append(para)
            continue
        if len(text) < min_length or text.startswith("#") or text.startswith("|"):
            kept.append(para)
            continue

        sig = simhash(text)
        if any(hamming_distance(sig, s) <= threshold for s, _ in seen):
            logger.info('疑似重复段落（仅诊断，保留正文）：%r', para)
        seen.append((sig, len(text)))
        kept.append(para)

    return "\n\n".join(kept)


def find_similar_documents(
    simhash_value: int,
    candidates: dict[str, int],
    threshold: int = DUPLICATE_HAMMING_THRESHOLD,
    exclude_source: str = "",
) -> list[tuple[str, int]]:
    """在候选指纹中查找与 ``simhash_value`` 海明距离 <= threshold 的文档。

    Args:
        simhash_value: 待比对指纹。
        candidates: {source: simhash} 映射。
        threshold: 海明距离阈值（<= 该值视为相似）。
        exclude_source: 排除的 source（替换更新时排除自身）。

    Returns:
        [(source, hamming_distance), ...]，按距离升序。
    """
    result = []
    for source, sig in candidates.items():
        if source == exclude_source or not isinstance(sig, int):
            continue
        dist = hamming_distance(simhash_value, sig)
        if dist <= threshold:
            result.append((source, dist))
    result.sort(key=lambda item: (item[1], item[0]))
    return result
