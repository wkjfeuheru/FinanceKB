"""测试集自动生成器：基于知识库文档生成高质量分层评估集。

生成结构（约 7:2:1）：
* 单跳事实（single_hop，~70%）：单块即可回答的事实题，标注单个证据 chunk id；
* 跨文档多跳（multi_hop，~20%）：需综合两个不同文档的事实，标注两个 chunk id；
* 负样本陷阱（negative，~10%）：前提错误/超范围问题，期望系统拒绝而非编造。

输出为与 :class:`TestSetLoader` 兼容的 Markdown 文件，每个条目标注
``- 问题类型：...`` 与 ``- 证据chunk：<id1>, <id2>`` 元数据行。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dataset import EvaluationDataset

from src.agent.prompts.chat import (
    MULTI_HOP_QA_PROMPT,
    NEGATIVE_QA_PROMPT,
    SINGLE_HOP_QA_PROMPT,
)
from src.core.config import get_model

logger = logging.getLogger(__name__)


def _invoke_many(chain: Any, payloads: list[dict[str, Any]], max_workers: int = 8) -> list[Any]:
    """并发调用生成链并保持输入顺序；单项失败以异常对象返回。"""
    def invoke(payload: dict[str, Any]) -> Any:
        try:
            return chain.invoke(payload)
        except Exception as exc:  # noqa: BLE001
            return exc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(invoke, payloads))


def _parse_llm_qa(raw: str) -> tuple[str, str] | None:
    """解析 LLM 输出：返回 (query, ground_truth)；格式非法返回 None。"""
    text = (raw or "").strip()
    answer_marker = "### 标准答案"
    marker_idx = text.find(answer_marker)
    if marker_idx == -1:
        return None
    question_part = text[:marker_idx]
    question_match = re.search(r"^##\s*问题\s*(.+?)\s*$", question_part, re.DOTALL)
    if not question_match:
        return None
    query = question_match.group(1).strip()
    if len(query) < 4:
        return None
    answer = text[marker_idx + len(answer_marker):].strip()
    # 金融问答的正确答案可能是比例、期限、电话等短值，不能用长度误剔除。
    if not answer:
        return None
    return query, answer


def _is_skip_output(raw: str) -> bool:
    """LLM 判定片段不适合出题时按 Prompt 约定只输出 SKIP。"""
    return (raw or "").strip().upper().startswith("SKIP")


# ---------------------------------------------------------------------------
# chunk 出题价值预过滤（目录/标题/图片描述等低知识密度块不值得出题）
# ---------------------------------------------------------------------------

# 过短内容阈值：标题、目录单项、页眉页脚等通常不足此长度
_MIN_CHUNK_CHARS = 150

# 图片视觉描述强特征：Docling 解析 PDF 时生成的插图描述文本（与金融知识无关）
_IMAGE_DESC_PATTERN = re.compile(r"一幅|画面|描绘|图中所示|插图显示")

# 编号条目行（目录/条款列表特征）：如 "（一）xxx"、"1. xxx"
_NUMBERED_LINE_PATTERN = re.compile(r"^[\(（]?[一二三四五六七八九十\d]+[\)）]?[、\.．\s]")


def _is_low_quality_chunk(content: str) -> bool:
    """判断 chunk 是否为低知识密度内容，不值得用于生成评估题。

    规则：
    1. 过短：标题、目录单项、页眉页脚等通常不足 150 字符；
    2. 图片视觉描述：Docling 解析 PDF 时生成的插图描述
       （如"一幅描绘江南水乡古典风貌的黑白风景画…"）与金融知识无关；
    3. 目录/结构列表：非空行中大部分为短编号条目（如目录页）。
    """
    text = content.strip()
    if len(text) < _MIN_CHUNK_CHARS:
        return True
    if _IMAGE_DESC_PATTERN.search(text):
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4:
        numbered = sum(1 for ln in lines if _NUMBERED_LINE_PATTERN.match(ln))
        short_lines = sum(1 for ln in lines if len(ln) < 60)
        if numbered / len(lines) >= 0.6 and short_lines / len(lines) >= 0.8:
            return True
    return False


def _iter_chunk_records(
    kb: Any,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    """取指定 source 的切块记录（含 id/content/source），最多 ``limit`` 条。"""
    try:
        client = kb._get_client()
        results = client.query(
            collection_name=kb.collection_name,
            filter=f'source == "{source.replace(chr(34), chr(92) + chr(34))}"',
            output_fields=["id", "content", "source", "version"],
            limit=limit,
        )
        return [
            {"id": r.get("id", ""), "content": (r.get("content") or "").strip(),
             "source": r.get("source", ""), "version": r.get("version", "") or "current"}
            for r in results
            if r.get("id") and (r.get("content") or "").strip()
        ]
    except Exception as exc:
        logger.warning("读取文档切块失败（%s）：%s", source, exc)
        return []


def _collect_chunk_pool(kb: Any, per_doc_limit: int) -> list[dict[str, Any]]:
    """收集全库切块池（每文档最多 per_doc_limit 块，剔除低质块）。"""
    pool: list[dict[str, Any]] = []
    skipped = 0
    for doc in kb.list_documents():
        source = doc.get("source", "")
        if not source:
            continue
        # 多取一倍再过滤，避免过滤后单文档可用候选不足
        records = _iter_chunk_records(kb, source, limit=per_doc_limit * 2)
        kept: list[dict[str, Any]] = []
        for rec in records:
            if _is_low_quality_chunk(rec["content"]):
                skipped += 1
                continue
            kept.append(rec)
        pool.extend(kept[:per_doc_limit])
    if skipped:
        logger.info("chunk 预过滤：剔除 %d 个低质块（目录/标题/图片描述等）", skipped)
    return pool


def _generate_single_hop(chain: Any, chunks: list[dict[str, Any]]) -> list[tuple[dict, str]]:
    """单跳：每块生成一个事实问答，返回 [(meta, markdown_block)]。"""
    out: list[tuple[dict, str]] = []
    payloads = [{"context": chunk["content"][:3000]} for chunk in chunks]
    for chunk, raw in zip(chunks, _invoke_many(chain, payloads), strict=True):
        if isinstance(raw, Exception):
            logger.warning("单跳生成失败（%s）：%s", chunk["source"], raw)
            continue
        if _is_skip_output(raw):
            logger.info("单跳：LLM 判定片段不适合出题，跳过（%s）", chunk["source"])
            continue
        parsed = _parse_llm_qa(raw)
        if parsed is None:
            logger.warning("单跳输出格式非法，跳过：%s", (raw or "")[:100].replace("\n", " "))
            continue
        query, answer = parsed
        meta = {
            "query": query, "ground_truth": answer,
            "question_type": "single_hop",
            "related_docs": chunk["source"],
            "chunk_ids": [chunk["id"]],
            "expected_behavior": "",
            "evidence_records": [chunk],
        }
        out.append((meta, _render_block(meta, len(out) + 1)))
    return out


def _generate_multi_hop(
    chain: Any, pool: list[dict[str, Any]], count: int, seed: int,
    candidate_multiplier: int = 3,
) -> list[tuple[dict, str]]:
    """跨文档多跳：从不同 source 抽取块对生成推理问答。

    抽样策略（避免"同一题刷满整个多跳子集"）：
    * 优先枚举**不同 source 组合**并打乱，而不是反复 ``rng.sample``——旧实现
      在 source 较少时会不断抽到同一对文档，最终产出多条完全相同的题目
      （实测 8 条 multi_hop 里存在同一问题重复出现）；
    * 对 (source_a, source_b, chunk_a, chunk_b) 与生成的 query 双重去重；
    * 只有 source 组合数不足时才允许同一组合换块重抽。
    """
    # 按 source 分组
    by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in pool:
        by_source.setdefault(chunk["source"], []).append(chunk)
    sources = sorted(by_source)
    rng = random.Random(seed)

    pairs: list[tuple[dict, dict]] = []
    if len(sources) >= 2:
        # 枚举所有跨文档块对后打乱，既覆盖文档组合又避免旧实现遇重复即提前退出。
        pairs = [
            (a, b)
            for i, a_src in enumerate(sources)
            for b_src in sources[i + 1:]
            for a in by_source[a_src]
            for b in by_source[b_src]
        ]
        rng.shuffle(pairs)
        pairs = pairs[: count * candidate_multiplier]

    out: list[tuple[dict, str]] = []
    seen_queries: set[str] = set()
    payloads = [{
        "source_a": a["source"], "context_a": a["content"][:2000],
        "source_b": b["source"], "context_b": b["content"][:2000],
    } for a, b in pairs]
    for (a, b), raw in zip(pairs, _invoke_many(chain, payloads), strict=True):
        if len(out) >= count:
            break
        if isinstance(raw, Exception):
            logger.warning("多跳生成失败：%s", raw)
            continue
        if _is_skip_output(raw):
            logger.info("多跳：LLM 判定片段不适合出题，跳过")
            continue
        parsed = _parse_llm_qa(raw)
        if parsed is None:
            continue
        query, answer = parsed
        # 同一问题重复入集会稀释多跳子集的指标（n 虚高、方差变大）
        dedup_key = query.strip()
        if dedup_key in seen_queries:
            logger.info("多跳：生成结果与已有问题重复，跳过（%s）", dedup_key[:30])
            continue
        seen_queries.add(dedup_key)
        meta = {
            "query": query, "ground_truth": answer,
            "question_type": "multi_hop",
            "related_docs": f"{a['source']}, {b['source']}",
            "chunk_ids": [a["id"], b["id"]],
            "expected_behavior": "",
            "evidence_records": [a, b],
        }
        out.append((meta, _render_block(meta, len(out) + 1)))
    return out


def _generate_negative(
    chain: Any, sources: list[str], count: int, attempt_multiplier: float = 2,
) -> list[tuple[dict, str]]:
    """负样本陷阱：前提错误/超范围问题。"""
    out: list[tuple[dict, str]] = []
    source_list = "\n".join(f"- {s}" for s in sources[:20]) or "（知识库为空）"
    payloads = [
        {"sources": f"{source_list}\n\n请生成互不重复的负样本变体 #{index + 1}。"}
        for index in range(math.ceil(count * attempt_multiplier))
    ]
    for raw in _invoke_many(chain, payloads):
        if len(out) >= count:
            break
        if isinstance(raw, Exception):
            logger.warning("负样本生成失败：%s", raw)
            continue
        parsed = _parse_llm_qa(raw)
        if parsed is None:
            continue
        query, answer = parsed
        meta = {
            "query": query, "ground_truth": answer,
            "question_type": "negative",
            "related_docs": "",
            "chunk_ids": [],
            "expected_behavior": "应拒绝回答或指出知识库无相关信息",
        }
        out.append((meta, _render_block(meta, len(out) + 1)))
    return out


def _render_block(meta: dict[str, Any], index: int) -> str:
    """渲染单条测试集的 Markdown 块。"""
    lines = [f"## {index}. {meta['query']}", "", f"- 问题类型：{meta['question_type']}"]
    if meta.get("related_docs"):
        lines.append(f"- 相关文档：{meta['related_docs']}")
    if meta.get("chunk_ids"):
        lines.append(f"- 证据chunk：{', '.join(meta['chunk_ids'])}")
    if meta.get("expected_behavior"):
        lines.append(f"- 期望行为：{meta['expected_behavior']}")
    lines += ["", "### 标准答案", "", meta["ground_truth"], "", "---"]
    return "\n".join(lines)


def generate_testset(
    kb: Any = None,
    *,
    count: int = 40,
    seed: int = 42,
    per_doc_limit: int = 8,
    out_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    """生成分层测试集（约 7:2:1）。

    Args:
        kb: 知识库实例（默认 ``get_knowledge_base()``）。
        count: 目标总题数（实际受知识库块数限制可能略少）。
        seed: 随机种子（抽样稳定）。
        per_doc_limit: 每文档纳入候选池的最大块数。
        out_path: 输出路径；None 时写 ``data/evaluation_qa_generated.md``。

    Returns:
        (条目列表, 输出文件路径)。LLM 不可用或知识库为空返回 ([], None)。
    """
    from src.index.milvus_kb import get_knowledge_base

    from .test_set import TestSetLoader

    if get_model() is None:
        logger.error("LLM 未配置，无法生成测试集")
        return [], None

    kb = kb or get_knowledge_base()
    documents = kb.list_documents()
    if not documents:
        logger.error("知识库为空，无法生成测试集")
        return [], None

    pool = _collect_chunk_pool(kb, per_doc_limit)
    if not pool:
        logger.error("知识库无可用切块，无法生成测试集")
        return [], None

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    def _chain(prompt: str):
        return ChatPromptTemplate.from_messages([("human", prompt)]) | get_model() | StrOutputParser()

    rng = random.Random(seed)
    chunks = list(pool)
    rng.shuffle(chunks)

    # 7:2:1 配比
    single_target = round(count * 0.7)
    multi_target = round(count * 0.2)
    negative_target = count - single_target - multi_target

    single_chunks = chunks[: min(len(chunks), single_target)]
    entries: list[dict[str, Any]] = []

    # 单跳
    entries.extend(m for m, _ in _generate_single_hop(_chain(SINGLE_HOP_QA_PROMPT), single_chunks))
    # 多跳（跨文档）
    entries.extend(m for m, _ in _generate_multi_hop(_chain(MULTI_HOP_QA_PROMPT), pool, multi_target, seed))
    # 负样本
    sources = [d.get("source", "") for d in documents if d.get("source")]
    entries.extend(m for m, _ in _generate_negative(_chain(NEGATIVE_QA_PROMPT), sources, negative_target))

    if not entries:
        logger.error("未能生成任何有效问答对")
        return [], None

    # 去重：不同生成器（尤其负样本）可能撞出相同问题，写盘前按问题文本去重
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for meta in entries:
        key = re.sub(r"\s+", "", meta["query"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(meta)
    if len(deduped) < len(entries):
        logger.warning("问题去重：剔除 %d 条重复", len(entries) - len(deduped))
    entries = deduped

    final_blocks: list[str] = []
    for i, meta in enumerate(entries, 1):
        final_blocks.append(_render_block(meta, i))

    if out_path is None:
        out_path = Path(__file__).resolve().parent / "data" / "evaluation_qa_generated.md"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 金融 RAG 分层评估问答集（约 7:2:1 = 单跳:跨文档多跳:负样本）\n"
        "# 由 LLM 基于当前知识库生成，请人工抽查后使用；每个条目标注证据 chunk id。\n\n"
    )
    out_path.write_text(header + "\n\n".join(final_blocks) + "\n", encoding="utf-8")

    # 自检：必须能被 TestSetLoader 解析且配比记录
    parsed_entries = TestSetLoader(out_path).load_test_set()
    dist: dict[str, int] = {}
    for entry in parsed_entries:
        dist[entry.question_type] = dist.get(entry.question_type, 0) + 1
    logger.info("生成测试集 %d 条（分布 %s）→ %s", len(parsed_entries), dist, out_path)
    return entries, out_path


def knowledge_base_fingerprint(kb: Any) -> str:
    """基于当前文档清单和文件指纹计算稳定的知识库指纹。"""
    documents = sorted(kb.list_documents(), key=lambda item: item.get("source", ""))
    fingerprint_store = kb.get_fingerprint_store()
    payload = []
    for document in documents:
        source = str(document.get("source", ""))
        stored = fingerprint_store.get(source) or {}
        payload.append({
            "source": source,
            "version": document.get("version", ""),
            "chunk_count": document.get("chunk_count", 0),
            "sha256": stored.get("sha256", ""),
            "retrieval_metadata": {
                key: document.get(key) or ""
                for key in ("report_type", "security_name", "industry_l1", "industry_l2", "broker", "category", "date")
            },
        })
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_evidence_resolver(kb: Any):
    """构造按 chunk id 查询当前 Milvus 记录的在线证据解析器。"""
    client = kb._get_client()

    def resolve(chunk_id: str) -> dict[str, Any] | None:
        escaped = chunk_id.replace('"', '\\"')
        rows = client.query(
            collection_name=kb.collection_name,
            filter=f'id == "{escaped}"',
            output_fields=["id", "content", "source", "version"],
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "id": str(row.get("id", "")),
            "content": str(row.get("content", "")),
            "source": str(row.get("source", "")),
            "version": str(row.get("version", "") or "current"),
        }

    return resolve


def _required_facts(answer: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[。！？；\n]", answer) if part.strip()]
    return parts[:5] or [answer.strip()]


def _candidate_tier(question_type: str, ordinal: int) -> tuple[str, str]:
    """为候选池预留约 1.5 倍 Gold 审核余量，其余通过自动门禁进入 Silver。"""
    from .dataset import DEFAULT_QUOTAS

    gold_target = DEFAULT_QUOTAS["gold"][question_type]
    gold_candidates = math.ceil(gold_target * 1.5)
    return ("gold", "pending") if ordinal <= gold_candidates else ("silver", "auto_pass")


def generate_evaluation_dataset(
    kb: Any = None,
    *,
    count: int = 240,
    seed: int = 42,
    per_doc_limit: int = 40,
    out_path: str | Path,
) -> tuple[EvaluationDataset, Path]:
    """生成权威 JSONL 候选集，并执行结构与在线证据门禁。"""
    from src.index.milvus_kb import get_knowledge_base

    from .dataset import DEFAULT_QUOTAS, DatasetValidationError, EvaluationDataset

    kb = kb or get_knowledge_base()
    documents = kb.list_documents()
    if get_model() is None:
        raise RuntimeError("LLM 未配置，无法生成评测候选集")
    if not documents:
        raise RuntimeError("知识库为空，无法生成评测候选集")
    pool = _collect_chunk_pool(kb, per_doc_limit)
    if not pool:
        raise RuntimeError("知识库无可用切块，无法生成评测候选集")

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    def chain(prompt: str):
        return ChatPromptTemplate.from_messages([("human", prompt)]) | get_model() | StrOutputParser()

    # 目标分布与冻结集 80:40:30 一致，而不是沿用旧版 7:2:1。
    single_target = round(count * 80 / 150)
    multi_target = round(count * 40 / 150)
    negative_target = count - single_target - multi_target
    rng = random.Random(seed)
    chunks = list(pool)
    rng.shuffle(chunks)
    generated: list[dict[str, Any]] = []
    single_chunks: list[dict[str, Any]] = []
    for index in range(single_target + max(20, round(single_target * 0.15))):
        base = chunks[index % len(chunks)]
        variant = index // len(chunks) + 1
        single_chunks.append({
            **base,
            # quote 必须源自 Milvus 原文，不能包含仅用于提示模型的变体指令。
            "evidence_content": base["content"],
            "content": f"{base['content']}\n\n请基于原文生成互不重复的事实题变体 #{variant}。",
        })
    singles = _generate_single_hop(chain(SINGLE_HOP_QA_PROMPT), single_chunks)
    generated.extend(meta for meta, _ in singles[:single_target])
    generated.extend(meta for meta, _ in _generate_multi_hop(
        chain(MULTI_HOP_QA_PROMPT), pool, multi_target, seed, candidate_multiplier=2,
    ))
    sources = [str(item.get("source", "")) for item in documents if item.get("source")]
    generated.extend(meta for meta, _ in _generate_negative(
        chain(NEGATIVE_QA_PROMPT), sources, negative_target, attempt_multiplier=1.5,
    ))

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for meta in generated:
        key = re.sub(r"\s+", "", meta["query"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(meta)

    counts: dict[str, int] = {}
    for meta in deduped:
        qtype = meta["question_type"]
        counts[qtype] = counts.get(qtype, 0) + 1
    required = {
        qtype: DEFAULT_QUOTAS["gold"][qtype] + DEFAULT_QUOTAS["silver"][qtype]
        for qtype in DEFAULT_QUOTAS["gold"]
    }
    short = {qtype: (counts.get(qtype, 0), expected) for qtype, expected in required.items()
             if counts.get(qtype, 0) < expected}
    if short:
        raise DatasetValidationError(f"候选生成后配额不足：{short}")

    fingerprint = knowledge_base_fingerprint(kb)
    generated_at = datetime.now(UTC).astimezone().isoformat()
    model = get_model()
    model_name = str(
        getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
    )
    ordinals: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, meta in enumerate(deduped, 1):
        qtype = meta["question_type"]
        ordinals[qtype] = ordinals.get(qtype, 0) + 1
        tier, status = _candidate_tier(qtype, ordinals[qtype])
        evidence = [
            {
                "source": record["source"],
                "document_version": str(record.get("version") or "current"),
                "chunk_id": str(record["id"]),
                "quote": str(record.get("evidence_content") or record["content"]).strip()[:800],
            }
            for record in meta.get("evidence_records", [])
        ]
        reference = "" if qtype == "negative" else meta["ground_truth"]
        rows.append({
            "id": f"eval-candidate-{index:04d}",
            "query": meta["query"],
            "question_type": qtype,
            "difficulty": "medium",
            "answer_type": "comparison" if qtype == "multi_hop" else "factual",
            "reference_answer": reference,
            "required_facts": [] if qtype == "negative" else _required_facts(reference),
            "evidence": evidence,
            "expected_behavior": meta.get("expected_behavior", ""),
            "tier": tier,
            "review_status": status,
            "reviewer": "",
            "review_note": "",
            "provenance": {
                "generator_model": model_name,
                "generated_at": generated_at,
                "kb_fingerprint": fingerprint,
                "seed": seed,
            },
        })
    dataset = EvaluationDataset.from_dicts(rows)
    dataset.validate(evidence_resolver=build_evidence_resolver(kb))
    output = dataset.save(out_path)
    return dataset, output


def _missing_related_docs(related_docs: str, existing: set[str]) -> list[str]:
    """找出相关文档里知识库没有的文件。

    文件名本身可以含逗号（例如「不改基本面，燃机订单…」）。先按分隔符切开，
    再把相邻片段用原分隔符拼回去，能对上已知文件名就不算缺失。
    """
    text = (related_docs or "").strip()
    if not text:
        return []
    stems = {stem for stem in existing if stem}
    if Path(text).stem in stems:
        return []
    parts = [part.strip() for part in re.split(r"[,，;；|]", text) if part.strip()]
    missing: list[str] = []
    index = 0
    while index < len(parts):
        matched_end = None
        for end in range(len(parts), index, -1):
            window = parts[index:end]
            candidates = [window[0]] if end == index + 1 else []
            for sep in ("，", ",", ";", "；", "|"):
                candidates.append(sep.join(window))
            if any(Path(candidate).stem in stems for candidate in candidates):
                matched_end = end
                break
        if matched_end is None:
            missing.append(parts[index])
            index += 1
        else:
            index = matched_end
    return missing


def filter_entries_by_kb(
    entries: list[Any],
    kb: Any,
    *,
    allow_missing: bool = False,
) -> tuple[list[Any], list[tuple[Any, list[str]]]]:
    """预检测试集条目：相关文档不在知识库的条目剔除（或保留）。

    Returns:
        (保留条目, [(被剔除条目, 缺失文档列表), ...])
    """
    existing = {
        Path(item.get("source", "")).stem
        for item in kb.list_documents()
    }
    kept: list[Any] = []
    removed: list[tuple[Any, list[str]]] = []
    for entry in entries:
        missing = _missing_related_docs(entry.related_docs or "", existing)
        if missing and not allow_missing:
            removed.append((entry, missing))
            continue
        kept.append(entry)
    if removed:
        logger.warning("剔除 %d 条相关文档不在知识库的测试条目", len(removed))
    return kept, removed
