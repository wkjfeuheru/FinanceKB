"""策略评估模块（基于 ragas 框架）。

使用 ragas 对 RAG 优化点 A/B 实验的检索与回答质量进行评估。**生成侧固定输出
RAGAS 四指标**（覆盖 RAG 的两条链路）：

* 检索链路：``context_precision``（检索噪音）、``context_recall``（知识盲区）；
* 生成链路：``faithfulness``（防幻觉）、``answer_relevancy``（防跑偏）。

核心组件：
* :class:`RagasBatchEvaluator` — 批量评估器：一次评估一臂的全部题目
  （打包为单个 EvaluationDataset，显著降低开销与指标抖动）；
* :func:`compute_composite_score` — 分层加权综合评分（历史遗留，当前无调用方，
  权重表里引用已下线指标的键为惰性键）；
* 本地检索指标（hit_rate / MRR / NDCG）与对抗层拒绝分数。
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

# 将项目级 libs 目录加入 sys.path（ragas 及其依赖安装在此）
# 确保项目根目录在 sys.path 中：既支持 `python -m finance_rag.src.eval.ragas_eval`，
# 也支持直接 `python finance_rag/src/eval/ragas_eval.py` 运行
# （直接运行脚本时 sys.path[0] 是脚本目录，会报 ModuleNotFoundError: config）。
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 将项目级 libs 目录加入 sys.path（ragas 及其依赖安装在此），优先于已安装版本
_LIBS_DIR = _PROJECT_ROOT / "libs"
if _LIBS_DIR.exists() and str(_LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(_LIBS_DIR))

# 兼容 langchain-community >= 0.4 — vertexai 模块已被移除，
# 但 ragas 0.4.3 在模块顶层导入了该路径。此处注入占位模块避免 ImportError。
try:
    from langchain_community.chat_models import vertexai  # noqa: F401
except ImportError:
    import types
    _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    _dummy.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy

from ragas.llms.base import LangchainLLMWrapper
from ragas.prompt.utils import extract_json

from finance_rag.src.core.config import (
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
    get_model,
)
from finance_rag.src.eval.metrics import (
    GENERATION_METRICS,
    compute_chunk_retrieval_metrics,
)
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base

logger = logging.getLogger(__name__)

#: 生成侧统一使用的 RAGAS 四指标（定义在 :mod:`metrics`，避免多处副本漂移）。
GENERATION_CHAIN_METRICS: tuple[str, ...] = GENERATION_METRICS


class JsonExtractLangchainLLMWrapper(LangchainLLMWrapper):
    """Extract the JSON payload from Ragas judge-model responses.

    DeepSeek may wrap JSON in Markdown or explanatory text.  Ragas validates
    judge responses against Pydantic models, so normalize every generated
    candidate before the prompt parser sees it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_outputs: list[str] = []

    def clear_captured_outputs(self) -> None:
        self.captured_outputs.clear()

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None,
    ):
        result = await super().agenerate_text(
            prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )
        for batch in result.generations:
            for generation in batch:
                if generation.text:
                    raw_text = generation.text
                    self.captured_outputs.append(raw_text)
                    generation.text = extract_json(raw_text)
        return result


def _json_safe_metric(value: Any) -> float | None:
    """Convert a metric value to strict-JSON-compatible form.

    Ragas may return NaN or infinity when a metric cannot be calculated for a
    sample.  Starlette deliberately rejects those values during JSON encoding,
    so represent unavailable metrics as JSON ``null`` instead.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 4) if math.isfinite(numeric) else None


def _text_units(text: str) -> set[str]:
    """Return stable Chinese bigrams and alphanumeric words for overlap scoring."""
    normalized = re.sub(r"\[\d+\]", " ", (text or "").lower())
    units = set(re.findall(r"[a-z0-9]+", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(sequence) == 1:
            units.add(sequence)
        else:
            units.update(sequence[i:i + 2] for i in range(len(sequence) - 1))
    return units


def _token_f1(left: str, right: str) -> float:
    left_units, right_units = _text_units(left), _text_units(right)
    if not left_units or not right_units:
        return 0.0
    common = len(left_units & right_units)
    if common == 0:
        return 0.0
    precision = common / len(left_units)
    recall = common / len(right_units)
    return 2 * precision * recall / (precision + recall)


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[。！？!?；;\n]+", text or "")
        if len(part.strip()) >= 4
    ]


def _supported_sentence_score(claims: str, evidence: list[str]) -> float:
    """Average support of each claim sentence by its best evidence chunk."""
    claim_list = _sentences(claims)
    evidence_list = [item for item in evidence if item]
    if not claim_list or not evidence_list:
        return 0.0
    scores = [
        max((_token_f1(claim, item) for item in evidence_list), default=0.0)
        for claim in claim_list
    ]
    return round(sum(scores) / len(scores), 4)


def _fallback_metrics(sample: dict[str, Any]) -> dict[str, float]:
    """Deterministic local fallbacks for LLM-judge metrics."""
    response = sample.get("response", "")
    contexts = sample.get("retrieved_contexts", []) or []
    return {
        "faithfulness": _supported_sentence_score(response, contexts),
    }


# ---------------------------------------------------------------------------
# 本地检索指标
# ---------------------------------------------------------------------------

def _split_doc_names(related_docs: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """把「相关文档」字段拆成去重后的文档名列表（保留原始写法）。"""
    if not isinstance(related_docs, str):
        return list(dict.fromkeys(str(item).strip() for item in (related_docs or []) if str(item).strip()))
    names: list[str] = []
    for part in re.split(r"[,，;；|]", related_docs or ""):
        name = part.strip()
        if name and name not in names:
            names.append(name)
    return names


def _compute_doc_coverage(
    retrieved_sources: list[str],
    related_docs: str,
    k: int = 5,
) -> dict[str, float | None]:
    """文档级覆盖度：区分「命中过一篇」与「全部命中」。

    ``hit_rate`` 只要求命中任意一篇，在多跳问题（相关文档 2 篇）上会饱和为 1.0，
    无法反映覆盖程度；本函数补充 0 / 0.5 / 1.0 的梯度。

    Args:
        retrieved_sources: 检索返回的 source 列表（按 rank 排序）。
        related_docs: 相关文档字段（逗号/分号分隔）。
        k: 检索深度。

    Returns:
        ``{"doc_coverage", "doc_recall"}``；无相关文档标注时返回 None（n/a）。
    """
    names = _split_doc_names(related_docs)
    if not names:
        return {"doc_coverage": None, "doc_recall": None}

    related_stems = {Path(name).stem.lower() for name in names}
    retrieved_stems = {
        Path(source).stem.lower() for source in retrieved_sources[:k] if source
    }
    covered = len(related_stems & retrieved_stems)
    total = len(related_stems)
    return {
        "doc_coverage": round(covered / total, 4) if total else None,
        "doc_recall": 1.0 if total and covered == total else 0.0,
    }


def _compute_evidence_metrics(
    retrieved_chunk_ids: list[str],
    evidence_chunk_ids: list[str] | tuple[str, ...] | None,
    k: int = 5,
) -> dict[str, float | None]:
    """块级证据指标：标注的证据块在 top-k 里被召回了几成、排第几。

    这是当前评测链路缺失的一层：``hit_rate`` 是文档级（命中一篇即 1.0），
    ``multi_hop_hit`` 要求整份文档集合被覆盖，两者之间没有梯度，
    也无法区分「压根没召回」与「召回了但被 rerank 截断」。

    Args:
        retrieved_chunk_ids: 检索返回的 chunk id 列表（按 rank 排序）。
        evidence_chunk_ids: 测试集标注的证据 chunk id；为空表示该条目未标注证据。
        k: 检索深度。

    Returns:
        ``{"evidence_hit", "evidence_coverage", "evidence_rank",
        "evidence_all_in_topk"}``。

        未标注证据（``evidence_chunk_ids`` 为空）时全部返回 ``None``，
        即 n/a —— 这样「测试集没有证据标注」不会污染成假 0.0。
    """
    evidence = [str(cid) for cid in (evidence_chunk_ids or []) if cid]
    if not evidence:
        return {
            "evidence_hit": None,
            "evidence_coverage": None,
            "evidence_rank": None,
            "evidence_all_in_topk": None,
        }

    top_k = [str(cid) for cid in retrieved_chunk_ids[:k] if cid]
    rank_by_id = {cid: rank for rank, cid in enumerate(top_k, 1)}
    hits = [rank_by_id[cid] for cid in set(evidence) if cid in rank_by_id]

    return {
        "evidence_hit": 1.0 if hits else 0.0,
        "evidence_coverage": round(len(hits) / len(set(evidence)), 4),
        # 未命中记 k+1（而不是 0），避免与「排第 1」混淆
        "evidence_rank": float(min(hits)) if hits else float(k + 1),
        "evidence_all_in_topk": 1.0 if len(hits) == len(set(evidence)) else 0.0,
    }


def _compute_chunk_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    evidence_chunk_ids: list[str] | tuple[str, ...] | None,
    k: int = 5,
) -> dict[str, float | None]:
    """计算块级 Recall@k、Precision@k 和 MRR。

    Precision 的分母固定为 ``k``，即使断崖检测使实际返回少于 ``k`` 条，
    也不会因结果变少而虚增精确率。
    """
    return compute_chunk_retrieval_metrics(retrieved_chunk_ids, evidence_chunk_ids, k)


def _compute_local_retrieval_metrics(
    retrieved_titles: list[str],
    related_docs: str,
    k: int = 5,
) -> dict[str, float]:
    """基于检索到的文档标题和相关文档字段计算本地检索指标。

    Args:
        retrieved_titles: 检索返回的文档标题列表（按 rank 排序）。
        related_docs: 相关文档字段（逗号/分号分隔的文档名）。
        k: 检索深度。

    Returns:
        含 hit_rate@k, MRR@k, NDCG@k 的字典。
    """
    if not related_docs or not retrieved_titles:
        return {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}

    # 标准化：去除扩展名进行比较
    related_set = {Path(name).stem.lower() for name in _split_doc_names(related_docs)}

    retrieved_stems = [Path(t).stem.lower() for t in retrieved_titles[:k]]

    # hit_rate@k: 至少有一个相关文档出现在 top-k 中
    hit = 1.0 if any(s in related_set for s in retrieved_stems) else 0.0

    # MRR@k: 第一个相关文档的倒数排名
    mrr = 0.0
    for rank, stem in enumerate(retrieved_stems, 1):
        if stem in related_set:
            mrr = 1.0 / rank
            break

    # NDCG@k: 简化版（相关性二值：相关=1，不相关=0）
    dcg = 0.0
    for rank, stem in enumerate(retrieved_stems, 1):
        if stem in related_set:
            dcg += 1.0 / (math.log2(rank + 1))
    # 理想 DCG（所有相关文档排在前面）
    ideal_count = min(len(related_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    ndcg = round(dcg / idcg, 4) if idcg > 0 else 0.0

    return {
        "hit_rate": round(hit, 4),
        "mrr": round(mrr, 4),
        "ndcg": ndcg,
    }


def build_context_text(contexts: list[str]) -> str:
    """将检索结果组装为 LLM 上下文文本。"""
    if not contexts:
        return "（未检索到相关文档）"
    chunks: list[str] = []
    for i, content in enumerate(contexts, 1):
        truncated = content[:2000] if content else ""
        chunks.append(f"[{i}] {truncated}")
    return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Ragas 指标与批量评估
# ---------------------------------------------------------------------------

def _get_metrics(llm=None, embeddings=None, *, fast: bool = False):
    """获取 ragas 评估指标列表。

    **生成侧固定为 RAGAS 四指标**，``fast`` 参数保留只为兼容既有调用方
    （``--fast`` / ``--full`` 不再改变指标集）：

    * 检索链路：``context_precision``（检索噪音）、``context_recall``（知识盲区）；
    * 生成链路：``faithfulness``（防幻觉）、``answer_relevancy``（防跑偏）。

    历史口径里的 ``answer_correctness`` / ``context_entity_recall`` 已下线：
    四指标之外的口径会让「同一份报告读两个标准」，且这两项在既有报告里
    与四项高度共线（历史 JSON 仍可读，只是不再产出）。
    """
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        Faithfulness,
        LLMContextRecall,
    )

    return [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        ContextPrecision(llm=llm),
        LLMContextRecall(llm=llm),
    ]


def _judge_model_name(model: Any) -> str:
    """取裁判模型标识（用于报告落盘溯源）。"""
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


class RagasBatchEvaluator:
    """Ragas 批量评估器：一次评估一臂的全部题目。

    ``evaluate_batch`` 将整臂样本打包为单个 ``EvaluationDataset`` 调用
    ``ragas.evaluate``（指标实例只初始化一次），再按样本拆分指标值，
    失败指标单独重试，faithfulness 保留确定性本地回退。
    """

    def __init__(self, kb=None):
        self._kb = kb or get_knowledge_base()

    def evaluate_batch(
        self,
        rows: list[dict[str, Any]],
        *,
        fast: bool = False,
        judge_model: Any = None,
        judge_label: str = "primary",
    ) -> dict[str, Any]:
        """批量评估（生成侧固定 RAGAS 四指标）。

        Args:
            rows: 样本列表，每项必须含 ``user_input`` / ``retrieved_contexts`` /
                  ``response`` / ``reference``（可附其它自定义字段，会原样
                  回填到 per_query 输出中）。
            fast: 兼容参数，不再改变指标集（生成侧恒为四指标）。
            judge_model: 裁判模型；``None`` 时用主模型。跨模型交叉校验时传入
                  第二裁判族的模型（结果里会标注 ``judge_label`` 便于溯源）。
            judge_label: 本次评估的裁判标签（``primary`` / ``secondary``）。

        Returns:
            ``{"metrics": {指标名: 均值}, "per_query": [{..., "metrics": {...},
            "answer_chars": int}], "metric_coverage": {指标名: 覆盖率},
            "metric_errors": {指标名: 错误信息}, "fallback_metrics": [...],
            "judge_label": str, "judge_model_name": str}``
        """
        judge = judge_model if judge_model is not None else get_model()
        if judge is None:
            raise ValueError("LLM 未配置（缺少 DEEPSEEK_API_KEY）")
        if not rows:
            return {
                "metrics": {}, "per_query": [], "metric_coverage": {},
                "metric_errors": {}, "fallback_metrics": [],
                "judge_label": judge_label, "judge_model_name": "",
            }

        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig

        eval_dataset = EvaluationDataset.from_list([
            {
                "user_input": row["user_input"],
                "retrieved_contexts": row.get("retrieved_contexts") or [],
                "response": row.get("response") or "",
                "reference": row.get("reference") or "",
            }
            for row in rows
        ])

        run_config = RunConfig(
            timeout=RAGAS_TIMEOUT_SECONDS,
            max_retries=RAGAS_MAX_RETRIES,
            max_workers=RAGAS_MAX_WORKERS,
        )

        # 包装 LLM（DeepSeek JSON 兼容）与本地嵌入
        evaluator_llm = JsonExtractLangchainLLMWrapper(
            judge,
            run_config=run_config,
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            self._kb.get_embeddings()
        )
        metrics = _get_metrics(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            fast=fast,
        )
        logger.info("开始 ragas 批量评估（%d 样本，裁判=%s），指标：%s",
                    len(rows), judge_label, [m.name for m in metrics])

        # 全部指标一次性并行评估；失败指标单独重试
        initial_result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            show_progress=False,
        )

        def _extract_metric_column(metric_name: str) -> list[float | None]:
            try:
                raw_values = initial_result[metric_name]
            except (KeyError, TypeError):
                return [None] * len(rows)
            values = [_json_safe_metric(v) for v in raw_values]
            return values

        columns: dict[str, list[float | None]] = {}
        metric_errors: dict[str, str] = {}
        for metric in metrics:
            columns[metric.name] = _extract_metric_column(metric.name)

        # 全部失败的指标单独重试（异常可见，便于定位真实原因）
        for metric in metrics:
            if any(value is not None for value in columns[metric.name]):
                continue
            evaluator_llm.clear_captured_outputs()
            try:
                retry_result = evaluate(
                    dataset=eval_dataset,
                    metrics=[metric],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                    show_progress=False,
                )
                columns[metric.name] = [
                    _json_safe_metric(v) for v in retry_result[metric.name]
                ]
            except Exception as exc:
                metric_errors[metric.name] = str(exc)
                raw_output = (
                    evaluator_llm.captured_outputs[-1]
                    if evaluator_llm.captured_outputs
                    else "<no LLM output>"
                )
                logger.warning(
                    "Ragas metric %s retry failed: %s; last LLM output: %s",
                    metric.name, exc, raw_output[:2000],
                )

        # faithfulness 确定性本地回退
        fallback_names: list[str] = []
        for i, row in enumerate(rows):
            if columns["faithfulness"][i] is None:
                columns["faithfulness"][i] = _fallback_metrics(row)[
                    "faithfulness"
                ]
                fallback_names.append("faithfulness")

        # 逐样本输出（保留调用方附加字段）
        per_query: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            item = {k: v for k, v in row.items() if k != "retrieved_contexts"}
            item["metrics"] = {
                name: columns[name][i] for name in columns
            }
            # 长度遥测：长度偏差的原始数据（不放进 metrics，避免拉高均值误读）
            item["answer_chars"] = len(str(row.get("response") or ""))
            per_query.append(item)

        # 聚合均值（仅统计有限值）与覆盖率（该指标有值的样本占比）
        aggregated: dict[str, float | None] = {}
        coverage: dict[str, float] = {}
        for name, values in columns.items():
            finite = [v for v in values if v is not None]
            aggregated[name] = (
                round(sum(finite) / len(finite), 4) if finite else None
            )
            coverage[name] = round(len(finite) / len(values), 4) if values else 0.0

        return {
            "metrics": aggregated,
            "per_query": per_query,
            "metric_coverage": coverage,
            "metric_errors": metric_errors,
            "fallback_metrics": sorted(set(fallback_names)),
            "judge_label": judge_label,
            "judge_model_name": _judge_model_name(judge),
        }


# ---------------------------------------------------------------------------
# 综合评分
# ---------------------------------------------------------------------------

# 默认权重（适用于 all 层级）。
# 注：`answer_correctness` 已从生成侧指标集下线，此处的键因此是**惰性键**——
# compute_composite_score 取不到值就跳过，不影响其它权重（该函数当前无调用方，
# 保留是为了不破坏历史报告里的权重口径）。
_DEFAULT_WEIGHTS = {
    "answer_correctness": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.15,
    "context_recall": 0.15,
    "context_precision": 0.10,
    "hit_rate": 0.10,
}

# 分层权重
_TIER_WEIGHTS: dict[str, dict[str, float]] = {
    "L1": {
        "answer_correctness": 0.15,
        "faithfulness": 0.15,
        "answer_relevancy": 0.10,
        "context_recall": 0.10,
        "context_precision": 0.10,
        "hit_rate": 0.20,
        "mrr": 0.10,
        "ndcg": 0.10,
    },
    "L2": {
        "answer_correctness": 0.30,
        "faithfulness": 0.25,
        "answer_relevancy": 0.15,
        "context_recall": 0.15,
        "context_precision": 0.10,
        "hit_rate": 0.05,
    },
    "L3": {
        "negative_rejection": 0.40,
        "faithfulness": 0.20,
        "answer_correctness": 0.15,
        "answer_relevancy": 0.15,
        "context_precision": 0.10,
    },
}


def compute_composite_score(
    metrics: dict[str, float | None],
    tier: str = "all",
) -> dict[str, Any]:
    """根据指标值和测试层级计算加权综合分。"""
    weights = _TIER_WEIGHTS.get(tier, _DEFAULT_WEIGHTS)
    total_weight = 0.0
    weighted_sum = 0.0
    included: dict[str, float] = {}
    for name, weight in weights.items():
        value = metrics.get(name)
        if value is not None and isinstance(value, (int, float)):
            weighted_sum += float(value) * weight
            total_weight += weight
            included[name] = float(value)
    if total_weight > 0:
        composite = round(weighted_sum / total_weight, 4)
    else:
        composite = 0.0
    return {
        "composite_score": composite,
        "included_metrics": included,
        "total_weight_used": round(total_weight, 4),
        "tier": tier,
    }
