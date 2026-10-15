"""研报元数据抽取：正则优先 → LLM 回退 → 受控词表校验。

链路设计要点（改动前请先读）：

- **正则可解释、可复现**，研报文件名高度规范（``名称(代码)…-券商-日期``），
  能确定性拿到的字段绝不交给 LLM。
- **行业只由 LLM 抽取**，但必须落在申万 2021 版受控词表内，并校验
  「二级属于所选一级」。理由：检索过滤走精确匹配（``==`` / ``in``），
  LLM 若输出「白酒」（申万二级实为「白酒Ⅱ」），过滤会**静默漏召**。
- **类型与类型必填字段必须齐备**，否则拒绝文档；原始抽取异类字段丢弃。
"""

from __future__ import annotations

import json
import copy
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from finance_rag.src.rag.ingestion.public_metadata import (
    public_audit_needs_review,
    validate_public_metadata_audit,
    validate_publication_date,
)

logger = logging.getLogger(__name__)

REPORT_TYPES: tuple[str, ...] = ("个股", "行业", "宏观")

_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[4] / "assets" / "taxonomy" / "sw_industry.json"
)

# ``贵州茅台(600519)`` / ``贵州茅台（600519）``
_SECURITY_NAME_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9*]{2,12})[（(]\d{6}[)）]")
# ``-中信证券-`` / ``_中信证券`` 等
_BROKER_RE = re.compile(r"([\u4e00-\u9fff]{2,6}证券)")

_FIELD_LIMITS = {
    "security_name": 64,
    "industry_l1": 32,
    "industry_l2": 32,
    "report_type": 16,
    "broker": 64,
}

_LLM_PROMPT = """你是金融研报元数据抽取器。请只输出 JSON，不要解释、不要代码块。

字段与取值要求：
- 先判断研究对象：单家公司属于个股，特定产业/技术/医疗科技的研究或白皮书属于行业，经济总量/政策/大类资产属于宏观。
- 非券商白皮书也按研究对象分类；缺少股票代码或跨产业应用不意味着宏观。行业主题必须根据主要研究对象选择最相关的申万一级，二级不确定可省略。
- security_name：个股报告必填证券简称，行业与宏观报告不得包含此字段。
- report_type：只能取 "个股" / "行业" / "宏观" 之一。
- broker：发布机构名称，如 "中信证券"；无法确定留空字符串。
- industry_l1：行业报告必填，**必须**从下面的一级行业清单中原样选择；个股与宏观报告不得包含此字段。
- industry_l2：行业报告可选，**必须**属于所填 industry_l1；个股与宏观报告不得包含此字段。

一级行业 -> 二级行业清单：
{taxonomy}

研报文件名：{filename}
研报标题：{title}
研报正文开头：
{head}

请按 report_type 只输出适用字段。例如宏观：{{"report_type": "宏观", "broker": ""}}。"""


@dataclass(frozen=True)
class ExtractedMetadata:
    """一篇研报的元数据抽取结果。"""

    security_name: str = ""
    industry_l1: str = ""
    industry_l2: str = ""
    report_type: str = ""
    broker: str = ""
    meta_source: str = "regex"
    needs_review: bool = False
    type_metadata_audit: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        """并入文档级 metadata 的字典形式。"""
        values = {
            "security_name": self.security_name,
            "industry_l1": self.industry_l1,
            "industry_l2": self.industry_l2,
            "report_type": self.report_type,
            "broker": self.broker,
            "meta_source": self.meta_source,
            "needs_review": self.needs_review,
        }
        if self.type_metadata_audit:
            values["type_metadata_audit"] = self.type_metadata_audit
        return build_canonical_metadata(values)


_TYPE_FIELDS = frozenset({"security_name", "industry_l1", "industry_l2"})
_APPLICABLE_FIELDS = {
    "个股": frozenset({"security_name"}),
    "行业": frozenset({"industry_l1", "industry_l2"}),
    "宏观": frozenset(),
}


def build_canonical_metadata(
    metadata: dict[str, Any], *, explicit_fields: set[str] | None = None
) -> dict[str, Any]:
    """完整校验文档元数据；人工明确提交的异类字段拒绝，抽取残留丢弃。"""
    report_type = str(metadata.get("report_type") or "").strip()
    if report_type not in REPORT_TYPES:
        raise ValueError("report_type 必填且仅允许 个股/行业/宏观")
    applicable = _APPLICABLE_FIELDS[report_type]
    foreign = (explicit_fields or set()) & (_TYPE_FIELDS - applicable)
    if foreign:
        raise ValueError(f"{report_type} 不允许提交字段：{', '.join(sorted(foreign))}")
    result: dict[str, Any] = {
        "report_type": report_type,
        "broker": _clip(metadata.get("broker"), "broker"),
        "meta_source": str(metadata.get("meta_source") or "regex").strip()[:16],
        "needs_review": bool(metadata.get("needs_review", False)),
    }
    if "public_metadata_audit" in metadata:
        audit = validate_public_metadata_audit(metadata["public_metadata_audit"])
        result["public_metadata_audit"] = audit
        result["needs_review"] |= public_audit_needs_review(audit)
    if "type_metadata_audit" in metadata:
        result["type_metadata_audit"] = copy.deepcopy(metadata["type_metadata_audit"])
        result["needs_review"] |= bool(result["type_metadata_audit"])
    for field, limit in (("category", 64), ("date", 32)):
        if field in metadata:
            raw_value = metadata[field]
            value = "" if raw_value is None else str(raw_value)
            if len(value.encode("utf-8")) > limit:
                raise ValueError(f"{field} 不能超过 {limit} 个 UTF-8 字节")
            if field == "date":
                value = validate_publication_date(value)
            result[field] = value
    if "version" in metadata:
        result["version"] = metadata["version"]
    if report_type == "个股":
        name = _clip(metadata.get("security_name"), "security_name")
        if not name:
            raise ValueError("个股 report_type 必须有 security_name")
        result["security_name"] = name
    elif report_type == "行业":
        taxonomy = load_industry_taxonomy()
        level1 = str(metadata.get("industry_l1") or "").strip()
        level2 = str(metadata.get("industry_l2") or "").strip()
        if not level1 and level2:
            parents = [name for name, children in taxonomy.items() if level2 in children]
            if len(parents) == 1:
                level1 = parents[0]
        if level1 not in taxonomy:
            raise ValueError("行业 report_type 必须有合法的 industry_l1（申万一级行业）")
        result["industry_l1"] = level1
        if level2:
            if level2 in taxonomy[level1]:
                result["industry_l2"] = level2
            else:
                result["needs_review"] = True
                result["type_metadata_audit"] = {"industry_l2": {
                    "status": "unknown", "value": level2,
                    "reasons": ["industry_l2_outside_selected_l1"],
                }}
    return result


def merge_metadata_patch(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """类型切换先清理旧字段，再合并显式提交值并执行完整校验。"""
    allowed = _TYPE_FIELDS | {"report_type", "broker", "category", "date"}
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"未知元数据字段：{', '.join(sorted(unknown))}")
    merged = dict(current)
    if "report_type" in patch and patch["report_type"] != current.get("report_type"):
        for field in _TYPE_FIELDS:
            merged.pop(field, None)
    merged.update(patch)
    merged["meta_source"] = "manual"
    if "public_metadata_audit" in current:
        audit = validate_public_metadata_audit(current["public_metadata_audit"])
        for field in ("date", "broker"):
            if field in patch:
                value = validate_publication_date(patch[field]) if field == "date" else _clip(patch[field], "broker")
                item = audit["fields"][field]
                item.update(value=value, status="manual", reasons=["manual_patch"])
                item["evidence"].append({"raw": str(patch[field] or ""), "value": value,
                                         "page": None, "location": None, "source": "manual_patch"})
        merged["public_metadata_audit"] = audit
    type_audit = copy.deepcopy(current.get("type_metadata_audit", {}))
    if "report_type" in patch and patch["report_type"] != current.get("report_type"):
        type_audit = {}
    elif set(patch) & {"industry_l1", "industry_l2"}:
        type_audit.pop("industry_l2", None)
    if type_audit:
        merged["type_metadata_audit"] = type_audit
    else:
        merged.pop("type_metadata_audit", None)
    untracked_review = bool(current.get("needs_review")) and not (
        current.get("public_metadata_audit") or current.get("type_metadata_audit"))
    merged["needs_review"] = bool(type_audit) or untracked_review
    return build_canonical_metadata(merged, explicit_fields=set(patch))


@lru_cache(maxsize=1)
def load_industry_taxonomy(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """加载申万一级 -> 二级受控词表。

    这是**词表**而不是「代码 -> 行业」映射表：行业值由 LLM 抽取后在此校验。
    文件缺失或损坏时返回空词表（此时所有行业值都会被丢弃并标记待确认，
    而不是放行任意自由文本）。
    """
    target = Path(path) if path is not None else _TAXONOMY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        industries = payload["industries"]
        return {
            str(level1): frozenset(str(item) for item in level2s)
            for level1, level2s in industries.items()
        }
    except Exception as exc:  # noqa: BLE001 - 词表不可用不应让入库失败
        logger.warning("加载申万行业词表失败（%s）：%s", target, exc)
        return {}


def _clip(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if len(text.encode("utf-8")) > _FIELD_LIMITS[field]:
        raise ValueError(f"{field} 不能超过 {_FIELD_LIMITS[field]} 个 UTF-8 字节")
    return text


def resolve_report_title(path: str | Path, source: str) -> str:
    """优先读取 PDF 内置标题；元数据缺失或文件不可读时退回源文件名。"""
    if Path(path).suffix.lower() == ".pdf":
        try:
            import fitz

            with fitz.open(path) as document:
                title = str((document.metadata or {}).get("title") or "").strip()
                if title:
                    return title
        except Exception as exc:  # noqa: BLE001 - 标题不能使解析入口失败
            logger.debug("PDF 标题不可读（%s）：%s", source, exc)
    return Path(source).stem


def _regex_security_name(filename: str, title: str) -> str:
    """从文件名/标题解析证券简称。"""
    scopes = [filename, title]
    for scope in scopes:
        match = _SECURITY_NAME_RE.search(scope or "")
        if match:
            return match.group(1)
    return ""


def _regex_broker(filename: str, title: str) -> str:
    for scope in (filename, title):
        match = _BROKER_RE.search(scope or "")
        if match:
            return match.group(1)
    return ""


def _regex_report_type(filename: str, title: str, security_name: str) -> str:
    """报告类型只用文件名/标题判定。

    正文开头常常罗列一堆股票代码，用它判类型会把行业研报误判成个股研报。
    """
    text = f"{filename or ''} {title or ''}"
    if "宏观" in text:
        return "宏观"
    if "行业" in text:
        return "行业"
    if security_name:
        return "个股"
    return ""


def _model_text(raw: Any) -> str:
    """取出模型返回的**文本内容**。

    langchain 的 ``model.invoke`` 返回的是 ``AIMessage`` 而不是字符串；直接
    ``str(AIMessage)`` 会把 ``additional_kwargs=...``、``response_metadata=...``
    一起带上，于是「首尾花括号之间」切出来的根本不是合法 JSON，解析必然失败。
    实测这会让**每一篇**文档的元数据都为空（研报文件名是哈希串，正则兜不住），
    而失败是静默的：只表现为 ``meta_source='regex'``、``needs_review=True``。
    """
    if raw is None:
        return ""
    content = getattr(raw, "content", None)
    if content is None and isinstance(raw, dict):
        content = raw.get("content")
    if content is None:
        return str(raw)
    if isinstance(content, list):
        # 部分模型返回分段内容：[{"type": "text", "text": "..."}]
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content)


def _parse_json_payload(raw: Any) -> dict[str, Any]:
    """从模型输出里取出 JSON 对象（容忍消息对象、```json 代码块与前后噪声）。"""
    if isinstance(raw, dict):
        return raw
    text = _model_text(raw).strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _llm_extract(*, filename: str, head: str, taxonomy: dict[str, frozenset[str]], title: str = "") -> dict[str, Any]:
    """让轻量模型抽取字段。任何失败都返回空字典，由调用方降级。"""
    from finance_rag.src.core.config import get_model, get_rewrite_model

    model = get_rewrite_model() or get_model()
    if model is None:
        return {}
    listing = "\n".join(
        f"- {level1}：{'、'.join(sorted(level2s))}" for level1, level2s in sorted(taxonomy.items())
    )
    prompt = _LLM_PROMPT.format(
        taxonomy=listing or "（词表不可用，全部留空）",
        filename=filename or "",
        title=title or "",
        head=(head or "")[:2000],
    )
    return _parse_json_payload(model.invoke(prompt))


def extract_metadata(
    filename: str,
    title: str = "",
    markdown_head: str = "",
    *,
    enable_llm: bool = True,
) -> ExtractedMetadata:
    """抽取一篇研报的元数据。

    模型不可用时降级到正则；仍缺少类型必填字段时明确拒绝文档。
    """
    taxonomy = load_industry_taxonomy()

    name = _regex_security_name(filename, title)
    values: dict[str, str] = {}
    if name:
        values["security_name"] = name
    broker = _regex_broker(filename, title)
    if broker:
        values["broker"] = broker
    report_type = _regex_report_type(filename, title, name)
    if report_type:
        values["report_type"] = report_type

    used_llm = False
    if enable_llm:
        try:
            payload = _llm_extract(filename=filename, title=title, head=markdown_head, taxonomy=taxonomy)
        except Exception as exc:  # noqa: BLE001 - 模型不可用不得阻断入库
            logger.warning("研报元数据 LLM 抽取失败，降级为正则结果：%s", exc)
            payload = {}
        if payload:
            # 正则拿到的值优先（可解释、可复现），LLM 只补空缺
            for key in _FIELD_LIMITS:
                value = payload.get(key)
                if value and not values.get(key):
                    values[key] = value
                    used_llm = True

    canonical = build_canonical_metadata({**values, "meta_source": "llm" if used_llm else "regex"})
    return ExtractedMetadata(**canonical)


def metadata_field_validators() -> dict[str, Callable[[str], bool]]:
    """研报元数据字段的取值校验器（入库、API 修正、检索过滤共用一套规则）。

    集中在一处的原因：这三条路径都以「精确匹配」的方式使用这些值，
    任何一条放宽，都会造成静默漏召或写入不可过滤的值。
    """
    taxonomy = load_industry_taxonomy()
    return {
        "security_name": lambda value: 0 < len(value) <= 64,
        "industry_l1": lambda value: value in taxonomy,
        "industry_l2": lambda value: any(value in children for children in taxonomy.values()),
        "report_type": lambda value: value in REPORT_TYPES,
        "broker": lambda value: 0 < len(value) <= 64,
    }


def validate_metadata_field(field: str, value: Any) -> str:
    """校验单个元数据字段值；非法或空返回空串。

    返回空串而不是抛异常：调用方（入库/API）按「留空」处理，
    空值不参与过滤，因此不会误收窄召回。
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    validator = metadata_field_validators().get(field)
    if validator is None or not validator(text):
        return ""
    return text


def apply_metadata(
    metadata: dict[str, Any] | None, extracted: ExtractedMetadata
) -> dict[str, Any]:
    """把抽取结果并入文档级 metadata（不改动调用方传入的字典）。"""
    merged = dict(metadata or {})
    merged.update(extracted.as_metadata())
    if metadata and "public_metadata_audit" in metadata:
        audit = validate_public_metadata_audit(metadata["public_metadata_audit"])
        # unknown/conflict 也不能用 LLM 身份猜测填空。
        merged["broker"] = audit["fields"]["broker"]["value"]
        merged["date"] = audit["fields"]["date"]["value"]
    merged["needs_review"] = bool((metadata or {}).get("needs_review")) or extracted.needs_review
    return build_canonical_metadata(merged)


__all__ = [
    "REPORT_TYPES",
    "ExtractedMetadata",
    "apply_metadata",
    "extract_metadata",
    "build_canonical_metadata",
    "merge_metadata_patch",
    "load_industry_taxonomy",
    "metadata_field_validators",
    "validate_metadata_field",
]
