"""研报元数据抽取与规范化：正则优先 → LLM 回退。

字段体系（精简版）：

- **核心字段**：``date``（生效日期）、``version``（文档版本）。
- **研报字段**：``report_type``（个股/行业/宏观/策略）与 ``broker``
  （发布机构）。平台只接收研报，所有文档都按研报规则校验。

链路设计要点（改动前请先读）：

- **正则可解释、可复现**，研报文件名高度规范（``名称(代码)…-券商-日期``），
  能确定性拿到的字段绝不交给 LLM。
- 检索过滤走精确匹配（``==`` / ``in``），因此所有字段值都必须先过
  :func:`build_canonical_metadata` 的完整校验，任何一条路径放宽都会
  造成静默漏召或写入不可过滤的值。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.chunking.public_metadata import validate_publication_date

logger = logging.getLogger(__name__)

REPORT_TYPES: tuple[str, ...] = ("个股", "行业", "宏观", "策略")

# ``贵州茅台(600519)`` / ``贵州茅台（600519）``：证券名+代码是强个股信号
_SECURITY_CODE_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9*]{2,12}[（(]\d{6}[)）]")
# ``-中信证券-`` / ``_中信证券`` 等
_BROKER_RE = re.compile(r"([\u4e00-\u9fff]{2,6}证券)")

_FIELD_LIMITS = {
    "broker": 64,
}

_LLM_PROMPT = """你是金融研报元数据抽取器。请只输出 JSON，不要解释、不要代码块。

字段与取值要求：
- 先判断研究对象：单家公司属于个股，特定产业/技术/医疗科技的研究或白皮书属于行业，经济总量/政策/大类资产属于宏观，市场配置/风格轮动/仓位与行业比较属于策略。
- 非券商白皮书也按研究对象分类；缺少股票代码或跨产业应用不意味着宏观。
- report_type：只能取 "个股" / "行业" / "宏观" / "策略" 之一。
- broker：发布机构名称，如 "中信证券"；无法确定留空字符串。

研报文件名：{filename}
研报标题：{title}
研报正文开头：
{head}

只输出适用字段，例如：{{"report_type": "宏观", "broker": ""}}"""


@dataclass(frozen=True)
class ExtractedMetadata:
    """一篇文档的元数据抽取结果。"""

    date: str = ""
    version: str = ""
    report_type: str = ""
    broker: str = ""

    def as_metadata(self) -> dict[str, Any]:
        """并入文档级 metadata 的字典形式。"""
        return build_canonical_metadata({
            "date": self.date,
            "version": self.version,
            "report_type": self.report_type,
            "broker": self.broker,
        })


def build_canonical_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """完整校验文档元数据；``report_type`` 必须取个股/行业/宏观/策略之一。

    ``category`` 已从平台模型中移除，返回值固定为空串以兼容历史 schema。
    """
    report_type = str(metadata.get("report_type") or "").strip()
    broker = _clip(metadata.get("broker"), "broker")
    if report_type not in REPORT_TYPES:
        raise ValueError("report_type 必填且仅允许 个股/行业/宏观/策略")

    result: dict[str, Any] = {
        "category": "",
        "date": validate_publication_date(metadata.get("date") or ""),
        "report_type": report_type,
        "broker": broker,
    }
    if "version" in metadata:
        result["version"] = metadata["version"]
    return result


def merge_metadata_patch(current: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """合并显式提交值（只带需要改的字段；空串表示清空）并执行完整校验。"""
    allowed = {"date", "version", "report_type", "broker"}
    unknown = set(patch) - allowed
    if unknown:
        raise ValueError(f"未知元数据字段：{', '.join(sorted(unknown))}")
    merged = dict(current)
    merged.update(patch)
    return build_canonical_metadata(merged)


def _clip(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if len(text.encode("utf-8")) > _FIELD_LIMITS[field]:
        raise ValueError(f"{field} 不能超过 {_FIELD_LIMITS[field]} 个 UTF-8 字节")
    return text


def resolve_report_title(path: str, source: str) -> str:
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


def _regex_broker(filename: str, title: str) -> str:
    for scope in (filename, title):
        match = _BROKER_RE.search(scope or "")
        if match:
            return match.group(1)
    return ""


def _regex_report_type(filename: str, title: str) -> str:
    """报告类型只用文件名/标题判定。

    正文开头常常罗列一堆股票代码，用它判类型会把行业研报误判成个股研报。
    """
    text = f"{filename or ''} {title or ''}"
    if _SECURITY_CODE_RE.search(text):
        return "个股"
    if "策略" in text:
        return "策略"
    if "宏观" in text:
        return "宏观"
    if "行业" in text:
        return "行业"
    return ""


def _model_text(raw: Any) -> str:
    """取出模型返回的**文本内容**。

    langchain 的 ``model.invoke`` 返回的是 ``AIMessage`` 而不是字符串；直接
    ``str(AIMessage)`` 会把 ``additional_kwargs=...``、``response_metadata=...``
    一起带上，于是「首尾花括号之间」切出来的根本不是合法 JSON，解析必然失败。
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


def _llm_extract(*, filename: str, head: str, title: str = "") -> dict[str, Any]:
    """让轻量模型抽取字段。任何失败都返回空字典，由调用方降级。"""
    from src.core.config import get_model, get_rewrite_model

    model = get_rewrite_model() or get_model()
    if model is None:
        return {}
    prompt = _LLM_PROMPT.format(
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
    """抽取一篇文档的元数据。

    模型不可用时降级到正则；仍缺少 report_type 时明确拒绝文档。
    """
    values: dict[str, str] = {}

    report_type = _regex_report_type(filename, title)
    if report_type:
        values["report_type"] = report_type
    broker = _regex_broker(filename, title)
    if broker:
        values["broker"] = broker

    if enable_llm:
        try:
            payload = _llm_extract(filename=filename, title=title, head=markdown_head)
        except Exception as exc:  # noqa: BLE001 - 模型不可用不得阻断入库
            logger.warning("研报元数据 LLM 抽取失败，降级为正则结果：%s", exc)
            payload = {}
        if payload:
            # 正则拿到的值优先（可解释、可复现），LLM 只补空缺
            for key in ("report_type", "broker"):
                value = payload.get(key)
                if value and not values.get(key):
                    values[key] = value

    canonical = build_canonical_metadata(values)
    return ExtractedMetadata(
        date=canonical["date"],
        version=canonical.get("version", ""),
        report_type=canonical["report_type"],
        broker=canonical["broker"],
    )


def metadata_field_validators() -> dict[str, Callable[[str], bool]]:
    """文档元数据字段的取值校验器（入库、API 修正、检索过滤共用一套规则）。

    集中在一处的原因：这三条路径都以「精确匹配」的方式使用这些值，
    任何一条放宽，都会造成静默漏召或写入不可过滤的值。
    """
    return {
        "date": lambda value: bool(validate_publication_date(value, allow_empty=False)),
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
    """把抽取结果并入文档级 metadata（不改动调用方传入的字典）。

    只补空缺：抽取链路不产 ``date``，空串（未知）不得覆盖公共元数据
    已解析出的 date / broker，否则入库日期会被静默清空。
    """
    merged = dict(metadata or {})
    merged.update({key: value for key, value in extracted.as_metadata().items() if value})
    return build_canonical_metadata(merged)


__all__ = [
    "REPORT_TYPES",
    "ExtractedMetadata",
    "apply_metadata",
    "extract_metadata",
    "build_canonical_metadata",
    "merge_metadata_patch",
    "metadata_field_validators",
    "validate_metadata_field",
]
