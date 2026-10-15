"""有来源证据的公共元数据；不从临时路径、时间戳或正文引用猜测。"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

RULES_VERSION = "public-metadata-v1"
_DATE = r"(?<!\d)(20\d{2})\s*[年/._-]\s*(\d{1,2})\s*[月/._-]\s*(\d{1,2})\s*日?(?!\d)"
_DATE_RE = re.compile(_DATE)
_COMPACT_DATE_RE = re.compile(r"(?<![A-Za-z0-9])(20\d{2})(\d{2})(\d{2})(?![A-Za-z0-9])")
_PUBLISHED_RE = re.compile(r"(?:发布日期|发布时间|报告日期|公布日期|出版日期)\s*[:：]?\s*(" + _DATE + r")")
_EXCLUDED_RE = re.compile(r"报告期|相关研究|相关报告|参考文献|资料来源|数据来源|基本数据|基础数据|市场数据|截至|收盘|股价|走势图|财务|盈利预测")
_LABEL_CONTEXT_EXCLUDED_RE = re.compile(r"相关研究|相关报告|参考文献|资料来源|数据来源|基本数据|基础数据|市场数据|截至|收盘|股价|走势图|财务|盈利预测|引用|参考")
_PDF_SECTION_RE = re.compile(r"相关研究|相关报告|基本数据|基础数据|市场数据|财务|盈利预测|股价|走势图|Related\s+research|Basic\s+data", re.IGNORECASE)
_HEADING_RE = re.compile(r"(?:相关研究|相关报告|基本数据|基础数据|市场数据|财务预测|参考文献|Related\s+research|Basic\s+data)\s*[:：]?", re.IGNORECASE)
_BROKERS = (
    "东吴证券", "中信证券", "华泰证券", "中金公司", "中国国际金融股份有限公司",
    "国金证券", "国信证券", "国信证券经济研究所", "中航证券", "华鑫证券", "诚通证券", "太平洋证券",
    "开源证券", "招商证券", "申万宏源证券", "申万宏源", "海通证券", "国泰君安证券",
    "国泰海通证券", "广发证券", "兴业证券", "中信建投证券", "光大证券", "天风证券",
    "东北证券", "西南证券", "东兴证券", "财通证券", "浙商证券", "平安证券",
    "银河证券", "中国银河证券", "长江证券", "源达信息", "格林大华期货", "格林大华期货有限公司", "南华期货",
    "中国信息通信研究院", "全球计算联盟", "中国信科", "易观分析", "浪潮信息", "IDC", "Alleima",
)
_ALIASES = {"中国国际金融股份有限公司": "中金公司", "中国银河证券": "银河证券", "申万宏源": "申万宏源证券",
            "格林大华期货有限公司": "格林大华期货", "国信证券经济研究所": "国信证券"}
_FULL_SIGNED_NAMES = frozenset({"格林大华期货有限公司", "国信证券经济研究所"})
_INSTITUTION_RE = re.compile("|".join(re.escape(value) for value in sorted(_BROKERS, key=len, reverse=True)))


def validate_publication_date(value: str, *, allow_empty: bool = True) -> str:
    """显式日期须为完整 ISO 日历日；空串可表示未知或人工清空。"""
    text = "" if value is None else str(value).strip()
    if not text and allow_empty:
        return ""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError("date 必须是完整有效的 YYYY-MM-DD 日期")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError("date 必须是完整有效的 YYYY-MM-DD 日期") from exc


@dataclass(frozen=True)
class PublicMetadataResult:
    date: str
    broker: str
    needs_review: bool
    audit: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {"date": self.date, "broker": self.broker, "needs_review": self.needs_review,
                "public_metadata_audit": copy.deepcopy(self.audit)}


def public_audit_needs_review(audit: dict[str, Any]) -> bool:
    return any(audit["fields"][field]["status"] in {"unknown", "conflict"} for field in ("date", "broker"))


def validate_public_metadata_audit(audit: Any) -> dict[str, Any]:
    """校验内部审计载荷形状；外部请求不得传入此字段。"""
    if not isinstance(audit, dict) or audit.get("rules_version") != RULES_VERSION:
        raise ValueError("public_metadata_audit 规则版本不合法")
    if not isinstance(audit.get("source"), str) or not isinstance(audit.get("pdf_sha256"), str):
        raise ValueError("public_metadata_audit 来源不合法")
    fields = audit.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("public_metadata_audit 缺少 fields")
    for name in ("date", "broker"):
        item = fields.get(name)
        if (not isinstance(item, dict) or item.get("status") not in {"known", "unknown", "conflict", "manual"}
                or not isinstance(item.get("value"), str) or not isinstance(item.get("evidence"), list)
                or not isinstance(item.get("reasons"), list)):
            raise ValueError(f"public_metadata_audit {name} 不合法")
        for evidence in item["evidence"]:
            if not isinstance(evidence, dict) or not {"raw", "page", "location", "source"} <= evidence.keys():
                raise ValueError(f"public_metadata_audit {name} 证据不合法")
    return copy.deepcopy(audit)


def _evidence(raw: str, value: str, source: str, page: int | None = None,
              location: Any = None) -> dict[str, Any]:
    return {"raw": raw, "value": value, "source": source, "page": page, "location": location}


def _add_dates(text: str, origin: str, evidence: list, reasons: list, *, page=None,
               location=None, compact=False) -> None:
    patterns = (_DATE_RE, _COMPACT_DATE_RE) if compact else (_DATE_RE,)
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                value = date(*(int(part) for part in match.groups())).isoformat()
            except ValueError:
                value = ""
                if "invalid_calendar_date" not in reasons:
                    reasons.append("invalid_calendar_date")
            evidence.append(_evidence(match.group(0), value, origin, page, location))


def _label_is_in_excluded_context(text: str, offset: int) -> bool:
    """标签归属取决于所在区段；相关研究条目可跨多行。"""
    line_start = text.rfind("\n", 0, offset) + 1
    prefix = text[line_start:offset]
    if _LABEL_CONTEXT_EXCLUDED_RE.search(prefix):
        return True
    if re.search(r"本(?:研究)?报告\s*$", prefix):
        return False
    for line in reversed(text[:line_start].splitlines()):
        stripped = line.strip()
        heading = re.sub(r"^#{1,6}\s*", "", stripped)
        if _HEADING_RE.fullmatch(heading):
            return True
        if stripped.startswith("#"):
            break
    return False


def _add_publishers(text: str, origin: str, evidence: list, *, page=None, location=None,
                    filename=False) -> None:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        line_offset = offset
        offset += len(raw_line)
        compact = re.sub(r"\s+", "", line).strip()
        if (_EXCLUDED_RE.search(compact) or re.search(r"引用|参考|据|认为|观点|整理", compact)
                or _label_is_in_excluded_context(text, line_offset)):
            continue
        for match in _INSTITUTION_RE.finditer(compact):
            name = match.group(0)
            before, after = compact[:match.start()], compact[match.end():]
            signed = (
                (not before and after in {"研究所", "研究院", "研究", "研究所发布", "证券研究所", "经济研究所", "有限公司", "股份有限公司"})
                or (not before and not after and name in _FULL_SIGNED_NAMES)
                or bool(re.search(r"(?:发布机构|出版机构|发行机构|发布单位|出品方)[:：]?$", before))
            )
            if filename:
                signed = (not before or before[-1:] in "-_— ") and (not after or after[:1] in "-_—.")
            if signed:
                evidence.append(_evidence(line, _ALIASES.get(name, name), origin, page, location))


def _pdf_in_excluded_section(block: tuple, blocks: list, *, max_gap: float | None = None) -> bool:
    return any(
        len(previous) >= 5 and isinstance(previous[4], str)
        and _PDF_SECTION_RE.search(previous[4])
        and previous[1] < block[1]
        and (max_gap is None or block[1] - previous[3] <= max_gap)
        and abs(previous[0] - block[0]) <= 80
        for previous in blocks
    )


def _pdf_has_signature_context(block: tuple, blocks: list, page_rect: Any, *, last_page: bool) -> bool:
    raw = block[4]
    if re.search(r"发布机构|出版机构|发行机构|发布单位|出品方", raw):
        return True
    if not last_page:
        return block[1] <= page_rect.height * 0.25 or block[3] >= page_rect.height * 0.85
    if block[1] >= page_rect.height * 0.75 and block[0] >= page_rect.width * 0.55 and any(
        len(other) >= 5 and isinstance(other[4], str)
        and re.search(r"免责声明|免责及评级说明|法律声明", other[4])
        for other in blocks
    ):
        return True
    return any(
        len(other) >= 5 and isinstance(other[4], str)
        and block[1] < other[1] <= block[1] + 150
        and abs(other[0] - block[0]) <= 80
        and re.search(r"地址|电话|官网|邮编|服务热线|客服", other[4])
        for other in blocks
    )


def _landscape_cover_date(block: tuple, blocks: list, page_rect: Any) -> bool:
    """横版稀疏封面：居中日期紧随更大的居中标题块。"""
    if page_rect.width <= page_rect.height or len(blocks) > 10:
        return False
    center = page_rect.width / 2
    date_height = block[3] - block[1]
    if not (block[0] >= page_rect.width * 0.25 and block[2] <= page_rect.width * 0.75
            and abs((block[0] + block[2]) / 2 - center) <= page_rect.width * 0.1
            and block[1] <= page_rect.height * 0.75):
        return False
    return any(
        len(other) >= 5 and isinstance(other[4], str)
        and not _PDF_SECTION_RE.search(other[4])
        and 0 < block[1] - other[3] <= 120
        and other[3] - other[1] >= date_height * 1.5
        and abs((other[0] + other[2]) / 2 - center) <= page_rect.width * 0.1
        for other in blocks
    )


def _portrait_cover_date(block: tuple, blocks: list, page_rect: Any) -> bool:
    if page_rect.width > page_rect.height or block[1] > page_rect.height * 0.35:
        return False
    return any(
        len(other) >= 5 and isinstance(other[4], str)
        and re.search(r"证券研究报告|研究报告|Research\s+report|周报|点评|深度报告|首次覆盖报告", other[4], re.IGNORECASE)
        and abs(other[1] - block[1]) <= 220
        for other in blocks if other is not block
    )


def _add_copyright_publishers(text: str, evidence: list, *, page: int, location: list) -> None:
    compact = re.sub(r"\s+", "", text)
    for match in _INSTITUTION_RE.finditer(compact):
        before = compact[:match.start()]
        name = match.group(0)
        # 只匹配明确自身报告归属，不把免责声明中随处出现的机构名当发行人。
        owned = bool(re.search(r"本(?:研究)?报告(?:的)?版权(?:仅为|属于|归|为)[“\"']?$", before))
        copyright = bool(re.search(r"版权所有(?:20\d{2})?[©]?$", before))
        if owned or copyright:
            evidence.append(_evidence(text, _ALIASES.get(name, name), "pdf_last_page_ownership", page, location))


def _resolve_field(evidence: list, reasons: list, *, explicit: Any, has_explicit: bool,
                   field: str, as_of: date) -> dict[str, Any]:
    values = {item["value"] for item in evidence if item["value"]}
    if has_explicit:
        value = validate_publication_date(explicit) if field == "date" else str(explicit or "").strip()
        if len(value.encode("utf-8")) > 64 and field == "broker":
            raise ValueError("broker 不能超过 64 个 UTF-8 字节")
        evidence.append(_evidence(str(explicit or ""), value, "explicit"))
        return {"value": value, "status": "manual", "evidence": evidence, "reasons": ["explicit_override"]}
    future = field == "date" and any(date.fromisoformat(value) > as_of for value in values)
    if future:
        reasons.append("future_date")
    if len(values) > 1:
        reasons.append("conflicting_reliable_sources")
        value, status = "", "conflict"
    elif future or "invalid_calendar_date" in reasons:
        value, status = "", "unknown"
    elif values:
        value, status = next(iter(values)), "known"
    else:
        value, status = "", "unknown"
        reasons.append("no_reliable_evidence")
    return {"value": value, "status": status, "evidence": evidence, "reasons": list(dict.fromkeys(reasons))}


def resolve_public_metadata(source: str, markdown: str = "", *, pdf_path: Path | None = None,
                            explicit: dict | None = None, as_of: date | None = None) -> PublicMetadataResult:
    """从原始 source、明确发布标签、原 PDF 首页发布布局和署名解析公共字段。"""
    explicit = explicit or {}
    as_of = as_of or date.today()
    evidence = {"date": [], "broker": []}
    reasons = {"date": [], "broker": []}
    source_name = Path(source).name
    if re.fullmatch(r"H3_AP\d+_\d+\.pdf", source_name, re.IGNORECASE):
        reasons["date"].append("distribution_identifier_excluded")
    if not re.fullmatch(r"H3_AP\d+_\d+\.pdf", source_name, re.IGNORECASE) and not (
        source_name.startswith(".") or ".uploading." in source_name
        or re.match(r"^[a-f0-9]{32}(?:\.|$)", source_name, re.IGNORECASE)
        or re.match(r"^[a-f0-9]{8}-[a-f0-9]{4}-", source_name, re.IGNORECASE)
    ):
        if not _EXCLUDED_RE.search(source_name):
            _add_dates(source_name, "original_source", evidence["date"], reasons["date"], compact=True)
        _add_publishers(source_name, "original_source", evidence["broker"], filename=True)
    for match in _PUBLISHED_RE.finditer(markdown or ""):
        # 引用/相关研究中的发布标签仍不是本报告发布日期。
        if _label_is_in_excluded_context(markdown or "", match.start()):
            continue
        _add_dates(match.group(1), "markdown_publication_label", evidence["date"], reasons["date"],
                   location={"offset": match.start()})
    _add_publishers(markdown or "", "markdown_signature", evidence["broker"])
    pdf_sha256 = ""
    if pdf_path is not None and Path(pdf_path).suffix.lower() == ".pdf":
        try:
            with Path(pdf_path).open("rb") as handle:
                pdf_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            import pymupdf

            with pymupdf.open(str(pdf_path)) as document:
                if not len(document):
                    raise ValueError("empty PDF")
                page = document[0]
                blocks = page.get_text("blocks")
                for block in blocks:
                    if len(block) < 5 or not isinstance(block[4], str):
                        continue
                    raw = block[4]
                    location = list(block[:4])
                    in_excluded_section = _pdf_in_excluded_section(block, blocks)
                    if not _pdf_in_excluded_section(block, blocks, max_gap=180) and _pdf_has_signature_context(
                        block, blocks, page.rect, last_page=False
                    ):
                        _add_publishers(raw, "pdf_first_page_signature", evidence["broker"], page=1, location=location)
                    compact = re.sub(r"\s+", "", raw)
                    for match in _PUBLISHED_RE.finditer(raw):
                        if not in_excluded_section and not _label_is_in_excluded_context(raw, match.start()):
                            _add_dates(match.group(1), "pdf_publication_label", evidence["date"], reasons["date"], page=1, location=location)
                    # 首页短独立日期行，或证券报告头部日期。拒绝表格、股价与相关报告区域。
                    date_matches = list(_DATE_RE.finditer(compact))
                    if len(date_matches) != 1 or _EXCLUDED_RE.search(raw) or len(compact) > 80:
                        continue
                    match = date_matches[0]
                    prefix, suffix = compact[:match.start()], compact[match.end():]
                    standalone = not prefix and not suffix
                    report_header = not standalone and prefix in {"", "证券研究报告|", "证券研究报告｜"} and (
                        suffix in {"", "宏观经济"} or "证券研究报告" in suffix)
                    # 无标签日期只认页首布局或稀疏横版封面上的居中标题-日期组合。
                    cover_position = (
                        report_header and block[1] <= page.rect.height * 0.6
                        or standalone and (_portrait_cover_date(block, blocks, page.rect)
                                           or _landscape_cover_date(block, blocks, page.rect))
                    )
                    if not in_excluded_section and cover_position:
                        _add_dates(match.group(0), "pdf_first_page_cover", evidence["date"], reasons["date"], page=1, location=location)
                if len(document) > 1:
                    last_page = document[-1]
                    last_blocks = last_page.get_text("blocks")
                    for block in last_blocks:
                        if len(block) >= 5 and isinstance(block[4], str):
                            if not _pdf_in_excluded_section(block, last_blocks, max_gap=180) and _pdf_has_signature_context(
                                block, last_blocks, last_page.rect, last_page=True
                            ):
                                _add_publishers(block[4], "pdf_last_page_signature", evidence["broker"],
                                                page=len(document), location=list(block[:4]))
                            _add_copyright_publishers(block[4], evidence["broker"], page=len(document), location=list(block[:4]))
        except Exception as exc:  # 文件损坏/无文本仅留下诊断，绝不回退到 mtime 或路径名。
            for field in reasons:
                reasons[field].append("pdf_read_error")
            reasons["date"].append(f"pdf_error:{type(exc).__name__}")
    fields = {field: _resolve_field(evidence[field], reasons[field], explicit=explicit.get(field),
                                   has_explicit=field in explicit, field=field, as_of=as_of)
              for field in ("date", "broker")}
    audit = {"rules_version": RULES_VERSION, "source": source, "pdf_sha256": pdf_sha256, "fields": fields}
    return PublicMetadataResult(fields["date"]["value"], fields["broker"]["value"], public_audit_needs_review(audit), audit)


__all__ = ["PublicMetadataResult", "resolve_public_metadata", "validate_publication_date"]
