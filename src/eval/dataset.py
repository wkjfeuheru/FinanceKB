"""版本化 JSONL 评测集、质量门禁与冻结清单。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_QUOTAS = {
    "gold": {"single_hop": 30, "multi_hop": 15, "negative": 15},
    "silver": {"single_hop": 50, "multi_hop": 25, "negative": 15},
}
QUESTION_TYPES = {"single_hop", "multi_hop", "negative"}
TIERS = {"gold", "silver"}
REVIEW_STATUSES = {"pending", "approved", "rejected", "auto_pass"}
DIFFICULTIES = {"easy", "medium", "hard"}
ANSWER_TYPES = {"factual", "list", "comparison", "procedural"}
STRATA = {"explicit", "cross_doc", "vague", "negative"}


def case_stratum(case: EvalCase) -> str:
    """分层标签。显式写入 ``provenance.stratum`` 时优先，否则按问题类型推断。"""
    raw = str((case.provenance or {}).get("stratum") or "").strip()
    if raw:
        return raw
    if case.question_type == "negative":
        return "negative"
    if case.question_type == "multi_hop":
        return "cross_doc"
    return "explicit"


class DatasetValidationError(ValueError):
    """评测集结构或质量门禁失败。"""


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    document_version: str
    chunk_id: str
    quote: str
    evidence_id: str = ""
    document_sha256: str = ""
    quote_start: int | None = None
    quote_end: int | None = None
    location: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceRef:
        return cls(
            source=str(value.get("source", "")).strip(),
            document_version=str(value.get("document_version", "")).strip(),
            chunk_id=str(value.get("chunk_id", "")).strip(),
            quote=str(value.get("quote", "")).strip(),
            evidence_id=str(value.get("evidence_id", "")).strip(),
            document_sha256=str(value.get("document_sha256", "")).strip(),
            quote_start=value.get("quote_start"),
            quote_end=value.get("quote_end"),
            location=dict(value.get("location") or {}),
        )


@dataclass(frozen=True)
class FactRequirement:
    id: str
    statement: str
    allowed_sources: tuple[str, ...]
    aliases: tuple[tuple[str, ...], ...]
    acceptable_evidence_sets: tuple[tuple[str, ...], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FactRequirement:
        return cls(
            str(value.get("id", "")).strip(), str(value.get("statement", "")).strip(),
            tuple(str(item).strip() for item in value.get("allowed_sources", [])),
            tuple(tuple(str(item).strip() for item in group) for group in value.get("aliases", [])),
            tuple(tuple(str(item).strip() for item in group) for group in value.get("acceptable_evidence_sets", [])),
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    question_type: str
    difficulty: str
    answer_type: str
    reference_answer: str
    required_facts: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    expected_behavior: str
    tier: str
    review_status: str
    reviewer: str
    review_note: str
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
    facts: tuple[FactRequirement, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvalCase:
        return cls(
            id=str(value.get("id", "")).strip(),
            query=str(value.get("query", "")).strip(),
            question_type=str(value.get("question_type", "")).strip(),
            difficulty=str(value.get("difficulty", "medium")).strip(),
            answer_type=str(value.get("answer_type", "factual")).strip(),
            reference_answer=str(value.get("reference_answer", "")).strip(),
            required_facts=tuple(str(item).strip() for item in value.get("required_facts", []) if str(item).strip()),
            evidence=tuple(EvidenceRef.from_dict(item) for item in value.get("evidence", [])),
            expected_behavior=str(value.get("expected_behavior", "")).strip(),
            tier=str(value.get("tier", "silver")).strip(),
            review_status=str(value.get("review_status", "pending")).strip(),
            reviewer=str(value.get("reviewer", "")).strip(),
            review_note=str(value.get("review_note", "")).strip(),
            provenance=dict(value.get("provenance") or {}),
            schema_version=str(value.get("schema_version", "1.0")),
            facts=tuple(FactRequirement.from_dict(item) for item in value.get("facts", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_facts"] = list(self.required_facts)
        value["evidence"] = [asdict(item) for item in self.evidence]
        value["facts"] = [{"id": item.id, "statement": item.statement,
                           "allowed_sources": list(item.allowed_sources),
                           "aliases": [list(group) for group in item.aliases],
                           "acceptable_evidence_sets": [list(group) for group in item.acceptable_evidence_sets]}
                          for item in self.facts]
        return value


@dataclass(frozen=True)
class DatasetAudit:
    valid: bool
    case_count: int
    checked_evidence: int
    errors: tuple[str, ...] = ()


EvidenceResolver = Callable[[str], dict[str, Any] | None]


def _safe_quote_boundaries(content: str, start: int, end: int) -> bool:
    """禁止从数值/英文 token 内取引文或切开 HTML 标签。"""
    import re
    for boundary in (start, end):
        if 0 < boundary < len(content):
            left, right = content[boundary - 1], content[boundary]
            if left.isascii() and right.isascii() and (left.isalnum() or left in "_.%+-") and (right.isalnum() or right in "_.%+-"):
                return False
        if content.rfind("<", 0, boundary) > content.rfind(">", 0, boundary):
            return False
    quote = content[start:end]
    tags = re.findall(r"<\s*(/?)\s*([A-Za-z][\w:-]*)[^>]*>", quote)
    stack: list[str] = []
    void = {"br", "hr", "img", "input", "meta", "link", "wbr"}
    for close, name in tags:
        name = name.lower()
        if name in void:
            continue
        if close:
            if not stack or stack.pop() != name:
                return False
        else:
            stack.append(name)
    return not stack


class EvaluationDataset:
    """统一评测集接口：加载、校验、审核和冻结。"""

    def __init__(self, cases: Iterable[EvalCase], *, source_path: Path | None = None):
        self.cases = tuple(cases)
        self.source_path = source_path

    @classmethod
    def from_dicts(cls, rows: Iterable[dict[str, Any]]) -> EvaluationDataset:
        return cls(EvalCase.from_dict(row) for row in rows)

    @classmethod
    def load(
        cls,
        path: str | Path,
        tier: str | None = None,
        review_status: str | None = None,
    ) -> EvaluationDataset:
        path = Path(path)
        cases: list[EvalCase] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    cases.append(EvalCase.from_dict(json.loads(line)))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise DatasetValidationError(f"第 {line_number} 行不是有效 JSON：{exc}") from exc
        if tier is not None:
            cases = [case for case in cases if case.tier == tier]
        if review_status is not None:
            cases = [case for case in cases if case.review_status == review_status]
        dataset = cls(cases, source_path=path)
        dataset.validate()
        return dataset

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for case in self.cases:
                handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def validate(
        self,
        *,
        evidence_resolver: EvidenceResolver | None = None,
        resolve_evidence: Callable[[EvidenceRef], dict[str, Any] | None] | None = None,
    ) -> DatasetAudit:
        errors: list[str] = []
        seen_ids: set[str] = set()
        seen_queries: set[str] = set()
        checked_evidence = 0
        for case in self.cases:
            prefix = case.id or "<missing-id>"
            if not case.id:
                errors.append("存在缺少 id 的条目")
            elif case.id in seen_ids:
                errors.append(f"{prefix}: id 重复")
            seen_ids.add(case.id)
            if not case.query:
                errors.append(f"{prefix}: 缺少问题")
            elif case.query in seen_queries:
                errors.append(f"{prefix}: 问题重复")
            seen_queries.add(case.query)
            if case.question_type not in QUESTION_TYPES:
                errors.append(f"{prefix}: question_type 非法")
            stratum = str((case.provenance or {}).get("stratum") or "").strip()
            if stratum and stratum not in STRATA:
                errors.append(f"{prefix}: stratum 非法：{stratum}")
            if case.tier not in TIERS:
                errors.append(f"{prefix}: tier 非法")
            if case.review_status not in REVIEW_STATUSES:
                errors.append(f"{prefix}: review_status 非法")
            if case.difficulty not in DIFFICULTIES:
                errors.append(f"{prefix}: difficulty 非法")
            if case.answer_type not in ANSWER_TYPES:
                errors.append(f"{prefix}: answer_type 非法")
            if case.schema_version not in {"1.0", "2.0"}:
                errors.append(f"{prefix}: schema_version 非法")
            if case.question_type == "negative":
                if not case.expected_behavior:
                    errors.append(f"{prefix}: 负样本缺少 expected_behavior")
                if case.evidence:
                    errors.append(f"{prefix}: 负样本不应包含 evidence")
                continue
            if not case.reference_answer:
                errors.append(f"{prefix}: 正例缺少 reference_answer")
            if not case.required_facts and not case.facts:
                errors.append(f"{prefix}: 正例缺少 required_facts")
            evidence_by_id = {item.evidence_id: item for item in case.evidence if item.evidence_id}
            if case.schema_version == "2.0":
                if len(evidence_by_id) != len(case.evidence):
                    errors.append(f"{prefix}: evidence_id 缺失或重复")
                if not case.facts:
                    errors.append(f"{prefix}: schema2 正例缺少 facts")
                if len({fact.id for fact in case.facts}) != len(case.facts):
                    errors.append(f"{prefix}: fact id 重复")
                for fact in case.facts:
                    if not fact.id or not fact.statement or not fact.allowed_sources or not fact.aliases or not fact.acceptable_evidence_sets:
                        errors.append(f"{prefix}: fact 字段不完整：{fact.id}")
                    if any(not group or any(not phrase for phrase in group) for group in fact.aliases):
                        errors.append(f"{prefix}: aliases 空组合：{fact.id}")
                    for group in fact.acceptable_evidence_sets:
                        if not group or len(set(group)) != len(group):
                            errors.append(f"{prefix}: evidence set 空或重复：{fact.id}")
                        for ref_id in group:
                            ref = evidence_by_id.get(ref_id)
                            if ref is None or ref.source not in fact.allowed_sources:
                                errors.append(f"{prefix}: fact evidence 不存在或 source 不允许：{ref_id}")
            required_evidence = 2 if case.question_type == "multi_hop" else 1
            if len(case.evidence) < required_evidence:
                errors.append(f"{prefix}: 证据数量不足")
            if case.question_type == "multi_hop" and len({item.source for item in case.evidence}) < 2:
                errors.append(f"{prefix}: 多跳证据必须来自不同文档")
            for evidence in case.evidence:
                if (
                    not evidence.source or not evidence.document_version
                    or (not evidence.chunk_id and not (case.schema_version == "2.0" and (evidence.location.get("mapping_status") == "unmapped" or (evidence.location.get("kind") in {"parent", "table"} and evidence.location.get("anchor_id"))))) or not evidence.quote
                ):
                    errors.append(f"{prefix}: evidence 缺少 source/document_version/chunk_id/quote")
                    continue
                if case.schema_version == "2.0" and not evidence.document_sha256:
                    errors.append(f"{prefix}: document_sha256 缺失：{evidence.evidence_id}")
                if evidence_resolver is None and resolve_evidence is None:
                    continue
                checked_evidence += 1
                if resolve_evidence is not None:
                    actual = resolve_evidence(evidence)
                elif evidence.location.get("kind", "chunk") != "chunk":
                    errors.append(f"{prefix}: 非chunk证据需要 resolve_evidence：{evidence.evidence_id}")
                    continue
                else:
                    actual = evidence_resolver(evidence.chunk_id)
                if not actual:
                    errors.append(f"{prefix}: chunk 不存在：{evidence.chunk_id}")
                    continue
                if str(actual.get("source", "")) != evidence.source:
                    errors.append(f"{prefix}: chunk/source 不匹配：{evidence.chunk_id}")
                if evidence.quote not in str(actual.get("content", "")):
                    errors.append(f"{prefix}: quote 无法在 chunk 中定位：{evidence.chunk_id}")
                if case.schema_version == "2.0":
                    from .metrics import compute_text_coverage
                    content = str(actual.get("content", ""))
                    start, end = evidence.quote_start, evidence.quote_end
                    if actual.get("document_sha256") != evidence.document_sha256:
                        errors.append(f"{prefix}: SHA 不匹配：{evidence.evidence_id}")
                    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(content)) or content[start:end] != evidence.quote:
                        errors.append(f"{prefix}: quote offset 不匹配：{evidence.evidence_id}")
                    elif not _safe_quote_boundaries(content, start, end):
                        errors.append(f"{prefix}: quote 截断数字/ASCII/HTML：{evidence.evidence_id}")
                    if compute_text_coverage([content], [evidence.quote]) != 1.0:
                        errors.append(f"{prefix}: selfcompute_text_coverage != 1：{evidence.evidence_id}")
        if errors:
            raise DatasetValidationError("；".join(errors))
        return DatasetAudit(True, len(self.cases), checked_evidence)

    def export_review(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "id", "tier", "review_status", "reviewer", "review_note",
            "question_type", "query", "reference_answer", "evidence_sources", "evidence_quotes",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in self.cases:
                writer.writerow({
                    "id": case.id,
                    "tier": case.tier,
                    "review_status": case.review_status,
                    "reviewer": case.reviewer,
                    "review_note": case.review_note,
                    "question_type": case.question_type,
                    "query": case.query,
                    "reference_answer": case.reference_answer,
                    "evidence_sources": " | ".join(item.source for item in case.evidence),
                    "evidence_quotes": " | ".join(item.quote for item in case.evidence),
                })
        return path

    def import_review(self, path: str | Path) -> EvaluationDataset:
        allowed = {"pending", "approved", "rejected", "auto_pass"}
        updates: dict[str, dict[str, str]] = {}
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                case_id = (row.get("id") or "").strip()
                status = (row.get("review_status") or "").strip()
                if case_id and status:
                    if status not in allowed:
                        raise DatasetValidationError(f"{case_id}: 审核状态非法：{status}")
                    updates[case_id] = {
                        "review_status": status,
                        "reviewer": (row.get("reviewer") or "").strip(),
                        "review_note": (row.get("review_note") or "").strip(),
                    }
        known = {case.id for case in self.cases}
        unknown = sorted(set(updates) - known)
        if unknown:
            raise DatasetValidationError(f"审核文件包含未知 id：{unknown}")
        return EvaluationDataset(
            (
                replace(case, **updates[case.id])
                if case.id in updates else case
                for case in self.cases
            ),
            source_path=self.source_path,
        )

    def freeze(
        self,
        output_path: str | Path,
        *,
        quotas: dict[str, dict[str, int]] | None = None,
        kb_fingerprint: str,
        dataset_version: str | None = None,
        max_source_share: float = 0.20,
    ) -> dict[str, Any]:
        self.validate()
        quotas = quotas or DEFAULT_QUOTAS
        selected: list[EvalCase] = []
        selected_source_counts: Counter[str] = Counter()
        for tier, type_quotas in quotas.items():
            for question_type, expected in type_quotas.items():
                eligible_status = "approved" if tier == "gold" else "auto_pass"
                candidates = sorted(
                    (
                        case for case in self.cases
                        if case.tier == tier
                        and case.question_type == question_type
                        and case.review_status == eligible_status
                    ),
                    key=lambda case: case.id,
                )
                if len(candidates) < expected:
                    if tier == "gold":
                        raise DatasetValidationError(
                            f"Gold {question_type} 需要 {expected} 条 approved，实际 {len(candidates)} 条"
                        )
                    raise DatasetValidationError(
                        f"Silver {question_type} 需要 {expected} 条 auto_pass，实际 {len(candidates)} 条"
                    )
                # 在满足各层配额的候选中优先选择当前引用次数较低的来源，
                # 让冻结集在来源上尽量均衡；无解时仍由下方严格 20% 门禁失败。
                remaining = list(candidates)
                for _ in range(expected):
                    def balance_key(case: EvalCase) -> tuple[float, float, str]:
                        sources = [item.source for item in case.evidence] or [""]
                        projected = selected_source_counts.copy()
                        projected.update(sources)
                        total = sum(projected.values()) or 1
                        return (
                            max(projected[source] / total for source in sources),
                            sum(projected[source] for source in sources) / len(sources),
                            case.id,
                        )

                    chosen = min(remaining, key=balance_key)
                    selected.append(chosen)
                    selected_source_counts.update(item.source for item in chosen.evidence)
                    remaining.remove(chosen)

        selected_ids = {case.id for case in selected}
        if len(selected_ids) != len(selected):
            raise DatasetValidationError("冻结选择包含重复 id")

        expected_total = sum(sum(values.values()) for values in quotas.values())
        if len(selected) != expected_total:
            raise DatasetValidationError(f"冻结数量错误：期望 {expected_total}，实际 {len(selected)}")

        fingerprints = {
            str(case.provenance.get("kb_fingerprint", ""))
            for case in selected
            if case.provenance.get("kb_fingerprint")
        }
        if fingerprints and fingerprints != {kb_fingerprint}:
            raise DatasetValidationError(
                f"知识库指纹漂移：数据集={sorted(fingerprints)} 当前={kb_fingerprint}"
            )

        source_counts = Counter(
            evidence.source
            for case in selected
            if case.question_type != "negative"
            for evidence in case.evidence
        )
        total_evidence = sum(source_counts.values())
        if total_evidence:
            over_limit = {
                source: count / total_evidence
                for source, count in source_counts.items()
                if count / total_evidence > max_source_share
            }
            if over_limit:
                raise DatasetValidationError(f"单文档证据占比超过 {max_source_share:.0%}：{over_limit}")

        frozen = EvaluationDataset(sorted(selected, key=lambda case: case.id))
        output_path = frozen.save(output_path)
        version = dataset_version or datetime.now(UTC).strftime("%Y%m%d")
        counts = {
            tier: dict(Counter(case.question_type for case in selected if case.tier == tier))
            for tier in sorted(quotas)
        }
        content_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset_version": version,
            "created_at": datetime.now(UTC).astimezone().isoformat(),
            "kb_fingerprint": kb_fingerprint,
            "dataset_sha256": content_hash,
            "case_count": len(selected),
            "counts": counts,
            "source_distribution": dict(sorted(source_counts.items())),
            "max_source_share": max_source_share,
        }
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
