"""离线、可重现的 agentic-v2 样本清单；只读取原始数据。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import warnings
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "agentic-v2"
PROTOCOL_RULES = {
    "version": PROTOCOL_VERSION,
    "dedup": "NFKC-casefold-no-whitespace-query+fact-annotations",
    "pairing": "exact-duplicates+explicit-paired_case_id-connected-components",
    "bootstrap": "paired-case-deltas-average-within-group-then-resample-groups",
    "gold": "declared-gold+approved+named-reviewer; never-inferred-from-filename",
    "near_duplicate_threshold": 0.9,
    "guardrails": "negative-mean-blocks-pass",
    "pdf_identity": "actual-KB-document-sources-union-scanned-sources; complete-valid-64hex-SHA256-required",
    "exact_chunk_metrics": "raw-selected-child-id-exact-match-at-5; excludes-expanded-parent-members",
    "context_evidence_metrics": "actual-budgeted-context-slots-at-5; same-source-complete-gold-quote-visible; child-membership-label-alone-is-not-evidence",
    "context_precision": "distinct-first-supporting-window-slots-divided-by-fixed-5; never-flatten-child-members",
    "required_fingerprints": ["dataset_fingerprint", "kb_fingerprint", "protocol_fingerprint", "judge_fingerprint"],
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


PROTOCOL_FINGERPRINT = _digest(PROTOCOL_RULES)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def fact_signature(row: dict[str, Any]) -> str:
    """忽略标注 ID，保留事实陈述、别名、来源与可接受证据约束。"""
    def canonical(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in sorted(value.items()) if key != "id"}
        if isinstance(value, (list, tuple)):
            return sorted((canonical(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return value

    return _digest(canonical({"required_facts": row.get("required_facts", []),
                              "facts": row.get("facts", []),
                              "expected_behavior": row.get("expected_behavior", "") if row.get("question_type") == "negative" else ""}))


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按精确重复和显式配对归组；模糊相似只标记人工审核。"""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        parents[find(right)] = find(left)

    exact: dict[tuple[str, str], int] = {}
    by_id: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        row["normalized_query"] = normalize_text(str(row.get("query", "")))
        row["fact_fingerprint"] = fact_signature(row)
        row["exact_fingerprint"] = _digest([row["normalized_query"], row["fact_fingerprint"]])
        row.setdefault("needs_review", False)
        row.setdefault("review_reasons", [])
        key = (row.get("pool", "evaluation"), row["exact_fingerprint"])
        if key in exact:
            join(exact[key], index)
        else:
            exact[key] = index
        by_id.setdefault((row.get("pool", "evaluation"), row["case_id"]), []).append(index)
    for index, row in enumerate(rows):
        paired = row.get("paired_case_id") or (row.get("provenance") or {}).get("paired_case_id")
        if not paired:
            continue
        targets = by_id.get((row.get("pool", "evaluation"), str(paired)), [])
        if not targets or len({rows[target]["exact_fingerprint"] for target in targets}) > 1:
            row["needs_review"] = True
            row["review_reasons"].append("unresolved_or_ambiguous_paired_case_id")
        else:
            for target in targets:
                join(index, target)
    components: dict[int, list[int]] = {}
    for index in range(len(rows)):
        components.setdefault(find(index), []).append(index)
    for members in components.values():
        group = "sg-" + _digest(sorted({rows[index]["exact_fingerprint"] for index in members}))[:24]
        for index in members:
            rows[index]["sample_group_id"] = group
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a, b = rows[left], rows[right]
            if a.get("pool") != b.get("pool") or a["sample_group_id"] == b["sample_group_id"]:
                continue
            query_a, query_b = a["normalized_query"], b["normalized_query"]
            if query_a and query_b and SequenceMatcher(None, query_a, query_b, autojunk=False).ratio() >= 0.9:
                for row in (a, b):
                    row["needs_review"] = True
                    if "unknown_near_duplicate" not in row["review_reasons"]:
                        row["review_reasons"].append("unknown_near_duplicate")
    for indexes in by_id.values():
        if len({rows[index]["exact_fingerprint"] for index in indexes}) > 1:
            for index in indexes:
                rows[index]["needs_review"] = True
                rows[index]["review_reasons"].append("case_id_annotation_conflict")
    return rows


def sample_group_ids(cases: list[Any]) -> dict[str, str]:
    rows = [{**case.to_dict(), "case_id": case.id, "pool": "evaluation"} for case in cases]
    return {row["case_id"]: row["sample_group_id"] for row in group_rows(rows)}


def _legacy_rows(path: Path) -> list[dict[str, Any]]:
    from .test_set import TestSetLoader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        entries = TestSetLoader(path).load_test_set()
    rows = []
    for index, entry in enumerate(entries, 1):
        negative = entry.question_type == "negative"
        facts = list(entry.required_facts) or [part.strip() for part in re.split(r"[。！？；\n]", entry.ground_truth) if part.strip()][:5]
        rows.append({"id": entry.unique_id or f"legacy-{index:04d}", "query": entry.query,
                     "question_type": entry.question_type, "required_facts": [] if negative else facts,
                     "expected_behavior": entry.expected_behavior, "reference_answer": entry.ground_truth,
                     "evidence_chunk_ids": list(entry.chunk_ids), "tier": "silver", "review_status": "pending",
                     "reviewer": "", "review_note": "只读旧 Markdown 转换；未查询知识库，证据与审核待确认",
                     "provenance": {"kind": "legacy_markdown", "human_approved": False}})
    return rows


def build_manifest(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []
    sources = []
    paths = sorted(set(data_dir.glob("*.jsonl")) | set(data_dir.glob("*generated*.md")))
    for path in paths:
        raw = path.read_bytes()
        loaded = _legacy_rows(path) if path.suffix == ".md" else [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        sources.append({"source_file": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "case_count": len(loaded)})
        for index, item in enumerate(loaded, 1):
            pool = "anchor" if "anchor_id" in item else ("regression" if "regression" in path.stem or (item.get("provenance") or {}).get("kind") == "regression" else "evaluation")
            case_id = str(item.get("id") or item.get("anchor_id") or "")
            if not case_id:
                raise ValueError(f"{path.name}:{index}: missing case id")
            rows.append({**item, "case_id": case_id, "query": item.get("query", item.get("user_input", "")),
                         "source_file": path.name, "source_row": index, "pool": pool,
                         "paired_case_id": str((item.get("provenance") or {}).get("paired_case_id") or item.get("paired_case_id") or ""),
                         "tier": item.get("tier", "silver"), "review_status": item.get("review_status", "pending"),
                         "reviewer": item.get("reviewer", ""), "review_note": item.get("review_note", "")})
    group_rows(rows)
    evaluation = [row for row in rows if row["pool"] == "evaluation"]
    approved = [row for row in evaluation if row["review_status"] == "approved" and row["reviewer"] and not row["needs_review"] and (row.get("provenance") or {}).get("human_approved") is not False]
    reviewed = [row for row in approved if row["tier"] == "gold"]
    for row in rows:
        row["eligible_gold"] = row in reviewed
    return {"protocol_version": PROTOCOL_VERSION, "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "protocol_rules": PROTOCOL_RULES, "dataset_fingerprint": _digest(sources), "sources": sources,
            "raw_case_count": len(evaluation), "exact_unique_case_count": len({row["exact_fingerprint"] for row in evaluation}),
            "independent_sample_count": len({row["sample_group_id"] for row in evaluation}),
            "reviewed_independent_sample_count": len({row["sample_group_id"] for row in approved}),
            "reviewed_gold_independent_sample_count": len({row["sample_group_id"] for row in reviewed}),
            "needs_review_count": sum(row["needs_review"] for row in evaluation),
            "anchor_count": sum(row["pool"] == "anchor" for row in rows),
            "regression_case_count": sum(row["pool"] == "regression" for row in rows),
            "declared_tier_counts": dict(sorted(Counter(row["tier"] for row in evaluation).items())),
            "claims": {"has_100_reviewed_independent_samples": len({row["sample_group_id"] for row in approved}) >= 100,
                       "historical_n5_context_precision_14_12_percent": "historical_only_not_agentic_v2_evidence"},
            "cases": rows}
