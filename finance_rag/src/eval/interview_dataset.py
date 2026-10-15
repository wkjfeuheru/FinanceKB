"""从 gold20 派生 30 题面试分层集。

20 道已审核题保持原问法，并标上 ``explicit`` / ``cross_doc`` / ``negative``。
另外 10 道配对题保留实体、机构和报告范围，独立自足。历史 vague ID/分层
保留用于已有入口兼容，不代表问题仍模糊。共享证据，不能把 30 题当独立样本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import EvaluationDataset

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLD20_PATH = DATA_DIR / "research_report_gold20.jsonl"
INTERVIEW30_PATH = DATA_DIR / "research_report_interview30.jsonl"

_STRATUM_BY_TYPE = {
    "single_hop": "explicit",
    "multi_hop": "cross_doc",
    "negative": "negative",
}

# 历史 ID 和配对关系保留；问句由修订后的核心题派生，不再隐去实体。
VAGUE_PAIRS: tuple[dict[str, Any], ...] = (
    {
        "id": "rr-vague-001",
        "paired_case_id": "rr-single-001",
    },
    {
        "id": "rr-vague-002",
        "paired_case_id": "rr-single-002",
    },
    {
        "id": "rr-vague-003",
        "paired_case_id": "rr-single-003",
    },
    {
        "id": "rr-vague-004",
        "paired_case_id": "rr-single-004",
    },
    {
        "id": "rr-vague-005",
        "paired_case_id": "rr-single-005",
    },
    {
        "id": "rr-vague-006",
        "paired_case_id": "rr-single-006",
    },
    {
        "id": "rr-vague-007",
        "paired_case_id": "rr-single-007",
    },
    {
        "id": "rr-vague-008",
        "paired_case_id": "rr-single-008",
    },
    {
        "id": "rr-vague-009",
        "paired_case_id": "rr-multi-004",
    },
    {
        "id": "rr-vague-010",
        "paired_case_id": "rr-multi-009",
    },
)


def load_gold20(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or GOLD20_PATH
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_interview_cases(gold_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """返回 30 条可直接交给 ``EvaluationDataset.from_dicts`` 的字典。"""
    gold = gold_rows if gold_rows is not None else load_gold20()
    by_id = {row["id"]: row for row in gold}
    cases: list[dict[str, Any]] = []
    for row in gold:
        case = json.loads(json.dumps(row))
        provenance = dict(case.get("provenance") or {})
        provenance["stratum"] = _STRATUM_BY_TYPE[case["question_type"]]
        provenance["interview_set"] = "research-report-interview30"
        case["provenance"] = provenance
        cases.append(case)

    for spec in VAGUE_PAIRS:
        source = by_id.get(spec["paired_case_id"])
        if source is None:
            raise KeyError(f"模糊题配对不存在：{spec['paired_case_id']}")
        case = json.loads(json.dumps(source))
        case["id"] = spec["id"]
        case["query"] = '请依据指定报告原文回答：' + source["query"]
        case["review_note"] = "独立自足的配对问法；共享原题证据，不是独立统计样本"
        provenance = dict(case.get("provenance") or {})
        provenance.update({
            "stratum": "vague",
            "paired_case_id": spec["paired_case_id"],
            "builder": "finance_rag.src.eval.interview_dataset",
            "query_style": "standalone_paired",
            "independent_sample": False,
            "interview_set": "research-report-interview30",
        })
        case["provenance"] = provenance
        cases.append(case)
    return cases


def write_interview_dataset(
    path: Path | None = None,
    gold_path: Path | None = None,
) -> Path:
    """校验并写出 30 题 JSONL。结构校验不连知识库。"""
    output = path or INTERVIEW30_PATH
    dataset = EvaluationDataset.from_dicts(build_interview_cases(load_gold20(gold_path)))
    dataset.validate()
    return dataset.save(output)
