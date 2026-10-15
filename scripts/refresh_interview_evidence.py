"""把评测集证据块 id 重新定位到当前集合。引文必须唯一命中。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_gold_set import _document_version, _load_chunks
from finance_rag.src.eval.dataset import EvaluationDataset
from finance_rag.src.eval.testset_generator import build_evidence_resolver
from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base


def refresh(path: Path, grouped: dict, resolver) -> None:
    dataset = EvaluationDataset.load(path)
    rows = []
    changed = 0
    for case in dataset.cases:
        row = case.to_dict()
        for evidence in row.get("evidence") or []:
            hits = [
                chunk for chunk in grouped.get(evidence["source"], [])
                if evidence["quote"] and evidence["quote"] in str(chunk.get("content") or "")
            ]
            if len(hits) != 1:
                raise SystemExit(f"{case.id} quote hits={len(hits)}")
            chunk = hits[0]
            new_id = str(chunk.get("id") or "")
            if new_id != evidence["chunk_id"]:
                changed += 1
            evidence["chunk_id"] = new_id
            evidence["document_version"] = _document_version(chunk)
        rows.append(row)
    updated = EvaluationDataset.from_dicts(rows)
    updated.validate(evidence_resolver=resolver)
    updated.save(path)
    print(f"{path.name}: cases={len(rows)} ids_updated={changed}")


def main() -> int:
    kb = get_knowledge_base()
    grouped = _load_chunks(kb)
    resolver = build_evidence_resolver(kb)
    data = ROOT / "finance_rag" / "src" / "eval" / "data"
    refresh(data / "research_report_interview30.jsonl", grouped, resolver)
    refresh(data / "research_report_gold20.jsonl", grouped, resolver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
