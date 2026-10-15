"""只读枚举评测资料，生成新协议清单；不调用模型或数据库。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.manifest import build_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "finance_rag/src/eval/data")
    parser.add_argument("--output", type=Path, default=ROOT / "scripts/results/agentic-v2/manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("protocol_version", "raw_case_count", "exact_unique_case_count", "independent_sample_count", "reviewed_gold_independent_sample_count", "needs_review_count", "anchor_count", "regression_case_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
