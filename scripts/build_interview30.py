"""写出 30 题面试分层集。结构校验不连接知识库。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_rag.src.eval.interview_dataset import INTERVIEW30_PATH, write_interview_dataset


def main() -> int:
    path = write_interview_dataset()
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
