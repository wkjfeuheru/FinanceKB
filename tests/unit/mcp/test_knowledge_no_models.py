from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_MODULES = (
    "onnxruntime",
    "sentence_transformers",
    "src.index.milvus_kb",
    "src.retrieve",
)


def test_knowledge_mcp_does_not_load_rag_or_model_modules() -> None:
    code = (
        "import sys\n"
        "import src.mcp.knowledge.server\n"
        f"forbidden = [name for name in {FORBIDDEN_MODULES!r} if name in sys.modules]\n"
        "print(','.join(forbidden))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
