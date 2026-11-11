"""从其他工作目录导入包时，保持惰性加载和随代码分发的资源可用。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(
        [sys.executable, '-B', '-c', code], cwd=cwd, env=env,
        capture_output=True, text=True, encoding='utf-8', timeout=30,
    )


def test_public_package_import_keeps_heavy_dependencies_lazy(tmp_path):
    result = _run(
        'import sys; import src; '
        'assert "KnowledgeBase" in src.__all__; '
        'assert not {"torch", "transformers", "sentence_transformers", "pymilvus"}.intersection(sys.modules)',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_shipped_resources_resolve_outside_project_working_directory(tmp_path):
    result = _run(
        'from src.core.config import EVAL_ANCHORS_PATH; '
        'from src.eval.anchors import load_anchors; '
        'assert load_anchors(EVAL_ANCHORS_PATH)',
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
