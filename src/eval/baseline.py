"""冻结基线注册表：一次冻结、长期比对。

设计要点
--------
* 基线是一条**冻结元组**（数据集指纹 + KB 指纹 + 配置快照 + 代码指纹 +
  裁判指纹 + 协议指纹），不是「每次实验重跑一遍的 A 臂」；
* 对比时直接加载 ``records.jsonl`` 参与配对 Bootstrap（``compare_runs`` 本来就
  接受 records 列表），候选臂只需跑一次；
* 指纹漂移时基线**不再可比**：``verify`` 返回 ``ok=False``，上层判定必须落到
  ``no_decision``，不允许拿漂移前后的数字下结论。

三个内置 profile 覆盖「底线 / 生产默认 / 全量优化」三层，与
``src/eval/retrieval.py`` 的 L0/L5 分层评测互补：分层评测看趋势，基线比对做决策。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics import METRICS_VERSION

PROFILES: dict[str, dict[str, Any]] = {
    "b0": {
        "label": "纯稠密检索（能力底线）",
        "strategy": "L0",
        "policy": {},
        "note": "无重排、无混合；用于证明优化链路确实带来增益。",
    },
    "b1": {
        "label": "生产默认（加权混合 + BGE 重排）",
        "strategy": "PRODUCTION",
        "policy": {},
        "note": "主基线：所有单因子实验的对照组。",
    },
    "b2": {
        "label": "Agentic 全量（改写 + 路由 + 反思）",
        "strategy": "PRODUCTION",
        "policy": {"mode": "agentic"},
        "note": "发布级对比的上界，成本与延迟必须并排看。",
    },
}

#: 冻结进 manifest 的配置键：只收影响检索/生成的开关，避免整份 .env 泄露密钥。
CONFIG_KEYS = (
    "ENABLE_RERANKER",
    "ENABLE_HYDE",
    "ENABLE_LANGGRAPH",
    "ENABLE_CLIFF_DETECTION",
    "ENABLE_REFUSAL",
    "CHAT_ENABLE_QUERY_REWRITE",
    "RETRIEVAL_FUSION_MODE",
    "RETRIEVAL_DENSE_WEIGHT",
    "RETRIEVAL_SPARSE_WEIGHT",
    "CHAT_TOP_K",
    "CHAT_RERANK_TOP_K",
    "AGENT_MAX_SUBPROBLEMS",
    "LANGGRAPH_MAX_REFLECT_ROUNDS",
    "EMBEDDING_MODEL",
    "EMBEDDING_VERSION",
)

FINGERPRINT_FIELDS = (
    "kb_fingerprint",
    "pdf_set_fingerprint",
    "chunk_content_fingerprint",
)


def baseline_root() -> Path:
    """基线根目录；``EVAL_BASELINE_DIR`` 可覆盖（CI 上指向缓存盘）。"""
    override = os.getenv("EVAL_BASELINE_DIR", "").strip()
    if override:
        return Path(override)
    from src.core.config import PROJECT_ROOT

    return PROJECT_ROOT / "src" / "eval" / "data" / "baselines"


def resolve_profile(profile: str) -> dict[str, Any]:
    spec = PROFILES.get(str(profile))
    if spec is None:
        raise ValueError(f"未知基线 profile：{profile}；可选 {', '.join(sorted(PROFILES))}")
    return dict(spec)


def build_baseline_strategy(kb: Any, profile: str):
    """把 profile 翻译成评测策略；复用生产执行器，不复制检索逻辑。"""
    from .retrieval import (
        _parent_resolver,
        _run_async,
        adapt_production_output,
        build_retrieval_strategy,
    )
    from .runner import EvaluationStrategy

    spec = resolve_profile(profile)
    if spec["strategy"] != "PRODUCTION":
        return build_retrieval_strategy(kb, spec["strategy"])

    from src.retrieve import pipeline as service

    policy = service.resolve_policy(**spec["policy"])

    def retrieve(query: str):
        request = service.RetrievalRequest(query)
        result = _run_async(lambda: service.execute_retrieval(request, policy, kb=kb))
        return adapt_production_output(result)

    return EvaluationStrategy(
        name=f"BASELINE:{profile}",
        retrieve=retrieve,
        config={
            "profile": profile,
            "strategy": spec["strategy"],
            "policy": dict(spec["policy"]),
            "embedding_model": getattr(kb, "embedding_model", None),
        },
        resolve_parent=_parent_resolver(kb),
    )


def config_snapshot() -> dict[str, Any]:
    from src.core import config

    return {key: getattr(config, key, None) for key in CONFIG_KEYS}


def judge_snapshot() -> dict[str, Any]:
    from src.core import config

    try:
        model = config.primary_judge_spec()
    except Exception:  # pragma: no cover - 配置缺失时不应阻断冻结
        model = "unavailable"
    from .manifest import PROTOCOL_FINGERPRINT

    payload = f"{model}|{METRICS_VERSION}|{PROTOCOL_FINGERPRINT}"
    return {"model": model, "fingerprint": hashlib.sha256(payload.encode()).hexdigest()}


def environment_snapshot(kb: Any | None = None, *, deep: bool = True) -> dict[str, Any]:
    """当前环境的可比性快照；任何一项失败都降级为 ``unavailable``，不抛异常。

    ``deep=False`` 跳过全集合 chunk 扫描（冻结时记录、比对时不重复扫），
    漂移检测仍由 ``kb_fingerprint``（文档 sha + chunk_count）兜底。
    """
    from src.core.config import PROJECT_ROOT

    from .manifest import PROTOCOL_FINGERPRINT, PROTOCOL_VERSION
    from .retrieval import knowledge_content_fingerprints
    from .runner import code_worktree_fingerprint

    snapshot: dict[str, Any] = {
        "protocol": {"version": PROTOCOL_VERSION, "fingerprint": PROTOCOL_FINGERPRINT},
        "judge": judge_snapshot(),
        "config": config_snapshot(),
        "code": {
            "worktree_sha256": code_worktree_fingerprint(PROJECT_ROOT),
        },
    }
    if kb is None:
        snapshot.update({key: "unavailable" for key in FINGERPRINT_FIELDS})
        return snapshot
    try:
        from .testset_generator import knowledge_base_fingerprint

        snapshot["kb_fingerprint"] = knowledge_base_fingerprint(kb)
    except Exception as exc:  # noqa: BLE001
        snapshot["kb_fingerprint"] = "unavailable"
        snapshot["kb_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    if deep:
        try:
            snapshot.update(knowledge_content_fingerprints(kb))
        except Exception as exc:  # noqa: BLE001
            snapshot["content_fingerprint_error"] = f"{type(exc).__name__}: {exc}"
    else:
        snapshot.update(
            {key: "unavailable" for key in FINGERPRINT_FIELDS if key != "kb_fingerprint"}
        )
    snapshot["collection"] = getattr(kb, "collection_name", "unknown")
    return snapshot


def _dataset_fingerprint(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def dataset_snapshot(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers: dict[str, int] = {}
    for row in rows:
        tier = str(row.get("tier") or "")
        tiers[tier] = tiers.get(tier, 0) + 1
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": _dataset_fingerprint(path, rows),
        "case_count": len(rows),
        "tier_counts": dict(sorted(tiers.items())),
    }


def measure_noise_floor(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = ("mrr", "recall_at_5", "precision_at_5"),
) -> dict[str, float]:
    """同配置两次运行的逐指标平均绝对偏移，作为「提升是否超出噪声」的下限。"""
    by_a = {row["case_id"]: row for row in records_a if row.get("case_id")}
    by_b = {row["case_id"]: row for row in records_b if row.get("case_id")}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return {}
    floor: dict[str, float] = {}
    for name in metrics:
        deltas = []
        for case_id in shared:
            left = (by_a[case_id].get("metrics") or {}).get(name)
            right = (by_b[case_id].get("metrics") or {}).get(name)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                deltas.append(abs(float(right) - float(left)))
        if deltas:
            floor[name] = round(sum(deltas) / len(deltas), 4)
    return floor


def freeze_baseline(
    profile: str,
    *,
    dataset_path: str | Path,
    version: str | None = None,
    sample: int = 0,
    seed: int = 7,
    include_silver: bool = False,
    outdir: str | Path | None = None,
    noise_floor: dict[str, float] | None = None,
) -> Path:
    """跑一次基线并冻结；产物目录即 manifest 所在目录。"""
    from .dataset import DatasetValidationError, EvaluationDataset
    from .runner import EvaluationRunner

    spec = resolve_profile(profile)
    dataset_path = Path(dataset_path)
    dataset = EvaluationDataset.load(dataset_path)
    cases = list(dataset.cases)
    if not include_silver:
        cases = [case for case in cases if case.tier == "gold"]
    if sample and 0 < sample < len(cases):
        cases = random.Random(seed).sample(cases, sample)
    if not cases:
        raise DatasetValidationError(
            f"基线没有可用条目：{dataset_path}（tier={'gold+silver' if include_silver else 'gold'}）"
        )
    selected = EvaluationDataset(cases, source_path=dataset_path)

    from src.index.milvus_kb import get_knowledge_base

    kb = get_knowledge_base()
    strategy = build_baseline_strategy(kb, profile)
    version = str(version or datetime.now().strftime("%Y%m%d-%H%M%S"))
    target = Path(outdir) if outdir else baseline_root() / profile / version
    if (target / "manifest.json").exists():
        raise FileExistsError(f"基线已冻结，换一个版本号：{target / 'manifest.json'}")

    rows = [case.to_dict() for case in cases]
    dataset_info = dataset_snapshot(dataset_path, rows)
    snapshot = environment_snapshot(kb)
    summary = EvaluationRunner().run(
        selected,
        strategy,
        "retrieval",
        target,
        metadata={
            "dataset_version": version,
            "seed": seed,
            "profile": profile,
            **{key: snapshot.get(key) for key in FINGERPRINT_FIELDS},
        },
    ).summary

    manifest = {
        "profile": profile,
        "label": spec["label"],
        "note": spec["note"],
        "version": version,
        "strategy": spec["strategy"],
        "policy": dict(spec["policy"]),
        "frozen_at": datetime.now().astimezone().isoformat(),
        "dataset": dataset_info,
        "kb": {
            **{key: snapshot.get(key, "unavailable") for key in FINGERPRINT_FIELDS},
            "collection": snapshot.get("collection", "unknown"),
        },
        "code": snapshot.get("code") or {},
        "config": snapshot.get("config") or {},
        "judge": snapshot.get("judge") or {},
        "protocol": snapshot.get("protocol") or {},
        "metrics_version": METRICS_VERSION,
        "noise_floor": dict(noise_floor or {}),
        "summary": {
            "n": summary.get("n"),
            "metrics": summary.get("metrics") or {},
            "dataset_fingerprint": summary.get("dataset_fingerprint"),
            "code_worktree_sha256": summary.get("code_worktree_sha256"),
            "git_head": summary.get("git_head"),
            "git_dirty": summary.get("git_dirty"),
        },
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _manifest_dir(profile: str, version: str | None) -> Path:
    candidate = Path(profile)
    if candidate.is_dir() and (candidate / "manifest.json").exists():
        return candidate
    root = baseline_root() / str(profile)
    if version:
        return root / str(version)
    versions = sorted(
        path for path in root.iterdir() if (path / "manifest.json").exists()
    ) if root.is_dir() else []
    if not versions:
        raise FileNotFoundError(
            f"没有已冻结的基线：{root}；先跑 evaluate.py baseline freeze --profile {profile}"
        )
    return versions[-1]


def load_baseline(profile: str, version: str | None = None) -> dict[str, Any]:
    target = _manifest_dir(profile, version)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    records_path = target / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if records_path.exists() else []
    summary_path = target / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"manifest": manifest, "records": records, "summary": summary, "path": target}


def verify_baseline(
    profile: str,
    version: str | None = None,
    *,
    kb: Any | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """对比当前环境与冻结元组；漂移字段逐条列出。"""
    data = load_baseline(profile, version)
    manifest = data["manifest"]
    if kb is None:
        try:
            from src.index.milvus_kb import get_knowledge_base

            kb = get_knowledge_base()
        except Exception:  # noqa: BLE001 - 离线校验时无 KB，按 unavailable 处理
            kb = None
    snapshot = environment_snapshot(kb, deep=deep)
    expected_collection = (manifest.get("kb") or {}).get("collection")
    actual_collection = snapshot.get("collection")
    if expected_collection and actual_collection and expected_collection != actual_collection:
        # 集合迁移后旧基线不再是同一份数据：显式标记 legacy，避免误读成普通漂移
        return {
            "profile": manifest.get("profile", profile),
            "version": manifest.get("version", version),
            "ok": False,
            "legacy": True,
            "reason": f"基线集合 {expected_collection} 与当前 {actual_collection} 不同，需新建基线",
            "drift": [{"field": "kb.collection", "expected": expected_collection,
                       "actual": actual_collection}],
            "path": str(data["path"]),
            "snapshot": snapshot,
        }
    drift: list[dict[str, Any]] = []

    def compare(field: str, expected: Any, actual: Any) -> None:
        if expected in (None, "", "unavailable") or actual in (None, "", "unavailable"):
            return
        if expected != actual:
            drift.append({"field": field, "expected": expected, "actual": actual})

    dataset = manifest.get("dataset") or {}
    dataset_path = Path(str(dataset.get("path") or ""))
    if dataset_path.is_file():
        compare(
            "dataset.sha256",
            dataset.get("sha256"),
            hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        )
    for field in FINGERPRINT_FIELDS:
        compare(f"kb.{field}", (manifest.get("kb") or {}).get(field), snapshot.get(field))
    compare("code.worktree_sha256", (manifest.get("code") or {}).get("worktree_sha256"), (snapshot.get("code") or {}).get("worktree_sha256"))
    compare("judge.fingerprint", (manifest.get("judge") or {}).get("fingerprint"), (snapshot.get("judge") or {}).get("fingerprint"))
    compare("protocol.fingerprint", (manifest.get("protocol") or {}).get("fingerprint"), (snapshot.get("protocol") or {}).get("fingerprint"))

    return {
        "profile": manifest.get("profile", profile),
        "version": manifest.get("version", version),
        "ok": not drift,
        "legacy": False,
        "drift": drift,
        "path": str(data["path"]),
        "snapshot": snapshot,
    }


def list_baselines() -> list[dict[str, Any]]:
    root = baseline_root()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for profile_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for version_dir in sorted(path for path in profile_dir.iterdir() if path.is_dir()):
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "profile": manifest.get("profile", profile_dir.name),
                    "version": manifest.get("version", version_dir.name),
                    "label": manifest.get("label", ""),
                    "frozen_at": manifest.get("frozen_at", ""),
                    "case_count": (manifest.get("dataset") or {}).get("case_count"),
                    "kb_fingerprint": (manifest.get("kb") or {}).get("kb_fingerprint"),
                    "path": str(version_dir),
                }
            )
    return rows
