"""人工标注校准：抽样导出、回填导入、一致性判定与 CLI 端到端。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from finance_rag.src.eval.cli import main
from finance_rag.src.eval.human_calibration import (
    calibration_report,
    export_human_sample,
    load_human_scores,
    load_run_arms,
    render_calibration_markdown,
    stratified_pick,
)

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _row(case_id: str, question_type: str, value: float, response: str = "回答") -> dict:
    return {
        "case_id": case_id,
        "tier": "gold",
        "question_type": question_type,
        "stratum": "explicit",
        "user_input": f"问题 {case_id}？",
        "response": response,
        "metrics": {
            "faithfulness": value,
            "answer_relevancy": value,
            "context_precision": value,
            "context_recall": value,
            "recall_at_5": 0.5,
        },
        "sources": [{"content": "上下文片段"}],
    }


# ---------------------------------------------------------------------------
# 分层抽样
# ---------------------------------------------------------------------------


def test_stratified_pick_covers_every_type_and_is_reproducible():
    rows = (
        [_row(f"s{i}", "single_hop", 0.5) for i in range(10)]
        + [_row(f"m{i}", "multi_hop", 0.5) for i in range(2)]
        + [_row(f"n{i}", "negative", 0.5) for i in range(2)]
    )
    picked = stratified_pick(rows, 3, seed=7)
    assert len(picked) == 3
    assert {row["question_type"] for row in picked} == {"single_hop", "multi_hop", "negative"}
    assert [row["case_id"] for row in stratified_pick(rows, 3, seed=7)] == [
        row["case_id"] for row in picked
    ]
    # 请求量超过总量时按实际返回，不报错
    assert len(stratified_pick(rows, 99, seed=7)) == len(rows)


# ---------------------------------------------------------------------------
# 导出 / 导入
# ---------------------------------------------------------------------------


def test_export_human_sample_writes_blank_human_columns(tmp_path):
    arms = {
        "arm_a": [_row("c1", "single_hop", 0.8), _row("c2", "multi_hop", 0.4)],
        "arm_b": [_row("c1", "single_hop", 0.6)],
    }
    out = export_human_sample(arms, out_csv=tmp_path / "sample.csv", n=10, seed=7)
    with out.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["arm"] for row in rows} == {"arm_a", "arm_b"}
    assert all(row["human_faithfulness"] == "" for row in rows)
    assert rows[0]["auto_faithfulness"] != ""
    assert rows[0]["contexts_preview"] == "上下文片段"
    assert rows[0]["response_chars"] == "2"


def test_load_human_scores_validates_range_and_skips_unreviewed(tmp_path):
    source = tmp_path / "filled.csv"
    fields = [
        "case_id", *(f"human_{metric}" for metric in METRICS), "human_note",
    ]
    with source.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "case_id": "c1", "human_faithfulness": "0.9",
            "human_answer_relevancy": "0.8", "human_context_precision": "",
            "human_context_recall": "", "human_note": "基本忠实",
        })
        writer.writerow({"case_id": "c2", "human_note": "今天没空评"})

    scores = load_human_scores(source)
    assert scores["c1"]["metrics"] == {"faithfulness": 0.9, "answer_relevancy": 0.8}
    assert scores["c1"]["note"] == "基本忠实"
    assert "c2" not in scores  # 未评行不进统计

    bad = tmp_path / "bad.csv"
    bad.write_text('case_id,human_faithfulness\nc1,1.5\n', encoding="utf-8")
    with pytest.raises(ValueError, match="0~1"):
        load_human_scores(bad)

    not_number = tmp_path / "not_number.csv"
    not_number.write_text('case_id,human_faithfulness\nc1,高\n', encoding="utf-8")
    with pytest.raises(ValueError, match="不是数字"):
        load_human_scores(not_number)


# ---------------------------------------------------------------------------
# 一致性判定
# ---------------------------------------------------------------------------


def test_calibration_report_aligned_when_judge_matches_human():
    auto = [_row(f"c{i}", "single_hop", 0.2 + i * 0.1) for i in range(6)]
    human = {
        row["case_id"]: {"metrics": {"faithfulness": 0.2 + i * 0.1}, "note": ""}
        for i, row in enumerate(auto)
    }
    report = calibration_report(auto, human, metrics=("faithfulness",))

    metric = report["metrics"]["faithfulness"]
    assert metric["n"] == 6
    assert metric["mae"] == 0.0
    assert metric["bias"] == 0.0
    assert metric["spearman"] == 1.0
    assert metric["agreement_within_0.2"] == 1.0
    assert metric["verdict"] == "aligned"
    assert report["n_matched"] == 6
    assert report["unreviewed_or_unmatched"] == 0


def test_calibration_report_flags_divergence_and_offsets():
    auto = [_row(f"c{i}", "single_hop", 0.5) for i in range(4)]
    # judge 系统性偏高 0.3 → 偏置大，判定 divergent
    human = {
        row["case_id"]: {"metrics": {"faithfulness": 0.2}, "note": ""} for row in auto
    }
    report = calibration_report(auto, human, metrics=("faithfulness",))
    metric = report["metrics"]["faithfulness"]
    assert metric["bias"] == -0.3
    assert metric["recommended_offset"] == -0.3
    assert metric["verdict"] == "divergent"
    assert "human" not in metric  # 不泄漏原始标注

    # 中等偏置 → offset（可校准）
    auto2 = [_row(f"d{i}", "single_hop", 0.5 + i * 0.1) for i in range(5)]
    human2 = {
        row["case_id"]: {"metrics": {"faithfulness": 0.35 + i * 0.1}, "note": ""}
        for i, row in enumerate(auto2)
    }
    report2 = calibration_report(auto2, human2, metrics=("faithfulness",))
    assert report2["metrics"]["faithfulness"]["verdict"] == "offset"


def test_calibration_report_no_data_and_unmatched_rows():
    auto = [_row("c1", "single_hop", 0.5)]
    human = {"missing": {"metrics": {"faithfulness": 0.5}, "note": ""}}
    report = calibration_report(auto, human, metrics=("faithfulness",))

    assert report["metrics"]["faithfulness"]["verdict"] == "no_data"
    assert report["n_matched"] == 0
    assert report["unreviewed_or_unmatched"] == 1
    assert "no_data" in "\n".join(render_calibration_markdown(report))


# ---------------------------------------------------------------------------
# 运行产物读取 + CLI 端到端
# ---------------------------------------------------------------------------


def _write_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "per_query": [
            _row("c1", "single_hop", 0.8),
            _row("c2", "multi_hop", 0.4),
            _row("c3", "negative", 0.2),
        ]
    }
    (run_dir / "arm_a.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def test_load_run_arms_reads_ab_artifacts_and_runner_records(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    (run_dir / "report.json").write_text('{"comparison": {}}', encoding="utf-8")
    (run_dir / "records.jsonl").write_text(
        json.dumps({"case_id": "r1", "tier": "gold", "metrics": {}}) + "\n",
        encoding="utf-8",
    )
    arms = load_run_arms(run_dir)
    assert set(arms) == {"arm_a", "records"}  # report.json 被跳过
    assert len(arms["arm_a"]) == 3


def test_calibration_cli_export_then_import(tmp_path):
    run_dir = _write_run_dir(tmp_path)

    csv_path = tmp_path / "human_sample.csv"

    assert main([
        "calibration", "export", "--run-dir", str(run_dir),
        "--n", "3", "--out", str(csv_path),
    ]) == 0
    assert csv_path.exists()

    # 模拟人工回填：judge 偏高 0.2
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        auto = float(row["auto_faithfulness"])
        row["human_faithfulness"] = f"{max(0.0, auto - 0.2):.4f}"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    assert main([
        "calibration", "import", "--run-dir", str(run_dir), "--csv", str(csv_path),
    ]) == 0
    report = json.loads((run_dir / "human_calibration.json").read_text(encoding="utf-8"))
    assert report["n_human"] == len(rows)
    assert report["metrics"]["faithfulness"]["bias"] == pytest.approx(-0.2, abs=1e-6)


def test_calibration_matches_legacy_artifacts_without_case_id(tmp_path):
    """早期产物没有 case_id：抽样与回填必须靠「臂 + 问题」稳定键对齐。

    真实来源：``scripts/results/ab/rerank_on_off/20260825-201550/*.json``
    的 per_query 只有 user_input，没有 case_id。
    """
    legacy = [
        {
            "user_input": f"问题 {i}？",
            "question_type": "single_hop",
            "response": "回答",
            "metrics": {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
                "context_recall": 0.5,
            },
            "sources": [{"content": "上下文片段"}],
        }
        for i in range(3)
    ]
    run_dir = tmp_path / "legacy_run"
    run_dir.mkdir()
    (run_dir / "arm_a.json").write_text(
        json.dumps({"per_query": legacy}, ensure_ascii=False), encoding="utf-8"
    )

    arms = load_run_arms(run_dir)
    csv_path = export_human_sample(
        arms, out_csv=run_dir / "human_sample.csv", n=3, seed=7
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_id"] == "arm_a:问题 0？"
    for row in rows:
        row["human_faithfulness"] = "0.6"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    human = load_human_scores(csv_path)
    auto_rows = [row for records in arms.values() for row in records]
    report = calibration_report(auto_rows, human, metrics=("faithfulness",))

    assert report["n_matched"] == 3  # 缺 case_id 也能全部配对
    assert report["metrics"]["faithfulness"]["bias"] == pytest.approx(-0.2)
