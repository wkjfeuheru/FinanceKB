"""固定旧条件回放：实际运行器、上下文与失败产物。"""

from copy import deepcopy
import hashlib
import importlib
import json

import pytest

from src.eval.dataset import EvaluationDataset
from src.eval.runner import EvaluationRunner


def _module():
    # 缺少功能时让 RED 是明确断言，而不是收集阶段导入异常。
    import importlib.util
    assert importlib.util.find_spec("scripts.replay_metadata_baseline"), "缺少固定条件只读回放 CLI"
    return importlib.import_module("scripts.replay_metadata_baseline")


def _dataset():
    return EvaluationDataset.from_dicts([
        {"id": "positive", "query": "东吴证券报告中收入是多少？", "question_type": "single_hop",
         "reference_answer": "收入100亿元", "required_facts": ["收入100亿元"],
         "evidence": [{"chunk_id": "gold", "source": "a.pdf", "document_version": "v1", "quote": "收入100亿元"}],
         "tier": "silver", "review_status": "auto_pass"},
        {"id": "negative", "query": "不存在的事实？", "question_type": "negative",
         "expected_behavior": "拒答", "tier": "silver", "review_status": "auto_pass"},
    ])


def _baseline(tmp_path, dataset=None):
    dataset = dataset or _dataset()
    folder = tmp_path / "baseline"
    folder.mkdir()
    dataset.save(folder / "dataset_snapshot.jsonl")
    config = {"strategy": "PRODUCTION", "mode": "retrieval",
              "dataset_fingerprint": EvaluationRunner._dataset_fingerprint(list(dataset.cases)),
              "kb_fingerprint": "old-kb", "pdf_set_fingerprint": "pdf-sha",
              "chunk_content_fingerprint": "chunk-sha"}
    (folder / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [{"case_id": case.id, "query": case.query, "status": "completed",
             "diagnostics": {"filters": {"broker": ["东吴证券"], "date": {"gte": "2026-09-01"}}}}
            for case in dataset.cases]
    _write_records(folder, rows)
    return folder, rows


def _write_records(folder, rows):
    (folder / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )


class MemoryKB:
    embedding_model = "fixture/model"

    def list_documents(self):
        return [{"source": "a.pdf", "version": "v1", "chunk_count": 1, "date": "2026-09-01"}]

    def get_fingerprint_store(self):
        return {"a.pdf": {"sha256": "pdf-fixture"}}


def _external_pipeline(monkeypatch, *, multiple=False, error=None, mutate=False, long_text=False):
    from src import engine as chat_service

    calls = []
    actual_retrieve = chat_service.retrieve_documents

    async def retrieve(query, **kwargs):
        calls.append((query, deepcopy(kwargs)))
        pipeline = await actual_retrieve(query, **kwargs)
        if mutate:
            kwargs["filters"]["broker"].append("中信证券")
        return pipeline

    class SearchKB:
        def hybrid_search(self, query, **kwargs):
            if error:
                raise RuntimeError(error)
            text = "无关" * 3000 + "收入100亿元" if long_text else "索引摘要"
            docs = [{"id": "gold", "source": "a.pdf", "content": text, "score": 1.0,
                     "parent_id": "table-id", "block_type": "text" if long_text else "table"}]
            self.last_retrieval_diagnostics = {
                "coarse_candidate_chunk_ids": ["gold"] + [f"x{i}" for i in range(49)],
                "coarse_candidates": deepcopy(docs), "fusion_mode": "weighted",
            }
            return docs

    class TableStore:
        def get_batch(self, ids):
            return [{"id": "table-id", "markdown": "| 指标 | 值 |\n| 收入 | 收入100亿元 |", "start_page": 2}]

    monkeypatch.setattr(chat_service, "retrieve_documents", retrieve)
    monkeypatch.setattr(chat_service, "_table_store_for", lambda collection: TableStore())
    kbs = [("finance_kb", SearchKB())]
    if multiple:
        kbs.append(("second_kb", SearchKB()))
    monkeypatch.setattr(chat_service, "iter_active_kbs", lambda: kbs)
    # 外部模型功能关闭；过滤计划、异步生产检索、合并及上下文方法全部真实执行。
    monkeypatch.setattr(chat_service, "CHAT_ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(chat_service, "ENABLE_HYDE", False)
    return calls


def test_replay_keeps_original_conditions_and_scores_real_expanded_context(tmp_path, monkeypatch):
    replay = _module()
    baseline, rows = _baseline(tmp_path)
    untouched = (baseline / "records.jsonl").read_bytes()
    calls = _external_pipeline(monkeypatch)
    plan = replay.load_baseline(baseline, _dataset())
    strategy = replay.build_replay_strategy(plan, MemoryKB())
    run = EvaluationRunner().run(_dataset(), strategy, "retrieval", tmp_path / "run")
    assert len(run.records) == 1  # 负例不触发生产生成或回放。
    row = run.records[0]
    assert row["status"] == "completed"
    assert row["metrics"]["recall_at_5"] == 1.0
    assert row["metrics"]["required_fact_coverage_at_5"] == 0.0
    assert row["metrics"]["context_required_fact_coverage"] == 1.0
    assert row["metrics"]["exact_candidate_recall_at_50"] == 1.0
    assert "收入100亿元" in row["diagnostics"]["context_text"]
    assert calls == [(_dataset().cases[0].query, {"filters": rows[0]["diagnostics"]["filters"],
                    "infer_filters": False, "k": 50, "rerank_top_n": 5})]
    assert strategy.name == "METADATA_REPLAY"
    assert strategy.evaluate_negative is None
    assert plan["baseline_records_sha256"] == hashlib.sha256(untouched).hexdigest()
    assert len(row["diagnostics"]["metadata_replay"]["original_filter_sha256"]) == 64
    assert (baseline / "records.jsonl").read_bytes() == untouched


def test_replay_scores_context_after_actual_text_budget(tmp_path, monkeypatch):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    _external_pipeline(monkeypatch, long_text=True)
    run = EvaluationRunner().run(_dataset(), replay.build_replay_strategy(replay.load_baseline(baseline, _dataset()), MemoryKB()),
                                 "retrieval", tmp_path / "run")
    assert run.records[0]["metrics"]["required_fact_coverage_at_5"] == 1.0
    assert run.records[0]["metrics"]["context_required_fact_coverage"] == 0.0


@pytest.mark.parametrize("change", ["query", "id", "invalid", "missing", "failed", "duplicate"])
def test_unreplayable_cases_fail_explicitly_without_retrieval(tmp_path, monkeypatch, change):
    replay = _module()
    baseline, rows = _baseline(tmp_path)
    if change == "query":
        rows[0]["query"] = "被改写的问题"
    elif change == "id":
        rows[0]["case_id"] = "other-case"
    elif change == "invalid":
        rows[0]["diagnostics"]["filters"]["date"] = {"gte": "2026-02-30"}
    elif change == "missing":
        del rows[0]["diagnostics"]["filters"]
    elif change == "failed":
        rows[0]["status"] = "failed"
    else:
        rows.append(deepcopy(rows[0]))
    _write_records(baseline, rows)
    calls = _external_pipeline(monkeypatch)
    plan = replay.load_baseline(baseline, _dataset())
    run = EvaluationRunner().run(_dataset(), replay.build_replay_strategy(plan, MemoryKB()), "retrieval", tmp_path / "run")
    assert run.records[0]["status"] == "failed"
    assert run.records[0]["error"]
    assert calls == []
    assert plan["cases"]["positive"]["failure_reason"]


def test_changed_facts_and_nonproduction_baseline_are_rejected(tmp_path):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    changed = _dataset().cases[0].to_dict()
    changed["required_facts"] = ["收入200亿元"]
    with pytest.raises(ValueError):
        replay.load_baseline(baseline, EvaluationDataset.from_dicts([changed, _dataset().cases[1].to_dict()]))
    config = json.loads((baseline / "run_config.json").read_text(encoding="utf-8"))
    config["strategy"] = "L5"
    (baseline / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        replay.load_baseline(baseline, _dataset())


def test_multiple_branches_do_not_invent_unified_candidate_scores(tmp_path, monkeypatch):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    _external_pipeline(monkeypatch, multiple=True)
    run = EvaluationRunner().run(_dataset(), replay.build_replay_strategy(replay.load_baseline(baseline, _dataset()), MemoryKB()),
                                 "retrieval", tmp_path / "run")
    row = run.records[0]
    assert row["status"] == "completed"
    assert row["diagnostics"]["unified_coarse_stage"] == "unavailable"
    assert row["diagnostics"]["candidate_chunk_ids"] == []
    assert row["metrics"]["exact_candidate_recall_at_20"] is None
    assert row["metrics"]["exact_candidate_recall_at_50"] is None
    assert row["metrics"]["exact_candidate_recall_at_actual_depth"] is None


def test_pipeline_mutation_does_not_change_original_filters(tmp_path, monkeypatch):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    original = (baseline / "records.jsonl").read_bytes()
    calls = _external_pipeline(monkeypatch, mutate=True)
    plan = replay.load_baseline(baseline, _dataset())
    strategy = replay.build_replay_strategy(plan, MemoryKB())
    strategy.retrieve(_dataset().cases[0].query)
    strategy.retrieve(_dataset().cases[0].query)
    assert calls[0][1] == calls[1][1]
    assert plan["cases"]["positive"]["original_filters"]["broker"] == ["东吴证券"]
    assert (baseline / "records.jsonl").read_bytes() == original


@pytest.mark.parametrize("failure", [False, True])
def test_cli_saves_provenance_controls_and_nonzero_on_failure(tmp_path, monkeypatch, failure):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    dataset_path = tmp_path / "core.jsonl"
    _dataset().save(dataset_path)
    _external_pipeline(monkeypatch, error="Milvus unavailable" if failure else None)
    from src.index import milvus_kb
    monkeypatch.setattr(milvus_kb, "get_knowledge_base", lambda: MemoryKB())
    code = replay.main(["--baseline-run-dir", str(baseline), "--dataset", str(dataset_path),
                        "--outdir", str(tmp_path / "out"), "--run-id", "fixture"])
    assert code == (1 if failure else 0)
    folder = tmp_path / "out" / "core" / "fixture-METADATA_REPLAY"
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    controls = json.loads((folder / "replay_controls.json").read_text(encoding="utf-8"))
    assert config["strategy"] == "METADATA_REPLAY"
    assert config["metadata"]["baseline_records_sha256"] == hashlib.sha256((baseline / "records.jsonl").read_bytes()).hexdigest()
    assert config["metadata"]["baseline_kb_fingerprint"] == "old-kb"
    assert config["metadata"]["comparable"] is False
    for field in ("kb_fingerprint", "dataset_fingerprint", "code_worktree_sha256", "code_diff_sha256",
                  "parser_clean_chunk_rules", "pdf_set_fingerprint", "chunk_content_fingerprint"):
        assert field in config
    assert controls["cases"]["positive"]["original_filters"]["broker"] == ["东吴证券"]
    assert controls["cases"]["positive"]["actual_call"]["infer_filters"] is False
    if failure:
        assert controls["cases"]["positive"]["failure_reason"]
        assert "Milvus unavailable" in controls["cases"]["positive"]["retrieval_calls"][0]["error"]
    else:
        assert controls["cases"]["positive"]["actual_filters"]["broker"] == ["东吴证券"]
    assert (folder / "dataset_snapshot.jsonl").read_bytes() == dataset_path.read_bytes()


def test_cli_retains_dataset_manifest_drift_gate(tmp_path, monkeypatch):
    replay = _module()
    baseline, _ = _baseline(tmp_path)
    path = tmp_path / "core.jsonl"
    _dataset().save(path)
    path.with_suffix(".manifest.json").write_text(json.dumps({"dataset_version": "core", "kb_fingerprint": "old-kb"}), encoding="utf-8")
    calls = _external_pipeline(monkeypatch)
    from src.index import milvus_kb
    monkeypatch.setattr(milvus_kb, "get_knowledge_base", lambda: MemoryKB())
    assert replay.main(["--baseline-run-dir", str(baseline), "--dataset", str(path),
                        "--outdir", str(tmp_path / "out"), "--run-id", "drift"]) == 1
    assert calls == []
