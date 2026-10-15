import pytest

from scripts.rebuild_research_kb import verify_prepared


def test_prepared_parse_cannot_reuse_a_changed_pdf(tmp_path):
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'changed')
    manifest = {'reports': {'a.pdf': {'sha256': 'wrong', 'status': 'completed'}}}
    with pytest.raises(ValueError, match='PDF'):
        verify_prepared(manifest, [pdf])


def test_prepared_parse_rejects_missing_or_failed_sources(tmp_path):
    pdf = tmp_path / 'a.pdf'
    pdf.write_bytes(b'pdf')
    with pytest.raises(ValueError, match='sources'):
        verify_prepared({'reports': {}}, [pdf])


def test_parse_semantics_are_validated_before_any_rebuild():
    from scripts.rebuild_research_kb import validate_parse_artifact
    with pytest.raises(ValueError, match='empty'):
        validate_parse_artifact({'source': 'a.pdf', 'pdf_sha256': 'sha', 'markdown': ''}, 'a.pdf', 'sha')


def test_baseline_gate_rejects_empty_and_failed_runs(tmp_path):
    from scripts.rebuild_research_kb import validate_baselines
    with pytest.raises(ValueError, match='baseline'):
        validate_baselines(tmp_path, 'dataset', 'content')


def test_ingest_batch_failure_cannot_exit_success():
    from scripts.ingest_reports import batch_exit_code
    assert batch_exit_code([{'status': 'failed'}], {'a.pdf': {'chunks': 0}}) == 1
    assert batch_exit_code([{'status': 'completed'}], {'a.pdf': {'chunks': 1}}) == 0


def _successful_baselines(directory):
    import json
    from finance_rag.src.eval.dataset import EvalCase, EvaluationDataset
    from finance_rag.src.eval.runner import EvaluationRunner
    cases = [EvalCase.from_dict({'id': f'case-{i}', 'query': f'问题{i}',
             'question_type': 'negative' if i >= 18 else 'single_hop'}) for i in range(20)]
    dataset = EvaluationDataset(cases)
    for layer in ('L0', 'L1', 'L2', 'L3', 'L5', 'PRODUCTION'):
        selected = cases if layer == 'PRODUCTION' else cases[:18]
        target = directory / f'run-{layer}'
        target.mkdir()
        payload = {'dataset_fingerprint': EvaluationRunner._dataset_fingerprint(selected),
                   'chunk_content_fingerprint': 'content', 'metrics_version': '2.0',
                   'annotation_schema_versions': ['2.0'], 'embedding_model': 'BAAI/bge-m3',
                   'embedding_pooling': 'cls', 'embedding_dim': 1024,
                   'chunk_max_tokens': 256, 'chunk_overlap_tokens': 25,
                   'dense_weight': .8, 'sparse_weight': .2,
                   'failed': 0, 'completed': len(selected), 'n': len(selected)}
        for name in ('summary.json', 'run_config.json'):
            (target / name).write_text(json.dumps(payload), encoding='utf8')
        (target / 'records.jsonl').write_text('\n'.join(json.dumps({'case_id': case.id,
             'status': 'completed'}) for case in selected), encoding='utf8')
        dataset.save(target / 'dataset_snapshot.jsonl')
    return dataset


def test_baseline_gate_uses_positive_subset_hash_and_full_production_hash(tmp_path):
    from scripts.rebuild_research_kb import validate_baselines
    dataset = _successful_baselines(tmp_path)
    validate_baselines(tmp_path, dataset, 'content')


def test_baseline_gate_rejects_wrong_case_id_despite_correct_count_and_hash(tmp_path):
    from scripts.rebuild_research_kb import validate_baselines
    dataset = _successful_baselines(tmp_path)
    records = tmp_path / 'run-L0/records.jsonl'
    records.write_text(records.read_text(encoding='utf8').replace('case-0"', 'foreign-case"'), encoding='utf8')
    with pytest.raises(ValueError, match='baseline'):
        validate_baselines(tmp_path, dataset, 'content')
