"""真实事实组合、来源边界和在线证据审计的回归契约。"""
import pytest

from finance_rag.src.eval.dataset import EvalCase, EvaluationDataset, DatasetValidationError
from finance_rag.src.eval import metrics as metric_module
from finance_rag.src.eval.runner import EvaluationRunner, EvaluationStrategy, RetrievalOutput


def patch_production_retrieval(monkeypatch, retrieve):
    from finance_rag.src.services import retrieval_service
    async def execute(request, policy, **kwargs):
        return await retrieve(request.query, k=request.k, rerank_top_n=request.rerank_top_n)
    monkeypatch.setattr(retrieval_service, "execute_retrieval", execute)


def patch_production_answer(monkeypatch, answer):
    from finance_rag.src.services import answer_service
    monkeypatch.setattr(answer_service, "answer_pipeline", answer)


def compute_structured_fact_metrics(*args):
    assert hasattr(metric_module, 'compute_structured_fact_metrics'), '缺少结构化事实计分接口'
    return metric_module.compute_structured_fact_metrics(*args)


def row():
    return {
        'id': 'v2', 'query': '两份报告业绩？', 'schema_version': '2.0',
        'question_type': 'multi_hop', 'difficulty': 'medium', 'answer_type': 'comparison',
        'reference_answer': '2026H1 A营收10亿元，B下降3%。',
        'required_facts': ['A营收10亿元', 'B下降3%'], 'tier': 'gold',
        'review_status': 'approved', 'reviewer': 'test', 'review_note': '', 'expected_behavior': '',
        'evidence': [
            {'evidence_id': 'a1', 'source': 'a.pdf', 'document_version': 'v', 'document_sha256': 'sha-a',
             'chunk_id': 'a', 'quote': '2026H1营收10亿元', 'quote_start': 0, 'quote_end': 13,
             'location': {'kind': 'chunk'}},
            {'evidence_id': 'b1', 'source': 'b.pdf', 'document_version': 'v', 'document_sha256': 'sha-b',
             'chunk_id': 'b', 'quote': '2026H1同比下降3%', 'quote_start': 0, 'quote_end': 13,
             'location': {'kind': 'chunk'}},
            {'evidence_id': 'a2', 'source': 'a.pdf', 'document_version': 'v', 'document_sha256': 'sha-a',
             'chunk_id': 'alternative', 'quote': '2026H1收入10亿元', 'quote_start': 0, 'quote_end': 13,
             'location': {'kind': 'chunk'}},
        ],
        'facts': [
            {'id': 'fa', 'statement': 'A营收', 'allowed_sources': ['a.pdf'],
             'aliases': [['2026H1', '营收', '10亿元'], ['2026H1', '收入', '10亿元']],
             'acceptable_evidence_sets': [['a1'], ['a2']]},
            {'id': 'fb', 'statement': 'B下降', 'allowed_sources': ['b.pdf'],
             'aliases': [['2026H1', '下降', '3%']], 'acceptable_evidence_sets': [['b1']]},
        ],
    }


def test_fact_options_are_or_and_groups_complete_at_last_ref_rank():
    value = row()
    value['facts'][0]['acceptable_evidence_sets'] = [['a1', 'a2'], ['a2']]
    case = EvalCase.from_dict(value)
    docs = [{'id': 'irrelevant', 'source': 'a.pdf', 'content': ''},
            {'id': 'alternative', 'source': 'a.pdf', 'content': '2026H1收入10亿元'},
            {'id': 'parent', 'covered_child_ids': ['b'], 'source': 'b.pdf', 'content': '2026H1同比下降3%'}]
    metrics = compute_structured_fact_metrics(case, docs)
    assert metrics['evidence_group_recall_at_5'] == 1.0
    assert metrics['fact_mrr'] == pytest.approx(0.4167)
    assert metrics['structured_required_fact_coverage_at_5'] == 1.0
    assert metrics['all_required_documents_at_5'] == 1.0


@pytest.mark.parametrize('docs', [
    [{'source': 'wrong.pdf', 'content': '2026H1营收10亿元'}],
    [{'source': 'a.pdf', 'content': '2026H1营收110亿元'}],
    [{'source': 'a.pdf', 'content': '2026H1营收10万元'}],
    [{'source': 'a.pdf', 'content': '2025H1营收10亿元'}],
    [{'source': 'a.pdf', 'content': '2026H1营收'}, {'source': 'a.pdf', 'content': '10亿元'}],
    [{'source': 'a.pdf', 'title': '2026H1营收10亿元', 'content': '其他正文'}],
])
def test_structured_alias_cannot_match_wrong_source_value_unit_period_or_joined_docs(docs):
    assert compute_structured_fact_metrics(EvalCase.from_dict(row()), docs)['structured_required_fact_coverage_at_5'] == 0.0


def test_cross_doc_missing_one_is_not_all_documents():
    result = compute_structured_fact_metrics(EvalCase.from_dict(row()), [{'id': 'a', 'source': 'a.pdf', 'content': '2026H1营收10亿元'}])
    assert result['evidence_group_recall_at_5'] == 0.5
    assert result['all_required_documents_at_5'] == 0.0


def test_old_schema_has_no_inferred_fact_equivalents():
    value = row()
    value.pop('schema_version')
    value.pop('facts')
    case = EvalCase.from_dict(value)
    assert case.schema_version == '1.0'
    assert compute_structured_fact_metrics(case, [])['fact_mrr'] is None


def test_old_schema_still_reports_required_document_completeness():
    value = row()
    value.pop('schema_version')
    value.pop('facts')
    metrics = compute_structured_fact_metrics(EvalCase.from_dict(value), [{'source': 'a.pdf'}])
    assert metrics['all_required_documents_at_5'] == 0.0


def test_annotation_online_self_coverage_and_safe_quote_boundaries():
    value = row()
    for ref in value['evidence']:
        ref['quote_end'] = len(ref['quote'])
    actual = {ref['evidence_id']: {'content': ref['quote'], 'source': ref['source'],
                                  'document_sha256': ref['document_sha256']} for ref in value['evidence']}
    assert EvaluationDataset.from_dicts([value]).validate(resolve_evidence=lambda ref: actual[ref.evidence_id]).checked_evidence == 3
    # 引文逐字符确实存在，但从英文数字 token 内截取，不可用。
    actual['a1']['content'] = 'X' + value['evidence'][0]['quote']
    value['evidence'][0].update(quote_start=1, quote_end=len(actual['a1']['content']))
    with pytest.raises(DatasetValidationError, match='截断|selfcompute'):
        EvaluationDataset.from_dicts([value]).validate(resolve_evidence=lambda ref: actual[ref.evidence_id])


def test_online_audit_requires_source_allowed_and_unique_ref_per_and_group():
    value = row()
    value['facts'][0]['acceptable_evidence_sets'] = [['a1', 'a1'], ['b1']]
    with pytest.raises(DatasetValidationError):
        EvaluationDataset.from_dicts([value]).validate()


def test_dirty_diff_fingerprint_and_rules_are_recorded(tmp_path):
    from finance_rag.src.eval.runner import EvaluationRunner
    docs = [{'id': 'a', 'source': 'a.pdf'}]
    strategy = EvaluationStrategy('fake', lambda query: RetrievalOutput(docs, docs))
    result = EvaluationRunner().run(EvaluationDataset.from_dicts([row()]), strategy, 'retrieval', tmp_path)
    assert len(result.summary['code_diff_sha256']) == 64
    assert len(result.summary['parser_clean_chunk_rules']) == 64
    assert result.summary['metrics_version'] == '2.0'
    assert result.summary['annotation_schema_versions'] == ['2.0']


def test_v2_unmapped_identity_round_trip():
    value = row()
    value['evidence'][0]['chunk_id'] = ''
    value['evidence'][0]['location'] = {'kind': 'pdf', 'mapping_status': 'unmapped'}
    dataset = EvaluationDataset.from_dicts([value])
    dataset.validate()
    assert dataset.cases[0].to_dict()['evidence'][0]['evidence_id'] == 'a1'
    assert dataset.cases[0].to_dict()['facts'][0]['acceptable_evidence_sets'] == [['a1'], ['a2']]


@pytest.mark.parametrize('mutation', ['sha', 'offset', 'source'])
def test_online_audit_rejects_sha_offset_and_source_mismatch(mutation):
    value = row()
    for ref in value['evidence']:
        ref['quote_end'] = len(ref['quote'])
    actual = {ref['evidence_id']: {'content': ref['quote'], 'source': ref['source'],
                                 'document_sha256': ref['document_sha256']} for ref in value['evidence']}
    if mutation == 'sha': actual['a1']['document_sha256'] = 'changed'
    if mutation == 'offset': value['evidence'][0]['quote_start'] = 1
    if mutation == 'source': actual['a1']['source'] = 'wrong.pdf'
    with pytest.raises(DatasetValidationError):
        EvaluationDataset.from_dicts([value]).validate(resolve_evidence=lambda ref: actual[ref.evidence_id])


def test_short_coarse_pool_has_null_fixed_depth_and_actual_depth_recall(tmp_path):
    docs = [{'id': 'a', 'source': 'a.pdf', 'content': '2026H1营收10亿元'}]
    strategy = EvaluationStrategy('fake', lambda query: RetrievalOutput(docs, docs))
    record = EvaluationRunner().run(EvaluationDataset.from_dicts([row()]), strategy, 'retrieval', tmp_path).records[0]
    assert record['metrics']['exact_candidate_recall_at_20'] is None
    assert record['metrics']['exact_candidate_recall_at_50'] is None
    assert record['metrics']['exact_candidate_recall_at_actual_depth'] == pytest.approx(0.3333)
    assert record['diagnostics']['coarse_actual_depth'] == 1


def test_production_reuses_real_context_budget_without_mutating_retrieved_docs(monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    original = [{'id': 'a', 'source': 'a.pdf', 'content': 'x' * 6000 + '2026H1营收10亿元'}]
    async def retrieve(query, **kwargs):
        return {'docs': original, 'diagnostics': {}, 'retrieval_error': None}
    patch_production_retrieval(monkeypatch, retrieve)
    strategy = build_retrieval_strategy(object(), 'PRODUCTION')
    output = strategy.retrieve('问题')
    assert len(output.results[0]['content']) > 6000
    assert len(output.context_results[0]['content']) == chat_service._context_budget('text')
    assert '2026H1营收10亿元' not in output.context_results[0]['content']
    assert original[0]['content'].endswith('2026H1营收10亿元')


@pytest.mark.asyncio
async def test_production_sync_adapter_works_inside_existing_event_loop(monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    async def retrieve(query, **kwargs):
        return {'docs': [{'id': 'a', 'source': 'a.pdf', 'content': '实际正文'}], 'retrieval_error': None}
    patch_production_retrieval(monkeypatch, retrieve)
    assert build_retrieval_strategy(object(), 'PRODUCTION').retrieve('问题').results[0]['id'] == 'a'


def test_production_negatives_run_existing_chat_and_do_not_dilute_positive_metrics(tmp_path, monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    async def retrieve(query, **kwargs):
        return {'docs': [{'id': 'a', 'source': 'a.pdf', 'content': '2026H1营收10亿元'},
                         {'id': 'b', 'source': 'b.pdf', 'content': '2026H1同比下降3%'}], 'retrieval_error': None}
    async def answer(query, pipeline, **kwargs):
        return {'answer': '无法回答', 'answer_status': 'refused', 'answer_rejected': True, 'sources': []}
    patch_production_retrieval(monkeypatch, retrieve)
    patch_production_answer(monkeypatch, answer)
    negative = {**row(), 'id': 'negative', 'query': '没有依据？', 'question_type': 'negative',
                'evidence': [], 'facts': [], 'required_facts': [], 'reference_answer': '', 'expected_behavior': '应拒答'}
    result = EvaluationRunner().run(EvaluationDataset.from_dicts([row(), negative]), build_retrieval_strategy(object(), 'PRODUCTION'), 'retrieval', tmp_path)
    assert len(result.records) == 2
    record = result.records[1]
    assert record['answer'] == '无法回答'
    assert record['diagnostics']['answer_rejected'] is True
    assert record['expected_behavior'] == '应拒答'
    assert result.summary['independent_positive']['n'] == 1
    assert result.summary['independent_positive']['metrics']['evidence_group_recall_at_5'] == 1.0


def test_code_fingerprint_tracks_dirty_tracked_and_relevant_untracked_python(tmp_path):
    import subprocess
    from finance_rag.src.eval.runner import code_worktree_fingerprint
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    directory = tmp_path / 'finance_rag' / 'src'
    directory.mkdir(parents=True)
    path = directory / 'behavior.py'
    path.write_text('x = 1\n', encoding='utf-8')
    first = code_worktree_fingerprint(tmp_path)
    path.write_text('x = 2\n', encoding='utf-8')
    assert code_worktree_fingerprint(tmp_path) != first
    second = code_worktree_fingerprint(tmp_path)
    (tmp_path / '.env').write_text('secret', encoding='utf-8')
    assert code_worktree_fingerprint(tmp_path) == second


def test_stage_group_survival_uses_real_document_pool_and_covered_children(tmp_path):
    coarse = [{'id': 'a', 'source': 'a.pdf', 'content': '2026H1营收10亿元'}]
    local = [{'id': 'parent-b', 'covered_child_ids': ['b'], 'source': 'b.pdf', 'content': '2026H1同比下降3%'}]
    rerank = coarse + local
    strategy = EvaluationStrategy('stages', lambda query: RetrievalOutput(coarse, local, diagnostics={
        'coarse_candidates': coarse, 'coarse_candidate_chunk_ids': ['a'],
        'document_candidates': local, 'document_candidate_chunk_ids': ['parent-b'],
        'rerank_candidates': rerank, 'rerank_candidate_chunk_ids': ['a', 'parent-b'],
    }), config={'use_rerank': True})
    record = EvaluationRunner().run(EvaluationDataset.from_dicts([row()]), strategy, 'retrieval', tmp_path).records[0]
    survival = record['diagnostics']['fact_group_survival']
    assert survival['global']['completion_ranks'] == {'fa': 1, 'fb': None}
    assert survival['document']['completion_ranks'] == {'fa': None, 'fb': 1}
    assert survival['rerank']['completion_ranks'] == {'fa': 1, 'fb': 2}
    assert survival['rerank']['actual_depth'] == 2


def test_production_expands_table_and_preserves_original_index_evidence(monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    original = [{'id': 'a', 'source': 'a.pdf', 'collection': 'c', 'parent_id': 'table-a',
                 'block_type': 'table', 'content': '紧凑摘要'}]
    async def retrieve(query, **kwargs): return {'docs': original}
    class TableStore:
        def get_batch(self, ids):
            return [{'id': 'table-a', 'markdown': '|时期|营收|\n|2026H1|10亿元|', 'start_page': 2, 'row_count': 1}]
    patch_production_retrieval(monkeypatch, retrieve)
    monkeypatch.setattr(chat_service, '_table_store_for', lambda collection: TableStore())
    result = build_retrieval_strategy(object(), 'PRODUCTION').retrieve('问题')
    assert result.results[0]['content'] == '紧凑摘要'
    assert '10亿元' in result.context_results[0]['content']
    assert result.context_results[0]['id'] == 'a'


def test_production_negative_error_is_failed_with_expected_behavior(tmp_path, monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    async def retrieve(query, **kwargs): return {'docs': [], 'decision': 'refuse'}
    async def answer(query, pipeline, **kwargs): raise RuntimeError('LLM服务失败')
    patch_production_retrieval(monkeypatch, retrieve)
    patch_production_answer(monkeypatch, answer)
    negative = {**row(), 'question_type': 'negative', 'evidence': [], 'facts': [],
                'expected_behavior': '应拒答'}
    record = EvaluationRunner().run(EvaluationDataset.from_dicts([negative]), build_retrieval_strategy(object(), 'PRODUCTION'), 'retrieval', tmp_path).records[0]
    assert record['status'] == 'failed'
    assert 'LLM服务失败' in record['error']
    assert record['expected_behavior'] == '应拒答'


def test_kb_content_fingerprint_reads_every_iterator_batch_and_changes_with_text():
    from finance_rag.src.eval.retrieval import knowledge_content_fingerprints
    class Iterator:
        def __init__(self, rows): self.batches = [[rows[0]], [rows[1]], []]
        def next(self): return self.batches.pop(0)
        def close(self): pass
    class KB:
        collection_name = 'test'
        rows = [{'id': 'a', 'source': 'a.pdf', 'content': '第一份'}, {'id': 'b', 'source': 'a.pdf', 'content': '第二份'}]
        def _get_client(self): return self
        def query_iterator(self, **kwargs): return Iterator(self.rows)
        def get_fingerprint_store(self): return {'a.pdf': {'sha256': 'pdf-sha'}}
    kb = KB()
    first = knowledge_content_fingerprints(kb)
    kb.rows[1]['content'] = 'changed'
    second = knowledge_content_fingerprints(kb)
    assert first['chunk_content_fingerprint'] != second['chunk_content_fingerprint']
    assert first['pdf_set_fingerprint'] == second['pdf_set_fingerprint']
    assert knowledge_content_fingerprints(object())['chunk_content_fingerprint'] == 'unavailable'


def test_online_resolver_uses_parent_table_stores_and_current_pdf_sha():
    from finance_rag.src.eval.retrieval import build_online_evidence_resolver
    class Store:
        def get(self, key): return {'id': key, 'source': 'a.pdf', 'content': '父块全文', 'markdown': '完整表格'}
    class KB:
        def get_fingerprint_store(self): return {'a.pdf': {'sha256': 'sha-a'}}
        def _get_parent_store(self): return Store()
        def _get_table_store(self): return Store()
    refs = row()['evidence']
    refs[0].update(chunk_id='', location={'kind': 'parent', 'anchor_id': 'p'})
    refs[1].update(chunk_id='', source='a.pdf', location={'kind': 'table', 'anchor_id': 't'})
    resolver = build_online_evidence_resolver(KB())
    case = EvalCase.from_dict({**row(), 'evidence': refs})
    assert resolver(case.evidence[0])['content'] == '父块全文'
    assert resolver(case.evidence[1])['content'] == '完整表格'
    assert resolver(case.evidence[0])['document_sha256'] == 'sha-a'


def test_pdf_fingerprint_supports_production_fingerprint_store_get_interface():
    from finance_rag.src.eval.retrieval import knowledge_content_fingerprints
    class Store:
        def get(self, source): return {'sha256': 'a' * 64}
    class KB:
        def get_fingerprint_store(self): return Store()
        def list_documents(self): return [{'source': 'a.pdf'}]
    assert len(knowledge_content_fingerprints(KB())['pdf_set_fingerprint']) == 64


def test_git_provenance_decodes_utf8_filenames_on_windows_gbk_locale(monkeypatch):
    import subprocess
    from finance_rag.src.eval import runner
    def git_output(command, **kwargs):
        raw = ('a' * 40 if command[1] == 'rev-parse' else ' M 中文报告.py\n').encode('utf-8')
        if kwargs.get('text'):
            return raw.decode(kwargs.get('encoding') or 'gbk')
        return raw
    monkeypatch.setattr(subprocess, 'check_output', git_output)
    monkeypatch.setattr(runner, 'code_worktree_fingerprint', lambda root: 'snapshot')
    monkeypatch.setattr(runner, 'code_diff_fingerprint', lambda root: 'diff')
    provenance = EvaluationRunner._run_provenance([], EvaluationStrategy('fake', lambda q: RetrievalOutput([], [])), {})
    assert provenance['git_head'] == 'a' * 40
    assert provenance['git_dirty'] is True


def test_negative_with_inherited_positive_stratum_never_enters_positive_denominator(tmp_path):
    negative = {**row(), 'question_type': 'negative', 'evidence': [], 'facts': [],
                'expected_behavior': '拒答', 'provenance': {'stratum': 'explicit'}}
    strategy = EvaluationStrategy('fake', lambda query: RetrievalOutput([], []),
                                  evaluate_negative=lambda case: {'answer': '无法回答', 'answer_rejected': True})
    run = EvaluationRunner().run(EvaluationDataset.from_dicts([negative]), strategy, 'retrieval', tmp_path)
    assert run.summary['independent_positive']['n'] == 0


@pytest.mark.parametrize('answer_result, expected_status, expected_score', [
    ({}, 'failed', None),
    ({'answer':'无法回答'}, 'failed', None),
    ({'answer':'', 'answer_status':'unverified'}, 'completed', None),
    ({'answer':'', 'answer_status':'refused', 'answer_rejected':True}, 'completed', 1.0),
    ({'answer':'', 'answer_status':'complete', 'answer_rejected':False}, 'completed', 0.0),
])
def test_production_negative_requires_completed_result_and_explicit_refusal(
    tmp_path, monkeypatch, answer_result, expected_status, expected_score,
):
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    async def retrieve(query, **kwargs): return {'docs': [], 'decision': 'refuse'}
    async def answer(query, pipeline, **kwargs): return answer_result
    patch_production_retrieval(monkeypatch, retrieve)
    patch_production_answer(monkeypatch, answer)
    negative = {**row(), 'question_type':'negative', 'evidence':[], 'facts':[], 'expected_behavior':'应拒答'}
    record = EvaluationRunner().run(EvaluationDataset.from_dicts([negative]), build_retrieval_strategy(object(), 'PRODUCTION'), 'retrieval', tmp_path).records[0]
    assert record['status'] == expected_status
    assert record['metrics'].get('negative_rejection') == expected_score
    if expected_status == 'failed':
        assert '完成结果' in record['error']
    elif expected_score is None:
        assert record['diagnostics']['rejection_verification'] == 'coverage_or_claim_verification_unknown'


def test_unverified_model_failure_is_not_counted_as_correct_negative(tmp_path):
    negative = {**row(), 'question_type':'negative', 'evidence':[], 'facts':[], 'expected_behavior':'应拒答'}
    strategy = EvaluationStrategy('PRODUCTION', lambda q: RetrievalOutput([], []),
        evaluate_negative=lambda case: {'answer':'无法回答，模型故障', 'answer_status':'unverified',
                                        'evaluation_status':'unverified', 'answer_rejected':False})
    record=EvaluationRunner().run(EvaluationDataset.from_dicts([negative]),strategy,'retrieval',tmp_path).records[0]
    assert record['metrics']['negative_rejection'] is None
    assert record['diagnostics']['rejection_verification']=='coverage_or_claim_verification_unknown'


def test_generation_uses_existing_actual_context_snapshot(tmp_path):
    received=[]
    docs=[{'id':'a','source':'a.pdf','content':'x'*2000+'窗口外事实'}]
    context=[{**docs[0],'content':'x'*2000}]
    def generate(case,contexts):
        received.extend(contexts)
        return '已生成'
    strategy=EvaluationStrategy('snapshot',lambda q:RetrievalOutput(docs,docs,context_results=context),generate=generate,
        evaluate_answer=lambda *a: {})
    EvaluationRunner().run(EvaluationDataset.from_dicts([row()]),strategy,'generation',tmp_path)
    assert received==['x'*2000]


def test_context_fact_coverage_uses_all_actual_context_docs_beyond_top_five(tmp_path):
    noise = [{'id': str(i), 'source': 'noise.pdf', 'content': '无关内容'} for i in range(5)]
    context = noise + [{'id': 'a', 'source': 'a.pdf', 'content': '2026H1营收10亿元'},
                       {'id': 'b', 'source': 'b.pdf', 'content': '2026H1同比下降3%'}]
    strategy = EvaluationStrategy('context', lambda query: RetrievalOutput(noise, context, context_results=context))
    record = EvaluationRunner().run(EvaluationDataset.from_dicts([row()]), strategy, 'retrieval', tmp_path).records[0]
    assert record['metrics']['structured_required_fact_coverage_at_5'] == 0.0
    assert record['metrics']['context_required_fact_coverage'] == 1.0


def test_production_diagnostics_preserve_actual_pipeline_filters_rewrite_depth_and_calls(monkeypatch):
    from finance_rag.src.services import chat_service
    from finance_rag.src.eval.retrieval import build_retrieval_strategy
    async def retrieve(query, **kwargs):
        return {'docs': [], 'filters': {'security_name': '实际公司'}, 'rewritten': '实际改写问题',
                'k': 73, 'rerank_top_n': 7,
                'diagnostics': {'retrieval_calls': [{'query': 'mixed', 'coarse_candidate_count': 0},
                                                     {'query': 'dense', 'coarse_candidate_count': 0}]}}
    patch_production_retrieval(monkeypatch, retrieve)
    diagnostic = build_retrieval_strategy(object(), 'PRODUCTION').retrieve('原问题').diagnostics
    assert diagnostic['filters'] == {'security_name': '实际公司'}
    assert diagnostic['rewritten_query'] == '实际改写问题'
    assert diagnostic['k'] == 73
    assert diagnostic['rerank_top_n'] == 7
    assert diagnostic['retrieval_calls'] == [{'query': 'mixed', 'coarse_candidate_count': 0},
                                              {'query': 'dense', 'coarse_candidate_count': 0}]
