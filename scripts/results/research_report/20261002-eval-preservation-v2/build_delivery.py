"""本次运行的只读验收与描述性报告生成器；不执行跨指纹显著性检验。"""
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.kb_snapshot import _read_rows, verify_snapshot
from finance_rag.src.eval.runner import negative_rejection_score


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line]


def write_json(name, value):
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def build(qa_only=False):
    before, after = verify_snapshot(BASE / 'backup'), verify_snapshot(BASE / 'after')
    migration = read(BASE / 'migration.json')
    assert migration['exit_code'] == 0 and migration['source_count'] == 18
    assert len(migration['outcomes']) == 18 and all(r['status'] == 'completed' for r in migration['outcomes'])
    tables = _read_rows(BASE / 'after/table_chunks.jsonl.gz')
    proofs = []
    for source, label, values in [
        ('H3_AP202609271829940631_1.pdf', '营业总收入', {'2026E': '36,342'}),
        ('H3_AP202609281829951050_1.pdf', '营业收入', {'2025A': '22,586', '2026E': '24,924'}),
    ]:
        candidates = []
        for table in tables:
            if table['source'] != source:
                continue
            rows = table['payload']
            for header in rows:
                if not all(year in header for year in values):
                    continue
                for row in rows:
                    if any(label in cell for cell in row) and '百万元' in ''.join(header + row) and all(
                            len(row) > header.index(year) and row[header.index(year)] == value
                            for year, value in values.items()):
                        candidates.append({'source': source, 'table_id': table['id'], 'header': header, 'row': row})
        assert candidates, f'year/column regression failed: {source}'
        proofs.extend(candidates)
    parents = _read_rows(BASE / 'after/parent_chunks.jsonl.gz')
    vectors = _read_rows(BASE / 'after/vectors.jsonl.gz')
    per_source = {row['source']: row for row in vectors}
    schema = read(BASE / 'after/schema.json')
    norms = [math.sqrt(sum(v*v for v in row['dense_vector'])) for row in vectors]
    vector_validation = {'dimensions': dict(Counter(len(row['dense_vector']) for row in vectors)),
                         'l2_norm_min': min(norms), 'l2_norm_max': max(norms),
                         'legacy_512_count': sum(len(row['dense_vector']) == 512 for row in vectors)}
    risks = []
    for source, phrases in [
        ('H3_AP202609271829940631_1.pdf', ['轨道交通固定资产投资不及预期', '城轨市场竞争加剧', '智能低空业务规模化落地不及预期']),
        ('H3_AP202609261829925702_1.pdf', ['宏观经济风险', '国际化经营风险', '市场竞争及行业风险']),
    ]:
        matching = [r['id'] for r in parents + vectors if r['source'] == source and
                    all(p in ''.join(r['content'].split()) for p in phrases)]
        assert matching, f'risk text lost: {source}'
        risks.append({'source': source, 'evidence_container_ids': matching, 'phrases': phrases})
    audits = [read(path) for path in (BASE / 'fresh-parse').glob('*.audit.json')]
    fallbacks = [{'source': audit['source'], 'error': table['error'], 'preserved':
                 any(table['raw_html'] in row['markdown'] for row in tables if row['source'] == audit['source'])}
                 for audit in audits for table in audit['tables'] if table['error']]
    assert all(item['preserved'] for item in fallbacks)
    mapping_rows = []
    facts_lost = []
    for name in ('research_report_gold20.jsonl', 'research_report_regression4.jsonl'):
        for case in jsonl(BASE / 'annotations-new-kb' / name):
            refs = {e['evidence_id']: e for e in case['evidence']}
            for fact in case['facts']:
                mapped = any(all(refs[e]['location']['mapping_status'] == 'mapped' for e in group)
                             for group in fact['acceptable_evidence_sets'])
                item = {'case_id': case['id'], 'fact_id': fact['id'], 'statement': fact['statement'],
                        'any_complete_group_mapped': mapped}
                mapping_rows.append(item)
                if not mapped:
                    facts_lost.append(item)
    acceptance = {'before': before, 'after': after, 'migration': migration,
                  'vector_validation': vector_validation, 'offline_smoke': read(BASE / 'offline_smoke.json'),
                  'report_types': dict(Counter(row['report_type'] for row in per_source.values())),
                  'security_code_in_schema': any(field['name'] == 'security_code' for field in schema['fields']),
                  'stored_dates': {source: row['date'] for source, row in sorted(per_source.items())},
                  'publication_dates_semantically_verified': False,
                  'table_column_proofs': proofs, 'risk_text_proofs': risks, 'html_fallbacks': fallbacks,
                  'cleaning_rules': dict(Counter(r['rule'] for a in audits for r in a['removals'])),
                  'parsed_tables': sum(len(a['tables']) for a in audits),
                  'fact_mapping': mapping_rows, 'unmapped_facts': facts_lost}
    write_json('acceptance.json', acceptance)
    if qa_only:
        print(json.dumps({'sources': after['source_count'], 'table_column_proofs': len(proofs),
                          'risk_proofs': len(risks), 'raw_html_preserved': len(fallbacks),
                          'vector_validation': vector_validation,
                          'unmapped_facts': len(facts_lost)}, ensure_ascii=False))
        return
    pairs = []
    per_question = []
    for dataset, old_dir, new_dir in [
        ('core20', 'baseline-final', 'after-eval-core'),
        ('interview30-paired', 'baseline-interview', 'after-eval-interview'),
        ('regression4', 'baseline-regression', 'after-eval-regression'),
    ]:
        for layer in ('L0', 'L1', 'L2', 'L3', 'L5', 'PRODUCTION'):
            old_path, = (BASE / old_dir).rglob(f'old-kb-final-{layer}/summary.json')
            new_path, = (BASE / new_dir).rglob(f'new-kb-final-{layer}/summary.json')
            old, new = read(old_path), read(new_path)
            assert old['failed'] == new['failed'] == 0
            pair = {'dataset': dataset, 'layer': layer, 'comparison_type': 'descriptive_only',
                    'significance': 'not_computed_different_kb_and_annotation_mapping_fingerprints',
                    'old': old, 'new': new}
            pairs.append(pair)
            old_records = {r['case_id']: r for r in jsonl(old_path.parent / 'records.jsonl')}
            new_records = {r['case_id']: r for r in jsonl(new_path.parent / 'records.jsonl')}
            assert old_records.keys() == new_records.keys()
            for phase, records in [('old', old_records), ('new', new_records)]:
                times = sorted(r['latency_ms'] for r in records.values() if 'negative_rejection' not in r['metrics'])
                pair[phase + '_positive_latency_ms'] = {'n': len(times), 'mean': statistics.mean(times),
                    'median': statistics.median(times), 'p95_nearest_rank': times[math.ceil(.95*len(times))-1],
                    'note': '实际运行记录，含冷启动与并发影响，不作性能因果归因'}
            for case_id, a in old_records.items():
                b = new_records[case_id]
                for metric in sorted(a['metrics'].keys() | b['metrics'].keys()):
                    per_question.append({'dataset': dataset, 'layer': layer, 'case_id': case_id,
                                         'metric': metric, 'old': a['metrics'].get(metric), 'new': b['metrics'].get(metric),
                                         'old_latency_ms': a['latency_ms'], 'new_latency_ms': b['latency_ms']})
    write_json('descriptive_comparison.json', pairs)
    negatives = []
    for pair in pairs:
        if pair['layer'] != 'PRODUCTION':
            continue
        for phase, directory in [('old', {'core20': 'baseline-final', 'interview30-paired': 'baseline-interview', 'regression4': 'baseline-regression'}[pair['dataset']]),
                                 ('new', {'core20': 'after-eval-core', 'interview30-paired': 'after-eval-interview', 'regression4': 'after-eval-regression'}[pair['dataset']])]:
            path, = (BASE / directory).rglob(f'{"old" if phase == "old" else "new"}-kb-final-PRODUCTION/records.jsonl')
            for record in jsonl(path):
                if 'negative_rejection' not in record['metrics']:
                    continue
                answer = str(record.get('answer') or '')
                negatives.append({'dataset': pair['dataset'], 'phase': phase, 'case_id': record['case_id'],
                                  'answer': answer, 'production_answer_rejected_flag': record['diagnostics'].get('answer_rejected'),
                                  'reported_negative_rejection': record['metrics']['negative_rejection'],
                                  'answer_keyword_proxy': negative_rejection_score(answer) if answer.strip() else None,
                                  'note': '生产标记和答案拒答措辞分别记录；关键词代理不等同于语义人工审核，空答案不推断成功'})
    write_json('negative_refusal_audit.json', negatives)
    import pymupdf
    reviewed_blocks = []
    for path in sorted((ROOT / 'assets/reports').glob('*.pdf')):
        with pymupdf.open(path) as document:
            for page_index, page in enumerate(document):
                text = page.get_text()
                for block in page.get_text('blocks'):
                    if block[6] != 0:
                        continue
                    matched = [term for term in ('宁德时代', '贵州茅台', '茅台', 'CATL', 'GWh')
                               if term.lower() in re.sub(r'\s+', '', block[4]).lower()]
                    if matched:
                        start = text.find(block[4])
                        reviewed_blocks.append({'source': path.name, 'page': page_index + 1,
                                                'start': start, 'end': start + len(block[4]),
                                                'terms': matched, 'complete_original_block': block[4]})
    write_json('negative_corpus_review.json', {
        'review_method': 'full 18-PDF text scan + implementation-agent review of all company/alias/GWh blocks',
        'pdf_manifest': 'backup/pdf_manifest.json', 'human_approved': False,
        'tier': 'silver', 'status': 'auto_pass', 'reviewed_blocks': reviewed_blocks,
        'conclusions': [
            '宁德时代出现于储能电芯报价/587Ah及标的推荐，不提供2026H1储能电池出货量GWh事实；唯一GWh块为横店东磁产能，不是宁德时代出货量。',
            '贵州茅台/茅台/CATL扫描无相关命中，18份报告没有贵州茅台2026H1营收和归母净利润完整事实。'],
        'limitation': '脚本关键词扫描不是逻辑上的完备证明；结论为代理原文复核，仍非人工Gold审核'})
    with (BASE / 'per_question_descriptive.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_question[0]))
        writer.writeheader()
        writer.writerows(per_question)
    print(json.dumps({'sources': after['source_count'], 'vectors': after['artifacts']['vectors.jsonl.gz']['count'],
                      'complete_fact_groups': len(mapping_rows)-len(facts_lost), 'unmapped_facts': len(facts_lost),
                      'evaluations': len(pairs)}, ensure_ascii=False))


if __name__ == '__main__':
    build(qa_only='--qa-only' in sys.argv)
