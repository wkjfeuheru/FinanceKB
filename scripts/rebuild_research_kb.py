"""分离新鲜解析与重建回灌，保留可验证解析产物和损失审计。"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_prepared(manifest: dict, files: list[Path]) -> None:
    reports = manifest['reports']
    if set(reports) != {path.name for path in files}:
        raise ValueError('prepared sources differ from original PDF set')
    for path in files:
        if reports[path.name].get('status') != 'completed' or digest(path) != reports[path.name]['sha256']:
            raise ValueError(f'prepared PDF missing/changed/failed: {path.name}')


def validate_parse_artifact(item: dict, source: str, sha: str):
    from finance_rag.src.rag.ingestion.mineru_parser import ParseResult
    from finance_rag.src.rag.ingestion.pdf_assets import ContentBlock
    if not isinstance(item.get('markdown'), str) or not item['markdown'].strip():
        raise ValueError(f'empty parse markdown: {source}')
    if item.get('source') != source or item.get('pdf_sha256') != sha:
        raise ValueError(f'parse artifact source/PDF mismatch: {source}')
    if item.get('parser') != 'mineru' or not item.get('parser_version') or not isinstance(item.get('stats'), dict):
        raise ValueError(f'invalid parser identity: {source}')
    blocks = item.get('blocks')
    captions = item.get('figure_captions')
    if not isinstance(blocks, list) or not blocks or any(
            not isinstance(b, dict) or not isinstance(b.get('text'), str) or not isinstance(b.get('page'), int)
            or isinstance(b['page'], bool) or b['page'] <= 0 for b in blocks):
        raise ValueError(f'invalid parsed blocks: {source}')
    if not isinstance(captions, dict) or any(not str(k).isdigit() or int(k) <= 0 or
            not isinstance(v, list) or any(not isinstance(t, str) for t in v) for k, v in captions.items()):
        raise ValueError(f'invalid figure captions: {source}')
    return ParseResult(markdown=item['markdown'], parser=item['parser'], parser_version=item['parser_version'],
                       stats=item['stats'], blocks=tuple(ContentBlock(**block) for block in blocks),
                       figure_captions={int(k): v for k, v in captions.items()})


def validate_baselines(directory: Path, dataset, content_hash: str) -> None:
    from finance_rag.src.eval.dataset import EvalCase
    from finance_rag.src.eval.runner import EvaluationRunner
    for layer in ('L0', 'L1', 'L2', 'L3', 'L5', 'PRODUCTION'):
        candidates = list(directory.rglob(f'*-{layer}/summary.json'))
        valid = False
        for path in candidates:
            try:
                summary = json.loads(path.read_text(encoding='utf8'))
                config = json.loads((path.parent / 'run_config.json').read_text(encoding='utf8'))
                records = [json.loads(line) for line in (path.parent / 'records.jsonl').read_text(encoding='utf8').splitlines() if line]
                count = 20 if layer == 'PRODUCTION' else 18
                selected = list(dataset.cases) if layer == 'PRODUCTION' else EvaluationRunner._select_cases(dataset, 'retrieval', False)
                selected_ids = {case.id for case in selected}
                snapshot_cases = [EvalCase.from_dict(json.loads(line)) for line in
                                  (path.parent / 'dataset_snapshot.jsonl').read_text(encoding='utf8').splitlines() if line]
                if EvaluationRunner._dataset_fingerprint(snapshot_cases) != EvaluationRunner._dataset_fingerprint(list(dataset.cases)):
                    continue
                expected = {'dataset_fingerprint': EvaluationRunner._dataset_fingerprint(selected), 'chunk_content_fingerprint': content_hash,
                            'metrics_version': '2.0', 'annotation_schema_versions': ['2.0'],
                            'embedding_model': 'BAAI/bge-m3', 'embedding_pooling': 'cls', 'embedding_dim': 1024,
                            'chunk_max_tokens': 256, 'chunk_overlap_tokens': 25,
                            'dense_weight': 0.8, 'sparse_weight': 0.2}
                if all(summary.get(k) == v and config.get(k) == v for k, v in expected.items()) and \
                        summary.get('failed') == 0 and summary.get('completed') == count and \
                        summary.get('n') == count and len(records) == count and \
                        len(selected) == count and len(selected_ids) == count and \
                        {r['case_id'] for r in records} == selected_ids and all(r['status'] == 'completed' for r in records):
                    valid = True
                    break
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        if not valid:
            raise ValueError(f'missing/invalid successful revised-label baseline: {layer}')


def loss_audit(source: str, markdown: str) -> dict:
    from finance_rag.src.rag.ingestion.cleaner import TextCleaner
    from finance_rag.src.rag.ingestion.html_table import restore_html_table
    cleaned = TextCleaner().clean_with_audit(markdown)
    tables = []
    for table in re.finditer(r'<table\b[^>]*>.*?</table>', markdown, re.I | re.S):
        layout = restore_html_table(table.group())
        tables.append({'start': table.start(), 'end': table.end(), 'rows': layout.rows,
                       'merges': list(layout.merges), 'error': layout.error,
                       'raw_html': table.group() if layout.error else None})
    return {'source': source, 'removals': [asdict(item) for item in cleaned.removals],
            'raw_chars': len(markdown), 'clean_chars': len(cleaned.text), 'tables': tables}


def seal_prepared(directory: Path) -> None:
    """验证全套解析与损失审计；新增封存清单，不修改初始解析记录。"""
    manifest = json.loads((directory / 'parse_manifest.json').read_text(encoding='utf8'))
    verify_prepared(manifest, sorted((ROOT / 'assets/reports').glob('*.pdf')))
    if manifest.get('fresh_parse') is not True:
        raise ValueError('fresh_parse is required')
    target = directory / 'parse_manifest.verified.json'
    if target.exists():
        raise ValueError('verified parse manifest is immutable')
    for source, spec in manifest['reports'].items():
        artifact = (directory / spec['artifact']).resolve()
        audit_path = (directory / spec['audit']).resolve()
        if artifact.parent != directory.resolve() or audit_path.parent != directory.resolve() or digest(artifact) != spec['artifact_sha256']:
            raise ValueError(f'parse artifact checksum/path mismatch: {source}')
        item = json.loads(artifact.read_text(encoding='utf8'))
        validate_parse_artifact(item, source, spec['sha256'])
        actual = json.loads(audit_path.read_text(encoding='utf8'))
        if actual != loss_audit(source, item['markdown']):
            raise ValueError(f'loss audit differs from raw parse: {source}')
        spec['audit_sha256'] = digest(audit_path)
    manifest['verified_at'] = time.time()
    manifest['original_manifest_sha256'] = digest(directory / 'parse_manifest.json')
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')


def prepare(directory: Path) -> int:
    from finance_rag.src.rag.ingestion.mineru_parser import parse_with_mineru
    from finance_rag.src.rag.ingestion.cleaner import TextCleaner
    from finance_rag.src.rag.ingestion.html_table import restore_html_table

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'parse_manifest.json'
    if manifest_path.exists():
        raise ValueError('prepared outputs are immutable; choose a new directory')
    manifest = {'fresh_parse': True, 'started_at': time.time(), 'reports': {}}
    files = sorted((ROOT / 'assets/reports').glob('*.pdf'))
    for index, path in enumerate(files, 1):
        print(f'[{index}/{len(files)}] Fresh MinerU parse: {path.name}', flush=True)
        started = time.perf_counter()
        record = {'sha256': digest(path)}
        try:
            result = parse_with_mineru(path, source=path.name)
            key = hashlib.sha256(path.name.encode('utf8')).hexdigest()[:16]
            artifact = {'source': path.name, 'pdf_sha256': record['sha256'],
                        'markdown': result.markdown, 'parser': result.parser,
                        'parser_version': result.parser_version, 'stats': result.stats,
                        'blocks': [asdict(block) for block in result.blocks],
                        'figure_captions': result.figure_captions}
            artifact_path = directory / f'{key}.parse.json'
            artifact_path.write_text(json.dumps(artifact, ensure_ascii=False), encoding='utf8')
            cleaned = TextCleaner().clean_with_audit(result.markdown)
            tables = []
            for table in re.finditer(r'<table\b[^>]*>.*?</table>', result.markdown, re.I | re.S):
                layout = restore_html_table(table.group())
                tables.append({'start': table.start(), 'end': table.end(), 'rows': layout.rows,
                               'merges': layout.merges, 'error': layout.error,
                               'raw_html': table.group() if layout.error else None})
            audit = {'source': path.name, 'removals': [asdict(item) for item in cleaned.removals],
                     'raw_chars': len(result.markdown), 'clean_chars': len(cleaned.text), 'tables': tables}
            (directory / f'{key}.audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
            record.update(status='completed', artifact=artifact_path.name, artifact_sha256=digest(artifact_path),
                          audit=f'{key}.audit.json', parser_version=result.parser_version,
                          audit_sha256=digest(directory / f'{key}.audit.json'),
                          raw_chars=len(result.markdown), images=len(result.images), **result.stats)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(record['error'], flush=True)
        record['latency_seconds'] = round(time.perf_counter() - started, 3)
        manifest['reports'][path.name] = record
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
        print(f"  {record['status']} {record['latency_seconds']}s", flush=True)
    return 0 if len(files) == 18 and all(r['status'] == 'completed' for r in manifest['reports'].values()) else 1


def ingest(prepared: Path, snapshot: Path, baseline: Path, dataset: Path, confirmation: str, report: Path) -> int:
    from scripts.kb_snapshot import verify_snapshot, _read_rows
    from finance_rag.src.infrastructure.vector_store.milvus_kb import get_knowledge_base
    from finance_rag.src.rag.ingestion.mineru_parser import ParseResult
    from finance_rag.src.rag.ingestion.pdf_assets import ContentBlock, extract_images
    import finance_rag.src.rag.ingestion as facade
    from scripts.ingest_reports import _run

    if confirmation != 'finance_kb':
        raise ValueError('--confirm-collection finance_kb is required')
    saved = verify_snapshot(snapshot)
    if saved.get('source_count') != 18 or not {'vectors.jsonl.gz', 'parent_chunks.jsonl.gz',
            'table_chunks.jsonl.gz', 'schema.json', 'fingerprints.json', 'pdf_manifest.json'} <= saved['artifacts'].keys():
        raise ValueError('complete 18-source backup is required before rebuilding')
    from finance_rag.src.eval.dataset import EvaluationDataset
    from finance_rag.src.eval.runner import EvaluationRunner
    rows = _read_rows(snapshot / 'vectors.jsonl.gz')
    chunks = sorted((r['id'], r['source'], hashlib.sha256(r['content'].encode('utf8')).hexdigest()) for r in rows)
    content_hash = hashlib.sha256(json.dumps(chunks, ensure_ascii=False).encode('utf8')).hexdigest()
    validate_baselines(baseline, EvaluationDataset.load(dataset), content_hash)
    manifest = json.loads((prepared / 'parse_manifest.verified.json').read_text(encoding='utf8'))
    if manifest.get('original_manifest_sha256') != digest(prepared / 'parse_manifest.json'):
        raise ValueError('original parse manifest changed')
    if manifest.get('fresh_parse') is not True:
        raise ValueError('fresh_parse is required')
    verify_prepared(manifest, sorted((ROOT / 'assets/reports').glob('*.pdf')))
    artifacts = {}
    for source, spec in manifest['reports'].items():
        path = (prepared / spec['artifact']).resolve()
        if path.parent != prepared.resolve() or digest(path) != spec['artifact_sha256']:
            raise ValueError(f'parse artifact checksum/path mismatch: {source}')
        audit = (prepared / spec['audit']).resolve()
        if audit.parent != prepared.resolve() or digest(audit) != spec.get('audit_sha256'):
            raise ValueError(f'audit artifact checksum/path mismatch: {source}')
        item = json.loads(path.read_text(encoding='utf8'))
        validate_parse_artifact(item, source, spec['sha256'])
        artifacts[source] = item

    def prepared_parse(path, *, source=''):
        item = artifacts[source]
        if digest(Path(path)) != item['pdf_sha256']:
            raise ValueError(f'upload PDF changed: {source}')
        return ParseResult(markdown=item['markdown'], parser=item['parser'],
                           parser_version=item['parser_version'], stats=item['stats'],
                           blocks=tuple(ContentBlock(**block) for block in item['blocks']),
                           figure_captions={int(k): v for k, v in item['figure_captions'].items()},
                           images=extract_images(path))

    # 只替换本次迁移进程的解析读入：使用刚刚完整重跑 MinerU 的可校验产物。
    # 之后继续走既有清洗/切块/图片描述/embedding/写库流水线。
    facade.parse_with_mineru = prepared_parse
    get_knowledge_base('finance_kb').rebuild_collection()
    args = argparse.Namespace(only=[], limit=0, category='investment_research', timeout=1800,
                              skip_ingest=False, report_json=str(report))
    return asyncio.run(_run(args))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'verify', 'ingest'))
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--snapshot')
    parser.add_argument('--baseline')
    parser.add_argument('--dataset')
    parser.add_argument('--confirm-collection', default='')
    parser.add_argument('--report', default='migration.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.stage == 'prepare':
        return prepare(Path(args.prepared))
    if args.stage == 'verify':
        seal_prepared(Path(args.prepared))
        return 0
    if not args.snapshot or not args.baseline or not args.dataset:
        parser.error('ingest requires --snapshot, --baseline and --dataset')
    return ingest(Path(args.prepared), Path(args.snapshot), Path(args.baseline), Path(args.dataset), args.confirm_collection, Path(args.report))


if __name__ == '__main__':
    raise SystemExit(main())
