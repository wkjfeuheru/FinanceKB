"""从完整 PDF 核验事实并发布可重映射的 silver 标注；从不依赖检索排名。"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / 'src/eval/data'
VERSION = 'research-report-facts-v2-20261002'


def normalize(text: str) -> str:
    return re.sub(r'\s+', '', text)


def _positions(text: str) -> tuple[str, list[int]]:
    positions = [i for i, char in enumerate(text) if not char.isspace()]
    return ''.join(text[i] for i in positions), positions


def locate_quote(text: str, phrases: list[str], paragraph_spans: list[tuple[int, int]] | None = None) -> tuple[str, int, int]:
    """用完整句子/标题定位；不按字符上限切断词、数字或标签。"""
    compact, positions = _positions(text)
    occurrences = []
    for phrase in phrases:
        phrase = normalize(phrase)
        hits = [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), compact)]
        if not hits:
            raise ValueError(f'phrase not present: {phrase}')
        occurrences.append(hits)
    choices = []
    for anchors in occurrences:
        for a, b in anchors:
            selected = [min(hits, key=lambda hit: abs(hit[0] - a)) for hits in occurrences]
            choices.append((min(s for s, _ in selected), max(e for _, e in selected)))
    first, last = min(choices, key=lambda span: span[1] - span[0])
    ranges = [(positions[first], positions[last - 1] + 1)]
    start, end = min(a for a, _ in ranges), max(b for _, b in ranges)
    # PDF 断行不是句子终点；双换行是 Markdown 段落边界。
    boundaries = list(re.finditer(r'[。！？]|\n\s*\n', text))
    previous = [m.end() for m in boundaries if m.end() <= start]
    following = [m.end() for m in boundaries if m.start() >= end]
    previous_start = max(previous, default=0)
    if paragraph_spans:
        containing = [span for span in paragraph_spans if span[0] <= start < span[1]]
        if containing:
            previous_start = max(previous_start, containing[-1][0])
    start = previous_start
    end = min(following, default=len(text))
    if paragraph_spans:
        # 使用 PDF 排版块边界，不把物理断行误当句首/句尾。
        containers = [span for span in paragraph_spans if span[0] < ranges[0][1] <= span[1]]
        if containers:
            end = min(end, containers[-1][1])
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[start:end], start, end


def map_evidence(evidence: dict, rows: list[dict]) -> dict:
    """优先映射完整原文；无映射保留身份及丢失状态，不删除事实。"""
    result = copy.deepcopy(evidence)
    quote = result['quote']
    needle = normalize(quote)
    for row in sorted(rows, key=lambda r: (r.get('kind', 'chunk') != 'chunk', len(r.get('content', '')), r['id'])):
        if row['source'] != result['source']:
            continue
        content = row.get('content', '')
        compact, positions = _positions(content)
        index = compact.find(needle)
        if index < 0 or not needle:
            continue
        start, end = positions[index], positions[index + len(needle) - 1] + 1
        result.update(chunk_id=row['id'] if row.get('kind', 'chunk') == 'chunk' else '',
                      quote=content[start:end], quote_start=start, quote_end=end)
        result['location'].update(kind=row.get('kind', 'chunk'), mapping_status='mapped',
                                  anchor_id=row['id'], start_page=row.get('start_page', 0),
                                  end_page=row.get('end_page', row.get('start_page', 0)))
        return result
    result.update(chunk_id='', quote_start=0, quote_end=len(quote))
    result['location'].update(kind='pdf', mapping_status='unmapped')
    return result


def remap_case(case: dict, rows: list[dict]) -> dict:
    """完整引文跨块时以原连续句子拆分，组合内要求全部命中。"""
    def remap_table_prefix(ref: dict) -> dict | None:
        """表格渲染尾部变更时，仅映射原样保留的完整表头和连续目标行。"""
        from src.eval.metrics import compute_text_coverage
        ref_id = ref['evidence_id']
        facts = [fact for fact in case['facts'] if any(ref_id in group for group in fact['acceptable_evidence_sets'])]
        if not facts or any(
            not fact.get('aliases') or ref['source'] not in fact.get('allowed_sources', [])
            or any(len(group) != 1 for group in fact['acceptable_evidence_sets'])
            for fact in facts
        ):
            return None
        quote = ref['quote']
        lines = quote.splitlines(keepends=True)
        if len(lines) < 3:
            return None
        header = lines[0].strip()
        separator = lines[1].strip()
        if not (header.startswith('|') and header.endswith('|') and separator.startswith('|') and separator.endswith('|')):
            return None
        columns = len(header.split('|')) - 2
        separator_cells = separator.split('|')[1:-1]
        if columns < 2 or len(separator_cells) != columns or not all(re.fullmatch(r'\s*:?-{3,}:?\s*', cell) for cell in separator_cells):
            return None
        offset = len(lines[0]) + len(lines[1])
        tables = [row for row in rows if row.get('kind') == 'table' and row.get('source') == ref['source']]
        for line in lines[2:]:
            complete_row = line.strip()
            if not (complete_row.startswith('|') and complete_row.endswith('|') and len(complete_row.split('|')) - 2 == columns):
                break
            offset += len(line)
            prefix = quote[:offset].rstrip('\r\n')
            if not all(any(group and compute_text_coverage([prefix], group) == 1.0 for group in fact['aliases']) for fact in facts):
                continue
            candidate = copy.deepcopy(ref)
            candidate['quote'] = prefix
            mapped_prefix = map_evidence(candidate, tables)
            if mapped_prefix['location']['mapping_status'] == 'mapped':
                mapped_prefix['location'].update(
                    mapping_reason='table_rendering_changed_target_rows_unchanged',
                    previous_quote_sha256=_hash(quote),
                )
                return mapped_prefix
        return None

    result = copy.deepcopy(case)
    replacements, evidence = {}, []
    for ref in case['evidence']:
        mapped = map_evidence(ref, rows)
        if mapped['location']['mapping_status'] == 'unmapped' and ref['location'].get('kind') == 'table':
            mapped = remap_table_prefix(ref) or mapped
        pieces = re.findall(r'[^。！？]*[。！？]|[^。！？]+$', ref['quote'])
        split_refs = []
        if mapped['location']['mapping_status'] == 'unmapped' and ref['location'].get('kind') != 'table' and len(pieces) > 1:
            offset = 0
            for piece in pieces:
                fragment = copy.deepcopy(ref)
                fragment.update(evidence_id=_hash(ref['evidence_id'] + f'|sentence|{offset}'), quote=piece)
                fragment['location'].update(parent_evidence_id=ref['evidence_id'], fragment_start=offset,
                                            fragment_end=offset + len(piece))
                split_refs.append(map_evidence(fragment, rows))
                offset += len(piece)
        if split_refs and all(r['location']['mapping_status'] == 'mapped' for r in split_refs):
            replacements[ref['evidence_id']] = [r['evidence_id'] for r in split_refs]
            evidence.extend(split_refs)
        else:
            replacements[ref['evidence_id']] = [ref['evidence_id']]
            evidence.append(mapped)
    result['evidence'] = list({r['evidence_id']: r for r in evidence}.values())
    for fact in result['facts']:
        fact['acceptable_evidence_sets'] = [list(dict.fromkeys(
            new for old in group for new in replacements[old])) for group in fact['acceptable_evidence_sets']]
    return result


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def _supports(content: str, aliases: list[list[str]]) -> list[str] | None:
    compact = normalize(content)
    return next((group for group in aliases if all(normalize(p) in compact for p in group)), None)


def audit_cases(cases: list[dict], rows: list[dict], pdf_pages: dict[str, list[str]],
                pdf_hashes: dict[str, str]) -> dict:
    from src.eval.dataset import EvaluationDataset
    indexed = {(row.get('kind', 'chunk'), row['id']): row for row in rows}
    for case in cases:
        for ref in case['evidence']:
            location = ref['location']
            page = pdf_pages[ref['source']][location['pdf_page'] - 1]
            if page[location['pdf_start']:location['pdf_end']] != location['original_quote']:
                raise ValueError(f"PDF anchor offset mismatch: {ref['evidence_id']}")
            if pdf_hashes[ref['source']] != ref['document_sha256']:
                raise ValueError(f"PDF hash mismatch: {ref['evidence_id']}")
    def resolve(ref):
        if ref.location['kind'] == 'pdf':
            return {'source': ref.source, 'content': ref.quote, 'document_sha256': pdf_hashes[ref.source]}
        row = indexed.get((ref.location['kind'], ref.location['anchor_id']))
        return {**row, 'document_sha256': pdf_hashes[ref.source]} if row else None
    audit = EvaluationDataset.from_dicts(cases).validate(resolve_evidence=resolve)
    return {'valid': audit.valid, 'cases': audit.case_count, 'checked_evidence': audit.checked_evidence,
            'review_method': 'whole-PDF offsets/SHA + stored-container exact offsets + self-coverage=1',
            'human_approved': False}


def build_case(old: dict, spec: dict, rows: list[dict], pdf_pages: dict[str, list[str]],
               pdf_hashes: dict[str, str], pdf_layouts: dict | None = None) -> dict:
    case = copy.deepcopy(old)
    case.update(schema_version='2.0', query=spec['query'], reference_answer=spec['answer'],
                tier='silver', review_status='auto_pass', reviewer='source-audit-script+implementation-agent',
                review_note='完整 PDF 事实核验与引文自匹配；未进行人工审核', evidence=[], facts=[])
    case['provenance'].update(annotation_version=VERSION, review_method='pdf-source-and-offset-audit',
                              human_approved=False, independent_sample=True)
    for number, fact in enumerate(spec.get('facts', []), 1):
        source = fact['source']
        aliases = fact['aliases']
        anchors = []
        # 整份 PDF，包含标题、要点和正文；不参考当前检索排名。
        for page, content in enumerate(pdf_pages[source], 1):
            for group in aliases:
                if not _supports(content, [group]):
                    continue
                spans = pdf_layouts[source][page - 1] if pdf_layouts else None
                original, pdf_start, pdf_end = locate_quote(content, group, spans)
                eid = _hash(f'{pdf_hashes[source]}|{page}|{pdf_start}|{pdf_end}')
                ref = {'evidence_id': eid, 'source': source, 'document_version': source,
                       'document_sha256': pdf_hashes[source], 'chunk_id': '', 'quote': original,
                       'quote_start': 0, 'quote_end': len(original),
                       'location': {'kind': 'pdf', 'original_quote': original,
                                    'pdf_page': page, 'pdf_start': pdf_start, 'pdf_end': pdf_end,
                                    'start_page': page, 'end_page': page,
                                    'anchor_id': eid, 'mapping_status': 'unmapped'}}
                anchors.append(map_evidence(ref, rows))
        if not anchors:
            raise ValueError(f"{case['id']} fact {number}: original PDF does not support {fact['statement']}")
        # 同一物理句子的多个当前块可为有效替代映射；不以弱关键词造事实。
        for row in rows:
            if row['source'] != source:
                continue
            terms = _supports(row.get('content', ''), aliases)
            if not terms:
                continue
            quote, start, end = locate_quote(row['content'], terms)
            matched = next((ref for ref in anchors if normalize(quote).lstrip('#▌■') in
                            normalize(ref['location']['original_quote'])), None)
            if row.get('kind') == 'table':
                # 表格的线性 PDF 提取顺序与 Markdown 不同。依靠完整表头、
                # 单位及目标值约束定位，并保留整表+PDF锚点供人工复核列关系。
                matched = anchors[0]
                quote, start, end = row['content'], 0, len(row['content'])
            if matched is None:
                continue
            ref = copy.deepcopy(matched)
            ref.update(evidence_id=_hash(matched['evidence_id'] + '|text|' + normalize(quote)),
                       chunk_id=row['id'] if row.get('kind', 'chunk') == 'chunk' else '',
                       quote=quote, quote_start=start, quote_end=end)
            ref['location'].update(kind=row.get('kind', 'chunk'), anchor_id=row['id'], mapping_status='mapped')
            anchors.append(ref)
        unique = {ref['evidence_id']: ref for ref in anchors}
        case['facts'].append({'id': f"{case['id']}-f{number}", 'statement': fact['statement'],
                              'allowed_sources': [source], 'aliases': aliases,
                              'acceptable_evidence_sets': [[eid] for eid in unique]})
        existing = {ref['evidence_id'] for ref in case['evidence']}
        case['evidence'].extend(ref for eid, ref in unique.items() if eid not in existing)
    case['required_facts'] = [fact['statement'] for fact in case['facts']]
    if case['question_type'] == 'negative':
        case['expected_behavior'] = spec.get('expected_behavior', old['expected_behavior'])
    return case


def main() -> int:
    import pymupdf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--specs', default=str(DATA / 'research_report_fact_specs_v2.json'))
    parser.add_argument('--remap', help='已有 schema2 JSONL，仅重映射，不改事实与证据身份')
    args = parser.parse_args()
    snapshot, out = Path(args.snapshot), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(snapshot / 'vectors.jsonl.gz')
    for name, kind, field in [('parent_chunks.jsonl.gz', 'parent', 'content'),
                              ('table_chunks.jsonl.gz', 'table', 'markdown')]:
        for row in _read_jsonl(snapshot / name):
            rows.append({**row, 'kind': kind, 'content': row[field]})
    pdf_pages, pdf_layouts, hashes = {}, {}, {}
    for path in sorted((ROOT / 'assets/reports').glob('*.pdf')):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with pymupdf.open(path) as document:
            pdf_pages[path.name] = [page.get_text() for page in document]
            pdf_layouts[path.name] = []
            for page, content in zip(document, pdf_pages[path.name]):
                spans, cursor = [], 0
                for block in page.get_text('blocks'):
                    if block[6] != 0:
                        continue
                    start = content.find(block[4], cursor)
                    if start >= 0:
                        cursor = start + len(block[4])
                        spans.append((start, cursor))
                pdf_layouts[path.name].append(spans)
    if args.remap:
        revised = _read_jsonl(Path(args.remap))
        revised = [remap_case(case, rows) for case in revised]
    else:
        specs = json.loads(Path(args.specs).read_text(encoding='utf8'))
        old = {case['id']: case for case in _read_jsonl(DATA / 'research_report_gold20.jsonl')}
        _write_jsonl(out / 'historical_gold20.jsonl', list(old.values()))
        revised = []
        for spec in specs:
            template = old.get(spec['id'], {'id': spec['id'], 'question_type': 'single_hop',
                       'difficulty': 'medium', 'answer_type': 'factual', 'provenance': {},
                       'expected_behavior': ''})
            revised.append(remap_case(build_case(template, spec, rows, pdf_pages, hashes, pdf_layouts), rows))
        changes = [{'id': case['id'], 'before': old.get(case['id']), 'after': case}
                   for case in revised]
        (out / 'annotation_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf8')
    audit = audit_cases(revised, rows, pdf_pages, hashes)
    (out / 'source_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    _write_jsonl(out / 'research_report_gold20.jsonl', [c for c in revised if not c['id'].startswith('rr-regression')])
    _write_jsonl(out / 'research_report_regression4.jsonl', [c for c in revised if c['id'].startswith('rr-regression')])
    mapping = [{'case_id': case['id'], **ref} for case in revised for ref in case['evidence']]
    (out / 'evidence_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'cases': len(revised), 'evidence': len(mapping),
                      'unmapped': sum(e['location']['mapping_status'] == 'unmapped' for e in mapping)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
