from scripts.revise_research_evaluation import locate_quote, map_evidence, normalize


def test_complete_quote_keeps_numbers_percent_words_and_tags():
    text = '## 业绩\n\n2026H1 营收76.5亿元，同比+10.8%；净利同比-3.6%。\n\n其他内容'
    quote, start, end = locate_quote(text, ['76.5亿元', '+10.8%'])
    assert quote == '2026H1 营收76.5亿元，同比+10.8%；净利同比-3.6%。'
    assert text[start:end] == quote


def test_mapping_is_source_bound_and_reports_loss_without_dropping():
    evidence = {'evidence_id': 'stable', 'source': 'a.pdf', 'quote': '2026年营收10亿元。',
                'location': {'kind': 'pdf', 'original_quote': '2026年营收10亿元。'}}
    rows = [{'id': 'wrong', 'source': 'b.pdf', 'content': evidence['quote']}]
    result = map_evidence(evidence, rows)
    assert result['evidence_id'] == 'stable'
    assert result['chunk_id'] == ''
    assert result['location']['mapping_status'] == 'unmapped'


def test_mapping_preserves_pdf_identity_and_exact_local_offsets():
    evidence = {'evidence_id': 'stable', 'source': 'a.pdf', 'quote': '2026年营收10亿元。',
                'location': {'kind': 'pdf', 'original_quote': '2026年营收10亿元。'}}
    content = '## 业绩\n2026 年营收 10 亿元。'
    result = map_evidence(evidence, [{'id': 'new', 'source': 'a.pdf', 'content': content}])
    assert result['evidence_id'] == 'stable'
    assert result['chunk_id'] == 'new'
    assert content[result['quote_start']:result['quote_end']] == result['quote']
    assert normalize(result['quote']) == normalize(evidence['quote'])


def test_legacy_builder_never_clips_numeric_quote():
    from scripts.build_gold_set import _find_evidence
    text = '2026H1营业收入76.5亿元，同比+10.8%；净利润12.0亿元，同比-3.6%。'
    result = _find_evidence({'a.pdf': [{'id': 'a', 'content': text}]},
                            {'source_hint': 'a.pdf', 'quote_hint': '76.5'})
    assert result[1] == text


def test_automated_review_cannot_promote_itself_to_gold():
    from scripts.build_gold_set import automated_review
    review = automated_review('implementation-agent')
    assert review['tier'] == 'silver'
    assert review['review_status'] == 'auto_pass'
    assert review['reviewer'] == 'implementation-agent'


def test_cross_block_mapping_requires_both_fragments_and_keeps_parent_identity():
    from scripts.revise_research_evaluation import remap_case
    a, b = '甲公司2026年收入10亿元。', '乙项目利润2亿元。'
    ref = {'evidence_id': 'original', 'source': 'a.pdf', 'quote': a + b,
           'location': {'kind': 'pdf', 'original_quote': a + b, 'pdf_start': 0, 'pdf_end': len(a+b)}}
    case = {'evidence': [ref], 'facts': [{'acceptable_evidence_sets': [['original']]}]}
    revised = remap_case(case, [{'id': 'a', 'source': 'a.pdf', 'content': a},
                                {'id': 'b', 'source': 'a.pdf', 'content': b}])
    group = revised['facts'][0]['acceptable_evidence_sets'][0]
    assert len(group) == 2
    assert {r['chunk_id'] for r in revised['evidence']} == {'a', 'b'}
    assert all(r['location']['parent_evidence_id'] == 'original' for r in revised['evidence'])


def test_mapping_prefers_original_chunk_identity_over_equal_parent():
    ref = {'evidence_id': 'stable', 'source': 'a.pdf', 'quote': '收入10亿元。',
           'location': {'kind': 'pdf', 'original_quote': '收入10亿元。'}}
    rows = [{'id': 'aaa-parent', 'kind': 'parent', 'source': 'a.pdf', 'content': ref['quote']},
            {'id': 'zzz-child', 'source': 'a.pdf', 'content': ref['quote']}]
    result = map_evidence(ref, rows)
    assert result['chunk_id'] == 'zzz-child'


def test_pdf_soft_line_break_is_not_a_sentence_boundary():
    text = '公司2026H1实现\n营业收入10亿元，同比增加。'
    quote, _, _ = locate_quote(text, ['营业收入10亿元'])
    assert quote == text


def _table_remap_case():
    quote = '| 年份 | 营收（亿元） | 利润（亿元） |\n| --- | --- | --- |\n| 2026 | 10 | 2 |\n| 2025 | 8 | A&B |'
    return {'evidence': [{'evidence_id': 'stable-table', 'source': 'a.pdf', 'document_sha256': 'pdf-sha',
                          'quote': quote, 'chunk_id': '', 'location': {'kind': 'table', 'anchor_id': 'old-table',
                          'original_quote': 'PDF年份营收原文', 'pdf_start': 20, 'pdf_end': 30}}],
            'facts': [{'id': 'revenue', 'statement': '2026年营收10亿元', 'allowed_sources': ['a.pdf'],
                       'aliases': [['年份', '营收（亿元）', '2026', '10']],
                       'acceptable_evidence_sets': [['stable-table']]}]}


def test_table_remap_accepts_only_complete_target_prefix_and_preserves_identity():
    import hashlib
    from scripts.revise_research_evaluation import remap_case
    case = _table_remap_case()
    old = case['evidence'][0]
    content = old['quote'].replace('A&B', 'A&amp;B')
    revised = remap_case(case, [{'kind': 'table', 'id': 'new-table', 'source': 'a.pdf', 'content': content}])
    ref = revised['evidence'][0]
    assert ref['location']['mapping_status'] == 'mapped'
    assert ref['quote'] == '| 年份 | 营收（亿元） | 利润（亿元） |\n| --- | --- | --- |\n| 2026 | 10 | 2 |'
    assert content[ref['quote_start']:ref['quote_end']] == ref['quote']
    assert ref['evidence_id'] == 'stable-table'
    assert ref['document_sha256'] == 'pdf-sha'
    assert ref['location']['anchor_id'] == 'new-table'
    assert ref['location']['original_quote'] == old['location']['original_quote']
    assert ref['location']['pdf_start'] == 20 and ref['location']['pdf_end'] == 30
    assert ref['location']['mapping_reason'] == 'table_rendering_changed_target_rows_unchanged'
    assert ref['location']['previous_quote_sha256'] == hashlib.sha256(old['quote'].encode('utf8')).hexdigest()
    assert revised['facts'] == case['facts']


def test_table_remap_rejects_changed_target_columns_wrong_source_and_missing_fact():
    from scripts.revise_research_evaluation import remap_case
    for source, replacement in [('a.pdf', '| 2026 | 2 | 10 |'), ('b.pdf', '| 2026 | 10 | 2 |'),
                                ('a.pdf', '| 2026 | 110 | 2 |')]:
        case = _table_remap_case()
        content = case['evidence'][0]['quote'].replace('| 2026 | 10 | 2 |', replacement).replace('A&B', 'A&amp;B')
        revised = remap_case(case, [{'kind': 'table', 'id': 'new-table', 'source': source, 'content': content}])
        assert revised['evidence'][0]['location']['mapping_status'] == 'unmapped'
        assert revised['facts'] == case['facts']


def test_table_remap_cannot_shorten_and_group_or_unannotated_alias():
    from scripts.revise_research_evaluation import remap_case
    for mutation in ('and', 'missing_alias'):
        case = _table_remap_case()
        if mutation == 'and':
            case['facts'][0]['acceptable_evidence_sets'] = [['stable-table', 'other']]
            case['evidence'].append({'evidence_id': 'other', 'source': 'a.pdf', 'quote': '另一证据',
                                     'location': {'kind': 'chunk'}})
        else:
            case['facts'][0]['aliases'] = [['2026', '99']]
        content = case['evidence'][0]['quote'].replace('A&B', 'A&amp;B')
        revised = remap_case(case, [{'kind': 'table', 'id': 'new-table', 'source': 'a.pdf', 'content': content}])
        assert revised['evidence'][0]['location']['mapping_status'] == 'unmapped'
        assert revised['facts'] == case['facts']
