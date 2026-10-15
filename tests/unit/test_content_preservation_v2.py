import pytest

from finance_rag.src.rag.ingestion.cleaner import TextCleaner, deduplicate_paragraphs
from finance_rag.src.rag.ingestion.chunker import _table_payload, _table_markdown, _table_index_text
from finance_rag.src.rag.retrieval.hybrid_retriever import HybridRetriever


def test_clean_preserves_risks_numbers_repeated_short_lines():
    text = '风险提示：需求下降\n风险揭示：可能亏损\n123\n3 / 42\n营业收入\n营业收入'
    assert TextCleaner().clean(text) == text


def test_clean_audit_records_removed_original_and_rule():
    result = TextCleaner().clean_with_audit('正文\n第 3 页\n版权所有公司\nhttps://example.com')
    assert result.text == '正文'
    assert {entry.text for entry in result.removals} == {'第 3 页', '版权所有公司', 'https://example.com'}
    assert all(entry.rule for entry in result.removals)


def test_similar_paragraphs_only_diagnosed(caplog):
    para = '公司预计营业收入持续增长，主要产品需求稳定且市场份额不断提升。' * 3
    text = para + '\n\n' + para.replace('，', ',')
    with caplog.at_level('INFO'):
        assert deduplicate_paragraphs(text) == text
    assert '疑似重复' in caplog.text


def test_table_retains_empty_positions_and_nested_spaces():
    html = '<table><tr><th>指标</th><th></th><th>2025</th><th></th></tr><tr><td>净<b>利</b>润</td><td></td><td>1 23</td><td></td></tr></table>'
    assert _table_payload(html) == [['指标', '', '2025', ''], ['净利润', '', '1 23', '']]


def test_table_spans_do_not_duplicate_numbers():
    html = '<table><tr><th rowspan="2">指标</th><th colspan="2">年度</th></tr><tr><th>2024</th><th>2025</th></tr><tr><td>收入</td><td colspan="2">100</td></tr></table>'
    assert _table_payload(html) == [['指标', '年度', ''], ['', '2024', '2025'], ['收入', '100', '']]
    md = _table_markdown(html)
    assert md.count('100') == 1
    assert 'R1C1:R2C1' in md and 'R1C2:R1C3' in md


def test_table_escapes_and_roundtrips():
    html = '<table><tr><td>A|B<br>C</td><td></td></tr></table>'
    md = _table_markdown(html)
    assert 'A\\|B<br>C' in md
    assert _table_payload(md) == [['A|B\nC', '']]


def test_literal_backslash_before_column_boundary_roundtrips():
    html = '<table><tr><td>A\\</td><td>B\\|C</td></tr></table>'
    assert _table_payload(_table_markdown(html)) == [['A\\', 'B\\|C']]


def test_single_column_html_roundtrips_without_separator_as_data():
    html = '<table><tr><th>指标</th></tr><tr><td>100</td></tr></table>'
    assert _table_payload(_table_markdown(html)) == [['指标'], ['100']]


def test_literal_encoded_br_is_distinct_from_real_linebreak():
    html = '<table><tr><td>&lt;br&gt; &amp; &#124;<br>尾行 &lt;span&gt;</td></tr></table>'
    assert _table_payload(_table_markdown(html)) == [['<br> & |\n尾行 <span>']]


def test_all_html_header_rows_enter_index():
    html = '<table><tr><th rowspan="2">指标</th><th colspan="2">年度</th></tr><tr><th>2024</th><th>2025</th></tr><tr><td>收入</td><td>10</td><td>20</td></tr></table>'
    index = _table_index_text(html)
    assert '2024' in index and '2025' in index and '年度' in index
    assert '收入' in index


@pytest.mark.parametrize('html', [
    '<table><tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">C</td></tr></table>',
    '<table><tr><td colspan="999999999">100</td></tr></table>',
    '<table><tr><td rowspan="x">A</td></tr></table>',
])
def test_invalid_table_retains_raw_html_and_logs(html, caplog):
    with caplog.at_level('WARNING'):
        assert _table_markdown(html) == html
    assert caplog.text


def expand(matches, parent=None, failure=False):
    class Store:
        def get_batch(self, ids):
            self.ids = ids
            if failure:
                raise RuntimeError('unavailable')
            return [parent] if parent else []
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._parent_store = Store()
    return retriever._expand_to_parents(matches, 100), retriever


def children():
    return [dict(id='c1', parent_id='p', content='首段', block_type='text'),
            dict(id='c2', parent_id='p', content='末段', block_type='text')]


@pytest.mark.parametrize('content', ['短', '长' * 3001, '无关正文' * 30, '首段' + '正文' * 30], ids=['short', 'long', 'unrelated', 'partial'])
def test_parent_fallback_preserves_all_distinct_children(content):
    result, _ = expand(children(), dict(id='p', content=content))
    assert [row['id'] for row in result] == ['c1', 'c2']
    assert [row['content'] for row in result] == ['首段', '末段']


@pytest.mark.parametrize('failure', [False, True])
def test_missing_failed_parent_preserves_all_children(failure):
    result, _ = expand(children(), failure=failure)
    assert [row['id'] for row in result] == ['c1', 'c2']


def test_valid_parent_covers_every_child_and_diagnostics_accumulate():
    body = '首段' + '正文' * 30 + '末段'
    result, retriever = expand(children(), dict(id='p', content=body, heading='标题'))
    assert len(result) == 1 and result[0]['content'] == body
    assert result[0]['covered_child_ids'] == ['c1', 'c2']
    assert result[0]['child_id'] == 'c1' and result[0]['heading'] == '标题'
    retriever._expand_to_parents(children(), 100)
    assert retriever._last_parent_expansion['input'] == 4
    assert retriever._last_parent_expansion['output'] == 2


def test_nontext_parent_not_queried_and_only_chunk_ids_deduplicated():
    rows = [dict(id='table1', parent_id='t', content='表', block_type='table'),
            dict(id='image1', parent_id='i', content='图', block_type='image')]
    result, retriever = expand(rows + [rows[0].copy()])
    assert [r['id'] for r in result] == ['table1', 'image1']
    assert not hasattr(retriever._parent_store, 'ids')
