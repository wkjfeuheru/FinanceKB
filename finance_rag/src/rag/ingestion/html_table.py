"""有界 HTML 表格布局还原：失败时保留原文，合并格只存一次文字。"""

from dataclasses import dataclass
from html.parser import HTMLParser
import logging

logger = logging.getLogger(__name__)
MAX_GRID_CELLS = 100_000
MAX_COLUMNS = 1_000
MAX_ROWS = 10_000


@dataclass(frozen=True)
class TableLayout:
    rows: list[list[str]]
    merges: tuple[str, ...] = ()
    header_rows: int = 0
    error: str = ''


class _Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None
        self.text = []
        self.depth = 0
        self.cell_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.depth += 1
            if self.depth > 1:
                raise ValueError('nested table')
        elif tag == 'tr':
            if self.row is not None:
                raise ValueError('unclosed row')
            self.row = []
        elif tag in ('td', 'th'):
            if self.row is None or self.cell is not None:
                raise ValueError('cell outside row or unclosed cell')
            attributes = dict(attrs)
            spans = []
            for name, maximum in (('rowspan', MAX_ROWS), ('colspan', MAX_COLUMNS)):
                raw = attributes.get(name, '1')
                if not raw or not raw.isascii() or not raw.isdigit() or len(raw) > 8:
                    raise ValueError('invalid ' + name)
                span = int(raw)
                if not 1 <= span <= maximum:
                    raise ValueError('oversized or invalid ' + name)
                spans.append(span)
            self.cell = (tag, *spans)
            self.text = []
        elif self.cell is not None and tag in ('br', 'p', 'div'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            if self.cell is None or self.cell[0] != tag:
                raise ValueError('mismatched cell close')
            self.row.append((*self.cell, ''.join(self.text).strip()))
            self.cell_count += 1
            if self.cell_count > MAX_GRID_CELLS:
                raise ValueError('too many cells in table')
            if len(self.row) > MAX_COLUMNS:
                raise ValueError('too many cells in row')
            self.cell = None
        elif tag == 'tr':
            if self.row is None or self.cell is not None:
                raise ValueError('invalid row close')
            self.rows.append(self.row)
            if len(self.rows) > MAX_ROWS:
                raise ValueError('too many rows')
            self.row = None
        elif tag == 'table':
            if self.row is not None or self.cell is not None:
                raise ValueError('unclosed row/cell')
            self.depth -= 1
        elif self.cell is not None and tag in ('p', 'div'):
            self.text.append('\n')

    def handle_data(self, data):
        if self.cell is not None:
            self.text.append(data)


def restore_html_table(html: str) -> TableLayout:
    try:
        parser = _Parser()
        parser.feed(html)
        parser.close()
        if parser.depth or parser.row is not None or parser.cell is not None:
            raise ValueError('incomplete table')
        occupied = {}
        merges = []
        width = 0
        header_rows = 0
        for r, row in enumerate(parser.rows):
            if row and all(cell[0] == 'th' for cell in row) and header_rows == r:
                header_rows += 1
            c = 0
            for _, rowspan, colspan, value in row:
                while (r, c) in occupied:
                    c += 1
                if r + rowspan > len(parser.rows):
                    raise ValueError('rowspan extends past table')
                if c + colspan > MAX_COLUMNS or (c + colspan) * len(parser.rows) > MAX_GRID_CELLS:
                    raise ValueError('grid limit exceeded')
                for rr in range(r, r + rowspan):
                    for cc in range(c, c + colspan):
                        if (rr, cc) in occupied:
                            raise ValueError('span collision')
                        occupied[rr, cc] = value if (rr, cc) == (r, c) else ''
                if rowspan > 1 or colspan > 1:
                    merges.append(f'R{r+1}C{c+1}:R{r+rowspan}C{c+colspan}')
                width = max(width, c + colspan)
                c += colspan
        rows = [[occupied.get((r, c), '') for c in range(width)] for r in range(len(parser.rows))]
        return TableLayout(rows, tuple(merges), header_rows)
    except (ValueError, TypeError) as exc:
        logger.warning('HTML 表格还原失败，保留原 HTML：%s', exc)
        return TableLayout([], error=str(exc))
