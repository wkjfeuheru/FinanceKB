import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'

async function visibleFilterChips(filters) {
  const source = await readFile(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')
  // 源文件是 CRLF 行尾，正则必须容忍 \r
  const filterChips = source.match(/function filterChips\(filters\) \{[\s\S]*?\r?\n\}\r?\n/)[0]
  const result = vm.runInNewContext(`${filterChips}\nfilterChips(filters)`, {
    filters,
    categoryLabel: (value) => value,
  })
  return JSON.parse(JSON.stringify(result))
}

test('已确认的发布日期集合在过滤标签中逐日可见', async () => {
  const chips = await visibleFilterChips({ date: ['2026-09-24', '2026-09-27'] })
  assert.deepEqual(chips, [
    { key: 'date', text: '日期：2026-09-24/2026-09-27' },
  ])
})

test('原有发布日期上下界标签保持可见', async () => {
  const chips = await visibleFilterChips({ date: { gte: '2026-09-01', lte: '2026-09-30' } })
  assert.deepEqual(chips, [
    { key: 'date_gte', text: '不早于 2026-09-01' },
    { key: 'date_lte', text: '不晚于 2026-09-30' },
  ])
})
