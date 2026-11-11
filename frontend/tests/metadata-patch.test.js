import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'

// 执行视图实际保存函数，隔离网络请求及消息展示；断言送出的 PATCH。
async function savedPayload(form) {
  const source = await readFile(new URL('../src/views/DocumentsView.vue', import.meta.url), 'utf8')
  const saveFunction = source.match(/async function saveMetadata\(\) \{[\s\S]*?\r?\n\}/)[0]
  let builder
  try {
    builder = (await import('../src/utils/metadata.js')).buildMetadataPatch
  } catch (error) {
    if (error.code !== 'ERR_MODULE_NOT_FOUND') throw error
  }
  let sent
  const context = {
    metadataDialog: { form, source: '研报.pdf', saving: false },
    patchDocumentMetadata: async (_, payload) => { sent = payload },
    loadDocuments: async () => {},
    ElMessage: { success() {}, error(message) { throw new Error(message) } },
    buildMetadataPatch: builder,
  }
  await vm.runInNewContext(`${saveFunction}\nsaveMetadata()`, context)
  return JSON.parse(JSON.stringify(sent))
}

test('编辑时回填已有发布日期，缺失日期保持空白', async () => {
  const source = await readFile(new URL('../src/views/DocumentsView.vue', import.meta.url), 'utf8')
  const openFunction = source.match(/function openMetadataDialog\(row\) \{[\s\S]*?\r?\n\}/)[0]
  const metadataDialog = {}
  const context = { metadataDialog }
  vm.runInNewContext(`${openFunction}\nopenMetadataDialog({source: '报告.pdf', date: '2026-09-28'})`, context)
  assert.equal(metadataDialog.form.date, '2026-09-28')
  vm.runInNewContext("openMetadataDialog({source: '报告.pdf'})", context)
  assert.equal(metadataDialog.form.date, '')
})

test('保存时提交全部核心字段，空日期清空', async () => {
  const payload = await savedPayload({ category: 'investment_research', version: 'v2', report_type: '个股',
    broker: '中信证券', date: '2026-09-28' })
  assert.deepEqual(payload, { category: 'investment_research', version: 'v2', report_type: '个股',
    broker: '中信证券', date: '2026-09-28' })
  const empty = await savedPayload({ category: '', version: '', report_type: '', broker: '', date: '' })
  assert.deepEqual(empty, { category: '', version: '', report_type: '', broker: '', date: '' })
})
