import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import vm from 'node:vm'

// 执行视图实际保存函数，隔离网络请求及消息展示；断言送出的 PATCH。
async function savedPayload(form) {
  const source = await readFile(new URL('../src/views/DocumentsView.vue', import.meta.url), 'utf8')
  const saveFunction = source.match(/async function saveMetadata\(\) \{[\s\S]*?\n\}\n/)[0]
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

for (const [type, expected] of [
  ['个股', { report_type: '个股', broker: '中信证券', security_name: '贵州茅台' }],
  ['行业', { report_type: '行业', broker: '中信证券', industry_l1: '电子', industry_l2: '半导体' }],
  ['宏观', { report_type: '宏观', broker: '中信证券' }],
]) {
  test(`保存 ${type} 时仅提交适用类型字段`, async () => {
    const payload = await savedPayload({ report_type: type, broker: '中信证券',
      security_name: '贵州茅台', industry_l1: '电子', industry_l2: '半导体' })
    assert.deepEqual(payload, expected)
  })
}
