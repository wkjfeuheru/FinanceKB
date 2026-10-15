import assert from 'node:assert/strict'
import test from 'node:test'
import { answerStatusLabel } from '../src/utils/answer.js'

test('未知校验不能显示通过', () => {
  assert.equal(answerStatusLabel({ answerStatus: 'unverified', citationValidation: { score: null, status: 'unknown' } }), '无法验证：未发布事实答案')
})
test('部分回答和拒答有不同状态', () => {
  assert.equal(answerStatusLabel({ answerStatus: 'partial' }), '部分回答：存在未覆盖事项')
  assert.equal(answerStatusLabel({ answerStatus: 'refused' }), '已拒答：证据不足或核验失败')
})
test('仅显式verified显示已核验', () => {
  assert.equal(answerStatusLabel({ citationValidation: { status: 'verified' } }), '事实已核验')
  assert.equal(answerStatusLabel({ citationValidation: { status: 'not_required' } }), '')
})
