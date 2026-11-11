export function coverageWarnings(msg) {
  return (msg?.warnings || []).filter((item) => typeof item === 'string' && item.startsWith('覆盖评估'))
}

export function answerStatusLabel(msg) {
  if (msg.answerStatus === 'unverified' || msg.citationValidation?.status === 'unknown') {
    return '无法验证：未发布事实答案'
  }
  if (msg.answerStatus === 'partial') return '部分回答：存在未覆盖事项'
  if (msg.answerStatus === 'refused') return '已拒答：证据不足或核验失败'
  if (msg.answerStatus === 'clarification') return '需要澄清问题范围'
  if (msg.citationValidation?.status === 'verified') return '事实已核验'
  return ''
}
