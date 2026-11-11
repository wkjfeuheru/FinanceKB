/**
 * SSE 事件流消费（与业务解耦，便于离线验证）。
 *
 * 背景（docs/rag-review.md RAG-14）：原先 EOF 只退出读取循环，不检查是否收到
 * `done`/`error`；而界面只靠这两个事件解除「回答中」。HTTP 200 之后连接提前
 * 结束时，发送按钮会一直禁用，用户无法恢复。
 *
 * 这里把「读取事件 → 记录协议终止状态 → 缺少终止事件即报中断」收敛到一处，
 * 并保证异常路径下读者被取消，避免悬挂连接。
 */

export function parseSSEBlock(block) {
  let eventType = 'message'
  let dataStr = ''
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      // 保留 data 内容中的空格，只去除 "data:" 前缀后的一个可选空格
      const raw = line.slice(5)
      dataStr += raw.startsWith(' ') ? raw.slice(1) : raw
    }
  }
  if (!dataStr) return null
  try {
    const data = JSON.parse(dataStr)
    return { type: eventType, ...data }
  } catch {
    // 截断/非法 JSON 不抛出：保留原始文本，交给上层按事件类型处理
    return { type: eventType, data: dataStr }
  }
}

/** 流式事件消费结果：`terminal` 为 'done' 或 'error'（协议级终止事件）。 */
export async function consumeEventStream(response, { onEvent } = {}) {
  if (!response?.body || typeof response.body.getReader !== 'function') {
    throw new Error('响应不支持流式读取，请检查服务是否可用')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let terminal = null

  const emit = (block) => {
    const event = parseSSEBlock(block)
    if (!event) return
    if (event.type === 'done' || event.type === 'error') terminal = event.type
    if (onEvent) onEvent(event)
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // 事件以空行分隔，兼容 \n\n 与 \r\n\r\n；buffer 保留未完成分片
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() || ''
      for (const block of blocks) emit(block)
    }
    buffer += decoder.decode()
    if (buffer.trim()) emit(buffer)
  } finally {
    // 只读到最后仍未拿到终止事件（或读取异常）：显式取消，避免连接悬挂
    if (terminal === null && typeof reader.cancel === 'function') {
      try {
        await reader.cancel()
      } catch {
        /* 取消失败不影响中断判定 */
      }
    }
  }

  if (terminal === null) {
    throw new Error('回答连接中断，请重试')
  }
  return { terminal }
}
