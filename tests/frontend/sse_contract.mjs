/**
 * SSE 消费与终态收尾的离线验证脚本（前端没有测试框架，这里用 Node 直接跑）。
 *
 * 覆盖 docs/rag-review.md RAG-14 的验收项：
 * 正常完成 / error 事件 / 缺少 done 的 EOF / 读取异常 / 主动取消；
 * 截断 JSON、跨网络分片的事件、CRLF 分隔符。
 *
 * 用法：node tests/frontend/sse_contract.mjs
 */

import assert from 'node:assert/strict'

import { consumeEventStream, parseSSEBlock } from '../../frontend/src/api/sse.js'

function streamOf(chunks) {
  const encoder = new TextEncoder()
  let index = 0
  return {
    body: {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) return { done: true, value: undefined }
            const value = chunks[index++]
            return { done: false, value: encoder.encode(value) }
          },
          async cancel() {
            this.cancelled = true
          },
          cancelled: false
        }
      }
    }
  }
}

function failingStream(chunks, error) {
  const encoder = new TextEncoder()
  let index = 0
  return {
    body: {
      getReader() {
        return {
          async read() {
            if (index >= chunks.length) throw error
            return { done: false, value: encoder.encode(chunks[index++]) }
          },
          async cancel() {}
        }
      }
    }
  }
}

const results = []
function check(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => results.push(['PASS', name]))
    .catch((error) => results.push(['FAIL', name, error.message]))
}

await check('正常完成：done 事件是终止信号', async () => {
  const events = []
  const response = streamOf([
    'event: sources\ndata: {"sources":[]}\n\n',
    'event: token\ndata: {"content":"你好"}\n\n',
    'event: done\ndata: {"answer_status":"complete"}\n\n'
  ])
  const { terminal } = await consumeEventStream(response, { onEvent: (e) => events.push(e.type) })
  assert.equal(terminal, 'done')
  assert.deepEqual(events, ['sources', 'token', 'done'])
})

await check('error 事件同样终止，不误判为中断', async () => {
  const response = streamOf(['event: error\ndata: {"message":"模型不可用"}\n\n'])
  const { terminal } = await consumeEventStream(response)
  assert.equal(terminal, 'error')
})

await check('缺少 done 的 EOF 必须报中断', async () => {
  const response = streamOf([
    'event: token\ndata: {"content":"半句"}\n\n'
  ])
  await assert.rejects(
    () => consumeEventStream(response),
    /回答连接中断/
  )
})

await check('读取异常向上抛出（由调用方收尾）', async () => {
  const response = failingStream(['event: token\ndata: {"content":"a"}\n\n'], new Error('network reset'))
  await assert.rejects(() => consumeEventStream(response), /network reset/)
})

await check('跨分片事件与 CRLF 分隔符都能解析', async () => {
  const events = []
  const response = streamOf([
    'event: tok',
    'en\r\ndata: {"con',
    'tent":"跨分片的回答"}\r\n\r\n',
    'event: done\r\ndata: {}\r\n\r\n'
  ])
  const { terminal } = await consumeEventStream(response, { onEvent: (e) => events.push(e) })
  assert.equal(terminal, 'done')
  assert.equal(events[0].content, '跨分片的回答')
  assert.equal(events[0].type, 'token')
})

await check('截断 JSON 不抛出，保留原始文本', async () => {
  const block = 'event: token\ndata: {"content":"未闭合'
  const event = parseSSEBlock(block)
  assert.equal(event.type, 'token')
  assert.equal(event.data, '{"content":"未闭合')
})

await check('无 data 的块不产生事件', async () => {
  assert.equal(parseSSEBlock('event: status\n'), null)
})

// --- chatStream 的终态收尾语义（stub fetch） ---

const { chatStream } = await import('../../frontend/src/api/index.js')

globalThis.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {}
}
globalThis.window = { location: { pathname: '/' } }

function stubFetch(handler) {
  globalThis.fetch = handler
}

async function settleOf(handler) {
  const finishes = []
  const errors = []
  stubFetch(handler)
  chatStream({ query: '测试' }, () => {}, {
    onError: (err) => errors.push(err),
    onFinish: (payload) => finishes.push(payload)
  })
  await new Promise((resolve) => setTimeout(resolve, 20))
  return { finishes, errors }
}

await check('chatStream：done 只收尾一次', async () => {
  const { finishes } = await settleOf(async () => ({
    ok: true,
    body: streamOf(['event: done\ndata: {}\n\n']).body
  }))
  assert.deepEqual(finishes.map((f) => f.status), ['done'])
})

await check('chatStream：EOF 缺少 done → transport-error 且只上报一次', async () => {
  const { finishes, errors } = await settleOf(async () => ({
    ok: true,
    body: streamOf(['event: token\ndata: {"content":"x"}\n\n']).body
  }))
  assert.deepEqual(finishes.map((f) => f.status), ['transport-error'])
  assert.equal(errors.length, 1)
  assert.match(errors[0].message, /中断/)
})

await check('chatStream：协议 error 事件不触发 onError（避免重复报错）', async () => {
  const { finishes, errors } = await settleOf(async () => ({
    ok: true,
    body: streamOf(['event: error\ndata: {"message":"模型不可用"}\n\n']).body
  }))
  assert.deepEqual(finishes.map((f) => f.status), ['error'])
  assert.equal(errors.length, 0)
})

await check('chatStream：HTTP 500 → transport-error 携带 detail', async () => {
  const { finishes, errors } = await settleOf(async () => ({
    ok: false,
    status: 500,
    json: async () => ({ detail: '入库队列已满' })
  }))
  assert.deepEqual(finishes.map((f) => f.status), ['transport-error'])
  assert.equal(errors[0].message, '入库队列已满')
})

await check('chatStream：主动取消 → aborted，不报网络错误', async () => {
  const failure = Object.assign(new Error('aborted'), { name: 'AbortError' })
  const { finishes, errors } = await settleOf(async () => { throw failure })
  assert.deepEqual(finishes.map((f) => f.status), ['aborted'])
  assert.equal(errors.length, 0)
})

let failed = 0
for (const row of results) {
  if (row[0] === 'FAIL') failed += 1
  console.log(`${row[0]}  ${row[1]}${row[2] ? `  → ${row[2]}` : ''}`)
}
console.log(`\n${results.length - failed}/${results.length} 通过`)
process.exitCode = failed ? 1 : 0
