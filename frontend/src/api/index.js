import axios from 'axios'

import { consumeEventStream } from './sse.js'

const TOKEN_KEY = 'finance_rag_token'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

const http = axios.create({
  baseURL: '/api',
  timeout: 60000
})

// 请求拦截器：自动附加 Authorization header
http.interceptors.request.use(config => {
  const token = getToken()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// uvicorn 热重载时，幂等 GET 请求对临时网络错误或 5xx 自动重试。
http.interceptors.response.use(
  response => response,
  async error => {
    const config = error.config
    const status = error.response?.status

    // 401：清 token 并跳登录页
    if (status === 401) {
      clearToken()
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
      return Promise.reject(error)
    }

    const retryable = config?.method?.toLowerCase() === 'get' &&
      (!error.response || (status >= 500 && status < 600))
    config.__retryCount = config.__retryCount || 0
    if (retryable && config.__retryCount < 2) {
      config.__retryCount += 1
      await new Promise(resolve => setTimeout(resolve, config.__retryCount * 600))
      return http(config)
    }
    return Promise.reject(error)
  }
)

// ---------------------------------------------------------------------------
// 健康检查
// ---------------------------------------------------------------------------

export const getHealth = () => http.get('/health').then(r => r.data)

// ---------------------------------------------------------------------------
// 鉴权
// ---------------------------------------------------------------------------

export const login = (username, password) =>
  http.post('/auth/login', { username, password }).then(r => {
    setToken(r.data.access_token)
    return r.data
  })

// ---------------------------------------------------------------------------
// 文档管理
// ---------------------------------------------------------------------------

export const listDocuments = (kb = '') => http.get('/documents', { params: { kb } }).then(r => r.data)
export const deleteDocument = (source, kb = '') => http.delete(`/documents/${encodeURIComponent(source)}`, { params: { kb } }).then(r => r.data)

// 上传文档（统一异步：返回 task_id 列表，需轮询 /tasks/{task_id} 获取结果）
export const uploadDocuments = (files, onProgress, kb = '') => {
  const form = new FormData()
  files.forEach(file => form.append('files', file))
  if (kb) form.append('kb', kb)
  return http.post('/documents/upload', form, {
    timeout: 300000,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100))
    }
  }).then(r => r.data)
}

export const getTaskStatus = (taskId) => http.get(`/tasks/${taskId}`).then(r => r.data)

export const getKbStats = (kb = '') => http.get('/kb/stats', { params: { kb } }).then(r => r.data)

// 研报元数据人工修正（不重新嵌入）。返回 422 时 detail 为中文说明。
export const patchDocumentMetadata = (source, payload, kb = '') =>
  http.patch(`/documents/${encodeURIComponent(source)}/metadata`, payload, {
    params: { kb }
  }).then(r => r.data)

// 原文页渲染：接口需要 JWT，<img src> 带不上 Authorization，
// 因此这里取回二进制再转 objectURL（调用方负责 revoke）。
export const fetchDocumentPage = async (source, page) => {
  const response = await http.get(
    `/documents/${encodeURIComponent(source)}/page/${page}`,
    { responseType: 'blob' }
  )
  return URL.createObjectURL(response.data)
}

// ---------------------------------------------------------------------------
// 非流式问答
// ---------------------------------------------------------------------------

export const chat = (payload) => http.post('/chat', payload).then(r => r.data)

// ---------------------------------------------------------------------------
// 流式问答（SSE）
// 接收回调 onEvent(event)，返回 AbortController 用于中止。
//
// handlers 支持两种写法：
//   chatStream(payload, onEvent, (err) => {...})                    // 旧签名
//   chatStream(payload, onEvent, { onError, onFinish })             // 推荐
//
// onFinish({ status, error }) 是**唯一**的终态收尾入口，每次请求只调用一次：
//   status = 'done'            收到 done 事件
//          | 'error'           收到 error 事件（正文已由 onEvent 呈现）
//          | 'transport-error' 网络/HTTP/读取中断等异常
//          | 'aborted'         用户主动取消
// ---------------------------------------------------------------------------

export function chatStream(payload, onEvent, handlers = {}) {
  const { onError, onFinish } =
    typeof handlers === 'function' ? { onError: handlers } : handlers || {}
  const controller = new AbortController()
  const token = getToken()
  let settled = false

  const settle = (status, error = null) => {
    if (settled) return
    settled = true
    if (status === 'transport-error' && onError) onError(error)
    if (onFinish) onFinish({ status, error })
  }

  const run = async () => {
    let response
    try {
      response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {})
        },
        body: JSON.stringify(payload),
        signal: controller.signal
      })
    } catch (err) {
      if (err?.name === 'AbortError') settle('aborted')
      else settle('transport-error', err)
      return
    }

    if (!response.ok) {
      // 与 axios 拦截器保持一致：401 清 token 并跳登录页
      if (response.status === 401) {
        clearToken()
        if (window.location.pathname !== '/login') {
          window.location.href = '/login'
        }
      }
      // 解析后端返回的友好 detail 信息，优先展示给用户
      let detail = ''
      try {
        const body = await response.json()
        detail = body.detail || ''
      } catch { /* 响应体非 JSON 时忽略 */ }
      settle('transport-error', new Error(detail || `HTTP ${response.status}`))
      return
    }

    try {
      const { terminal } = await consumeEventStream(response, { onEvent })
      // 协议级 error 事件的正文已由 onEvent 呈现，这里只做终态收尾
      settle(terminal === 'error' ? 'error' : 'done')
    } catch (err) {
      if (err?.name === 'AbortError') settle('aborted')
      else settle('transport-error', err)
    }
  }

  run()

  return controller
}


