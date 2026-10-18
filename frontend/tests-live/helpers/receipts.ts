import type { Page } from '@playwright/test'

export type ReceiptKind = 'pause-post' | 'pause-get' | 'answer-post' | 'decision-get'

export interface Receipt {
  /** Order the request was issued in (assigned at request time). */
  seq: number
  kind: ReceiptKind
  url: string
  method: string
  request: unknown
  /** Undefined until a response arrives; stays undefined for aborted/failed requests. */
  response?: unknown
  status?: number
}

// Matched on URL path (query ignored):
//   POST .../goals/{gid}/pause-control                  -> pause-post
//   GET  .../pause-control/{id}                         -> pause-get
//   POST .../goals/{gid}/decisions/{did}/answer         -> answer-post
//   POST .../goals/{gid}/decisions/agent-definition-review/batch-answer -> answer-post
//   GET  .../goals/{gid}/decisions/{did}                -> decision-get
// Answer response shape: { answer_receipt: { action_id, decision_ids, application_state }, ... }; a bare
// { action_id, decision_ids, application_state } body is accepted too.
const MATCHERS: Array<[ReceiptKind, string, RegExp]> = [
  ['pause-post', 'POST', /\/goals\/[^/]+\/pause-control$/],
  ['pause-get', 'GET', /\/pause-control\/[^/]+$/],
  ['answer-post', 'POST', /\/goals\/[^/]+\/decisions\/(?:[^/]+\/answer|agent-definition-review\/batch-answer)$/],
  ['decision-get', 'GET', /\/goals\/[^/]+\/decisions\/[^/]+$/],
]

const parse = (text: string | null | undefined): unknown => {
  if (!text) return null
  try { return JSON.parse(text) } catch { return text }
}

export interface Receipts {
  all: Receipt[]
  /** Requests issued, counted at request time (so aborted/dropped requests are included). */
  postCount: (kind: ReceiptKind) => number
  /** Latest answer POST that received a response; null if none. */
  lastAnswerReceipt: () => {
    action_id: unknown
    decision_ids: unknown
    application_state: unknown
    urlDecisionId: string | null
  } | null
}

export function captureReceipts(page: Page): Receipts {
  const all: Receipt[] = []
  const byRequest = new Map<unknown, Receipt>()
  page.on('request', (req) => {
    const path = new URL(req.url()).pathname
    const hit = MATCHERS.find(([, m, re]) => req.method() === m && re.test(path))
    if (!hit) return
    const entry: Receipt = { seq: all.length, kind: hit[0], url: req.url(), method: req.method(), request: parse(req.postData()) }
    all.push(entry)
    byRequest.set(req, entry)
  })
  page.on('response', async (res) => {
    const entry = byRequest.get(res.request())
    if (!entry) return
    entry.status = res.status()
    entry.response = parse(await res.text().catch(() => null))
  })
  return {
    all,
    postCount: (kind) => all.filter((r) => r.kind === kind).length,
    lastAnswerReceipt: () => {
      const last = [...all].reverse().find((r) => r.kind === 'answer-post' && r.response !== undefined)
      if (!last) return null
      const top = (last.response ?? {}) as Record<string, unknown>
      const body = (top.answer_receipt ?? top) as Record<string, unknown>
      return {
        action_id: body.action_id,
        decision_ids: body.decision_ids,
        application_state: body.application_state,
        urlDecisionId: new URL(last.url).pathname.match(/\/decisions\/([^/]+)\/answer$/)?.[1] ?? null,
      }
    },
  }
}
