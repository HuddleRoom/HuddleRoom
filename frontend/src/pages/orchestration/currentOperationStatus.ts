import { parseUtc } from '@/lib/time'
import type { OrchestrationStatusLine } from '@/lib/types'

export type StatusOccurrence = { key: string; occurred_at: string | null }
export type CurrentOperationAnnouncementCandidate = StatusOccurrence & { source: 'attention' | 'event'; text: string }

function timestamp(value: string | null): number | null {
  if (!value) return null
  const time = parseUtc(value).getTime()
  return Number.isFinite(time) ? time : null
}

export function compareStatusOccurrences(left: StatusOccurrence, right: StatusOccurrence): number {
  const leftTime = timestamp(left.occurred_at)
  const rightTime = timestamp(right.occurred_at)
  if (leftTime === null && rightTime !== null) return 1
  if (rightTime === null && leftTime !== null) return -1
  if (leftTime !== null && rightTime !== null && leftTime !== rightTime) return rightTime - leftTime
  return left.key < right.key ? -1 : left.key > right.key ? 1 : 0
}

export function currentOperationAnnouncementCandidates(statusLine: OrchestrationStatusLine): readonly CurrentOperationAnnouncementCandidate[] {
  const candidates: CurrentOperationAnnouncementCandidate[] = []
  const attention = statusLine.attention_summary
  if (attention && attention.kind !== 'needs_you') {
    candidates.push({ key: attention.key, occurred_at: attention.occurred_at, source: 'attention', text: attention.summary })
  }
  const event = statusLine.last_material_event
  if (event && !['human_question', 'human_answer', 'tick', 'heartbeat'].includes(event.kind)
    && timestamp(event.occurred_at) !== null) {
    candidates.push({ key: event.key, occurred_at: event.occurred_at, source: 'event', text: event.summary })
  }
  return candidates.sort(compareStatusOccurrences)
}
