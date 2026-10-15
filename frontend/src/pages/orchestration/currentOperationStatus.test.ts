import { describe, expect, it } from 'vitest'
import { compareStatusOccurrences, currentOperationAnnouncementCandidates } from './currentOperationStatus'
import type { OrchestrationStatusLine } from '@/lib/types'

const line = (kind = 'accepted_evidence'): OrchestrationStatusLine => ({
  current_state: { kind: 'unknown', summary: 'No active worker confirmed.', activities: [] },
  attention_summary: null,
  last_material_event: { key: 'event:1', kind, summary: 'Verification accepted.', subject: null, occurred_at: '2026-10-14T12:00:00Z' },
})

describe('current operation pure selection', () => {
  it('normalizes UTC, orders newest first and breaks ties by stable identity', () => {
    expect(compareStatusOccurrences({ key: 'b', occurred_at: '2026-10-14T12:00:00' }, { key: 'a', occurred_at: '2026-10-14T14:00:00+02:00' })).toBeGreaterThan(0)
    expect(compareStatusOccurrences({ key: 'older', occurred_at: '2026-10-13T12:00:00Z' }, { key: 'newer', occurred_at: '2026-10-14T12:00:00Z' })).toBeGreaterThan(0)
    expect(compareStatusOccurrences({ key: 'invalid', occurred_at: 'bad-date' }, { key: 'valid', occurred_at: '2026-10-14T12:00:00Z' })).toBeGreaterThan(0)
  })
  it.each(['human_question', 'human_answer', 'tick', 'heartbeat'])('leaves %s announcements to their existing owner', (kind) => {
    expect(currentOperationAnnouncementCandidates(line(kind))).toEqual([])
  })
  it('retains changed actionable attention even when its legacy occurrence time is unknown', () => {
    const status = line(); status.last_material_event = null
    status.attention_summary = { key: 'blocker:1', kind: 'needs_attention', summary: 'Work needs attention. Reason unknown.', subject: null, unblock_actor: 'unknown', unblock_actor_label: 'Unknown', occurred_at: null, next_check_at: null, additional_count: 0 }
    expect(currentOperationAnnouncementCandidates(status)).toEqual([{ key: 'blocker:1', occurred_at: null, source: 'attention', text: 'Work needs attention. Reason unknown.' }])
  })
  it('emits concrete timed nonhuman candidates and excludes owner decision attention', () => {
    const status = line()
    status.attention_summary = { key: 'decision:1', kind: 'needs_you', summary: 'Owner needed.', subject: null,
      unblock_actor: 'owner', unblock_actor_label: 'Owner', occurred_at: '2026-10-14T13:00:00Z', next_check_at: null, additional_count: 0 }
    expect(currentOperationAnnouncementCandidates(status)).toEqual([{ key: 'event:1', occurred_at: '2026-10-14T12:00:00Z', source: 'event', text: 'Verification accepted.' }])
    status.last_material_event = null
    expect(currentOperationAnnouncementCandidates(status)).toEqual([])
  })
})
