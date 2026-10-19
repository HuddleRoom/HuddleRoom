import React, { act, type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { descendants, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { absolute } from '@/lib/time'
import type { NeedsYouReference, OrchestrationAuthorityDecisionRecord, OrchestrationStatusLine, OrchestrationWorkflowFacts } from '@/lib/types'
import { CurrentOperationStatus } from './CurrentOperationStatus.tsx'
import { GoalAnnouncerProvider } from './goalAnnouncer'

const NOW = '2026-10-14T12:00:00Z'
function reference(kind: NeedsYouReference['kind'], id = 'record-1', project_id = 'project-1', label: string | null = 'Checkout'): NeedsYouReference {
  return { kind, id, project_id, goal_id: 'goal-1', label }
}
function line(): OrchestrationStatusLine {
  return { current_state: { kind: 'working', summary: 'Maya was reported validating Checkout.', activities: [
    { key: 'session:1', state: 'reported_active', summary: 'Maya was reported validating Checkout.', subject: reference('task'), actor: reference('agent', 'agent-1', 'project-1', 'Maya'), observed_at: NOW },
  ] }, attention_summary: { key: 'wait:1', kind: 'waiting', summary: 'Planning is waiting. Reason unknown.',
    subject: reference('task', 'planning', 'project-1', 'Planning'), unblock_actor: 'unknown', unblock_actor_label: 'Unknown', next_check_at: '2099-10-14T12:00:00Z', occurred_at: NOW, additional_count: 1 },
    last_material_event: { key: 'event:1', kind: 'accepted_evidence', summary: 'Verification accepted for Checkout.', subject: reference('gate'), occurred_at: NOW } }
}
function wf(overrides: Partial<OrchestrationWorkflowFacts> = {}): OrchestrationWorkflowFacts {
  return { run_id: 'run-1', control_version: 1, phase: 'baseline', phase_label: 'Baseline', process: null, process_type: null, process_status: null, task: null,
    operation_summary: 'Baseline step running.', occurred_at: '2026-10-14T12:00:00Z', pending_decision_id: null, additional_pending_count: 0,
    application: null, additional_application_count: 0, guidance: { kind: 'none', label: 'Nothing to do', decision_id: null, reason: null }, ...overrides }
}
function props(statusLine: OrchestrationStatusLine | null = line()): ComponentProps<typeof CurrentOperationStatus> {
  return { statusLine, condition: 'waiting', activeProjectId: 'project-1', dataUpdatedAt: Date.parse(NOW), isFetching: false, isError: false, onRevealSource: vi.fn(), goalKey: 'goal-1' }
}
const markup = (p = props()) => renderToStaticMarkup(<MemoryRouter><CurrentOperationStatus {...p} /></MemoryRouter>)

async function mounted(initial = props()) {
  const render = (next = initial) => <MemoryRouter><GoalAnnouncerProvider><CurrentOperationStatus {...next} /></GoalAnnouncerProvider></MemoryRouter>
  const view = await mountWithTestDom(() => render(), act)
  return { ...view, rerender: (next: typeof initial) => view.rerender(() => render(next)) }
}

describe('CurrentOperationStatus', () => {
  it('preserves separate concurrent branches and activity/attention/event/freshness reading order', () => {
    const html = markup()
    const positions = ['Maya was reported validating Checkout.', 'Planning is waiting.', 'Last update:', 'Status refreshed'].map((text) => html.indexOf(text))
    expect(positions.every((position) => position >= 0)).toBe(true)
    expect(positions).toEqual([...positions].sort((a, b) => a - b))
    expect(html).toContain('Who can unblock:')
    expect(html).toContain('Unknown')
    expect(html).toContain('Next check:')
    expect(html).toMatch(/break-words|break-all|overflow-wrap/)
    expect((html.match(/Status refreshed/g) ?? [])).toHaveLength(1)
  })
  it.each(['unknown', 'working', 'waiting', 'needs_you', 'needs_attention', 'paused', 'stopped', 'cancelled', 'completed'] as const)('supports the %s condition without false fallback', (condition) => {
    const status = line()
    status.current_state = { kind: 'unknown', summary: 'No active worker confirmed.', activities: [] }
    status.attention_summary = null; status.last_material_event = null
    const html = markup({ ...props(status), condition })
    expect(html).toContain(condition === 'paused' ? 'Paused — new work is held.' : 'No active worker confirmed.')
    expect(html).toContain('No recorded update yet')
    expect(html).not.toContain('Work in progress')
  })
  it('renders refresh/cached-error honesty without inventing activity on an initial absence', () => {
    expect(markup({ ...props(), isFetching: true })).toContain('refreshing')
    expect(markup({ ...props(), isError: true })).toContain('Status last refreshed')
    expect(markup({ ...props(), isError: true })).toContain('refresh failed; this status may be out of date')
    expect(markup(props(null))).not.toContain('No active worker confirmed')
  })
  it.each(['2026-10-14T12:00:00', '2026-10-14T14:00:00+02:00'])('uses semantic UTC time and absolute accessible time for %s', (occurred_at) => {
    const status = line(); status.last_material_event!.occurred_at = occurred_at
    const html = markup(props(status))
    expect(html).toContain('<time')
    expect(html).toContain('dateTime=')
    expect(html).toContain(absolute(occurred_at))
  })
  it('does not expose malformed times and labels overdue checks as unknown results', () => {
    const status = line(); status.last_material_event!.occurred_at = 'INVALID-TIMESTAMP'
    status.attention_summary!.next_check_at = '2020-01-01T00:00:00Z'
    const html = markup(props(status))
    expect(html).not.toContain('INVALID-TIMESTAMP')
    expect(html.toLowerCase()).toContain('passed')
    expect(html.toLowerCase()).toContain('unknown')
    status.attention_summary!.next_check_at = null
    expect(markup(props(status))).toContain('Next check: Unknown')
  })
  it.each([['task', '/tasks?task=record-1'], ['agent', '/agents/record-1'], ['goal', '/orchestration/record-1'], ['meeting', '/meetings/record-1']] as const)('uses the existing %s destination', async (kind, href) => {
    const status = line(); status.current_state.activities[0].subject = reference(kind)
    const view = await mounted(props(status))
    try { expect(descendants(view.container).some((node) => node.tagName === 'A' && node.getAttribute('href') === href)).toBe(true) }
    finally { view.cleanup() }
  })
  it.each(['gate', 'process', 'action'] as const)('reveals %s through the existing callback', async (kind) => {
    const status = line(); status.current_state.activities[0].subject = reference(kind)
    const p = props(status); const view = await mounted(p)
    try {
      const control = descendants(view.container).find((node) => node.tagName === 'BUTTON' && textOf(node).includes('Checkout'))!
      expect(control).toBeDefined()
      await act(async () => control.click())
      expect(p.onRevealSource).toHaveBeenCalledWith({ kind, id: 'record-1', goalId: 'goal-1' })
    } finally { view.cleanup() }
  })
  it.each([reference('task', 'foreign', 'foreign-project', 'Foreign'), reference('task', 'missing', 'project-1', null)])('keeps unsafe/unresolved references unlinked', (subject) => {
    const status = line(); status.current_state.activities[0].subject = subject
    expect(markup(props(status))).not.toContain(`/tasks?task=${subject.id}`)
  })
  it('renders each reported branch sentence once and exposes the actual safe actor destination', () => {
    const html = markup()
    expect((html.match(/Maya was reported validating Checkout\./g) ?? [])).toHaveLength(1)
    expect(html).toContain('href="/agents/agent-1"')
  })
  it('renders every concurrent reported and unknown branch once with actual scoped links', () => {
    const status = line()
    const second = { ...status.current_state.activities[0], key: 'session:2', summary: 'Ari was reported building Billing.', subject: reference('task', 'billing', 'project-1', 'Billing'), actor: reference('agent', 'ari', 'project-1', 'Ari') }
    const unknown = { ...second, key: 'session:3', state: 'liveness_unknown' as const, summary: 'Liveness unknown for Noor on Review.', subject: reference('task', 'review', 'project-1', 'Review'), actor: reference('agent', 'noor', 'project-1', 'Noor'), observed_at: null }
    status.current_state.summary += ` ${second.summary}`
    status.current_state.activities.push(second, unknown)
    const html = markup(props(status))
    for (const activity of status.current_state.activities) {
      expect(html.split(activity.summary)).toHaveLength(2)
      expect(html).toContain(`/tasks?task=${activity.subject!.id}`)
      expect(html).toContain(`/agents/${activity.actor!.id}`)
    }
  })
  it('treats a zero query timestamp as absent rather than a 1970 success', () => {
    const html = markup({ ...props(), dataUpdatedAt: 0 })
    expect(html).not.toContain('1970')
    expect(html).toContain('Status not yet refreshed')
  })
  it.each(['needs_you', 'needs_attention'] as const)('does not invent a scheduled check cue for %s attention', (kind) => {
    const status = line(); status.attention_summary = { ...status.attention_summary!, kind, next_check_at: null }
    expect(markup(props(status))).not.toContain('Next check:')
  })
  it('announces a semantic attention revision once even when the source key is stable', async () => {
    const p = props(); const view = await mounted(p)
    const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status')!
    try {
      const changed = line(); changed.attention_summary = { ...changed.attention_summary!, summary: 'Planning now needs maintainer repair.', unblock_actor: 'maintainer', unblock_actor_label: 'Maintainer' }
      await view.rerender({ ...p, statusLine: changed })
      expect(textOf(live())).toContain('Planning now needs maintainer repair.')
      const first = live().textContent
      await view.rerender({ ...p, statusLine: changed, isFetching: true })
      expect(live().textContent).toBe(first)
      const newCheck = { ...changed, attention_summary: { ...changed.attention_summary!, next_check_at: '2099-10-15T12:00:00Z' } }
      await view.rerender({ ...p, statusLine: newCheck })
      expect(live().textContent).not.toBe(first)
      const second = live().textContent
      await view.rerender({ ...p, statusLine: newCheck })
      expect(live().textContent).toBe(second)
    } finally { view.cleanup() }
  })
  it('announces a newly actionable undated blocker once without treating old events as new', async () => {
    const p = props(); const view = await mounted(p)
    const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status')!
    try {
      const blocked = line(); blocked.attention_summary = { ...blocked.attention_summary!, key: 'blocker:new', kind: 'needs_attention', summary: 'New repair needed.', occurred_at: null }
      await view.rerender({ ...p, statusLine: blocked })
      expect(textOf(live())).toBe('New repair needed.')
      const once = live().textContent
      await view.rerender({ ...p, statusLine: blocked, isFetching: true })
      expect(live().textContent).toBe(once)
    } finally { view.cleanup() }
  })
  it('announces only new nonhuman events once across repeated, out-of-order, freshness and goal switches', async () => {
    const p = props(); const view = await mounted(p)
    const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status')!
    try {
      expect(textOf(live())).toBe('')
      const next = line(); next.last_material_event = { ...next.last_material_event!, key: 'event:2', occurred_at: '2026-10-14T13:00:00Z', summary: 'New evidence accepted.' }
      await view.rerender({ ...p, statusLine: next })
      expect(textOf(live())).toBe('New evidence accepted.')
      const announced = live().textContent
      for (const state of [{ ...p, statusLine: next }, p, { ...p, statusLine: next, isFetching: true }, { ...p, statusLine: next, isError: true }, { ...p, statusLine: next, goalKey: 'goal-2' }]) {
        await view.rerender(state); expect(live().textContent).toBe(announced)
      }
      const human = line(); human.last_material_event = { ...human.last_material_event!, key: 'question:1', kind: 'human_question', occurred_at: '2026-10-15T12:00:00Z' }
      await view.rerender({ ...p, goalKey: 'goal-2', statusLine: human })
      expect(live().textContent).toBe(announced)
    } finally { view.cleanup() }
  })
})


it('shows the persisted reported-active observation time separately from event and fetch times', () => {
  const status = line()
  status.current_state.activities[0].observed_at = '2026-10-14T11:00:00'
  const html = markup({ ...props(status), dataUpdatedAt: Date.parse('2026-10-14T13:00:00Z') })
  expect(html).toContain('dateTime="2026-10-14T11:00:00.000Z"')
  expect(html).toContain(`title="${absolute('2026-10-14T11:00:00Z')}"`)
  expect(html).toContain('dateTime="2026-10-14T12:00:00.000Z"')
  expect(html).toContain('dateTime="2026-10-14T13:00:00.000Z"')
})

it('delivers simultaneous changed attention and material event through the shared region', async () => {
  const p = props(); const view = await mounted(p)
  try {
    const next = line()
    next.attention_summary = { ...next.attention_summary!, key: 'wait:new', summary: 'Maintainer repair required.', occurred_at: '2026-10-14T13:00:00Z' }
    next.last_material_event = { ...next.last_material_event!, key: 'evidence:new', summary: 'Checkout verification accepted.', occurred_at: '2026-10-14T13:00:00Z' }
    await view.rerender({ ...p, statusLine: next })
    const live = descendants(view.container).find((node) => node.getAttribute('role') === 'status')!
    expect(textOf(live)).toContain('Maintainer repair required.')
    expect(textOf(live)).toContain('Checkout verification accepted.')
  } finally { view.cleanup() }
})


it('preserves an older-run gate name as text when its record has no supported current-detail destination', async () => {
  const status = line()
  const historicalGate = { ...reference('gate', 'older-run-gate', 'project-1', 'Historical Checkout criterion'), goal_id: null }
  status.last_material_event = { ...status.last_material_event!, subject: historicalGate }
  const p = props(status); const view = await mounted(p)
  try {
    expect(textOf(view.container)).toContain('Historical Checkout criterion')
    expect(descendants(view.container).some((node) => ['BUTTON', 'A'].includes(node.tagName) && textOf(node).includes('Historical Checkout criterion'))).toBe(false)
    expect(p.onRevealSource).not.toHaveBeenCalled()
  } finally { view.cleanup() }
})

describe('CurrentOperationStatus workflow facts', () => {
  const withWf = (w: OrchestrationWorkflowFacts | null, condition: ComponentProps<typeof CurrentOperationStatus>['condition'] = 'waiting') => {
    const status = line(); status.workflow = w
    return markup({ ...props(status), condition })
  }
  it('shows phase, recorded operation without a worker, and never says live', () => {
    const html = withWf(wf())
    expect(html).toContain('Phase: Baseline')
    expect(html).toContain('Baseline step running.')
    expect(html).toContain('Recorded at')
    expect(html).not.toMatch(/\blive\b/i)
  })
  it('paused with a pending decision shows the badge, decision and Resume guidance', () => {
    const html = withWf(wf({ pending_decision_id: 'd1', additional_pending_count: 2, guidance: { kind: 'resume', label: 'Resume goal', decision_id: null, reason: null } }), 'paused')
    expect(html).toContain('Paused')
    expect(html).toContain('Waiting for your decision:')
    expect(html).toContain('Decision details not loaded yet')
    expect(html).toContain('Answer in Needs You')
    expect(html).toContain('2 more')
    expect(html).not.toContain('Maya was reported validating Checkout.</p>')
    expect(html).toContain('Paused — new work is held. Work already started is not cancelled and may still finish.')
    expect(html).toContain('Next: Resume goal')
  })
  it.each([
    ['pending', null, 'Saved · not applied — Not in effect yet.'],
    ['blocked', 'gate closed', 'Blocked — gate closed'],
    ['failed', 'boom', 'Failed — boom'],
  ] as const)('renders %s application wording', (state, reason, text) => {
    expect(withWf(wf({ application: { action_id: 'a', decision_ids: ['d'], state, reason }, additional_application_count: 1 }))).toContain(text)
    expect(withWf(wf({ application: { action_id: 'a', decision_ids: ['d'], state, reason }, additional_application_count: 1 }))).toContain('1 more')
  })
  it('renders an unavailable fallback when workflow is null', () => {
    const html = withWf(null)
    expect(html).toContain('Workflow details unavailable.')
    expect(html).not.toContain('None')
  })
  it('shows the question only when pending_direction matches, and R5 unknown time', () => {
    const status = line(); status.workflow = wf({ pending_decision_id: 'd1', occurred_at: null })
    const dir = { id: 'd1', question: 'Ship it?' } as OrchestrationAuthorityDecisionRecord
    const html = markup({ ...props(status), pendingDirection: dir })
    expect(html).toContain('&quot;Ship it?&quot;')
    expect(html).toContain('Recorded time unknown')
    expect(markup({ ...props(status), pendingDirection: { ...dir, id: 'other' } })).not.toContain('Ship it?')
  })
})
