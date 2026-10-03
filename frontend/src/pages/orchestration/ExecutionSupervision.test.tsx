import React, { act, type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { descendants, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type { OrchestrationAuthorityDecisionRecord, OrchestrationSupervision } from '@/lib/types'
import { ExecutionSupervision } from './ExecutionSupervision'
import { GoalAnnouncerProvider } from './goalAnnouncer'

const pendingDirection: OrchestrationAuthorityDecisionRecord = {
  id: 'direction-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'deployment_target', title: 'Choose deployment target',
  status: 'pending', authority: 'human', authority_agent_id: null, source_process_run_id: null, question: 'Where should this deploy?',
  context: null, options: [], recommendation: null, consequences: null, selected_option: null, reason: null,
  decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false, created_warning_id: null,
  related_gate_id: null, related_action_id: null, asked_at: '2026-09-10T00:00:00Z', decided_at: null,
  created_at: '2026-09-10T00:00:00Z', updated_at: '2026-09-10T00:00:00Z',
}

const supervision = {
  condition: 'needs_you', operation: 'Direction required', next_action: 'Answer pending direction', rationale: 'Choose deployment target',
  criterion: { key: 'deployment', description: 'Deploy successfully', status: 'open' },
  verified_progress: [{ key: 'build', status: 'verified', summary: 'Build passed' }],
  useful_learning: [{ learning_key: 'cache', status: 'accepted', description: 'Cache miss explained' }],
  accepted_evidence: [{ id: 'evidence-1', run_id: 'run-1', gate_id: 'gate-1', source_type: 'test', source_id: 'report-1', observed_event_id: 'event-1', producer_agent_id: 'producer-1', verdict: 'accepted', evidence_metadata: { summary: 'All checks passed', artifact_id: 'artifact-1', secret: 'private_value' }, created_at: '2026-09-10T01:00:00Z', updated_at: '2026-09-10T01:00:00Z' }],
  workers: [{ session_id: 'session-1', task_id: 'task-1', agent_id: 'agent-1', runner_id: 'runner-1', task_title: 'Run checks', session_status: 'running', observed_liveness: 'live', observed_at: '2026-09-10T01:01:00Z' }],
  waits: [{ id: 'wait-1', owner: { type: 'session', id: 'session-1' }, event: 'session.completed', matcher: { session_id: 'session-1', secret: 'private_value' }, due_recheck_at: '2026-09-10T02:00:00Z', fallback: { reason: 'Timeout', action_type: 'retry', expected_result: 'Completion', secret: 'private_value' } }],
  recovery_history: [{ session_id: 'session-1', classification: 'stalled', disposition: 'retry', backend_observation: 'missing', assessed_at: '2026-09-10T01:02:00Z', action_id: 'action-1', wait_id: 'wait-1' }], pending_direction: pendingDirection,
  budget: { consumed: { tokens: '30' }, committed: { tokens: '30' }, reserved: {}, remaining: { tokens: '70' } },
  transition: { key: 'deployment.finished:1', kind: 'needs_you', message: 'Deployment finished' },
} satisfies OrchestrationSupervision

function snapshot(overrides: Partial<OrchestrationSupervision> = {}): OrchestrationSupervision {
  return { ...supervision, ...overrides }
}

async function mountSupervision(initial: OrchestrationSupervision | null) {
  const render = (next: OrchestrationSupervision | null = initial) => <GoalAnnouncerProvider><ExecutionSupervision supervision={next} /></GoalAnnouncerProvider>
  const view = await mountWithTestDom(() => render(), act)
  return { ...view, rerender: (next: ComponentProps<typeof ExecutionSupervision>['supervision']) => view.rerender(() => render(next)) }
}

describe('ExecutionSupervision', () => {
  it('renders all read-only sections in semantic order, with responsive wrapping and no controls', () => {
    const html = renderToStaticMarkup(<ExecutionSupervision supervision={supervision} />)
    const labels = ['Verified progress', 'Useful learning', 'Accepted evidence', 'Workers', 'Open waits', 'Recovery linkage', 'Criterion', 'Criterion status', 'Pending question']
    for (const label of ['Current operation', ...labels, 'Budget', 'consumed', 'committed', 'reserved', 'remaining']) expect(html).toContain(label)
    expect(html).toContain('committed tokens 30')
    expect(html).toContain('flex flex-wrap')
    expect(html).toContain('grid-cols-1')
    expect(html).toContain('sm:grid-cols-2')
    expect(html).toContain('<h2')
    expect((html.match(/<dt/g) ?? [])).toHaveLength(2)
    expect(html).toContain('Deploy successfully')
    expect(html).toContain('Where should this deploy?')
    expect(html).toContain('Answer in Needs you below.')
    for (const value of ['build', 'verified', 'Build passed', 'cache', 'accepted', 'Cache miss explained', 'evidence-1', 'gate-1', 'report-1', 'event-1', 'producer-1', 'artifact-1', 'session-1', 'task-1', 'agent-1', 'runner-1', 'wait-1', 'session.completed', 'Timeout', 'action-1', 'stalled', 'missing']) expect(html).toContain(value)
    expect(html).not.toContain('private_value')
    expect(html).not.toContain('{&quot;')
    expect(['Criterion', 'Criterion status', 'Verified progress', 'Useful learning', 'Accepted evidence', 'Workers', 'Open waits', 'Recovery linkage', 'Pending question'].map((label) => html.indexOf(label))).toEqual([
      ...['Criterion', 'Criterion status', 'Verified progress', 'Useful learning', 'Accepted evidence', 'Workers', 'Open waits', 'Recovery linkage', 'Pending question'].map((label) => html.indexOf(label)),
    ].sort((a, b) => a - b))
    expect(html).not.toMatch(/<(?:button|input|select|textarea|a)\b/)
  })

  it('uses None fallbacks for absent optional facts', () => {
    const html = renderToStaticMarkup(<ExecutionSupervision supervision={snapshot({
      criterion: null, pending_direction: null, workers: [], waits: [], verified_progress: [], accepted_evidence: [], useful_learning: [], recovery_history: [],
      budget: { consumed: {}, committed: {}, reserved: {}, remaining: {} },
    })} />)
    expect((html.match(/>None</g) ?? [])).toHaveLength(9)
    expect((html.match(/—/g) ?? [])).toHaveLength(4)
  })

  it('keeps passive snapshots silent and announces each new direction or material transition once', async () => {
    const working = snapshot({ condition: 'working', transition: null })
    const view = await mountSupervision(working)
    try {
      // The component's own local aria-live region was removed — transitions now
      // flow through the shared page-level polite region (GoalAnnouncerProvider).
      const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
      expect(textOf(live())).toBe('')

      await view.rerender(working) // repeat
      await view.rerender(null) // absent supervision
      await view.rerender(snapshot({ condition: 'working', transition: { key: 'tick:1', kind: 'tick', message: 'Tick' } }))
      await view.rerender(snapshot({ condition: 'working', transition: { key: 'heartbeat:1', kind: 'heartbeat', message: 'Heartbeat' } }))
      await view.rerender(snapshot({ condition: 'working', transition: { key: 'heartbeat:1', kind: 'heartbeat', message: 'Heartbeat' } }))
      expect(textOf(live())).toBe('')

      const direction = snapshot({ transition: { key: 'deployment.finished:1', kind: 'pending_direction', message: 'Deployment finished' } })
      await view.rerender(direction)
      expect(textOf(live())).toBe('Needs your direction: Deployment finished')
      await view.rerender(direction)
      expect(textOf(live())).toBe('Needs your direction: Deployment finished')

      const material = snapshot({ condition: 'working', transition: { key: 'evidence:1:accepted', kind: 'accepted_evidence', message: 'Test report accepted' } })
      await view.rerender(material)
      expect(textOf(live())).toBe('Test report accepted')
      await view.rerender(material)
      expect(textOf(live())).toBe('Test report accepted')

      vi.useFakeTimers()
      await act(async () => { vi.advanceTimersByTime(10_000) })
      await view.rerender(material)
      expect(textOf(live())).toBe('Test report accepted')
    } finally {
      vi.useRealTimers()
      view.cleanup()
    }
  })

  it('omits the optional surface without supervision data', () => {
    expect(renderToStaticMarkup(<ExecutionSupervision supervision={null} />)).toBe('')
  })

  it.each([
    ['working', 'Working'], ['waiting', 'Waiting'], ['needs_you', 'Needs you'], ['needs_attention', 'Needs attention'],
    ['paused', 'Paused'], ['stopped', 'Stopped'], ['cancelled', 'Cancelled'], ['completed', 'Completed'],
  ] as const)('maps %s to its semantic status badge', (condition, badge) => {
    const html = renderToStaticMarkup(<ExecutionSupervision supervision={snapshot({ condition })} />)
    expect(html).toContain(`>${badge}</span>`)
    if (condition === 'waiting') expect(html).toContain('color:#0369A1')
    if (condition === 'stopped') expect(html).toContain('color:#5a6270')
  })
})
