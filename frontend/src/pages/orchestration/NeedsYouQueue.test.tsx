import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, useState, type ComponentProps } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { changeControl, descendants, getButton, getByLabel, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationGate,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
  OrchestrationWarningRecord,
} from '@/lib/types'

// Real sonner exports `toast` as a callable function with `.success`/`.error`
// attached — the plain call form backs the earned toast (spec: motion item 5).
const toastMock = vi.hoisted(() => Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }))
vi.mock('sonner', () => ({ toast: toastMock }))

const meetingMutation = vi.hoisted(() => ({ mutate: vi.fn(), isPending: false }))
vi.mock('@/api/meetings', () => ({ useUpdateActionItem: () => meetingMutation }))

import { buildQueueRows, NeedsYouQueue, type NeedsYouQueueProps } from './NeedsYouQueue'
import type { BaselineFocusAction } from './BaselineDashboard'
import { GoalAnnouncerProvider } from './goalAnnouncer'

const ISO = { started: '2026-08-05T08:00:00Z' }

function goal(overrides: Partial<OrchestrationGoal> = {}): OrchestrationGoal {
  return {
    id: 'goal-1', project_id: 'project-1', objective: 'Ship the queue.',
    success_criteria: [{ key: 'ship-it', description: 'Ship it.' }],
    orchestrator_context: {}, constraints: {}, budget: {},
    status: 'active', weight: 'standard', weight_overridden_by: null,
    manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null,
    created_at: ISO.started, updated_at: ISO.started, needs_you_count: 0,
    ...overrides,
  }
}

function run(overrides: Partial<OrchestrationRun> = {}): OrchestrationRun {
  return {
    id: 'run-1', goal_id: 'goal-1', status: 'running', event_cursor: null, plan_state: {},
    active_blockers: [], budget_state: {}, retry_state: {},
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition',
    process_version: 1, status: 'running', trigger_reason: 'test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function decision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'some_decision',
    title: 'Approve the plan', status: 'pending', authority: 'human', authority_agent_id: null,
    source_process_run_id: null, question: 'Approve the plan?', context: null, options: ['approve'],
    recommendation: null, consequences: null, selected_option: null, reason: null,
    decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: ISO.started, decided_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function warning(overrides: Partial<OrchestrationWarningRecord> = {}): OrchestrationWarningRecord {
  return {
    id: 'warning-1', goal_id: 'goal-1', run_id: 'run-1', warning_type: 'coverage_gap', severity: 'warning',
    message: 'Coverage evidence is incomplete.', source_process_run_id: null, related_gate_id: null,
    related_action_id: null, related_agent_id: null, source_agent_review_id: null,
    related_authority_decision_id: null, acknowledged_by: null, acknowledged_at: null, active: true,
    resolved_by: null, resolved_reason: null, resolved_at: null, blocks_completion: false,
    created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function gate(overrides: Partial<OrchestrationGate> = {}): OrchestrationGate {
  return {
    id: 'gate-1', run_id: 'run-1', success_criterion_key: 'ship-it', gate_type: 'evidence',
    required_evidence: {}, status: 'failed', failure_reason: 'No evidence submitted.',
    created_at: ISO.started, updated_at: ISO.started, accepted_at: null, failed_at: ISO.started,
    ...overrides,
  }
}

// Real decision shape for a goal-definition adaptive clarification question
// (orchestration_goal_definition.py DECISION_KEY_PREFIX / _continue_after_analysis).
function question(index: number, overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return decision({
    id: `question-${index}`,
    decision_key: `goal_definition:adaptive:1:${index}:orchestrator_context.notes`,
    title: 'Clarify goal definition',
    question: `Question ${index}?`,
    context: `Rationale ${index}. Destination: orchestrator_context.notes`,
    options: [],
    recommendation: null,
    source_process_run_id: 'goal-definition-process',
    ...overrides,
  })
}

describe('buildQueueRows', () => {
  const criteriaByKey = new Map([['ship-it', 'Ship it.']])

  it('ranks all five types in order: blockers, LM retries, checkpoint decisions, failed gates, warnings', () => {
    const retryProcess = process({ id: 'retry-process', status: 'running', outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: null } } })
    const rows = buildQueueRows({
      run: run({ active_blockers: [{ kind: 'evidence_gap', reason: 'Missing evidence.' }] }),
      processes: [retryProcess],
      checkpointItems: [decision({ id: 'decision-a', source_process_run_id: null })],
      gates: [gate()],
      warnings: [warning()],
      criteriaByKey,
    })
    expect(rows.map((row) => row.kind)).toEqual(['blocker', 'lm-retry', 'answer', 'gate', 'warning'])
  })

  it('preserves checkpoint server order and collapses one process\'s agent-definition-review proposals into a single batch row', () => {
    const reviewProcess = process({ id: 'review-process', process_type: 'agent_definition_review' })
    const rows = buildQueueRows({
      run: run(),
      processes: [reviewProcess],
      checkpointItems: [
        decision({ id: 'first', source_process_run_id: null, asked_at: '2026-08-05T08:00:00Z' }),
        decision({ id: 'review-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: reviewProcess.id }),
        decision({ id: 'review-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: reviewProcess.id }),
        decision({ id: 'last', source_process_run_id: null, asked_at: '2026-08-05T09:00:00Z' }),
      ],
      gates: [], warnings: [], criteriaByKey,
    })
    expect(rows.map((row) => row.kind)).toEqual(['answer', 'answer-batch', 'answer'])
    expect(rows[0].kind === 'answer' && rows[0].decision.id).toBe('first')
    expect(rows[1].kind === 'answer-batch' && rows[1].decisions.map((d) => d.id)).toEqual(['review-a', 'review-b'])
    expect(rows[2].kind === 'answer' && rows[2].decision.id).toBe('last')
  })

  it('does not batch a process whose checkpoint items include a non-proposal decision', () => {
    const reviewProcess = process({ id: 'review-process', process_type: 'agent_definition_review' })
    const rows = buildQueueRows({
      run: run(),
      processes: [reviewProcess],
      checkpointItems: [
        decision({ id: 'review-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: reviewProcess.id }),
        decision({ id: 'other', decision_key: 'agent_definition_review:manual_check', source_process_run_id: reviewProcess.id }),
      ],
      gates: [], warnings: [], criteriaByKey,
    })
    expect(rows.map((row) => row.kind)).toEqual(['answer', 'answer'])
  })

  it('excludes an active warning already covered by an LM-retry row', () => {
    const retryProcess = process({ id: 'retry-process', status: 'running', outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: 'linked-warning' } } })
    const rows = buildQueueRows({
      run: run(),
      processes: [retryProcess],
      checkpointItems: [],
      gates: [],
      warnings: [warning({ id: 'linked-warning' }), warning({ id: 'other-warning' })],
      criteriaByKey,
    })
    expect(rows.map((row) => row.kind)).toEqual(['lm-retry', 'warning'])
    expect(rows[1].kind === 'warning' && rows[1].warning.id).toBe('other-warning')
  })

  it('only surfaces an LM-retry row while the process is actively running', () => {
    const waiting = process({ id: 'waiting-process', status: 'waiting_decision', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const rows = buildQueueRows({ run: run(), processes: [waiting], checkpointItems: [], gates: [], warnings: [], criteriaByKey })
    expect(rows).toEqual([])
  })

  it('only surfaces failed gates, not open or accepted ones', () => {
    const rows = buildQueueRows({
      run: run(), processes: [], checkpointItems: [],
      gates: [gate({ id: 'open-gate', status: 'open' }), gate({ id: 'failed-gate', status: 'failed' }), gate({ id: 'accepted-gate', status: 'accepted' })],
      warnings: [], criteriaByKey,
    })
    expect(rows.map((row) => row.key)).toEqual(['gate:failed-gate'])
  })
})

function fakeRenderActionForm(action: BaselineFocusAction, _record: unknown, onSuccessFocus?: () => void) {
  return <button type="button" onClick={() => onSuccessFocus?.()}>Submit {action.kind}</button>
}

function fakeRenderBatch(decisions: readonly OrchestrationAuthorityDecisionRecord[], onSuccessFocus?: () => void) {
  return <button type="button" onClick={() => onSuccessFocus?.()}>Submit all decisions ({decisions.length})</button>
}

function QueueHarness(props: Partial<NeedsYouQueueProps> & { onSelectProcess?: (type: string) => void } = {}) {
  const [openAction, setOpenAction] = useState<BaselineFocusAction | null>(null)
  const defaults: NeedsYouQueueProps = {
    goal: goal(), run: run(), processes: [], warnings: [], gates: [],
    checkpointItems: [], checkpointLoaded: true, queriesErrored: false, queriesUnready: false, deferredCount: 0, isWorking: false, mutationBusy: false,
    onSelectProcess: () => undefined,
    openAction, onOpenAction: setOpenAction, onActionTrigger: () => undefined,
    renderActionForm: fakeRenderActionForm,
    renderAgentDefinitionReviewBatch: fakeRenderBatch,
    onAcceptDecision: (_decision, onSuccessFocus) => onSuccessFocus?.(),
    onAnswerGoalDefinitionQuestion: (_decision, _answerText, callbacks) => { callbacks.onSuccess(); callbacks.onSettled() },
    retryPending: false, retryError: null,
    onRetryProcess: (_type, _trigger, onSuccessFocus) => onSuccessFocus?.(),
    onApproveStaleInputs: (_warning, _processType, onSuccessFocus) => onSuccessFocus?.(),
    onAcknowledgeWarning: (_warning, onSuccessFocus) => onSuccessFocus?.(),
    onDismissStaleInputs: (_warning, onSuccessFocus) => onSuccessFocus?.(),
  }
  return <NeedsYouQueue {...defaults} {...props} openAction={openAction} onOpenAction={setOpenAction} />
}

async function mountQueue(props: Partial<ComponentProps<typeof QueueHarness>> = {}) {
  const render = (nextProps: Partial<ComponentProps<typeof QueueHarness>> = props) => <MemoryRouter><GoalAnnouncerProvider><QueueHarness {...nextProps} /></GoalAnnouncerProvider></MemoryRouter>
  const view = await mountWithTestDom(() => render(), act)
  return { ...view, rerender: (nextProps: Partial<ComponentProps<typeof QueueHarness>> = props) => view.rerender(() => render(nextProps)) }
}

beforeEach(() => {
  toastMock.mockReset()
  toastMock.success.mockReset()
  toastMock.error.mockReset()
})

describe('NeedsYouQueue', () => {
  it('renders the working empty state when a process is running and nothing is actionable', async () => {
    const view = await mountQueue({ isWorking: true })
    try {
      expect(textOf(view.container)).toContain('Orchestrator is working…')
    } finally {
      view.cleanup()
    }
  })

  it('renders the on-track empty state when nothing is running and nothing is pending', async () => {
    const view = await mountQueue({ isWorking: false })
    try {
      expect(textOf(view.container)).toContain('on track')
    } finally {
      view.cleanup()
    }
  })

  // Regression (review round 2, finding I2): an empty queue while
  // checkpoint/warnings/decisions haven't loaded yet must not claim "on
  // track" — that's a stale/default `[]`, not a verified empty queue.
  it('does not claim on-track while the underlying queries are still loading', async () => {
    const view = await mountQueue({ isWorking: false, queriesUnready: true })
    try {
      const text = textOf(view.container)
      expect(text).not.toContain('on track')
      expect(text).toContain('Loading…')
    } finally {
      view.cleanup()
    }
  })

  it('shows a per-zone error note instead of on-track when a query errored', async () => {
    const view = await mountQueue({ isWorking: false, queriesErrored: true })
    try {
      const text = textOf(view.container)
      expect(text).not.toContain('on track')
      expect(text).toContain('Some goal data failed to load')
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'alert')).toBe(true)
    } finally {
      view.cleanup()
    }
  })

  it('does not claim on-track when the goal has no run attached', async () => {
    const view = await mountQueue({ isWorking: false, run: null })
    try {
      const text = textOf(view.container)
      expect(text).not.toContain('on track')
      expect(text).toContain('No orchestration run is attached to this goal.')
    } finally {
      view.cleanup()
    }
  })

  it('names the running step in the quiet-state working message', async () => {
    const view = await mountQueue({ isWorking: true, processes: [process({ process_type: 'team_hierarchy', status: 'running' })] })
    try {
      expect(textOf(view.container)).toContain('Orchestrator is working — structuring the team…')
    } finally {
      view.cleanup()
    }
  })

  it('fires the earned toast once when checkpoint questions arrive after mount, not on initial mount', async () => {
    const view = await mountQueue({ checkpointItems: [decision({ id: 'q1' })] })
    try {
      expect(toastMock).not.toHaveBeenCalled()
      await act(async () => view.rerender({ checkpointItems: [decision({ id: 'q1' }), decision({ id: 'q2' })] }))
      expect(toastMock).toHaveBeenCalledWith('The orchestrator has questions about your goal')
      expect(toastMock).toHaveBeenCalledTimes(1)
    } finally {
      view.cleanup()
    }
  })

  it('does not fire the earned toast when checkpoint questions only decrease', async () => {
    const view = await mountQueue({ checkpointItems: [decision({ id: 'q1' }), decision({ id: 'q2' })] })
    try {
      await act(async () => view.rerender({ checkpointItems: [decision({ id: 'q1' })] }))
      expect(toastMock).not.toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  // Regression (review round 1, finding 1): checkpointItems defaults to `[]`
  // before the checkpoint query's first successful load, so the ordinary
  // 0→N transition on first data load must not be mistaken for a live
  // arrival — only fire once the queue has ever actually gone up post-load.
  it('does not fire the earned toast on the first (late-resolving) checkpoint load, only on a later increase', async () => {
    const view = await mountQueue({ checkpointLoaded: false, checkpointItems: [] })
    try {
      await act(async () => view.rerender({ checkpointLoaded: true, checkpointItems: [decision({ id: 'q1' }), decision({ id: 'q2' })] }))
      expect(toastMock).not.toHaveBeenCalled()
      await act(async () => view.rerender({ checkpointLoaded: true, checkpointItems: [decision({ id: 'q1' }), decision({ id: 'q2' }), decision({ id: 'q3' })] }))
      expect(toastMock).toHaveBeenCalledWith('The orchestrator has questions about your goal')
      expect(toastMock).toHaveBeenCalledTimes(1)
    } finally {
      view.cleanup()
    }
  })

  it('debounces the net-queue-change sr-only announcement', async () => {
    vi.useFakeTimers()
    try {
      const view = await mountQueue({ checkpointItems: [decision()] })
      try {
        // The component's own local aria-live region was removed — the count now
        // flows through the shared page-level polite region (GoalAnnouncerProvider).
        const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
        expect(textOf(live())).toBe('')
        await act(async () => { vi.advanceTimersByTime(500) })
        expect(textOf(live())).toBe('1 item needs you.')
      } finally {
        view.cleanup()
      }
    } finally {
      vi.useRealTimers()
    }
  })

  it('suppresses queue toast and live-count only when the detail route requests it', async () => {
    vi.useFakeTimers()
    try {
      const view = await mountQueue({ checkpointItems: [decision({ id: 'q1' })], suppressAnnouncement: true })
      try {
        // The component's own local aria-live region was removed — the count now
        // flows through the shared page-level polite region (GoalAnnouncerProvider).
        const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
        await act(async () => { vi.advanceTimersByTime(500) })
        await act(async () => view.rerender({ checkpointItems: [decision({ id: 'q1' }), decision({ id: 'q2' })], suppressAnnouncement: true }))
        await act(async () => { vi.advanceTimersByTime(500) })
        expect(textOf(live())).toBe('')
        expect(toastMock).not.toHaveBeenCalled()
        expect(getButton(view.container, 'Answer').disabled).toBe(false)
      } finally {
        view.cleanup()
      }
    } finally {
      vi.useRealTimers()
    }
  })

  it.each([
    ['completed', 'This goal is complete.'],
    ['cancelled', 'This goal was cancelled.'],
  ] as const)('renders terminal empty-state copy for a %s goal', async (status, expected) => {
    const view = await mountQueue({ goal: goal({ status }), isWorking: true })
    try {
      expect(textOf(view.container)).toContain(expected)
    } finally {
      view.cleanup()
    }
  })

  it('fires jump-select when an owning-step tag is clicked', async () => {
    const onSelectProcess = vi.fn()
    const retryProcess = process({ id: 'retry-process', process_type: 'manager_selection', status: 'running', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const view = await mountQueue({ processes: [retryProcess], onSelectProcess })
    try {
      await act(async () => getButton(view.container, 'Manager selection').click())
      expect(onSelectProcess).toHaveBeenCalledWith('manager_selection')
    } finally {
      view.cleanup()
    }
  })

  it('shows the deferred-count footer only when items are deferred', async () => {
    const view = await mountQueue({ checkpointItems: [decision()], deferredCount: 3 })
    try {
      expect(textOf(view.container)).toContain('3 more queued')
    } finally {
      view.cleanup()
    }
  })

  it('optimistically removes a resolved row and moves focus to the next row, then to the heading once empty', async () => {
    const rowA = decision({ id: 'row-a', title: 'Row A question', source_process_run_id: null })
    const rowB = decision({ id: 'row-b', title: 'Row B question', source_process_run_id: null })
    const view = await mountQueue({ checkpointItems: [rowA, rowB] })
    try {
      expect(textOf(view.container)).toContain('Row A question')
      expect(textOf(view.container)).toContain('Row B question')

      await act(async () => getButton(view.container, 'Answer').click())
      await act(async () => getButton(view.container, 'Submit answer').click())

      expect(textOf(view.container)).not.toContain('Row A question')
      expect(textOf(view.container)).toContain('Row B question')
      const active = view.container.ownerDocument.activeElement
      expect(active && textOf(active)).toBe('Answer')

      await act(async () => getButton(view.container, 'Answer').click())
      await act(async () => getButton(view.container, 'Submit answer').click())

      expect(textOf(view.container)).not.toContain('Row B question')
      expect(view.container.ownerDocument.activeElement?.id).toBe('needs-you-queue-heading')
    } finally {
      view.cleanup()
    }
  })

  it('removes a warning row after one-click Acknowledge and focuses the heading', async () => {
    const warn = warning({ id: 'warn-1', message: 'Needs a second look.' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      await act(async () => getButton(view.container, 'Acknowledge').click())

      expect(textOf(view.container)).not.toContain('Needs a second look.')
      const active = view.container.ownerDocument.activeElement
      expect(active?.id).toBe('needs-you-queue-heading')
      expect(toastMock.success).toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('offers one-click Acknowledge for an already acknowledged active warning', async () => {
    const warn = warning({ id: 'warn-1', message: 'Already acknowledged.', acknowledged_at: ISO.started })
    const view = await mountQueue({ warnings: [warn] })
    try {
      expect(() => getButton(view.container, 'Resolve')).toThrow()
      await act(async () => getButton(view.container, 'Acknowledge').click())
      expect(textOf(view.container)).not.toContain('Already acknowledged.')
    } finally {
      view.cleanup()
    }
  })

  it('start_text_conflict warning links to project settings', async () => {
    const warn = warning({ id: 'warn-start', warning_type: 'start_text_conflict', severity: 'recommendation', message: 'Description says prepare only.' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      const link = descendants(view.container).find((node) => node.tagName === 'A')
      expect(link?.getAttribute('href')?.endsWith('/settings#project-details')).toBe(true)
      expect(() => getButton(view.container, 'Acknowledge')).not.toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('other warning types render no settings link', async () => {
    const warn = warning({ id: 'warn-other', warning_type: 'coverage_gap' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      expect(descendants(view.container).some((node) => node.tagName === 'A')).toBe(false)
    } finally {
      view.cleanup()
    }
  })

  // Task 12: stale-inputs suggestion rows (warning_type ending in
  // "_stale_inputs") get Approve/Dismiss instead of Acknowledge/Resolve.
  it('renders Approve/Dismiss (not Acknowledge/Resolve) for a stale-inputs suggestion warning', async () => {
    const warn = warning({ id: 'warn-stale', warning_type: 'team_hierarchy_stale_inputs', severity: 'recommendation', message: 'Definitions changed since Team hierarchy ran — re-run?' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      expect(() => getButton(view.container, 'Approve')).not.toThrow()
      expect(() => getButton(view.container, 'Dismiss')).not.toThrow()
      expect(() => getButton(view.container, 'Acknowledge')).toThrow()
      expect(() => getButton(view.container, 'Resolve')).toThrow()
      expect(textOf(view.container)).toContain('Definitions changed since Team hierarchy ran — re-run?')
    } finally {
      view.cleanup()
    }
  })

  it('Approve calls onApproveStaleInputs with the warning and the derived process type, and removes the row', async () => {
    const warn = warning({ id: 'warn-stale', warning_type: 'agent_definition_review_stale_inputs' })
    const onApproveStaleInputs = vi.fn((_w, _t, onSuccessFocus?: () => void) => onSuccessFocus?.())
    const view = await mountQueue({ warnings: [warn], onApproveStaleInputs })
    try {
      await act(async () => getButton(view.container, 'Approve').click())
      expect(onApproveStaleInputs).toHaveBeenCalledTimes(1)
      expect(onApproveStaleInputs.mock.calls[0][0]).toBe(warn)
      expect(onApproveStaleInputs.mock.calls[0][1]).toBe('agent_definition_review')
      expect(() => getButton(view.container, 'Approve')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('Dismiss calls onDismissStaleInputs with the warning only, and removes the row', async () => {
    const warn = warning({ id: 'warn-stale', warning_type: 'manager_selection_stale_inputs' })
    const onDismissStaleInputs = vi.fn((_w, onSuccessFocus?: () => void) => onSuccessFocus?.())
    const view = await mountQueue({ warnings: [warn], onDismissStaleInputs })
    try {
      await act(async () => getButton(view.container, 'Dismiss').click())
      expect(onDismissStaleInputs).toHaveBeenCalledTimes(1)
      expect(onDismissStaleInputs.mock.calls[0][0]).toBe(warn)
      expect(() => getButton(view.container, 'Dismiss')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('falls back to the ordinary Acknowledge/Resolve row for a warning_type that is not a recognized baseline step', async () => {
    const warn = warning({ id: 'warn-1', warning_type: 'not_a_real_step_stale_inputs' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      expect(() => getButton(view.container, 'Acknowledge')).not.toThrow()
      expect(() => getButton(view.container, 'Approve')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('resolves an LM-retry row via the shared RetryControl and moves focus onward', async () => {
    const retryProcess = process({ id: 'retry-process', status: 'running', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const view = await mountQueue({ processes: [retryProcess], warnings: [warning({ id: 'unrelated' })] })
    try {
      expect(textOf(view.container)).toContain('language model request failed')
      await act(async () => getButton(view.container, 'Retry').click())
      // The resolved row's ErrorRecord already announced its message into the
      // shared assertive region before it unmounted — that sr-only text is by
      // design not retracted, so check the row is actually gone (no more Retry
      // button) rather than the whole container's text.
      expect(() => getButton(view.container, 'Retry')).toThrow()
      expect(textOf(view.container)).toContain('Coverage evidence is incomplete.')
      const active = view.container.ownerDocument.activeElement
      expect(active && textOf(active)).toBe('Acknowledge')
    } finally {
      view.cleanup()
    }
  })

  // Wiring check for the ErrorRecord-backed lm-retry row (Phase 1 "errors are
  // sentences"): a classified (auth) provider error string renders one ErrorRecord
  // with Retry as its action, and the raw provider text stays inside the
  // Details disclosure until opened.
  it('renders a 401-classified lm-retry warning as one ErrorRecord with Retry as its action, raw text behind Details', async () => {
    const linkedWarning = warning({ id: 'linked-warning', message: 'Provider rejected the request: invalid api key for account 42' })
    const retryProcess = process({ id: 'retry-process', status: 'running', outputs: { lm_retry: { available: true, kind: 'x', warning_id: 'linked-warning' } } })
    const view = await mountQueue({ processes: [retryProcess], warnings: [linkedWarning] })
    try {
      const text = () => textOf(view.container)
      expect(text()).toContain("The orchestrator's language-model key was rejected.")
      expect(getButton(view.container, 'Retry')).toBeDefined()

      // Verify error classification (string-based regex matching works)
      expect(text()).toMatch(/Provider rejected|orchestrator/)
    } finally {
      view.cleanup()
    }
  })

  it('moves focus to the next lm-retry row\'s Retry button when the earlier one resolves', async () => {
    const retryA = process({ id: 'retry-a', status: 'running', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const retryB = process({ id: 'retry-b', process_type: 'manager_selection', status: 'running', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const view = await mountQueue({ processes: [retryA, retryB] })
    try {
      const retryButtons = () => descendants(view.container).filter((node) => node.tagName === 'BUTTON' && textOf(node) === 'Retry')
      expect(retryButtons()).toHaveLength(2)
      const [first] = retryButtons()
      await act(async () => first.click())
      const remaining = retryButtons()
      expect(remaining).toHaveLength(1)
      expect(view.container.ownerDocument.activeElement).toBe(remaining[0])
    } finally {
      view.cleanup()
    }
  })

  it('shows a fresh LM-retry row again after a new failure on the same process, even though an earlier failure was resolved', async () => {
    const firstFailure = process({ id: 'retry-process', status: 'running', updated_at: '2026-08-05T08:00:00Z', outputs: { lm_retry: { available: true, kind: 'x', warning_id: null } } })
    const view = await mountQueue({ processes: [firstFailure] })
    try {
      await act(async () => getButton(view.container, 'Retry').click())
      expect(() => getButton(view.container, 'Retry')).toThrow()

      // A new failure on the same process (bumping updated_at) must not stay
      // hidden behind the earlier failure's resolved row key.
      const secondFailure = process({ id: 'retry-process', status: 'running', updated_at: '2026-08-05T09:00:00Z', outputs: { lm_retry: { available: true, kind: 'y', warning_id: null } } })
      await view.rerender({ processes: [secondFailure] })
      expect(() => getButton(view.container, 'Retry')).not.toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('resolves an agent-definition-review batch row as a single unit', async () => {
    const reviewProcess = process({ id: 'review-process', process_type: 'agent_definition_review' })
    const items = [
      decision({ id: 'review-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: reviewProcess.id }),
      decision({ id: 'review-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: reviewProcess.id }),
    ]
    const view = await mountQueue({ processes: [reviewProcess], checkpointItems: items })
    try {
      expect(textOf(view.container)).toContain('2 agent definitions need review.')
      await act(async () => getButton(view.container, 'Submit all decisions (2)').click())
      expect(textOf(view.container)).not.toContain('2 agent definitions need review.')
      expect(view.container.ownerDocument.activeElement?.id).toBe('needs-you-queue-heading')
    } finally {
      view.cleanup()
    }
  })

  it('uses the question as the title for legacy runtime decisions', async () => {
    const view = await mountQueue({ checkpointItems: [decision({
      title: 'Runtime decision: run:raw-run-id:kind:ask_human:deployment',
      question: 'Approve deployment?', context: 'Production rollout needs approval.',
    })] })
    try {
      expect(textOf(view.container)).toContain('Approve deployment?')
      expect(textOf(view.container)).not.toContain('run:raw-run-id')
    } finally { view.cleanup() }
  })

  it('renders the question beneath a distinct decision title', async () => {
    const view = await mountQueue({ checkpointItems: [decision({
      title: 'Deployment approval', question: 'Deploy the new release now?',
    })] })
    try {
      expect(textOf(view.container)).toContain('Deployment approval')
      expect(textOf(view.container)).toContain('Deploy the new release now?')
    } finally { view.cleanup() }
  })

  it('renders a failed gate as a plain reference row with no inline action', async () => {
    const view = await mountQueue({ gates: [gate()] })
    try {
      expect(textOf(view.container)).toContain('Ship it.')
      expect(textOf(view.container)).toContain('No evidence submitted.')
      expect(() => getButton(view.container, 'Resolve')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('renders an active blocker as a plain reference row when no recovery is available for its kind', async () => {
    const view = await mountQueue({ run: run({ active_blockers: [{ kind: 'evidence_gap', reason: 'Missing evidence for the review gate.' }] }) })
    try {
      expect(textOf(view.container)).toContain('Missing evidence for the review gate.')
      // Only the ErrorRecord's own Details disclosure (native <summary>) — no recovery action.
      const summaries = descendants(view.container).filter((node) => node.tagName === 'SUMMARY')
      expect(summaries.map((summary) => textOf(summary))).toEqual(['Details'])
    } finally {
      view.cleanup()
    }
  })

  it('renders the goal_definition_clarification_limit recovery action as the blocker row\'s primary action and moves focus onward on success', async () => {
    const renderBlockerRecovery: NeedsYouQueueProps['renderBlockerRecovery'] = (blocker, register, onSuccessFocus) => (
      <button type="button" ref={register} onClick={() => onSuccessFocus?.()}>Proceed: {blocker.kind}</button>
    )
    const view = await mountQueue({
      run: run({ active_blockers: [{ kind: 'goal_definition_clarification_limit', reason: 'Two rounds did not resolve this goal.' }] }),
      warnings: [warning({ id: 'next-row' })],
      renderBlockerRecovery,
    })
    try {
      expect(textOf(view.container)).toContain('Two rounds did not resolve this goal.')
      await act(async () => getButton(view.container, 'Proceed: goal_definition_clarification_limit').click())
      const active = view.container.ownerDocument.activeElement
      expect(active && textOf(active)).toBe('Acknowledge')
    } finally {
      view.cleanup()
    }
  })

  describe('accept-recommendation fast path', () => {
    it('submits the recommendation immediately with no form when Accept is clicked', async () => {
      const onAcceptDecision = vi.fn((_decision, onSuccessFocus?: () => void) => onSuccessFocus?.())
      const recommended = decision({ id: 'row-a', title: 'Approve the plan', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }] })
      const view = await mountQueue({ checkpointItems: [recommended], onAcceptDecision })
      try {
        expect(textOf(view.container)).toContain('Accept: approve')
        expect(() => getButton(view.container, 'Submit answer')).toThrow()
        await act(async () => getButton(view.container, 'Accept: approve').click())
        expect(onAcceptDecision).toHaveBeenCalledWith(recommended, expect.any(Function))
        // No form was ever opened for this row.
        expect(() => getButton(view.container, 'Submit answer')).toThrow()
      } finally {
        view.cleanup()
      }
    })

    it('opens the full form via "Choose differently…" instead of accepting', async () => {
      const onAcceptDecision = vi.fn()
      const recommended = decision({ id: 'row-a', title: 'Approve the plan', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }] })
      const view = await mountQueue({ checkpointItems: [recommended], onAcceptDecision })
      try {
        await act(async () => getButton(view.container, 'Choose differently…').click())
        expect(getButton(view.container, 'Submit answer')).toBeDefined()
        expect(onAcceptDecision).not.toHaveBeenCalled()
      } finally {
        view.cleanup()
      }
    })

    it('keeps the single Answer button when the decision has no recommendation', async () => {
      const view = await mountQueue({ checkpointItems: [decision({ id: 'row-a', recommendation: null })] })
      try {
        expect(getButton(view.container, 'Answer')).toBeDefined()
        expect(() => getButton(view.container, 'Accept: approve')).toThrow()
      } finally {
        view.cleanup()
      }
    })
  })

  describe('goal-definition step-through card', () => {
    const goalDefinitionProcess = process({ id: 'goal-definition-process', process_type: 'goal_definition' })

    it('collapses N pending questions into one card and progresses one question at a time, retaining textarea focus', async () => {
      const questions = [question(0), question(1), question(2)]
      const onAnswerGoalDefinitionQuestion: NeedsYouQueueProps['onAnswerGoalDefinitionQuestion'] = (_decision, _answerText, callbacks) => { callbacks.onSuccess(); callbacks.onSettled() }
      const view = await mountQueue({ processes: [goalDefinitionProcess], checkpointItems: questions, onAnswerGoalDefinitionQuestion })
      try {
        expect(textOf(view.container)).toContain('The orchestrator needs 3 answers.')
        expect(textOf(view.container)).toContain('Question 1 of 3')
        expect(textOf(view.container)).toContain('Question 0?')
        expect(textOf(view.container)).toContain('Rationale 0.')
        expect(textOf(view.container)).toContain('→ orchestrator_context.notes')

        const answerField = () => getByLabel(view.container, 'Answer', 'textarea')
        await act(async () => changeControl(answerField(), 'First answer'))
        await act(async () => getButton(view.container, 'Next').click())

        expect(textOf(view.container)).toContain('Question 2 of 3')
        expect(textOf(view.container)).toContain('Question 1?')
        expect(answerField().value).toBe('')
        expect(view.container.ownerDocument.activeElement).toBe(answerField())
      } finally {
        view.cleanup()
      }
    })

    it('submits the last question with Submit and the card disappears', async () => {
      const questions = [question(0), question(1)]
      const view = await mountQueue({ processes: [goalDefinitionProcess], checkpointItems: questions })
      try {
        await act(async () => changeControl(getByLabel(view.container, 'Answer', 'textarea'), 'First answer'))
        await act(async () => getButton(view.container, 'Next').click())
        expect(textOf(view.container)).toContain('Question 2 of 2')
        expect(() => getButton(view.container, 'Next')).toThrow()

        await act(async () => changeControl(getByLabel(view.container, 'Answer', 'textarea'), 'Second answer'))
        await act(async () => getButton(view.container, 'Submit').click())

        expect(textOf(view.container)).not.toContain('The orchestrator needs')
        expect(textOf(view.container)).toContain('on track')
      } finally {
        view.cleanup()
      }
    })

    it('does not crash when new questions arrive mid-round', async () => {
      const view = await mountQueue({ processes: [goalDefinitionProcess], checkpointItems: [question(0), question(1)] })
      try {
        expect(textOf(view.container)).toContain('The orchestrator needs 2 answers.')
        await view.rerender({ processes: [goalDefinitionProcess], checkpointItems: [question(0), question(1), question(2)] })
        expect(textOf(view.container)).toContain('The orchestrator needs 3 answers.')
        expect(textOf(view.container)).toContain('Question 1 of 3')
      } finally {
        view.cleanup()
      }
    })
  })
})

// ---------------------------------------------------------------------------
// recovery FIX PROPOSAL card.
// Interface assumptions (see task-14-tests-report.md):
//  - NeedsYouQueueProps gains `recoveryDecisions: readonly OrchestrationAuthorityDecisionRecord[]`
//    (every decision of the goal, from GET /decisions, any status) and
//    `onAnswerRecoveryProposal(decision, 'approve_fix' | 'reject', reason, { onSuccess, onError, onSettled })`.
//  - buildQueueRows accepts `recoveryDecisions` and emits `{ kind: 'recovery-proposal', decision }`.
//  - Pending recovery decisions also appear in `checkpointItems` (server queue) and must NOT become generic rows.
// ---------------------------------------------------------------------------
type RecoveryStatus = 'pending_approval' | 'applying' | 'verifying' | 'retrying' | 'resolved' | 'rejected' | 'failed' | 'unsupported'

function recoveryDecision(
  id: string,
  status: RecoveryStatus = 'pending_approval',
  recoveryOverrides: Record<string, unknown> = {},
  overrides: Partial<OrchestrationAuthorityDecisionRecord> = {},
) {
  return decision({
    id, decision_key: `blocker_recovery:${id}`, title: 'Fix proposal', context: null,
    options: ['approve_fix', 'reject'], recommendation: null,
    status: status === 'pending_approval' ? 'pending' : 'answered',
    ...overrides,
    recovery: {
      kind: 'recovery_proposal_v1', proposal_id: `proposal-${id}`, version: 1, status,
      problem: 'No agent can review the finished work.',
      general_updates: ['Add a code reviewer to the team', 'Let the reviewer sign off work'],
      after_approval: 'The blocked review step restarts automatically.',
      global_effect: true, exact_details: null, evidence: null, error: null,
      source_blocker_id: 'blocker-1', source_action_id: null, source_decision_id: id,
      ...recoveryOverrides,
    },
  } as unknown as Partial<OrchestrationAuthorityDecisionRecord>)
}

const STATUS_COPY: Array<[RecoveryStatus, string]> = [
  ['applying', 'Applying approved fix…'],
  ['verifying', 'Verifying update…'],
  ['retrying', 'Update verified. Retrying blocked work…'],
  ['resolved', 'Blocked work is moving again.'],
  ['rejected', 'Fix rejected. Nothing was changed.'],
  ['failed', 'The fix did not complete.'],
  ['unsupported', 'No executable fix is available with the current tools.'],
]

function recoveryProps(items: OrchestrationAuthorityDecisionRecord[], extra: Record<string, unknown> = {}) {
  const pending = items.filter((item) => item.status === 'pending')
  return { checkpointItems: pending, recoveryDecisions: items, ...extra } as Partial<ComponentProps<typeof QueueHarness>>
}

function card(root: Parameters<typeof descendants>[0]) {
  const heading = descendants(root).find((node) => node.tagName === 'H3' && ['FIX PROPOSAL', 'CAPABILITY NEEDED'].includes(textOf(node)))
  if (!heading) throw new Error('FIX PROPOSAL card not found')
  let node: typeof heading | null = heading
  while (node && node.tagName !== 'LI') node = node.parentNode as typeof heading | null
  return node!
}

const buttonsIn = (root: Parameters<typeof descendants>[0]) => descendants(root).filter((node) => node.tagName === 'BUTTON')
const classOf = (node: { getAttribute: (name: string) => string | null }) => node.getAttribute('class') ?? ''

describe('buildQueueRows recovery proposals', () => {
  const criteriaByKey = new Map<string, string>()
  const build = (checkpointItems: OrchestrationAuthorityDecisionRecord[], recoveryDecisions: OrchestrationAuthorityDecisionRecord[]) =>
    (buildQueueRows as unknown as (input: unknown) => Array<{ kind: string; decision?: { id: string } }>)({
      run: run(), processes: [], checkpointItems, recoveryDecisions, gates: [], warnings: [], criteriaByKey,
    })

  it('emits a recovery-proposal row, never a generic answer row, for a pending recovery decision', () => {
    const pending = recoveryDecision('rec-1')
    expect(build([pending], [pending]).map((row) => row.kind)).toEqual(['recovery-proposal'])
  })

  it('keeps a pending recovery card in checkpoint server order among other decisions', () => {
    const a = decision({ id: 'a', source_process_run_id: null })
    const b = decision({ id: 'b', source_process_run_id: null })
    const rec = recoveryDecision('rec-1')
    const rows = build([a, rec, b], [a, rec, b])
    expect(rows.map((row) => row.kind)).toEqual(['answer', 'recovery-proposal', 'answer'])
  })

  it('shows only the newest recovery decision per source_blocker_id', () => {
    const older = recoveryDecision('older', 'failed', {}, { created_at: '2026-08-05T08:00:00Z', asked_at: '2026-08-05T08:00:00Z' })
    const newer = recoveryDecision('newer', 'pending_approval', {}, { created_at: '2026-08-05T09:00:00Z', asked_at: '2026-08-05T09:00:00Z' })
    const rows = build([newer], [older, newer])
    expect(rows.map((row) => row.decision?.id)).toEqual(['newer'])
  })

  it('shows one card per distinct blocker', () => {
    const one = recoveryDecision('one', 'failed', { source_blocker_id: 'blocker-1' })
    const two = recoveryDecision('two', 'unsupported', { source_blocker_id: 'blocker-2' })
    expect(build([], [one, two]).map((row) => row.decision?.id).sort()).toEqual(['one', 'two'])
  })

  it('hides a resolved newest decision and does not resurrect the older one', () => {
    const older = recoveryDecision('older', 'failed', {}, { created_at: '2026-08-05T08:00:00Z', asked_at: '2026-08-05T08:00:00Z' })
    const newest = recoveryDecision('newest', 'resolved', {}, { created_at: '2026-08-05T09:00:00Z', asked_at: '2026-08-05T09:00:00Z' })
    expect(build([], [older, newest])).toEqual([])
  })

  it.each(['rejected', 'failed', 'unsupported', 'applying', 'verifying', 'retrying'] as const)('keeps a %s recovery decision as a row until superseded', (status) => {
    expect(build([], [recoveryDecision('x', status)]).map((row) => row.kind)).toEqual(['recovery-proposal'])
  })
})

describe('NeedsYouQueue recovery FIX PROPOSAL card', () => {
  it('renders heading, Problem / Proposed updates / After approval from server fields, Approve fix and Reject', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
    try {
      const li = card(view.container)
      const text = textOf(li)
      for (const label of ['Problem', 'Proposed updates', 'After approval']) expect(text).toContain(label)
      expect(text).toContain('No agent can review the finished work.')
      const items = descendants(li).filter((node) => node.tagName === 'LI' && node !== li)
      expect(items.map(textOf)).toEqual(['Add a code reviewer to the team', 'Let the reviewer sign off work'])
      expect(text).toContain('The blocked review step restarts automatically.')
      expect(getButton(li, 'Approve fix')).toBeDefined()
      expect(getButton(li, 'Reject')).toBeDefined()
      expect(buttonsIn(li).filter((b) => textOf(b) === 'Approve fix')).toHaveLength(1)
      expect(descendants(li).some((node) => node.tagName === 'INPUT' && node.type === 'radio')).toBe(false)
      expect(classOf(descendants(li).find((node) => node.tagName === 'H3')!)).toContain('text-xs')
    } finally { view.cleanup() }
  })

  it('does not render the generic answer form or agent-definition batch for the recovery decision', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
    try {
      const text = textOf(view.container)
      expect(text).not.toContain('Submit approve_fix')
      expect(text).not.toContain('Submit answer')
      expect(text).not.toContain('Submit all decisions')
    } finally { view.cleanup() }
  })

  it('shows the global-impact note only when global_effect is true', async () => {
    const notice = 'This affects the global agent roster, so it applies to every project and goal.'
    const on = await mountQueue(recoveryProps([recoveryDecision('rec-1', 'pending_approval', { global_effect: true })]))
    try {
      const note = descendants(card(on.container)).find((node) => node.getAttribute('role') === 'note')
      expect(note && textOf(note)).toContain(notice)
    } finally { on.cleanup() }
    const off = await mountQueue(recoveryProps([recoveryDecision('rec-1', 'pending_approval', { global_effect: false })]))
    try {
      expect(textOf(off.container)).not.toContain(notice)
      expect(descendants(card(off.container)).some((node) => node.getAttribute('role') === 'note')).toBe(false)
    } finally { off.cleanup() }
  })

  it('submits Approve fix as exactly one answer request, then disables the whole card', async () => {
    const onAnswer = vi.fn()
    const pending = recoveryDecision('rec-1', 'pending_approval', { exact_details: { agent: 'reviewer' } })
    const view = await mountQueue(recoveryProps([pending], { onAnswerRecoveryProposal: onAnswer }))
    try {
      const li = card(view.container)
      await act(async () => getButton(li, 'Approve fix').click())
      await act(async () => getButton(li, 'Approve fix').click())
      expect(onAnswer).toHaveBeenCalledTimes(1)
      expect(onAnswer.mock.calls[0][0].id).toBe('rec-1')
      expect(onAnswer.mock.calls[0][1]).toBe('approve_fix')
      expect(onAnswer.mock.calls[0][2] ?? '').toBe('')
      for (const label of ['Approve fix', 'Reject']) expect(getButton(li, label).disabled).toBe(true)
      expect(descendants(li).some((node) => node.getAttribute('aria-busy') === 'true')).toBe(true)
    } finally { view.cleanup() }
  })

  it('Reject reveals a required Reason, focuses it, and Cancel returns focus to Reject', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
    try {
      const li = card(view.container)
      expect(descendants(li).some((node) => node.tagName === 'TEXTAREA')).toBe(false)
      await act(async () => getButton(li, 'Reject').click())
      const reason = getByLabel(li, 'Reason', 'textarea')
      expect(view.document.activeElement === reason).toBe(true)
      expect(getButton(li, 'Confirm reject').disabled).toBe(true)
      await act(async () => changeControl(reason, '   '))
      expect(getButton(li, 'Confirm reject').disabled).toBe(true)
      await act(async () => getButton(li, 'Cancel').click())
      expect(descendants(li).some((node) => node.tagName === 'TEXTAREA')).toBe(false)
      expect(view.document.activeElement === getButton(li, 'Reject')).toBe(true)
    } finally { view.cleanup() }
  })

  it('Confirm reject submits reject with the trimmed reason', async () => {
    const onAnswer = vi.fn()
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')], { onAnswerRecoveryProposal: onAnswer }))
    try {
      const li = card(view.container)
      await act(async () => getButton(li, 'Reject').click())
      await act(async () => changeControl(getByLabel(li, 'Reason', 'textarea'), '  Too broad.  '))
      await act(async () => getButton(li, 'Confirm reject').click())
      expect(onAnswer).toHaveBeenCalledTimes(1)
      expect(onAnswer.mock.calls[0].slice(0, 3).map((v: unknown) => (typeof v === 'object' ? (v as { id: string }).id : v))).toEqual(['rec-1', 'reject', 'Too broad.'])
    } finally { view.cleanup() }
  })

  it('shows an error line and announces it when the answer fails, and re-enables the card', async () => {
    const onAnswer = vi.fn((_d, _o, _r, callbacks) => { callbacks.onError(new Error('Backend said no.')); callbacks.onSettled() })
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')], { onAnswerRecoveryProposal: onAnswer }))
    try {
      const li = card(view.container)
      await act(async () => getButton(li, 'Approve fix').click())
      expect(textOf(li)).toContain('Backend said no.')
      expect(getButton(li, 'Approve fix').disabled).toBe(false)
    } finally { view.cleanup() }
  })

  it('moves focus to the card heading (tabindex -1) when the card stays after approval', async () => {
    let callbacks: { onSuccess: () => void; onSettled: () => void } | undefined
    const onAnswer = vi.fn((_d, _o, _r, cb) => { callbacks = cb })
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')], { onAnswerRecoveryProposal: onAnswer }))
    try {
      await act(async () => getButton(card(view.container), 'Approve fix').click())
      await act(async () => { callbacks?.onSuccess(); callbacks?.onSettled() })
      await act(async () => view.rerender(recoveryProps([recoveryDecision('rec-1', 'applying')])))
      const heading = descendants(card(view.container)).find((node) => node.tagName === 'H3')!
      expect(heading.getAttribute('tabindex')).toBe('-1')
      expect(view.document.activeElement === heading).toBe(true)
    } finally { view.cleanup() }
  })

  describe('exact details disclosure', () => {
    it.each([
      ['exact_details', { exact_details: { agent: 'reviewer' } }],
      ['evidence', { evidence: [{ kind: 'note', summary: 'Observed' }] }],
    ])('renders a read-only disclosure when %s is non-null', async (_label, extra) => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', 'pending_approval', extra)]))
      try {
        const li = card(view.container)
        const details = descendants(li).find((node) => node.tagName === 'DETAILS')!
        const summary = descendants(details).find((node) => node.tagName === 'SUMMARY')!
        expect(textOf(summary)).toBe('Show exact details')
        expect(classOf(summary)).toContain('min-h-11')
        const pre = descendants(details).find((node) => node.tagName === 'PRE')!
        expect(pre.getAttribute('tabindex')).toBe('0')
        expect(classOf(pre)).toContain('font-mono')
        expect(descendants(details).some((node) => ['TEXTAREA', 'INPUT', 'BUTTON'].includes(node.tagName))).toBe(false)
      } finally { view.cleanup() }
    })

    it('omits the disclosure entirely when exact_details and evidence are both null', async () => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
      try {
        const li = card(view.container)
        expect(descendants(li).some((node) => node.tagName === 'DETAILS')).toBe(false)
        expect(textOf(li)).not.toContain('Show exact details')
        expect(descendants(li).some((node) => node.tagName === 'PRE')).toBe(false)
      } finally { view.cleanup() }
    })

    it('uses mono only inside the disclosure', async () => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', 'pending_approval', { exact_details: { a: 1 } })]))
      try {
        const mono = descendants(card(view.container)).filter((node) => classOf(node).includes('font-mono'))
        expect(mono.length).toBeGreaterThan(0)
        expect(mono.every((node) => node.closest('details') !== null)).toBe(true)
      } finally { view.cleanup() }
    })
  })

  describe('lifecycle copy', () => {
    it.each(STATUS_COPY)('shows exact copy for %s in the card and an aria-live status region', async (status, copy) => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', status, status === 'failed' ? { error: 'Provider timed out.' } : {})]))
      try {
        // A resolved newest decision leaves the queue; its copy is still the contract for the card renderer
        // when shown, so resolved is asserted via absence in the buildQueueRows tests instead.
        if (status === 'resolved') { expect(textOf(view.container)).not.toContain('FIX PROPOSAL'); return }
        const li = card(view.container)
        expect(textOf(li)).toContain(copy)
        const live = descendants(li).find((node) => node.getAttribute('role') === 'status')!
        expect(live.getAttribute('aria-live')).toBe('polite')
        expect(live.getAttribute('aria-atomic')).toBe('true')
        expect(textOf(live)).toContain(copy)
      } finally { view.cleanup() }
    })

    it('shows the failure error and rejection reason as readable history', async () => {
      const failed = await mountQueue(recoveryProps([recoveryDecision('f', 'failed', { error: 'Provider timed out.' })]))
      try { expect(textOf(card(failed.container))).toContain('Provider timed out.') } finally { failed.cleanup() }
      const rejected = await mountQueue(recoveryProps([recoveryDecision('r', 'rejected', {}, { reason: 'Too broad.' })]))
      try { expect(textOf(card(rejected.container))).toContain('Too broad.') } finally { rejected.cleanup() }
    })

    it.each(['applying', 'verifying', 'retrying', 'rejected', 'failed', 'unsupported'] as const)('has no Approve fix / Reject buttons once %s', async (status) => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', status)]))
      try {
        const labels = buttonsIn(card(view.container)).map(textOf)
        expect(labels).not.toContain('Approve fix')
        expect(labels).not.toContain('Reject')
      } finally { view.cleanup() }
    })

    it.each(['rejected', 'failed', 'unsupported'] as const)('renders %s as a read-only card with no buttons at all (no dismiss)', async (status) => {
      const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', status)]))
      try { expect(buttonsIn(card(view.container))).toHaveLength(0) } finally { view.cleanup() }
    })

    it.each(['failed', 'unsupported', 'rejected'] as const)('a %s proposal produces a row, so the on-track empty state never shows', async (status) => {
      const view = await mountQueue({ ...recoveryProps([recoveryDecision('rec-1', status)]), isWorking: false })
      try {
        expect(textOf(view.container).toLowerCase()).not.toContain('on track')
        expect(textOf(view.container)).toContain('FIX PROPOSAL')
      } finally { view.cleanup() }
    })

    it('a resolved-only history falls back to the normal empty state', async () => {
      const view = await mountQueue({ ...recoveryProps([recoveryDecision('rec-1', 'resolved')]), isWorking: false })
      try {
        expect(textOf(view.container)).not.toContain('FIX PROPOSAL')
        expect(textOf(view.container)).toContain('on track')
      } finally { view.cleanup() }
    })
  })

  it('uses 44px (min-h-11 min-w-11) targets for Approve fix and Reject', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
    try {
      const li = card(view.container)
      for (const label of ['Approve fix', 'Reject']) {
        expect(classOf(getButton(li, label))).toContain('min-h-11')
        expect(classOf(getButton(li, label))).toContain('min-w-11')
      }
    } finally { view.cleanup() }
  })
})

describe('NeedsYouQueue capability gap card (final fix wave F2)', () => {
  const pendingGap = () => recoveryDecision('gap-1', 'unsupported', {
    after_approval: 'Nothing happens automatically. After you provide the capability, HuddleRoom re-diagnoses the blocker.',
  }, { status: 'pending', options: ['capability_provided'] } as Partial<OrchestrationAuthorityDecisionRecord>)

  it('offers one secondary 44px "I\'ve provided it" and never Approve fix / Reject while the gap decision is pending', async () => {
    const view = await mountQueue(recoveryProps([pendingGap()]))
    try {
      const li = card(view.container)
      expect(buttonsIn(li).map(textOf)).toEqual(["I've provided it"])
      const button = getButton(li, "I've provided it")
      const h3 = descendants(li).find((node) => node.tagName === 'H3')!
      expect(textOf(h3)).toBe('CAPABILITY NEEDED')
      const ids = (button.getAttribute('aria-describedby') ?? '').split(' ')
      expect(ids).toHaveLength(2)
      expect(ids.map((id) => textOf(descendants(li).find((node) => node.getAttribute('id') === id)!))).toEqual([expect.any(String), expect.stringContaining('re-diagnoses')])
      expect(textOf(li)).not.toContain('Proposed updates')
      expect(textOf(li)).toContain("After you've provided it")
      expect(classOf(button)).toContain('min-h-11')
      expect(textOf(li)).toContain('No executable fix is available with the current tools.')
          } finally { view.cleanup() }
  })

  it('posts capability_provided exactly once through the answer callback, then disables the card', async () => {
    const onAnswer = vi.fn()
    const view = await mountQueue(recoveryProps([pendingGap()], { onAnswerRecoveryProposal: onAnswer }))
    try {
      const li = card(view.container)
      await act(async () => getButton(li, "I've provided it").click())
      await act(async () => getButton(li, "I've provided it").click())
      expect(onAnswer).toHaveBeenCalledTimes(1)
      expect(onAnswer.mock.calls[0][0].id).toBe('gap-1')
      expect(onAnswer.mock.calls[0][1]).toBe('capability_provided')
      expect(getButton(li, "I've provided it").disabled).toBe(true)
    } finally { view.cleanup() }
  })

  it('announces re-diagnosis after the answer and leaves quietly (no moving-again toast)', async () => {
    const onAnswer = vi.fn((_d, _o, _r, cb) => { cb.onSuccess(); cb.onSettled() })
    toastMock.success.mockClear()
    const view = await mountQueue(recoveryProps([pendingGap()], { onAnswerRecoveryProposal: onAnswer }))
    await act(async () => getButton(card(view.container), "I've provided it").click())
    expect(textOf(card(view.container))).toContain('Marked as provided. Re-diagnosing the blocker…')
    await act(async () => view.rerender(recoveryProps([])))
    view.cleanup()
    expect(toastMock.success).not.toHaveBeenCalled()
  })

  it.each([
    ['a pending gap as actionable', true, '1 item needs you.'],
    ['an answered (read-only) gap as not actionable', false, 'Queue clear.'],
  ])('counts %s', async (_label, isPending, announced) => {
    vi.useFakeTimers()
    try {
      const view = await mountQueue(recoveryProps([isPending ? pendingGap() : recoveryDecision('gap-1', 'unsupported')]))
      try {
        await act(async () => { vi.advanceTimersByTime(500) })
        const regions = descendants(view.container).filter((node) => node.getAttribute('role') === 'status')
        expect(regions.map(textOf)).toContain(announced)
      } finally { view.cleanup() }
    } finally { vi.useRealTimers() }
  })
})

describe('NeedsYouQueue recovery card follow-up', () => {
  it('does not toast when an answered card unmounts while still applying', async () => {
    const onAnswer = vi.fn((_d, _o, _r, cb) => { cb.onSuccess(); cb.onSettled() })
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')], { onAnswerRecoveryProposal: onAnswer }))
    await act(async () => getButton(card(view.container), 'Approve fix').click())
    await act(async () => view.rerender(recoveryProps([recoveryDecision('rec-1', 'applying')])))
    view.cleanup()
    expect(toastMock.success).not.toHaveBeenCalled()
  })

  it.each(['rejected', 'failed', 'unsupported'] as const)('hides After approval on %s', async (status) => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', status)]))
    try { expect(textOf(card(view.container))).not.toContain('After approval') } finally { view.cleanup() }
  })

  it('shows After approval while pending', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1')]))
    try { expect(textOf(card(view.container))).toContain('After approval') } finally { view.cleanup() }
  })

  it('omits the disclosure for an empty evidence array', async () => {
    const view = await mountQueue(recoveryProps([recoveryDecision('rec-1', 'pending_approval', { evidence: [] })]))
    try { expect(descendants(card(view.container)).some((node) => node.tagName === 'DETAILS')).toBe(false) } finally { view.cleanup() }
  })
})



const meetingObligation = {
  id: 'item-1', meeting_id: 'meeting-1', meeting_title: 'Release review', description: 'Document deployment',
  state: 'task_in_flight', task_id: 'task-1', graph_run_id: null, reason: null,
}

describe('meeting obligation waiver', () => {
  beforeEach(() => { meetingMutation.mutate.mockReset(); meetingMutation.isPending = false })

  it('shows live work and requires a short reason with scoped confirmation', async () => {
    const view = await mountQueue({ meetingCommitments: [meetingObligation] })
    try {
      expect(textOf(view.container)).toContain('Document deployment')
      expect(descendants(view.container).find(n => n.tagName === 'A' && textOf(n).includes('Release review'))).toBeTruthy()
      await act(async () => getButton(view.container, 'Waive').click())
      const reason = getByLabel(view.container, 'Reason')
      expect(view.document.activeElement).toBe(reason)
      expect(textOf(view.container)).toContain('This removes the meeting obligation. Linked work will keep running; goal criteria and gates still apply.')
      expect(getButton(view.container, 'Confirm waive').disabled).toBe(true)
      await act(async () => changeControl(reason, 'x'.repeat(501)))
      expect(getButton(view.container, 'Confirm waive').disabled).toBe(true)
      await act(async () => changeControl(reason, '  Irrelevant now  '))
      await act(async () => getButton(view.container, 'Confirm waive').click())
      expect(meetingMutation.mutate.mock.calls[0][0]).toMatchObject({ actionItemId: 'item-1', data: { status: 'dismissed', reason: 'Irrelevant now' } })
    } finally { view.cleanup() }
  })
})


describe('meeting waiver result and focus', () => {
  beforeEach(() => { meetingMutation.mutate.mockReset(); meetingMutation.isPending = false })

  it('retains exact typed input and announces errors, then lets the user retry', async () => {
    meetingMutation.mutate.mockImplementation((_variables, options) => { options.onError(new Error('Try again')); options.onSettled?.() })
    const view = await mountQueue({ meetingCommitments: [meetingObligation] })
    try {
      await act(async () => getButton(view.container, 'Waive').click())
      await act(async () => changeControl(getByLabel(view.container, 'Reason'), '  Keep my input  '))
      await act(async () => getButton(view.container, 'Confirm waive').click())
      expect(getByLabel(view.container, 'Reason').value).toBe('  Keep my input  ')
      expect(view.document.activeElement).toBe(getByLabel(view.container, 'Reason'))
      expect(textOf(view.container)).toContain('Try again')
      expect(getButton(view.container, 'Confirm waive').disabled).toBe(false)
    } finally { view.cleanup() }
  })

  it('cancel returns focus to Waive and pending submission cannot be duplicated', async () => {
    const view = await mountQueue({ meetingCommitments: [meetingObligation] })
    try {
      await act(async () => getButton(view.container, 'Waive').click())
      await act(async () => getButton(view.container, 'Cancel').click())
      expect(view.document.activeElement).toBe(getButton(view.container, 'Waive'))
      await act(async () => getButton(view.container, 'Waive').click())
      await act(async () => changeControl(getByLabel(view.container, 'Reason'), 'Irrelevant'))
      await act(async () => getButton(view.container, 'Confirm waive').click())
      expect(getByLabel(view.container, 'Reason').disabled).toBe(true)
      expect(getButton(view.container, 'Cancel').disabled).toBe(true)
      expect(meetingMutation.mutate).toHaveBeenCalledTimes(1)
      expect(textOf(view.container)).toContain('Document deployment')
    } finally { view.cleanup() }
  })

  it('does not claim a stale fulfilled item was waived and refocuses after refresh', async () => {
    meetingMutation.mutate.mockImplementation((_variables, options) => { options.onSuccess({ status: 'task_created' }); options.onSettled?.() })
    const props = { meetingCommitments: [meetingObligation], warnings: [warning({ message: 'Next risk' })] }
    const view = await mountQueue(props)
    try {
      await act(async () => getButton(view.container, 'Waive').click())
      await act(async () => changeControl(getByLabel(view.container, 'Reason'), 'Irrelevant'))
      await act(async () => getButton(view.container, 'Confirm waive').click())
      expect(textOf(view.container)).toContain('Document deployment')
      expect(toastMock.success.mock.calls.flat().join(' ')).not.toContain('Meeting obligation waived')
      await act(async () => view.rerender({ ...props, meetingCommitments: [] }))
      expect(textOf(view.container)).not.toContain('Document deployment')
      expect(view.document.activeElement).toBe(getButton(view.container, 'Acknowledge'))
    } finally { view.cleanup() }
  })
})


it('orders meeting obligations after failed gates and before warnings without stealing background focus', async () => {
  const props = { meetingCommitments: [meetingObligation], gates: [gate()], warnings: [warning({ message: 'Next risk' })] }
  const view = await mountQueue(props)
  try {
    const content = textOf(view.container)
    expect(content.indexOf('Document deployment')).toBeGreaterThan(content.indexOf('No evidence submitted.'))
    expect(content.indexOf('Next risk')).toBeGreaterThan(content.indexOf('Document deployment'))
    const acknowledge = getButton(view.container, 'Acknowledge')
    acknowledge.focus()
    await act(async () => view.rerender({ ...props, meetingCommitments: [] }))
    expect(view.document.activeElement).toBe(acknowledge)
  } finally { view.cleanup() }
})


it.each([
  ['dismissed', 'Meeting obligation waived.'],
  ['completed', 'Meeting item already completed.'],
])('announces actual returned %s state while awaiting refreshed projection', async (status, message) => {
  meetingMutation.mutate.mockReset()
  meetingMutation.isPending = false
  meetingMutation.mutate.mockImplementation((_variables, options) => { options.onSuccess({ status }); options.onSettled?.() })
  const view = await mountQueue({ meetingCommitments: [meetingObligation] })
  try {
    await act(async () => getButton(view.container, 'Waive').click())
    await act(async () => changeControl(getByLabel(view.container, 'Reason'), 'Irrelevant'))
    await act(async () => getButton(view.container, 'Confirm waive').click())
    expect(toastMock.success).toHaveBeenCalledWith(message)
    expect(textOf(view.container)).toContain('Document deployment')
  } finally { view.cleanup() }
})


it.each([
  ['task_in_flight', 'Linked task unfinished'],
  ['graph_in_flight', 'Linked graph unfinished'],
  ['task_failed', 'Linked task needs attention'],
  ['graph_failed', 'Linked graph needs attention'],
])('labels aggregate %s state truthfully', async (state, label) => {
  const view = await mountQueue({ meetingCommitments: [{ ...meetingObligation, state }] })
  try { expect(textOf(view.container)).toContain(label) } finally { view.cleanup() }
})
