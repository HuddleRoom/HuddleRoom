import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, useState, type ComponentProps } from 'react'
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
    onDismissStaleInputs: (_warning, onSuccessFocus) => onSuccessFocus?.(),
  }
  return <NeedsYouQueue {...defaults} {...props} openAction={openAction} onOpenAction={setOpenAction} />
}

async function mountQueue(props: Partial<ComponentProps<typeof QueueHarness>> = {}) {
  const render = (nextProps: Partial<ComponentProps<typeof QueueHarness>> = props) => <GoalAnnouncerProvider><QueueHarness {...nextProps} /></GoalAnnouncerProvider>
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

  it('keeps a warning row visible after Acknowledge (it does not clear the warning) and returns focus to it', async () => {
    const warn = warning({ id: 'warn-1', message: 'Needs a second look.' })
    const view = await mountQueue({ warnings: [warn] })
    try {
      await act(async () => getButton(view.container, 'Acknowledge').click())
      await act(async () => getButton(view.container, 'Submit acknowledge').click())

      expect(textOf(view.container)).toContain('Needs a second look.')
      const active = view.container.ownerDocument.activeElement
      expect(active && textOf(active)).toBe('Acknowledge')
      expect(toastMock.success).toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('offers Resolve instead of Acknowledge once a warning is already acknowledged, and removes it on resolve', async () => {
    const warn = warning({ id: 'warn-1', message: 'Already acknowledged.', acknowledged_at: ISO.started })
    const view = await mountQueue({ warnings: [warn] })
    try {
      expect(() => getButton(view.container, 'Acknowledge')).toThrow()
      await act(async () => getButton(view.container, 'Resolve').click())
      await act(async () => getButton(view.container, 'Submit resolve').click())
      expect(textOf(view.container)).not.toContain('Already acknowledged.')
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
