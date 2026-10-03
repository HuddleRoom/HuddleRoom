import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { changeControl, descendants, getButton, getByLabel, mountWithTestDom, textOf, type TestElement } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import { toast } from 'sonner'
import type {
  OrchestrationAgentReviewRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationDecisionOption,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
  OrchestrationWarningRecord,
} from '@/lib/types'

const api = vi.hoisted(() => ({
  dashboard: null as unknown,
  health: { data: { debug_enabled: false } } as unknown,
  memorySection: { data: undefined, isLoading: false, isError: false, refetch: vi.fn() },
  answer: { isPending: false, mutate: vi.fn() },
  answerAgentDefinitionBatch: { isPending: false, mutate: vi.fn() },
  skip: { isPending: false, mutate: vi.fn() },
  acknowledge: { isPending: false, mutate: vi.fn() },
  resolve: { isPending: false, mutate: vi.fn() },
  run: { isPending: false, mutate: vi.fn() },
  rerun: { isPending: false, mutate: vi.fn() },
  retry: { isPending: false, mutate: vi.fn() },
  authorize: { isPending: false, mutate: vi.fn() },
  debugStep: { isPending: false, mutate: vi.fn() },
  debugRerun: { isPending: false, mutate: vi.fn() },
}))

vi.mock('sonner', () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}))

// HierarchyGraph (Phase 12: lazy-loaded) renders reactflow, which needs real browser
// APIs (ResizeObserver, window.addEventListener, …) this test suite's lightweight DOM
// shim doesn't provide. Mock it to a plain div carrying the class the tests assert on —
// the mock applies to the dynamic import the same as a static one.
vi.mock('reactflow', () => ({
  default: ({ children }: { children?: React.ReactNode }) => React.createElement('div', { className: 'react-flow' }, children),
  Position: { Top: 'top', Bottom: 'bottom' },
}))

vi.mock('@/api/orchestration', () => ({
  useBaselineDashboard: () => api.dashboard,
  useOrchestrationHealth: () => api.health,
  useOrchestrationMemorySection: () => api.memorySection,
  useAnswerOrchestrationDecision: () => api.answer,
  useAnswerAgentDefinitionReviewBatch: () => api.answerAgentDefinitionBatch,
  useSkipOrchestrationProcess: () => api.skip,
  useAcknowledgeOrchestrationWarning: () => api.acknowledge,
  useResolveOrchestrationWarning: () => api.resolve,
  useRunOrchestrationBaselineStep: () => api.run,
  useRerunOrchestrationBaselineStep: () => api.rerun,
  useRetryOrchestrationBaselineStep: () => api.retry,
  useAuthorizeOrchestrationBaseline: () => api.authorize,
  // Retained while the D3 component replaces the debug-only controls.
  useStepOrchestrationBaseline: () => api.debugStep,
  useRerunLastOrchestrationBaseline: () => api.debugRerun,
}))

import { BaselineActionForm, BaselineDashboard } from './BaselineDashboard'
import { GoalAnnouncerProvider } from './goalAnnouncer'

const ISO = {
  started: '2026-08-05T08:00:00Z',
  asked: '2026-08-05T08:10:00Z',
  answered: '2026-08-05T08:20:00Z',
  flagged: '2026-08-05T08:30:00Z',
  acknowledged: '2026-08-05T08:40:00Z',
  resolved: '2026-08-05T08:50:00Z',
  completed: '2026-08-05T09:00:00Z',
}

function query<T>(data: T | undefined, overrides: Partial<{ isLoading: boolean; isError: boolean }> = {}) {
  const merged = { isLoading: false, isError: false, ...overrides }
  return { data, refetch: vi.fn(), ...merged, isSuccess: !merged.isLoading && !merged.isError && data !== undefined }
}

function goal(overrides: Partial<OrchestrationGoal> = {}): OrchestrationGoal {
  return {
    id: 'goal-00000000-0000-4000-8000-000000000001',
    project_id: 'project-1',
    objective: 'Ship the operator dashboard.',
    success_criteria: [], orchestrator_context: {}, constraints: {}, budget: {},
    status: 'active', weight: 'standard', weight_overridden_by: null,
    manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null,
    created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function run(overrides: Partial<OrchestrationRun> = {}): OrchestrationRun {
  return {
    id: 'run-00000000-0000-4000-8000-000000000001',
    goal_id: 'goal-00000000-0000-4000-8000-000000000001',
    status: 'running', event_cursor: null, plan_state: {}, active_blockers: [], budget_state: {}, retry_state: {},
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-00000000-0000-4000-8000-000000000001',
    goal_id: 'goal-00000000-0000-4000-8000-000000000001',
    run_id: 'run-00000000-0000-4000-8000-000000000001', process_type: 'goal_definition',
    process_version: 1, status: 'running', trigger_reason: 'operator_test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function decision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'decision-00000000-0000-4000-8000-000000000001',
    goal_id: 'goal-00000000-0000-4000-8000-000000000001',
    run_id: 'run-00000000-0000-4000-8000-000000000001', decision_key: 'review_scope',
    title: 'Approve the review scope', status: 'pending', authority: 'human', authority_agent_id: null,
    source_process_run_id: null, question: 'Approve the review scope?', context: null, options: ['approve'],
    recommendation: 'approve', consequences: null, selected_option: null, reason: null,
    decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: ISO.asked, decided_at: null, created_at: ISO.asked, updated_at: ISO.asked,
    ...overrides,
  }
}

function warning(overrides: Partial<OrchestrationWarningRecord> = {}): OrchestrationWarningRecord {
  return {
    id: 'warning-00000000-0000-4000-8000-000000000001',
    goal_id: 'goal-00000000-0000-4000-8000-000000000001',
    run_id: 'run-00000000-0000-4000-8000-000000000001', warning_type: 'coverage_gap', severity: 'warning',
    message: 'Coverage evidence is incomplete.', source_process_run_id: null, related_gate_id: null,
    related_action_id: null, related_agent_id: null, source_agent_review_id: null,
    related_authority_decision_id: null, acknowledged_by: null, acknowledged_at: null, active: true,
    resolved_by: null, resolved_reason: null, resolved_at: null, blocks_completion: false,
    created_at: ISO.flagged, updated_at: ISO.flagged,
    ...overrides,
  }
}

function review(overrides: Partial<OrchestrationAgentReviewRecord> = {}): OrchestrationAgentReviewRecord {
  return {
    id: 'review-00000000-0000-4000-8000-000000000001',
    goal_id: 'goal-00000000-0000-4000-8000-000000000001',
    run_id: 'run-00000000-0000-4000-8000-000000000001',
    agent_id: 'agent-1', source_process_run_id: null, review_context: null,
    proposed_work_functions: [], definition_snapshot: {}, fit_summary: '', strengths: [], risks: [],
    recommended_changes: [], approved_for_work_functions: [], created_at: ISO.flagged, updated_at: ISO.flagged,
    ...overrides,
  }
}

function dashboardData(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    processes: query<OrchestrationProcessRunRecord[]>([]),
    decisions: query<OrchestrationAuthorityDecisionRecord[]>([]),
    checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [], deferred_count: 0, max_questions: 3 }),
    warnings: query<OrchestrationWarningRecord[]>([]),
    agentReviews: query([]),
    memory: query({ preface: { objective: null, goal_status: 'active', goal_weight: 'standard', run_status: 'running', current_process: null, manager: null, hierarchy: null, constraints: null, active_warnings: [], recent_decisions: [], open_blockers: [], skipped_processes: [], introduction: null, always_loaded: [], toc: [] }, toc: [], always_loaded: [] }),
    ...overrides,
  }
}

function renderDashboard(props: Partial<ComponentProps<typeof BaselineDashboard>> = {}) {
  return renderToStaticMarkup(
    <BaselineDashboard projectId="project-1" goalId="goal-00000000-0000-4000-8000-000000000001"
      goal={goal()} run={run()} gates={[]} mutationBusy={false} acquireMutation={() => true}
      releaseMutation={() => undefined} onMessage={() => undefined} {...props} />,
  )
}

async function mountDashboard(props: Partial<ComponentProps<typeof BaselineDashboard>> = {}) {
  const renderDashboardComponent = (nextProps = props) => (
    <GoalAnnouncerProvider>
      <BaselineDashboard projectId="project-1" goalId="goal-00000000-0000-4000-8000-000000000001"
        goal={goal()} run={run()} gates={[]} mutationBusy={false} acquireMutation={() => true}
        releaseMutation={() => undefined} onMessage={() => undefined} {...nextProps} />
    </GoalAnnouncerProvider>
  )
  const view = await mountWithTestDom(() => renderDashboardComponent(), act)
  return { ...view, rerender: (nextProps: Partial<ComponentProps<typeof BaselineDashboard>> = props) => view.rerender(() => renderDashboardComponent(nextProps)) }
}

function control(markup: string, label: string) {
  return new RegExp(`<button[^>]*aria-label="${label}"[^>]*>[^<]*</button>`).test(markup)
}

function controlElement(root: Parameters<typeof descendants>[0], label: string) {
  const element = descendants(root).find((node) => node.tagName === 'BUTTON'
    && node.getAttribute('aria-label') === label)
  if (!element) throw new Error(`Control not found: ${label}`)
  return element
}

beforeEach(() => {
  api.dashboard = dashboardData()
  api.health = { data: { debug_enabled: false } }
  for (const mutation of [api.answer, api.answerAgentDefinitionBatch, api.skip, api.acknowledge, api.resolve, api.run, api.rerun, api.retry, api.debugStep, api.debugRerun]) {
    mutation.isPending = false
    mutation.mutate.mockReset()
  }
})

describe('Baseline dashboard D3 composition', () => {
  it.each([
    ['goal_definition', 'Goal definition'], ['manager_selection', 'Manager selection'],
    ['agent_definition_review', 'Agent definition review'], ['team_hierarchy', 'Team hierarchy'],
    ['effectiveness_review', 'Effectiveness review'], ['goal_closeout', 'Goal closeout'],
  ] as const)('shows Retry only for an LM-linked warning on %s and retains unrelated warning actions', (processType, label) => {
    const focused = process({ id: `${processType}-process`, process_type: processType, outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: 'lm-warning' } } })
    api.dashboard = dashboardData({
      processes: query([focused]),
      warnings: query([
        warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'LM request failed.' }),
        warning({ id: 'ordinary-warning', source_process_run_id: focused.id, message: 'Ordinary warning.' }),
      ]),
    })

    const markup = renderDashboard()
    expect(markup).toContain('Retry continues the failed request and keeps this run. Re-run starts a fresh run from the saved inputs.')
    expect(markup).toContain('>Retry</button>')
    expect(markup).not.toContain(`aria-label="Run ${label}"`)
    const lmWarning = markup.slice(markup.indexOf('LM request failed.'), markup.indexOf('Ordinary warning.'))
    expect(lmWarning).not.toContain('Acknowledge')
    expect(lmWarning).not.toContain('Resolve')
    // The Needs-you queue exposes one primary action per warning at a time:
    // Acknowledge while unacknowledged, Resolve once it has been.
    const ordinaryWarning = markup.slice(markup.indexOf('Ordinary warning.'))
    expect(ordinaryWarning).toContain('Acknowledge')
    expect(ordinaryWarning).not.toContain('Resolve')
  })

  it('keeps ordinary warning actions and Skip when an LM retry descriptor is unavailable', () => {
    const focused = process({ id: 'unavailable-retry-process', outputs: { lm_retry: { available: false, kind: null, warning_id: 'lm-warning' } } })
    api.dashboard = dashboardData({
      processes: query([focused]),
      warnings: query([warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'Retry is unavailable.' })]),
    })

    const markup = renderDashboard()
    expect(markup).not.toContain('>Retry</button>')
    expect(markup).toContain('Acknowledge')
    expect(markup).not.toContain('Resolve')
    expect(markup).toContain('>Skip</button>')
  })

  it('offers Re-run alongside Retry for a retry-stuck step (running or waiting-decision with a live retry checkpoint), and for a terminal one', async () => {
    const retryableRunning = process({
      id: 'retryable-running',
      status: 'running',
      outputs: { lm_retry: { available: true, kind: 'goal_analysis', warning_id: 'running-warning' } },
    })
    const retryableWaiting = process({
      id: 'retryable-waiting',
      process_type: 'manager_selection',
      status: 'waiting_decision',
      outputs: { lm_retry: { available: true, kind: 'manager_selection', warning_id: 'waiting-warning' } },
    })
    const terminalSkipped = process({
      id: 'terminal-skipped',
      process_type: 'team_hierarchy',
      status: 'skipped',
      completed_at: ISO.completed,
    })
    api.dashboard = dashboardData({ processes: query([retryableRunning, retryableWaiting, terminalSkipped]) })

    const view = await mountDashboard()
    try {
      // Running + retryable: the RetryControl shows in both the ProcessFocus
      // header (Phase B) and the Needs-you queue's LM-retry row (Phase C).
      // Re-run now shows too (review round 4, finding U2 — mirrors the
      // pre-redesign stepper's retryableStuck rule and the backend rerun
      // route's own eligibility, both of which already accept this state).
      // Run stays hidden either way (showRetryControl suppresses it).
      await act(async () => controlElement(view.container, 'Select Goal definition').click())
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON' && textOf(node) === 'Retry')).toHaveLength(2)
      expect(textOf(view.container)).toContain('Retry continues the failed request and keeps this run. Re-run starts a fresh run from the saved inputs.')
      expect(() => controlElement(view.container, 'Run Goal definition')).toThrow()
      await act(async () => controlElement(view.container, 'Re-run Goal definition').click())
      expect(api.rerun.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', processType: 'goal_definition' },
        expect.any(Object),
      )

      // Waiting on a decision with a live retry checkpoint is the same
      // "retry-stuck" state (the backend's rerun route accepts it too) — Re-run
      // shows, but there's still no header Retry/Run for a waiting-decision
      // step (Retry only reflects an *actively running* request). The queue's
      // LM-retry row for the still-running goal_definition step is goal-wide,
      // so it keeps showing regardless of which step is selected.
      await act(async () => controlElement(view.container, 'Select Manager selection').click())
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON' && textOf(node) === 'Retry')).toHaveLength(1)
      expect(() => controlElement(view.container, 'Run Manager selection')).toThrow()
      expect(() => controlElement(view.container, 'Re-run Manager selection')).not.toThrow()

      // A terminal step offers Re-run and wires it through the rerun mutation; the
      // queue's goal_definition retry row still shows (goal-wide, not selection-scoped).
      await act(async () => controlElement(view.container, 'Select Team hierarchy').click())
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON' && textOf(node) === 'Retry')).toHaveLength(1)
      await act(async () => controlElement(view.container, 'Re-run Team hierarchy').click())
      expect(api.rerun.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', processType: 'team_hierarchy' },
        expect.any(Object),
      )
    } finally {
      view.cleanup()
    }
  })

  it('keeps Re-run available for a skipped process with stale LM retry metadata', () => {
    const skipped = process({
      id: 'skipped-retryable',
      process_type: 'agent_definition_review',
      status: 'skipped',
      completed_at: ISO.completed,
      outputs: { lm_retry: { available: true, kind: 'agent_definition_review', warning_id: 'stale-warning' } },
    })
    api.dashboard = dashboardData({ processes: query([skipped]) })

    const markup = renderDashboard()
    expect(control(markup, 'Re-run Agent definition review')).toBe(true)
    expect(markup).not.toContain('>Retry</button>')
  })

  // Regression (review round 4, finding U2): a step stuck in "needs retry"
  // (running + a live LM-retry checkpoint) offered only Retry and Skip —
  // Re-run was gated to terminal steps only. All three now show together.
  it('shows Retry, Re-run, and Skip together for a retry-state step', () => {
    const stuck = process({ id: 'stuck-process', status: 'running', outputs: { lm_retry: { available: true, kind: 'goal_analysis', warning_id: 'stuck-warning' } } })
    api.dashboard = dashboardData({ processes: query([stuck]) })

    const markup = renderDashboard()
    expect(markup).toContain('>Retry</button>')
    expect(control(markup, 'Re-run Goal definition')).toBe(true)
    expect(markup).toContain('>Skip</button>')
    expect(control(markup, 'Run Goal definition')).toBe(false)
  })

  it('retries the selected LM request and reports an error in the focused process', async () => {
    const focused = process({ id: 'retry-process', outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: 'lm-warning' } } })
    const onMessage = vi.fn()
    api.dashboard = dashboardData({
      processes: query([focused]),
      warnings: query([warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'LM request failed.' })]),
    })
    api.retry.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new Error('Provider unavailable'))
      callbacks.onSettled?.()
    })

    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(view.container, 'Retry').click())
      expect(api.retry.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', processType: 'goal_definition' },
        expect.any(Object),
      )
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'alert' && textOf(node).includes('Provider unavailable'))).toBe(true)
    } finally {
      view.cleanup()
    }
  })

  it('shows pending retry state, restores focus after success, and reports the continuation', async () => {
    const focused = process({ id: 'retry-success-process', outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: 'lm-warning' } } })
    const onMessage = vi.fn()
    let callbacks: { onSuccess?: () => void; onSettled?: () => void } | undefined
    api.dashboard = dashboardData({
      processes: query([focused]),
      warnings: query([warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'LM request failed.' })]),
    })
    api.retry.mutate.mockImplementation((_variables, options) => {
      api.retry.isPending = true
      callbacks = options
    })

    // Two Retry buttons render now (ProcessFocus header + Needs-you queue row)
    // — scope to the header's, which restores focus to the process heading.
    const focusPanel = () => descendants(view.container).find((node) => node.getAttribute('aria-labelledby') === 'baseline-process-focus-heading')!
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(focusPanel(), 'Retry').click())
      await view.rerender()
      const pendingRetry = getButton(focusPanel(), 'Retrying…')
      expect(pendingRetry.getAttribute('disabled')).toBe('')

      api.retry.isPending = false
      api.dashboard = dashboardData({
        processes: query([process({ ...focused, outputs: { lm_retry: { available: false, kind: null, warning_id: null } } })]),
        warnings: query([warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'LM request failed.' })]),
      })
      await act(async () => { callbacks?.onSuccess?.(); callbacks?.onSettled?.() })
      await view.rerender()
      expect(onMessage).toHaveBeenCalledWith('Request succeeded. Continuing Goal definition.')
      expect(view.container.ownerDocument.activeElement?.id).toBe('baseline-process-focus-heading')
    } finally {
      view.cleanup()
    }
  })

  it('refreshes stale retry state by ApiError status and reports the precise message', async () => {
    const focused = process({ id: 'stale-retry-process', outputs: { lm_retry: { available: true, kind: 'provider_error', warning_id: 'lm-warning' } } })
    const onMessage = vi.fn()
    const processes = query([focused])
    const warnings = query([warning({ id: 'lm-warning', source_process_run_id: focused.id, message: 'LM request failed.' })])
    api.dashboard = dashboardData({ processes, warnings })
    api.retry.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new ApiError(409, 'Conflict without status text'))
      callbacks.onSettled?.()
    })

    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(view.container, 'Retry').click())
      expect(processes.refetch).toHaveBeenCalled()
      expect(warnings.refetch).toHaveBeenCalled()
      expect(onMessage).toHaveBeenCalledWith('This retry is no longer available. Refreshing process state.')
    } finally {
      view.cleanup()
    }
  })
  it('shows Run only on the frontier step and Re-run only on a terminal one, in the header of whichever step is selected', async () => {
    const steps = {
      goal_definition: { select: 'Select Goal definition', control: 'Goal definition' },
      manager_selection: { select: 'Select Manager selection', control: 'Manager selection' },
      agent_definition_review: { select: 'Select Reviewing agent definitions', control: 'Agent definition review' },
      team_hierarchy: { select: 'Select Team hierarchy', control: 'Team hierarchy' },
      goal_closeout: { select: 'Select Goal closeout', control: 'Goal closeout' },
    } as const
    function tryFind(container: Parameters<typeof descendants>[0], label: string) {
      try {
        return controlElement(container, label)
      } catch {
        return null
      }
    }
    async function check(container: Parameters<typeof descendants>[0], step: keyof typeof steps, expected: { run?: boolean; rerun?: boolean }, name: string) {
      const { select, control } = steps[step]
      await act(async () => controlElement(container, select).click())
      expect(!!tryFind(container, `Run ${control}`), `${name}: Run ${control}`).toBe(expected.run ?? false)
      expect(!!tryFind(container, `Re-run ${control}`), `${name}: Re-run ${control}`).toBe(expected.rerun ?? false)
    }

    // First step, nothing started yet: frontier is goal_definition.
    {
      api.dashboard = dashboardData({ processes: query([]) })
      const view = await mountDashboard()
      try {
        await check(view.container, 'goal_definition', { run: true }, 'first missing step')
        await check(view.container, 'manager_selection', {}, 'missing successor before predecessor terminal')
      } finally { view.cleanup() }
    }

    // goal_definition terminal: frontier moves to manager_selection; goal_definition itself offers Re-run.
    {
      api.dashboard = dashboardData({ processes: query([process({ status: 'completed', completed_at: ISO.completed })]) })
      const view = await mountDashboard()
      try {
        await check(view.container, 'goal_definition', { rerun: true }, 'completed first step')
        await check(view.container, 'manager_selection', { run: true }, 'manager after terminal goal definition')
      } finally { view.cleanup() }
    }

    // A step waiting on a decision is neither runnable nor rerunnable — resolved through the decision, not a chain control.
    {
      api.dashboard = dashboardData({ processes: query([process({ status: 'waiting_decision' })]) })
      const view = await mountDashboard()
      try {
        await check(view.container, 'goal_definition', {}, 'waiting decision')
      } finally { view.cleanup() }
    }

    // A step stuck on something else (e.g. its own decision) doesn't block a later
    // step whose own predecessors already cleared.
    {
      api.dashboard = dashboardData({ processes: query([
        process({ status: 'waiting_decision' }),
        process({ id: 'manager-selection', process_type: 'manager_selection', status: 'completed', completed_at: ISO.completed }),
      ]) })
      const view = await mountDashboard()
      try {
        await check(view.container, 'goal_definition', {}, 'goal definition still waiting on its decision')
        await check(view.container, 'agent_definition_review', { run: true }, 'agent review runnable despite the earlier stuck step')
      } finally { view.cleanup() }
    }

    // Closeout eligibility follows the same terminal-baseline chain and service-owned business rules as before.
    const terminalBaseline = [
      process({ id: 'goal-definition', status: 'completed', completed_at: ISO.completed }),
      process({ id: 'manager-selection', process_type: 'manager_selection', status: 'completed', completed_at: ISO.completed }),
      process({ id: 'agent-review', process_type: 'agent_definition_review', status: 'completed', completed_at: ISO.completed }),
      process({ id: 'hierarchy', process_type: 'team_hierarchy', status: 'skipped', completed_at: ISO.completed }),
      process({ id: 'effectiveness', process_type: 'effectiveness_review', status: 'skipped', completed_at: ISO.completed }),
    ]
    const closeoutCases: Array<{
      name: string
      processes: OrchestrationProcessRunRecord[]
      goal?: OrchestrationGoal
      run?: OrchestrationRun
      warnings?: OrchestrationWarningRecord[]
      expectRun: boolean
    }> = [
      { name: 'eligible closeout from all terminal baseline predecessors', processes: terminalBaseline, expectRun: true },
      { name: 'eligible blocked-goal closeout', processes: terminalBaseline, goal: goal({ status: 'blocked' }), expectRun: true },
      { name: 'blocked run remains tickable', processes: terminalBaseline, run: run({ status: 'blocked' }), expectRun: true },
      { name: 'warning and budget manifests remain service-owned', processes: terminalBaseline, warnings: [warning({ active: true, acknowledged_at: null })], run: run({ budget_state: { status: 'exceeded', overridden: false } }), expectRun: true },
      { name: 'closeout blocked by an effectiveness review awaiting a decision', processes: [...terminalBaseline.slice(0, 4), process({ id: 'effectiveness', process_type: 'effectiveness_review', status: 'waiting_decision' })], expectRun: false },
      { name: 'closeout blocked by active run blocker', processes: terminalBaseline, run: run({ active_blockers: [{ kind: 'evidence_gap' }] }), expectRun: false },
      { name: 'closeout blocked by terminal goal', processes: terminalBaseline, goal: goal({ status: 'completed' }), expectRun: false },
    ]
    for (const row of closeoutCases) {
      api.dashboard = dashboardData({ processes: query(row.processes), warnings: query(row.warnings ?? []) })
      const view = await mountDashboard({ goal: row.goal ?? goal(), run: row.run ?? run() })
      try {
        await check(view.container, 'goal_closeout', { run: row.expectRun }, row.name)
      } finally { view.cleanup() }
    }
  })

  it('runs and reruns the selected step through the public baseline mutations and protects pending controls', async () => {
    api.dashboard = dashboardData({ processes: query([]) })
    const runView = await mountDashboard()
    try {
      await act(async () => controlElement(runView.container, 'Run Goal definition').click())
      expect(api.run.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', processType: 'goal_definition' },
        expect.any(Object),
      )
    } finally {
      runView.cleanup()
    }

    api.dashboard = dashboardData({ processes: query([process({ status: 'completed', completed_at: ISO.completed })]) })
    const rerunView = await mountDashboard()
    try {
      await act(async () => controlElement(rerunView.container, 'Re-run Goal definition').click())
      expect(api.rerun.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', processType: 'goal_definition' },
        expect.any(Object),
      )
    } finally {
      rerunView.cleanup()
    }

    api.run.mutate.mockReset()
    api.run.isPending = true
    api.dashboard = dashboardData({ processes: query([]) })
    const pendingView = await mountDashboard()
    try {
      const pendingRun = controlElement(pendingView.container, 'Run Goal definition')
      expect(pendingRun.disabled).toBe(true)
      await act(async () => pendingRun.click())
      expect(api.run.mutate).not.toHaveBeenCalled()
    } finally {
      pendingView.cleanup()
    }
  })

  it('keeps the process focus panel on the run step once the Run mutation succeeds', async () => {
    api.dashboard = dashboardData({ processes: query([process({ status: 'completed', completed_at: ISO.completed })]) })
    api.run.mutate.mockImplementation((_variables, options) => { options.onSuccess?.(); options.onSettled?.() })
    const view = await mountDashboard()
    try {
      expect(controlElement(view.container, 'Select Goal definition').getAttribute('aria-pressed')).toBe('true')
      await act(async () => controlElement(view.container, 'Select Manager selection').click())
      await act(async () => controlElement(view.container, 'Run Manager selection').click())
      expect(controlElement(view.container, 'Select Manager selection').getAttribute('aria-pressed')).toBe('true')
      expect(controlElement(view.container, 'Select Goal definition').getAttribute('aria-pressed')).toBe('false')
    } finally {
      view.cleanup()
    }
  })

  it('gives every step-select button a pointer cursor and only the unselected ones a hover affordance', async () => {
    api.dashboard = dashboardData({ processes: query([
      process({ status: 'completed', completed_at: ISO.completed }),
      process({ id: 'manager', process_type: 'manager_selection', status: 'completed', completed_at: ISO.completed }),
    ]) })
    const view = await mountDashboard()
    try {
      const selected = controlElement(view.container, 'Select Goal definition')
      const unselected = controlElement(view.container, 'Select Manager selection')
      expect(selected.getAttribute('aria-pressed')).toBe('true')
      // Both stay clickable, so both carry cursor-pointer; hover ring is unselected-only.
      expect(selected.getAttribute('class')).toContain('cursor-pointer')
      expect(unselected.getAttribute('class')).toContain('cursor-pointer')
      expect(unselected.getAttribute('class')).toContain('hover:bg-huddleroom-surface')
      expect(selected.getAttribute('class')).not.toContain('hover:bg-huddleroom-surface')
    } finally {
      view.cleanup()
    }
  })

  it('shows a retryable goal-analyzer failure inline with its existing Run affordance', async () => {
    api.dashboard = dashboardData({
      processes: query([process({
        status: 'running',
        outputs: { error: 'RuntimeError: temporary analyzer failure', retryable: true },
      })]),
    })

    const view = await mountDashboard({
      goal: goal({ status: 'blocked' }),
      run: run({
        status: 'blocked',
        active_blockers: [{ kind: 'goal_definition_analyzer_error', reason: 'Goal analysis failed: RuntimeError: temporary analyzer failure' }],
      }),
    })
    try {
      const alert = descendants(view.container).find((node) => node.getAttribute('role') === 'alert')
      expect(alert).toBeDefined()
      expect(textOf(alert!)).toContain('Goal analysis failed: RuntimeError: temporary analyzer failure')
      expect(controlElement(view.container, 'Run Goal definition')).toBeDefined()
    } finally {
      view.cleanup()
    }
  })

  it.each([
    {
      name: 'Run',
      processes: [] as OrchestrationProcessRunRecord[],
      control: 'Run Goal definition',
      pendingText: 'Running…',
      mutation: api.run,
    },
    {
      name: 'Re-run',
      processes: [process({ status: 'completed', completed_at: ISO.completed })],
      control: 'Re-run Goal definition',
      pendingText: 'Re-running…',
      mutation: api.rerun,
    },
  ])('shows $name pending feedback on the header control while the mutation is in flight', async ({ processes, control: controlLabel, pendingText, mutation }) => {
    api.dashboard = dashboardData({ processes: query(processes) })
    mutation.mutate.mockImplementation(() => { mutation.isPending = true })
    const view = await mountDashboard()
    try {
      await act(async () => controlElement(view.container, controlLabel).click())
      await view.rerender()

      expect(textOf(controlElement(view.container, controlLabel))).toBe(pendingText)
    } finally {
      view.cleanup()
      mutation.isPending = false
    }
  })

  it('auto-selects delayed running data once, then preserves a manual selection through refetch', async () => {
    api.dashboard = dashboardData({ processes: query<OrchestrationProcessRunRecord[]>(undefined, { isLoading: true }) })
    const view = await mountDashboard()
    try {
      const review = process({ id: 'review', process_type: 'agent_definition_review', status: 'running' })
      const hierarchy = process({ id: 'hierarchy', process_type: 'team_hierarchy', status: 'completed', completed_at: ISO.completed })
      api.dashboard = dashboardData({ processes: query([review, hierarchy]) })
      await view.rerender()
      expect(controlElement(view.container, 'Select Reviewing agent definitions').getAttribute('aria-pressed')).toBe('true')
      expect(controlElement(view.container, 'Select Reviewing agent definitions').getAttribute('data-selected')).toBe('true')
      expect(controlElement(view.container, 'Select Team hierarchy').getAttribute('data-selected')).toBe('false')

      await act(async () => controlElement(view.container, 'Select Team hierarchy').click())
      expect(controlElement(view.container, 'Select Team hierarchy').getAttribute('aria-pressed')).toBe('true')
      expect(controlElement(view.container, 'Select Reviewing agent definitions').getAttribute('aria-pressed')).toBe('false')
      expect(controlElement(view.container, 'Select Team hierarchy').getAttribute('data-selected')).toBe('true')
      expect(controlElement(view.container, 'Select Reviewing agent definitions').getAttribute('data-selected')).toBe('false')

      api.dashboard = dashboardData({ processes: query([review, hierarchy]) })
      await view.rerender()
      expect(controlElement(view.container, 'Select Team hierarchy').getAttribute('aria-pressed')).toBe('true')
    } finally {
      view.cleanup()
    }
  })

  it('reports the effective selected process type on change (Activity-tab sync)', async () => {
    const onSelectedProcessTypeChange = vi.fn()
    const review = process({ id: 'review', process_type: 'agent_definition_review', status: 'running' })
    api.dashboard = dashboardData({ processes: query([review]) })
    const view = await mountDashboard({ onSelectedProcessTypeChange })
    try {
      expect(onSelectedProcessTypeChange).toHaveBeenCalledWith('agent_definition_review')
      onSelectedProcessTypeChange.mockClear()
      await act(async () => controlElement(view.container, 'Select Goal definition').click())
      expect(onSelectedProcessTypeChange).toHaveBeenCalledWith('goal_definition')
    } finally {
      view.cleanup()
    }
  })

  it('renders complete agent-review records as readable standard-mode cards', () => {
    const reviewProcess = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'completed', completed_at: ISO.completed })
    api.dashboard = dashboardData({
      processes: query([reviewProcess]),
      agentReviews: query([review({
        source_process_run_id: reviewProcess.id,
        definition_snapshot: {
          name: 'Ada Reviewer', role: 'release reviewer', provider: 'openai', model: 'gpt-5.6',
          capabilities: ['security review', 'release verification'],
          config: { tools: ['browser', 'filesystem'] },
        },
        fit_summary: 'Strong fit for release review.',
        proposed_work_functions: ['review release evidence'],
        approved_for_work_functions: ['verify security controls'],
        risks: ['No production access'],
        recommended_changes: ['Add staging credentials'],
      })]),
    })

    const reviewMarkup = renderDashboard()
    for (const text of [
      'Ada Reviewer', 'release reviewer', 'openai / gpt-5.6',
      'Capabilities: security review, release verification',
      'Tools: browser, filesystem',
      'Proposed work functions: review release evidence',
      'Approved work functions: verify security controls',
      'Risks: No production access',
      'Recommendations: Add staging credentials',
      'Assessment: Strong fit for release review.',
    ]) expect(reviewMarkup).toContain(text)
  })

  it('humanizes backend skipped-process warnings outside debug data', () => {
    const focused = process({ id: 'skipped-process', status: 'skipped', completed_at: ISO.completed })
    const rawMessage = "Process 'goal_definition' was skipped by human:user-123: Operator chose a manual manager."
    api.dashboard = dashboardData({
      processes: query([focused]),
      warnings: query([warning({ source_process_run_id: focused.id, message: rawMessage })]),
    })

    const standard = renderDashboard()
    expect(standard).toContain('Goal definition was skipped: Operator chose a manual manager.')
    for (const raw of ['goal_definition', 'human:user-123', rawMessage]) expect(standard).not.toContain(raw)
  })

  it('keeps a chain mutation conflict in the focused process panel', async () => {
    api.dashboard = dashboardData({ processes: query([]) })
    api.run.mutate.mockImplementation((_variables, options) => {
      options.onError?.(new Error('409 Conflict'))
      options.onSettled?.()
    })
    const view = await mountDashboard()
    try {
      await act(async () => controlElement(view.container, 'Run Goal definition').click())

      // The focus panel's own local role="alert" was removed — the error text
      // still renders inline (visibly, inside the panel), and is separately
      // announced through the shared assertive region (GoalAnnouncerProvider).
      const focusPanel = descendants(view.container).find((node) => node.getAttribute('aria-labelledby') === 'baseline-process-focus-heading')
      expect(focusPanel && descendants(focusPanel).some((node) => node.getAttribute('role') === 'alert')).toBe(false)
      expect(focusPanel && textOf(focusPanel)).toContain('409 Conflict')
      const assertiveAnnouncements = descendants(view.container).filter((node) => node.getAttribute('role') === 'alert' && node.getAttribute('aria-live') === 'assertive')
      expect(assertiveAnnouncements.some((node) => textOf(node) === '409 Conflict')).toBe(true)
    } finally {
      view.cleanup()
    }
  })

  it('resolves backend hierarchy agent UUIDs to reviewed human names in standard mode', () => {
    const implementerId = '11111111-1111-4111-8111-111111111111'
    const weakFitId = '22222222-2222-4222-8222-222222222222'
    const reviewProcessId = '33333333-3333-4333-8333-333333333333'
    const implementerReviewId = '44444444-4444-4444-8444-444444444444'
    const weakFitReviewId = '55555555-5555-4555-8555-555555555555'
    api.dashboard = dashboardData({
      processes: query([process({
        id: 'hierarchy-process', process_type: 'team_hierarchy', status: 'completed', completed_at: ISO.completed,
        outputs: {
          source_agent_review_process_id: reviewProcessId,
          review_ids: [implementerReviewId, weakFitReviewId],
          weight: 'substantial', compressed: false,
          required_work_functions: ['implementation', 'validation'],
          candidate_agents: { implementation: [implementerId], validation: [] },
          role_to_agent: { implementation: implementerId },
          hierarchy: { manager: { kind: 'none', id: null }, team_leads: [], contributors: [implementerId], reviewers: [] },
          weak_fits: [{ work_function: 'validation', agent_id: weakFitId }],
          responsibility_conflicts: [],
          independent_verification: { required: true, possible: false, verifier_agent_ids: [], overridden_by_decision_id: null },
          missing_work_functions: ['validation'], suggestions: [],
        },
      })]),
      agentReviews: query([
        review({ id: implementerReviewId, agent_id: implementerId, source_process_run_id: reviewProcessId, definition_snapshot: { name: 'Ada Implementer' }, approved_for_work_functions: ['implementation'] }),
        review({ id: weakFitReviewId, agent_id: weakFitId, source_process_run_id: reviewProcessId, definition_snapshot: { name: 'Grace Validator' }, approved_for_work_functions: ['validation'] }),
      ]),
    })

    const hierarchyMarkup = renderDashboard()
    for (const text of [
      'implementation: Ada Implementer',
      'Missing work functions: validation',
      'Weak fits: validation · Grace Validator',
    ]) expect(hierarchyMarkup).toContain(text)
    expect(hierarchyMarkup).not.toContain(implementerId)
    expect(hierarchyMarkup).not.toContain(weakFitId)
  })

  it.each([
    // Backend stores selected_manager as "agent:<uuid>" / "human:<uuid>" / "human" / "" (spec 8).
    ['a reviewed agent id', 'agent:77777777-7777-4777-8777-777777777777', 'Manager: Ada Implementer'],
    ['a human with id', 'human:88888888-8888-4888-8888-888888888888', 'Manager: You (human)'],
    ['a bare human', 'human', 'Manager: You (human)'],
    ['an empty selection', '', 'Manager: None selected'],
  ])('resolves the manager-selection card for %s', (_case, selectedManager, expectedManagerLine) => {
    const managerId = '77777777-7777-4777-8777-777777777777'
    api.dashboard = dashboardData({
      processes: query([process({
        process_type: 'manager_selection', status: 'completed', completed_at: ISO.completed,
        outputs: {
          selected_manager: selectedManager,
          authority_model: 'agent_manager',
          manager_fit_rationale: 'Strong domain expertise across regulated deployments.',
          // Candidate objects carry key/label/score/signals (no name/reason); label is "<name> (<role>)".
          candidates: [
            { key: 'agent:77777777-7777-4777-8777-777777777777', label: 'Ada Implementer (implementation)', score: 8 },
            { key: 'agent:99999999-9999-4999-8999-999999999999', label: 'Grace Validator (validation)', score: 5 },
          ],
        },
      })]),
      agentReviews: query([review({ agent_id: managerId, definition_snapshot: { name: 'Ada Implementer' } })]),
    })

    const markup = renderDashboard()
    expect(markup).toContain(expectedManagerLine)
    expect(markup).toContain('Authority model: Agent Manager')
    expect(markup).toContain('Fit rationale: Strong domain expertise across regulated deployments.')
    expect(markup).toContain('Candidates considered:')
    expect(markup).toContain('Ada Implementer (implementation)')
    expect(markup).toContain('Grace Validator (validation)')
    expect(markup).not.toContain(managerId)
  })

  it.each([
    ['available', { required: true, possible: true, verifier_agent_ids: ['66666666-6666-4666-8666-666666666666'], overridden_by_decision_id: null }, 'Independent verification available'],
    ['unavailable', { required: true, possible: false, verifier_agent_ids: [], overridden_by_decision_id: null }, 'Independent verification unavailable'],
    ['not required', { required: false, possible: true, verifier_agent_ids: [], overridden_by_decision_id: null }, 'Independent verification not required'],
    ['not recorded', undefined, 'Independent verification not recorded'],
  ])('renders independent verification as %s', (_state, independentVerification, expected) => {
    api.dashboard = dashboardData({
      processes: query([process({
        process_type: 'team_hierarchy', status: 'completed', completed_at: ISO.completed,
        outputs: independentVerification === undefined ? {} : { independent_verification: independentVerification },
      })]),
    })

    const markup = renderDashboard()
    expect(markup).toContain(expected)
    for (const other of ['Independent verification available', 'Independent verification unavailable', 'Independent verification not required', 'Independent verification not recorded']) {
      if (other !== expected) expect(markup).not.toContain(other)
    }
  })

  it('never renders an empty Z4 shell for a step that has not started — it falls back to the static process description', () => {
    api.dashboard = dashboardData({ processes: query([]) })
    const markup = renderDashboard()
    expect(markup).toContain('The orchestrator reads the goal, asks any clarifying questions it needs')
  })

  it('renders the goal-definition outcome card with weight, clarifications, and success criteria', () => {
    api.dashboard = dashboardData({
      processes: query([process({
        status: 'completed', completed_at: ISO.completed,
        outputs: {
          weight: 'standard',
          clarifications: [{ question: 'What is the deadline?', answer: 'End of quarter.', destination: 'orchestrator_context.deadline', round: 1 }],
        },
      })]),
    })
    const markup = renderDashboard({ goal: goal({ success_criteria: [{ key: 'ship-it', description: 'Ship it.' }] }) })
    expect(markup).toContain('How the orchestrator scoped this goal')
    expect(markup).toContain('Weight: standard')
    expect(markup).toContain('What is the deadline?')
    expect(markup).toContain('Answer: End of quarter.')
    expect(markup).toContain('deadline')
    expect(markup).toContain('Ship it.')
  })

  it('renders the goal-definition clarification-limit special case with unresolved questions', () => {
    api.dashboard = dashboardData({
      processes: query([process({
        status: 'completed', completed_at: ISO.completed,
        outputs: { clarification_limit_reached: true, unresolved_questions: ['Which environment should this target?'] },
      })]),
    })
    const markup = renderDashboard()
    expect(markup).toContain('How the orchestrator scoped this goal')
    expect(markup).toContain('proceeding on its best understanding')
    expect(markup).toContain('Which environment should this target?')
  })

  it('renders the agent-definition-review roll-up line with a compressed note above the review cards', () => {
    const reviewProcess = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'completed', completed_at: ISO.completed, outputs: { warning_count: 2, compressed: true } })
    api.dashboard = dashboardData({
      processes: query([reviewProcess]),
      agentReviews: query([review({ source_process_run_id: reviewProcess.id, definition_snapshot: { name: 'Ada Reviewer' } })]),
    })
    const markup = renderDashboard()
    expect(markup).toContain('Who the orchestrator reviewed and approved')
    expect(markup).toContain('1 agent reviewed · 2 warnings')
    expect(markup).toContain('Compressed (trivial goal)')
  })

  it('falls back to the static description for agent-definition-review when no reviews were recorded', () => {
    api.dashboard = dashboardData({ processes: query([process({ process_type: 'agent_definition_review', status: 'completed', completed_at: ISO.completed })]) })
    const markup = renderDashboard()
    expect(markup).not.toContain('Who the orchestrator reviewed and approved')
    expect(markup).toContain('The orchestrator reviews each proposed agent definition for fit')
  })

  it('renders the effectiveness-review outcome card with colored pass/fail checks and the recommended/decided line', () => {
    api.dashboard = dashboardData({
      processes: query([process({
        process_type: 'effectiveness_review', status: 'completed', completed_at: ISO.completed,
        outputs: {
          triggers: [{ name: 'inactivity', token: 'x', detail: 'No progress for 48 hours' }],
          checks: [{ name: 'goal_complete', passed: true, detail: 'Objective and success criteria are complete.' }, { name: 'manager_valid', passed: false, detail: 'Selected manager is inconsistent or inactive.' }],
          recommended_disposition: 'revise', selected_disposition: 'continue', decision_reason: 'Operator judged the gap non-material.',
        },
      })]),
    })
    const markup = renderDashboard()
    expect(markup).toContain('Why the orchestrator flagged this goal for review')
    expect(markup).toContain('No progress for 48 hours')
    expect(markup).toContain('Objective and success criteria are complete.')
    expect(markup).toContain('Selected manager is inconsistent or inactive.')
    expect(markup).toContain('Recommended: revise → Decided: continue')
    expect(markup).toContain('Reason: Operator judged the gap non-material.')
  })

  it('renders the goal-closeout outcome card, resolving the sign-off decision and parsing its context', () => {
    const closeoutProcess = process({
      id: 'closeout-process', process_type: 'goal_closeout', status: 'completed', completed_at: ISO.completed,
      outputs: { mode: 'completion', completion_authorized: true, full_closeout: true, signoff_decision_id: 'signoff-1', selected_option: 'approve_completion' },
    })
    const signoffDecision = decision({
      id: 'signoff-1', source_process_run_id: closeoutProcess.id, authority: 'human', selected_option: 'approve_completion', reason: 'Evidence covers every criterion.',
      context: JSON.stringify({
        declared_success_criteria: [{ key: 'ship-it' }],
        criterion_evidence: [{ criterion_key: 'ship-it', evidence_ids: ['ev-1', 'ev-2'] }],
        accepted_risks: [{ warning_id: 'w1', type: 'coverage_gap', severity: 'warning', acknowledged_by: 'user-1' }],
        overridden_gates: [],
      }),
    })
    api.dashboard = dashboardData({ processes: query([closeoutProcess]), decisions: query([signoffDecision]) })
    const markup = renderDashboard()
    expect(markup).toContain('Why this goal was allowed to close')
    expect(markup).toContain('Approved by you')
    expect(markup).toContain('Evidence covers every criterion.')
    expect(markup).toContain('ship-it: 2 evidence item(s)')
    expect(markup).toContain('coverage_gap')
  })

  it('renders the goal-closeout cancellation case without a sign-off decision', () => {
    api.dashboard = dashboardData({
      processes: query([process({ process_type: 'goal_closeout', status: 'completed', completed_at: ISO.completed, outputs: { mode: 'cancellation', completion_authorized: false, full_closeout: false } })]),
    })
    const markup = renderDashboard()
    expect(markup).toContain('Why this goal was cancelled')
    expect(markup).toContain('cancelled before completion')
  })

  it('renders the goal-closeout trivial auto-completion case with no sign-off decision required', () => {
    api.dashboard = dashboardData({
      processes: query([process({ process_type: 'goal_closeout', status: 'completed', completed_at: ISO.completed, outputs: { mode: 'completion', completion_authorized: true, full_closeout: false } })]),
    })
    const markup = renderDashboard()
    expect(markup).toContain('Why this goal was allowed to close')
    expect(markup).toContain('does not require a sign-off decision')
  })

  it('does not claim automatic closeout while decisions are still loading and a sign-off decision is expected', () => {
    const closeoutProcess = process({
      process_type: 'goal_closeout', status: 'completed', completed_at: ISO.completed,
      outputs: { mode: 'completion', completion_authorized: true, full_closeout: true, signoff_decision_id: 'signoff-1' },
    })
    api.dashboard = dashboardData({
      processes: query([closeoutProcess]),
      // Readiness (processState) only gates on the processes query — the
      // decisions query can still be in flight when this renders.
      decisions: query<OrchestrationAuthorityDecisionRecord[]>(undefined, { isLoading: true }),
    })
    const markup = renderDashboard()
    expect(markup).not.toContain('Completed automatically')
    expect(markup).not.toContain('does not require a sign-off decision')
    expect(markup).toContain('The orchestrator confirms the evidence justifies closing the goal')
  })

  it('does not claim automatic closeout when the decisions query errored (settled, but not successfully)', () => {
    const closeoutProcess = process({
      process_type: 'goal_closeout', status: 'completed', completed_at: ISO.completed,
      outputs: { mode: 'completion', completion_authorized: true, full_closeout: true, signoff_decision_id: 'signoff-1' },
    })
    api.dashboard = dashboardData({
      processes: query([closeoutProcess]),
      // isLoading is false once a query errors — a bare loading flag can't
      // distinguish this from "loaded, genuinely no signoff".
      decisions: query<OrchestrationAuthorityDecisionRecord[]>(undefined, { isLoading: false, isError: true }),
    })
    const markup = renderDashboard()
    expect(markup).not.toContain('Completed automatically')
    expect(markup).not.toContain('does not require a sign-off decision')
    expect(markup).toContain('The orchestrator confirms the evidence justifies closing the goal')
  })

  it.each(['running', 'waiting_decision'] as const)('keeps Skip reachable for a %s current process and sends its required reason', async (status) => {
    api.dashboard = dashboardData({ processes: query([process({ status })]) })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Skip').click())
      const reason = getByLabel(view.container, 'Reason (required)', 'textarea')
      await act(async () => changeControl(reason, 'Operator accepted the documented gap.'))
      await act(async () => getButton(view.container, 'Skip process').click())

      expect(api.skip.mutate).toHaveBeenCalledWith(
        {
          goalId: 'goal-00000000-0000-4000-8000-000000000001',
          processType: 'goal_definition',
          reason: 'Operator accepted the documented gap.',
        },
        expect.any(Object),
      )
    } finally {
      view.cleanup()
    }
  })

  it('submits all agent-definition review decisions together', async () => {
    const focused = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const reviewDecisions = [
      decision({ id: 'decision-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve', context: JSON.stringify({ original_description: null, proposed_description: 'Description proposed from null' }) }),
      decision({ id: 'decision-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve', context: JSON.stringify({ original_description: '', proposed_description: 'Proposed description', original_persona: 'Current persona', proposed_persona: 'Proposed persona' }) }),
      decision({ id: 'decision-c', decision_key: 'agent_definition_review:proposal:agent-3', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve' }),
    ]
    api.dashboard = dashboardData({
      processes: query([focused]),
      decisions: query(reviewDecisions),
      // The Needs-you queue's agent-definition-review batch grouping reads the
      // checkpoint (server-prioritized pending questions), not the raw decisions list.
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: reviewDecisions, deferred_count: 0, max_questions: 5 }),
    })

    const view = await mountDashboard()
    try {
      expect(() => getButton(view.container, 'Answer')).toThrow()
      expect(view.container.textContent?.match(/Current: \(empty\)/g)).toHaveLength(2)
      expect(view.container.textContent).toContain('Proposed: Description proposed from null')
      expect(view.container.textContent).toContain('Proposed: Proposed description')
      expect(view.container.textContent).toContain('System prompt')
      expect(view.container.textContent).toContain('Current: Current persona')
      expect(view.container.textContent).toContain('Proposed: Proposed persona')

      // SPR #87: every proposal offers exactly one Approve and one Reject
      // control (plus Edit). All three decisions share the same default
      // title/question, so rows are found by DOM order, not text.
      const articles = descendants(view.container).filter((node) => node.tagName === 'ARTICLE')
      expect(articles).toHaveLength(3)
      const [rowA, rowB, rowC] = articles

      await act(async () => getButton(rowA, 'Approve').click())

      await act(async () => getButton(rowB, 'Edit').click())
      expect(getByLabel(rowB, 'Description', 'textarea').value).toBe('Proposed description')
      expect(getByLabel(rowB, 'Persona', 'textarea').value).toBe('Proposed persona')
      await act(async () => changeControl(getByLabel(rowB, 'Description', 'textarea'), 'Edited description'))
      await act(async () => changeControl(getByLabel(rowB, 'Persona', 'textarea'), 'Edited persona'))
      await act(async () => getButton(rowB, 'Confirm edit').click())

      await act(async () => getButton(rowC, 'Reject').click())
      await act(async () => changeControl(getByLabel(rowC, 'Reason (required)', 'textarea'), 'Not ready yet.'))
      // Last row decided — the batch auto-submits, no separate "submit all" click.
      await act(async () => getButton(rowC, 'Confirm reject').click())

      expect(api.answerAgentDefinitionBatch.mutate).toHaveBeenCalledWith(
        {
          goalId: 'goal-00000000-0000-4000-8000-000000000001',
          answers: [
            { decisionId: 'decision-a', selectedOption: 'approve', reason: '', editedDescription: 'Description proposed from null', editedPersona: '' },
            { decisionId: 'decision-b', selectedOption: 'edit', reason: '', editedDescription: 'Edited description', editedPersona: 'Edited persona' },
            { decisionId: 'decision-c', selectedOption: 'reject', reason: 'Not ready yet.', editedDescription: '', editedPersona: '' },
          ],
        },
        expect.any(Object),
      )
    } finally {
      view.cleanup()
    }
  })

  it('approves every remaining agent-definition proposal with one click', async () => {
    const focused = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const reviewDecisions = [
      decision({ id: 'decision-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve' }),
      decision({ id: 'decision-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve' }),
    ]
    api.dashboard = dashboardData({
      processes: query([focused]),
      decisions: query(reviewDecisions),
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: reviewDecisions, deferred_count: 0, max_questions: 5 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Approve all remaining (2)').click())
      expect(api.answerAgentDefinitionBatch.mutate).toHaveBeenCalledWith(
        {
          goalId: 'goal-00000000-0000-4000-8000-000000000001',
          answers: [
            { decisionId: 'decision-a', selectedOption: 'approve', reason: '', editedDescription: '', editedPersona: '' },
            { decisionId: 'decision-b', selectedOption: 'approve', reason: '', editedDescription: '', editedPersona: '' },
          ],
        },
        expect.any(Object),
      )
    } finally {
      view.cleanup()
    }
  })

  it('recovers from a 409 on batch submit by refetching proposals and announcing the reload', async () => {
    const focused = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const reviewDecisions = [
      decision({ id: 'decision-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve' }),
      decision({ id: 'decision-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: focused.id, options: ['approve', 'reject', 'edit'], recommendation: 'approve' }),
    ]
    const processes = query([focused])
    const decisions = query(reviewDecisions)
    const warnings = query<OrchestrationWarningRecord[]>([])
    api.dashboard = dashboardData({
      processes,
      decisions,
      warnings,
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: reviewDecisions, deferred_count: 0, max_questions: 5 }),
    })
    api.answerAgentDefinitionBatch.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new ApiError(409, 'Conflict: pending proposal set changed'))
      callbacks.onSettled?.()
    })
    const onMessage = vi.fn()

    const view = await mountDashboard({ onMessage })
    try {
      // Approve both decisions — the second Approve is what completes the
      // batch and triggers the (failing) submit.
      const [rowA, rowB] = descendants(view.container).filter((node) => node.tagName === 'ARTICLE')
      await act(async () => getButton(rowA, 'Approve').click())
      await act(async () => getButton(rowB, 'Approve').click())

      // Verify refetch calls and message
      expect(decisions.refetch).toHaveBeenCalled()
      expect(processes.refetch).toHaveBeenCalled()
      expect(warnings.refetch).toHaveBeenCalled()
      expect(onMessage).toHaveBeenCalledWith('The proposals changed and were reloaded — please review the updated agent-definition proposals.')
    } finally {
      view.cleanup()
    }
  })

  it('reviews complete team proposals independently before showing the hierarchy', async () => {
    const hierarchy = process({ id: 'hierarchy-process', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const definition = (name: string, prompt: string) => ({ name, role: 'worker', description: `${name} description`, provider: 'openai', model: 'gpt-5', system_prompt: prompt, adapter_type: 'api', cli_runtime: null, capabilities: ['build', 'review'], config: { temperature: 0 } })
    const proposals = [
      decision({ id: 'proposal-a', decision_key: 'team_hierarchy:agent:a', title: 'Review Alpha', source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], context: JSON.stringify({ proposal: { proposal_id: 'a', definition: definition('Alpha', 'FULL ALPHA PROMPT\nNever truncate this handoff.') } }) }),
      decision({ id: 'proposal-b', decision_key: 'team_hierarchy:agent:b', title: 'Review Beta', source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], context: JSON.stringify({ proposal: { proposal_id: 'b', definition: definition('Beta', 'FULL BETA PROMPT\nKeep every boundary visible.') } }) }),
    ]
    api.dashboard = dashboardData({ processes: query([hierarchy]), decisions: query(proposals), checkpoint: query({ goal_id: goal().id, items: proposals, deferred_count: 0, max_questions: 5 }) })
    const view = await mountDashboard()
    try {
      expect(textOf(view.container)).toContain('FULL ALPHA PROMPT Never truncate this handoff.')
      expect(textOf(view.container)).toContain('FULL BETA PROMPT Keep every boundary visible.')
      expect(textOf(view.container)).toContain('Waiting for 2 proposed agent decisions before hierarchy approval.')
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON' && textOf(node) === 'Approve')).toHaveLength(2)
      const alpha = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Review Alpha'))!
      await act(async () => getButton(alpha, 'Edit').click())
      await act(async () => changeControl(getByLabel(alpha, 'Name', 'input'), 'Edited Alpha'))
      await act(async () => changeControl(getByLabel(alpha, 'Config (JSON object)', 'textarea'), '{"temperature":0,"limit":2}'))
      await act(async () => getButton(alpha, 'Submit edit').click())
      expect(api.answer.mutate).toHaveBeenCalledWith({ goalId: goal().id, decisionId: 'proposal-a', selectedOption: 'edit', reason: '', editedAgent: { ...definition('Alpha', 'FULL ALPHA PROMPT\nNever truncate this handoff.'), name: 'Edited Alpha', config: { temperature: 0, limit: 2 } } }, expect.any(Object))
      expect(getButton(descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Review Beta'))!, 'Approve')).toBeDefined()

      const hierarchyProposal = { assignments: [{ work_function: 'build', agent_ref: 'alpha-id' }], reporting_lines: [{ agent_ref: 'alpha-id', reports_to: 'manager' }], documented_gaps: ['independent_review'], approved_but_unused_agent_ids: ['beta-id'], rationale: 'Smallest complete team.', self_review: 'Coverage checked.' }
      const approval = decision({ id: 'approval', decision_key: 'team_hierarchy:approval', title: 'Approve team hierarchy', source_process_run_id: hierarchy.id, options: [{ key: 'approve_with_documented_gaps', proposal: hierarchyProposal }] })
      api.dashboard = dashboardData({ processes: query([hierarchy]), decisions: query([approval]), checkpoint: query({ goal_id: goal().id, items: [approval], deferred_count: 0, max_questions: 5 }) })
      await view.rerender()
      // The earlier "Waiting for…" state was already announced into the shared
      // polite region before the proposals resolved — that sr-only text is by
      // design not retracted, so check for the absence of the visible paragraph
      // specifically rather than the whole container's text.
      expect(descendants(view.container).some((node) => node.tagName === 'P' && !node.getAttribute('aria-live') && textOf(node).includes('Waiting for 2 proposed agent decisions'))).toBe(false)
      expect(textOf(view.container)).toContain('independent_review')
      expect(textOf(view.container)).toContain('beta-id')
      expect(textOf(view.container)).toContain('alpha-id')
    } finally { view.cleanup() }
  })

  it('shows manager comparison in the collapsed queue row without a second action', async () => {
    const manager = process({ id: 'manager-process', process_type: 'manager_selection', status: 'waiting_decision' })
    const review = decision({ id: 'manager-review', decision_key: 'manager_selection:review_override', source_process_run_id: manager.id, recommendation: 'approve', options: ['approve', 'reject'], context: 'Deterministic: agent:a\nLLM: agent:b\nRationale: Better leadership fit.' })
    api.dashboard = dashboardData({ processes: query([manager]), decisions: query([review]), checkpoint: query({ goal_id: goal().id, items: [review], deferred_count: 0, max_questions: 5 }) })
    const view = await mountDashboard()
    try {
      expect(textOf(view.container)).toContain('Deterministic: agent:a')
      expect(textOf(view.container)).toContain('LLM: agent:b')
      expect(textOf(view.container)).toContain('Rationale: Better leadership fit.')
      await act(async () => getButton(view.container, 'Choose differently…').click())
      expect(getButton(view.container, 'Submit answer')).toBeDefined()
      expect(descendants(view.container).some((node) => node.tagName === 'INPUT' && node.value === 'reject')).toBe(true)
    } finally { view.cleanup() }
  })

  it('keeps context-shaped non-manager decisions on the standard two-action path', async () => {
    const review = decision({ decision_key: 'team_hierarchy:manager_review_note', recommendation: 'approve', options: ['approve', 'reject'], context: 'Deterministic: a\nLLM: b\nRationale: shaped but unrelated.' })
    api.dashboard = dashboardData({ checkpoint: query({ goal_id: goal().id, items: [review], deferred_count: 0, max_questions: 5 }) })
    const view = await mountDashboard()
    try { expect(getButton(view.container, 'Choose differently…')).toBeDefined() } finally { view.cleanup() }
  })

  it('derives hierarchy status only from the current hierarchy process', () => {
    const oldRun = process({ id: 'hierarchy-a', process_type: 'team_hierarchy', status: 'complete', superseded_by_id: 'hierarchy-b' })
    const currentRun = process({ id: 'hierarchy-b', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const oldProposal = { assignments: [{ work_function: 'old-work', agent_ref: 'old-agent' }], reporting_lines: [], documented_gaps: [], approved_but_unused_agent_ids: [] }
    const oldApproval = decision({ id: 'old-approval', decision_key: 'team_hierarchy:approval', status: 'answered', source_process_run_id: oldRun.id, options: [{ key: 'approve', proposal: oldProposal }] })
    const currentAgent = decision({ id: 'current-agent', decision_key: 'team_hierarchy:agent:new', source_process_run_id: currentRun.id, options: ['approve', 'edit', 'reject'], context: JSON.stringify({ proposal: { definition: { name: 'New', role: 'worker', provider: 'openai', model: 'gpt-5' } } }) })
    api.dashboard = dashboardData({ processes: query([oldRun, currentRun]), decisions: query([oldApproval, currentAgent]), checkpoint: query({ goal_id: goal().id, items: [currentAgent], deferred_count: 0, max_questions: 5 }) })
    const markup = renderDashboard()
    expect(markup).toContain('Waiting for 1 proposed agent decision before hierarchy approval.')
    expect(markup).not.toContain('old-work')
  })

  it('focuses the next unresolved proposal after mutation success refreshes the queue', async () => {
    const hierarchy = process({ id: 'focus-hierarchy', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const proposal = (id: string, name: string) => decision({ id, decision_key: `team_hierarchy:agent:${id}`, title: `Review ${name}`, source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], context: JSON.stringify({ proposal: { definition: { name, role: 'worker', provider: 'openai', model: 'gpt-5' } } }) })
    const alpha = proposal('focus-alpha', 'Alpha')
    const beta = proposal('focus-beta', 'Beta')
    const checkpoint = (items: OrchestrationAuthorityDecisionRecord[]) => query({ goal_id: goal().id, items, deferred_count: 0, max_questions: 5 })
    api.dashboard = dashboardData({ processes: query([hierarchy]), decisions: query([alpha, beta]), checkpoint: checkpoint([alpha, beta]) })
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const view = await mountDashboard()
    try {
      const alphaRow = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Review Alpha'))!
      const betaRow = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Review Beta'))!
      const betaApprove = getButton(betaRow, 'Approve')
      await act(async () => getButton(alphaRow, 'Approve').click())
      try {
        await act(async () => { callbacks?.onSuccess?.({ decision: alpha }) })
      } catch (error) { throw new Error((error as Error).stack) }
      expect(toast.success).toHaveBeenCalledWith('Proposal decision received.')
      const refreshedBeta = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Review Beta'))!
      expect(betaApprove === getButton(refreshedBeta, 'Approve')).toBe(true)
      expect(view.document.activeElement === getButton(refreshedBeta, 'Approve')).toBe(true)
    } finally { view.cleanup() }
  })

  it('submits the accept-recommendation fast path through the answer mutation with no form and no reason', async () => {
    const recommended = decision({ id: 'decision-1', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }] })
    api.dashboard = dashboardData({
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [recommended], deferred_count: 0, max_questions: 5 }),
    })
    const view = await mountDashboard()
    try {
      expect(() => getButton(view.container, 'Submit answer')).toThrow()
      await act(async () => getButton(view.container, 'Accept: approve').click())
      expect(api.answer.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', decisionId: 'decision-1', selectedOption: 'approve', reason: '' },
        expect.any(Object),
      )
      expect(() => getButton(view.container, 'Submit answer')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  // Regression (review round 2, finding I1): the accept-recommendation fast
  // path has no open form (targetsOpenForm is false), so a failed request
  // must still surface an error — the button silently re-enabling with no
  // feedback is a bug, not "no form, no error UI needed".
  it('surfaces and announces an error when the accept-recommendation fast path fails', async () => {
    const recommended = decision({ id: 'decision-1', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }] })
    api.dashboard = dashboardData({
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [recommended], deferred_count: 0, max_questions: 5 }),
    })
    api.answer.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new Error('Answer rejected by validator.'))
      callbacks.onSettled?.()
    })
    const onMessage = vi.fn()
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(view.container, 'Accept: approve').click())
      expect(onMessage).toHaveBeenCalledWith('Answer rejected by validator.')
    } finally {
      view.cleanup()
    }
  })

  it('accepting one decision does not close or clear an unrelated decision’s open form', async () => {
    const decisionA = decision({ id: 'decision-a', title: 'Decision A', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }] })
    const decisionB = decision({ id: 'decision-b', title: 'Decision B', recommendation: 'defer', options: ['approve_with_conditions', 'defer'] })
    api.dashboard = dashboardData({
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [decisionA, decisionB], deferred_count: 0, max_questions: 5 }),
    })
    // Must actually resolve (fire onSuccess) so the accept path exercises the
    // openAction-closing/clearing logic the regression targets — a mock that
    // never calls back would let an unconditional close/clear slip through
    // undetected.
    api.answer.mutate.mockImplementation((_variables, callbacks) => { callbacks.onSuccess?.(); callbacks.onSettled?.() })
    const view = await mountDashboard()
    try {
      const rowA = () => descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Decision A'))!
      const rowB = () => descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes('Decision B'))!

      // Open decision A's full form via "Choose differently…" and pick an
      // overriding option + reason (proves the form's state, not just its
      // open/closed flag, survives).
      await act(async () => getButton(rowA(), 'Choose differently…').click())
      const reviseOption = descendants(rowA()).find((node) => node.tagName === 'INPUT' && node.value === 'revise')!
      await act(async () => changeControl(reviseOption, 'revise'))
      const reasonField = () => getByLabel(view.container, 'Reason (required when overriding the recommendation)', 'textarea')
      await act(async () => changeControl(reasonField(), 'Operator override rationale.'))

      // Accept decision B's recommendation directly, without touching A's form.
      await act(async () => getButton(rowB(), 'Accept: defer').click())
      expect(api.answer.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', decisionId: 'decision-b', selectedOption: 'defer', reason: '' },
        expect.any(Object),
      )

      // Decision A's form is untouched: still open, still shows the selected
      // option and typed reason.
      expect(getButton(view.container, 'Submit answer')).toBeDefined()
      expect(reviseOption.checked).toBe(true)
      expect(reasonField().value).toBe('Operator override rationale.')
    } finally {
      view.cleanup()
    }
  })

  it('submits the goal-definition step-through card through the answer mutation one question at a time', async () => {
    const goalDefinitionProcess = process({ id: 'goal-definition-process', process_type: 'goal_definition' })
    const questions = [
      decision({ id: 'q0', decision_key: 'goal_definition:adaptive:1:0:orchestrator_context.notes', question: 'Question 0?', context: 'Rationale 0. Destination: orchestrator_context.notes', options: [], recommendation: null, source_process_run_id: goalDefinitionProcess.id }),
      decision({ id: 'q1', decision_key: 'goal_definition:adaptive:1:1:orchestrator_context.notes', question: 'Question 1?', context: 'Rationale 1. Destination: orchestrator_context.notes', options: [], recommendation: null, source_process_run_id: goalDefinitionProcess.id }),
    ]
    api.dashboard = dashboardData({
      processes: query([goalDefinitionProcess]),
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: questions, deferred_count: 0, max_questions: 5 }),
    })
    api.answer.mutate.mockImplementation((_variables, callbacks) => { callbacks.onSuccess?.(); callbacks.onSettled?.() })
    const view = await mountDashboard()
    try {
      expect(textOf(view.container)).toContain('The orchestrator needs 2 answers.')
      await act(async () => changeControl(getByLabel(view.container, 'Answer', 'textarea'), 'First answer'))
      await act(async () => getButton(view.container, 'Next').click())
      expect(api.answer.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', decisionId: 'q0', selectedOption: 'First answer', reason: '' },
        expect.any(Object),
      )
      expect(textOf(view.container)).toContain('Question 2 of 2')

      await act(async () => changeControl(getByLabel(view.container, 'Answer', 'textarea'), 'Second answer'))
      await act(async () => getButton(view.container, 'Submit').click())
      expect(api.answer.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', decisionId: 'q1', selectedOption: 'Second answer', reason: '' },
        expect.any(Object),
      )
      expect(textOf(view.container)).not.toContain('The orchestrator needs')
    } finally {
      view.cleanup()
    }
  })

  it('surfaces the waiting-decision step in the now-line even when another step is running, without a contradictory "next"', () => {
    const waiting = process({ id: 'waiting-process', process_type: 'manager_selection', status: 'waiting_decision' })
    api.dashboard = dashboardData({
      processes: query([process({ id: 'running-process', process_type: 'agent_definition_review', status: 'running' }), waiting]),
      decisions: query([decision({ source_process_run_id: waiting.id, status: 'pending', authority: 'human' })]),
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [decision({ source_process_run_id: waiting.id, status: 'pending', authority: 'human' })], deferred_count: 0, max_questions: 5 }),
    })

    const markup = renderDashboard()
    // Regression (review round 1, finding 2): agent_definition_review is
    // sequence-adjacent to manager_selection but is already running — "next:
    // agent_definition_review" would be a contradiction, so it must not appear.
    expect(markup).toContain('Selecting a manager · 1 question for you</span>')
    expect(markup).not.toContain('next:')
  })

  it('keeps raw agent-review config behind the selected process debug toggle', async () => {
    const focused = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'completed', completed_at: ISO.completed })
    api.dashboard = dashboardData({
      processes: query([focused]),
      agentReviews: query([review({
        source_process_run_id: focused.id,
        definition_snapshot: {
          name: 'Debuggable Reviewer',
          config: { adapter_type: 'openai', model: 'gpt-5.6', system_prompt: 'raw-review-config' },
        },
      })]),
    })

    expect(renderDashboard()).not.toContain('raw-review-config')

    api.health = { data: { debug_enabled: true } }
    const view = await mountDashboard()
    try {
      const summary = descendants(view.container).find((node) => node.tagName === 'SUMMARY' && textOf(node) === 'Raw data')
      const details = summary?.parentNode as TestElement | null | undefined
      expect(summary).toBeDefined()
      expect(details?.getAttribute('open')).toBeNull()

      await act(async () => summary!.click())

      expect(details?.getAttribute('open')).toBe('')
      expect(textOf(details!)).toContain('raw-review-config')
      expect(textOf(details!)).toContain('definition_snapshot')
    } finally {
      view.cleanup()
    }
  })

  it('renders the team-hierarchy reactflow graph only once manager + role data is complete, falling back to text otherwise', async () => {
    // HierarchyGraph is lazy-loaded (Phase 12): the synchronous renderDashboard/renderToStaticMarkup
    // helper can only ever observe the Suspense fallback, so this test mounts via the real
    // client renderer (mountDashboard) and awaits the dynamic import resolving.
    const hierarchyProcess = process({
      id: 'hierarchy-graphable', process_type: 'team_hierarchy', status: 'completed', completed_at: ISO.completed,
      outputs: {
        hierarchy: { manager: { kind: 'agent', id: 'agent-lead' } },
        role_to_agent: { researcher: 'agent-lead' },
      },
    })
    api.dashboard = dashboardData({
      processes: query([hierarchyProcess]),
      agentReviews: query([review({ agent_id: 'agent-lead', source_process_run_id: hierarchyProcess.id, definition_snapshot: { name: 'Lead Agent' } })]),
    })
    const view = await mountDashboard()
    try {
      // HierarchyGraph is lazy-loaded; the dynamic import is a real async module
      // transform/load in the test runner, not just a microtask — poll with a real
      // delay until it resolves and re-renders.
      let hasReactFlow = false
      for (let attempt = 0; attempt < 100 && !hasReactFlow; attempt += 1) {
        await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)) })
        await view.rerender()
        hasReactFlow = descendants(view.container).some((node) => (node.getAttribute?.('class') ?? '').includes('react-flow'))
      }
      expect(hasReactFlow).toBe(true)
      expect(textOf(view.container)).toContain('Lead Agent')
      expect(textOf(view.container)).toContain('Manager: Lead Agent')
    } finally {
      view.cleanup()
    }

    api.dashboard = dashboardData({ processes: query([process({ process_type: 'team_hierarchy', status: 'skipped', completed_at: ISO.completed })]) })
    const fallbackMarkup = renderDashboard()
    expect(fallbackMarkup).not.toContain('react-flow')
    expect(fallbackMarkup).toContain('Manager: Not recorded')
  })

  it('colors a running process badge amber and a completed process badge green', () => {
    api.dashboard = dashboardData({
      processes: query([
        process({ status: 'running' }),
        process({ id: 'manager', process_type: 'manager_selection', status: 'completed', completed_at: ISO.completed }),
      ]),
    })

    const markup = renderDashboard()
    expect(markup).toContain(
      'style="color:#B45309"><span aria-hidden="true" class="text-[8px] leading-none">●</span>In progress<',
    )
    expect(markup).toContain(
      'style="color:#15803D"><span aria-hidden="true" class="text-[8px] leading-none">●</span>Done<',
    )
  })

  it('renders the goal_definition_clarification_limit recovery action as the queue blocker row\'s primary action', async () => {
    const renderBlockerRecovery: ComponentProps<typeof BaselineDashboard>['renderBlockerRecovery'] = (blocker, register, onSuccessFocus) => (
      <button type="button" ref={register} onClick={() => onSuccessFocus?.()}>Proceed with current understanding</button>
    )
    const view = await mountDashboard({
      run: run({ active_blockers: [{ kind: 'goal_definition_clarification_limit', reason: 'Two rounds did not resolve this goal.' }] }),
      renderBlockerRecovery,
    })
    try {
      expect(textOf(view.container)).toContain('Two rounds did not resolve this goal.')
      expect(getButton(view.container, 'Proceed with current understanding')).toBeDefined()
    } finally {
      view.cleanup()
    }
  })

  it('renders loading state with skeleton cards', () => {
    api.dashboard = dashboardData({
      processes: query<OrchestrationProcessRunRecord[]>(undefined, { isLoading: true, isError: false }),
    })

    const markup = renderDashboard()
    expect(markup).toContain('Loading orchestrator steps')
    expect(markup).toContain('aria-busy="true"')
    for (const falseState of [
      'No pending work', 'Goal definition', 'Not started',
      'No events recorded for this process.', 'No decisions recorded for this process.', 'No team hierarchy recorded.',
    ]) expect(markup).not.toContain(falseState)
    expect(control(markup, 'Run Goal definition')).toBe(false)
  })

  it('renders error state without a local alert role (announced through the shared region instead)', () => {
    api.dashboard = dashboardData({
      processes: query<OrchestrationProcessRunRecord[]>(undefined, { isLoading: false, isError: true }),
    })

    // ProcessChain dropped its own role="alert" — the error is announced
    // through GoalAnnouncerProvider's shared assertive region instead. That
    // announcement is effect-driven, so it can't be observed via this
    // static-markup render; only the visible copy and the absence of a local
    // alert role are checked here.
    const markup = renderDashboard()
    expect(markup).toContain('Orchestrator steps are unavailable.')
    expect(markup).not.toContain('role="alert"')
    for (const falseState of [
      'No pending work', 'Goal definition', 'Not started',
      'No events recorded for this process.', 'No decisions recorded for this process.', 'No team hierarchy recorded.',
    ]) expect(markup).not.toContain(falseState)
    expect(control(markup, 'Run Goal definition')).toBe(false)
  })
})

describe('Baseline action form regressions', () => {
  function form(
    options: OrchestrationDecisionOption[],
    selectedOption: string | null,
    recommendation: string | null,
    overrides: Partial<OrchestrationAuthorityDecisionRecord> = {},
  ) {
    return renderToStaticMarkup(
      <BaselineActionForm
        target={{ kind: 'answer', action: { kind: 'answer', id: 'decision-1' }, decision: decision({ options, recommendation, ...overrides }) }}
        mutationBusy={false} submitting={false} selectedOption={selectedOption} reason="" error={null}
        firstOptionRef={React.createRef<HTMLInputElement>()} reasonRef={React.createRef<HTMLTextAreaElement>()}
        submitRef={React.createRef<HTMLButtonElement>()} onAnswer={() => undefined} onSkip={() => undefined}
        onAcknowledge={() => undefined} onResolve={() => undefined} onSelectedOptionChange={() => undefined}
        onReasonChange={() => undefined} onCancel={() => undefined}
      />,
    )
  }

  it('requires a reason only when overriding the recommendation', () => {
    const recommended = form(['approve', { key: 'revise', label: 'Revise scope' }], 'approve', 'approve')
    const override = form(['approve', { key: 'revise', label: 'Revise scope' }], 'revise', 'approve')
    expect(recommended).toContain('Reason (required when overriding the recommendation)')
    expect(recommended).not.toMatch(/<textarea[^>]*required=""/)
    expect(override).toMatch(/<textarea[^>]*required=""/)
  })

  it('keeps optionless decisions as required free-text answers', () => {
    const markup = form([], null, null)
    expect(markup).toContain('>Answer</label>')
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Submit answer<\/button>/)
    const textareas = markup.match(/<textarea[^>]*>/g) ?? []
    expect(textareas.some((textarea) => !textarea.includes('required=""'))).toBe(true)
  })

  it('renders Reason field required when free-text answer differs from recommendation', () => {
    const markup = form([], 'custom-answer', 'recommend-value')
    expect(markup).toContain('>Answer</label>')
    expect(markup).toContain('Reason (required when overriding the recommendation)')
    expect(markup).toMatch(/<textarea[^>]*required=""/)
  })

  function formWithContext(context: string | null) {
    return renderToStaticMarkup(
      <BaselineActionForm
        target={{ kind: 'answer', action: { kind: 'answer', id: 'decision-1' }, decision: decision({ options: ['approve'], recommendation: 'approve', context }) }}
        mutationBusy={false} submitting={false} selectedOption="approve" reason="" error={null}
        firstOptionRef={React.createRef<HTMLInputElement>()} reasonRef={React.createRef<HTMLTextAreaElement>()}
        submitRef={React.createRef<HTMLButtonElement>()} onAnswer={() => undefined} onSkip={() => undefined}
        onAcknowledge={() => undefined} onResolve={() => undefined} onSelectedOptionChange={() => undefined}
        onReasonChange={() => undefined} onCancel={() => undefined}
      />,
    )
  }

  it('renders original vs proposed description and persona for an agent-definition-review decision', () => {
    // Real decision.context keys (orchestration_agent_definition_review.py:648-651).
    const context = JSON.stringify({
      reason: 'Persona lacked incident-response detail.',
      original_description: 'Handles routine deployments.',
      proposed_description: 'Handles routine deployments and incident response.',
      original_persona: 'Calm, detail-oriented engineer.',
      proposed_persona: 'Calm, detail-oriented engineer with incident command experience.',
    })

    const markup = formWithContext(context)
    expect(markup).toContain('Persona lacked incident-response detail.')
    expect(markup).toContain('Original: Handles routine deployments.')
    expect(markup).toContain('Proposed: Handles routine deployments and incident response.')
    expect(markup).toContain('Original: Calm, detail-oriented engineer.')
    expect(markup).toContain('Proposed: Calm, detail-oriented engineer with incident command experience.')
  })

  it('renders no diff block and does not crash for a non-review decision with no context', () => {
    const markup = formWithContext(null)
    expect(markup).not.toContain('Original:')
    expect(markup).not.toContain('Proposed:')
  })

  it('renders no diff block and does not crash for a decision context that is not a review payload', () => {
    const markup = formWithContext(JSON.stringify({ some_other_field: 'value' }))
    expect(markup).not.toContain('Original:')
    expect(markup).not.toContain('Proposed:')
  })
})
