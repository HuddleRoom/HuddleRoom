import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { changeControl, descendants, getButton, getByLabel, mountWithTestDom, textOf, TestEvent, type TestElement } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import { toast } from 'sonner'
import type {
  NeedsYouReference,
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
  answer: { isPending: false, mutate: vi.fn(), acceptRuntimeResult: vi.fn() },
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
  toast: Object.assign(vi.fn(), { success: vi.fn(), error: vi.fn() }),
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

const meetingMutation = vi.hoisted(() => ({ mutate: vi.fn(), isPending: false }))
vi.mock('@/api/meetings', () => ({ useUpdateActionItem: () => meetingMutation }))

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

function query<T>(data: T | undefined, overrides: Partial<{ isLoading: boolean; isError: boolean; isFetching: boolean }> = {}) {
  const merged = { isLoading: false, isError: false, isFetching: false, ...overrides }
  const isSuccess = !merged.isLoading && !merged.isError && data !== undefined
  return {
    data,
    refetch: vi.fn().mockResolvedValue({ data, isSuccess, isError: merged.isError }),
    ...merged,
    isSuccess,
  }
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

function task4RuntimeQuestion(
  number: number,
  overrides: Partial<OrchestrationAuthorityDecisionRecord> = {},
) {
  const minute = String(number).padStart(2, '0')
  const askedAt = `2026-08-05T08:${minute}:00Z`
  return decision({
    id: `runtime-q${number}`,
    decision_key: `runtime:question:${number}`,
    title: `Runtime question Q${number}`,
    question: `Explain the choice for Q${number}.`,
    options: [{ key: 'acknowledge' }],
    recommendation: null,
    runtime_identity: `runtime-identity-q${number}`,
    contract_version: `contract-q${number}:v1`,
    asked_at: askedAt,
    created_at: askedAt,
    updated_at: askedAt,
    ...overrides,
  })
}

function queueRow(root: TestElement, title: string) {
  const row = descendants(root).find((node) => node.tagName === 'LI' && textOf(node).includes(title))
  if (!row) throw new Error(`Queue row not found: ${title}`)
  return row
}

function formSubmit(root: TestElement, label = 'Answer') {
  const form = descendants(root).find((node) => node.tagName === 'FORM')
  if (!form) throw new Error(`Form not found under ${textOf(root)}`)
  return getButton(form, label)
}

function formFieldset(root: TestElement) {
  const fieldset = descendants(root).find((node) => node.tagName === 'FIELDSET')
  if (!fieldset) throw new Error(`Fieldset not found under ${textOf(root)}`)
  return fieldset
}

function focusControl(element: TestElement) {
  Object.assign(element, {
    attachEvent: () => {},
    detachEvent: () => {},
  })
  element.focus()
  element.dispatchEvent(new TestEvent('focusin'))
}

function currentReactProp(element: TestElement, name: string) {
  const entry = Object.entries(element).filter(([key]) => key.startsWith('__reactProps$')).at(-1)
  return (entry?.[1] as Record<string, unknown> | undefined)?.[name]
}

function dashboardData(overrides: Partial<Record<string, unknown>> = {}) {
  const overriddenCheckpoint = overrides.checkpoint as { data?: { items?: OrchestrationAuthorityDecisionRecord[] } } | undefined
  const defaultDecisions = overriddenCheckpoint?.data?.items ?? []
  return {
    processes: query<OrchestrationProcessRunRecord[]>([]),
    decisions: query<OrchestrationAuthorityDecisionRecord[]>(defaultDecisions),
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
    <MemoryRouter><GoalAnnouncerProvider>
      <BaselineDashboard projectId="project-1" goalId="goal-00000000-0000-4000-8000-000000000001"
        goal={goal()} run={run()} gates={[]} mutationBusy={false} acquireMutation={() => true}
        releaseMutation={() => undefined} onMessage={() => undefined} {...nextProps} />
    </GoalAnnouncerProvider></MemoryRouter>
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

function visibleLeafText(root: Parameters<typeof descendants>[0]) {
  const pieces: string[] = []
  const visit = (node: Parameters<typeof descendants>[0]) => {
    if ('tagName' in node && node.closest('details') !== null) return
    if (node.nodeType === 3) pieces.push(node.textContent)
    else node.childNodes.forEach((child) => visit(child))
  }
  visit(root)
  return pieces.join(' ').replace(/\s+/g, ' ').trim()
}

function withoutDetails(markup: string) {
  return markup.replace(/<details\b[\s\S]*?<\/details>/g, '')
}

beforeEach(() => {
  api.dashboard = dashboardData()
  api.health = { data: { debug_enabled: false } }
  vi.mocked(toast).mockReset()
  vi.mocked(toast.success).mockReset()
  vi.mocked(toast.error).mockReset()
  for (const mutation of [api.answer, api.answerAgentDefinitionBatch, api.skip, api.acknowledge, api.resolve, api.run, api.rerun, api.retry, api.authorize, api.debugStep, api.debugRerun]) {
    mutation.isPending = false
    mutation.mutate.mockReset()
  }
  api.answer.acceptRuntimeResult.mockReset()
})

describe('Baseline dashboard D3 composition', () => {
  it('SPR211 threads full-history reference metadata and exact source callbacks into Needs You', async () => {
    const sourceProcess = process({ id: '60000000-0000-4000-8000-000000000006' })
    const reference: NeedsYouReference = {
      kind: 'process', id: sourceProcess.id, project_id: 'project-1',
      goal_id: 'goal-00000000-0000-4000-8000-000000000001', label: 'Goal definition source',
    }
    const record = decision({
      id: 'referenced-decision', recommendation: null,
      question: `Review ${sourceProcess.id}.`, needs_you_references: [reference],
    })
    api.dashboard = dashboardData({
      processes: query([sourceProcess]),
      decisions: query([record]),
      checkpoint: query({ goal_id: record.goal_id, items: [record], deferred_count: 0, max_questions: 3 }),
    })
    const onRevealSource = vi.fn()
    const view = await mountDashboard({ needsYouReferences: [reference], onRevealSource })
    try {
      expect(visibleLeafText(view.container)).not.toContain(sourceProcess.id)
      await act(async () => getButton(view.container, 'Goal definition source').click())
      expect(onRevealSource).toHaveBeenCalledWith({
        kind: 'process', id: sourceProcess.id,
        goalId: 'goal-00000000-0000-4000-8000-000000000001',
      })
    } finally { view.cleanup() }
  })

  it('SPR210 reveals a warning source from superseded process history without resurfacing that process as current', async () => {
    const superseded = process({
      id: 'superseded-source-process',
      process_type: 'goal_definition',
      status: 'completed',
      superseded_by_id: 'current-source-process',
    })
    const current = process({
      id: 'current-source-process',
      process_type: 'goal_definition',
      status: 'running',
    })
    const sourceWarning = warning({
      id: 'warning-with-superseded-source',
      source_process_run_id: superseded.id,
    })
    const reference: NeedsYouReference = {
      kind: 'process', id: superseded.id, project_id: 'project-1',
      goal_id: goal().id, label: 'Original goal definition attempt',
    }
    api.dashboard = dashboardData({
      processes: query([superseded, current]),
      warnings: query([sourceWarning]),
    })
    const onRevealSource = vi.fn()
    const view = await mountDashboard({ needsYouReferences: [reference], onRevealSource })
    try {
      const warningRow = queueRow(view.container, sourceWarning.message)
      await act(async () => getButton(warningRow, reference.label!).click())
      expect(onRevealSource).toHaveBeenCalledWith({
        kind: 'process', id: superseded.id, goalId: goal().id,
      })
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON'
        && node.getAttribute('aria-label') === 'Run Goal definition')).toHaveLength(1)
    } finally { view.cleanup() }
  })

  it('SPR210 the queue-use answer form keeps the full multiline question visible exactly once with context and consequences', async () => {
    const question = 'Should the release proceed?\nUse the verified fallback if the primary path is unsafe.'
    const normalizedQuestion = question.replace('\n', ' ')
    const context = 'Review the final evidence.'
    const consequences = 'The fallback delays release but preserves verification.'
    const record = decision({
      id: 'queue-question', recommendation: null, title: 'Release approval',
      question, context, consequences,
    })
    api.dashboard = dashboardData({
      decisions: query([record]),
      checkpoint: query({ goal_id: record.goal_id, items: [record], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      expect(visibleLeafText(view.container).split(normalizedQuestion)).toHaveLength(2)
      expect(visibleLeafText(view.container)).toContain(context)
      expect(visibleLeafText(view.container)).toContain(consequences)
      await act(async () => getButton(view.container, 'Answer').click())
      expect(visibleLeafText(view.container).split(normalizedQuestion)).toHaveLength(2)
      expect(visibleLeafText(view.container)).toContain(context)
      expect(visibleLeafText(view.container)).toContain(consequences)
      expect(getButton(view.container, 'Submit answer')).toBeDefined()
    } finally { view.cleanup() }
  })

  it('SPR211 sanitizes recommendation, radio, button, and manager copy while submitting original option and reason values', async () => {
    const knownId = '10000000-0000-4000-8000-000000000001'
    const unknownId = '20000000-0000-4000-8000-000000000002'
    const reference: NeedsYouReference = {
      kind: 'agent', id: knownId, project_id: 'project-1', goal_id: goal().id, label: 'Named manager',
    }
    const record = decision({
      id: 'safe-manager-controls', decision_key: 'manager_selection:choose',
      title: 'Choose a manager', question: `Choose ${knownId} or ${unknownId}.`,
      context: `Deterministic: ${knownId}\nLLM: ${unknownId}\nRationale: Compare ${knownId} with ${unknownId}.`,
      options: [
        { key: knownId, label: `Assign ${knownId}` },
        { key: unknownId, label: `Assign ${unknownId}` },
      ],
      recommendation: knownId, needs_you_references: [reference],
    })
    api.dashboard = dashboardData({
      decisions: query([record]),
      checkpoint: query({ goal_id: record.goal_id, items: [record], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard({ needsYouReferences: [reference] })
    try {
      const closedCopy = visibleLeafText(view.container)
      expect(closedCopy).toContain('Named manager')
      expect(closedCopy).toContain('Reference unavailable')
      expect(closedCopy).not.toContain(knownId)
      expect(closedCopy).not.toContain(unknownId)
      expect(getButton(view.container, 'Accept: Assign Named manager')).toBeDefined()

      await act(async () => getButton(view.container, 'Choose differently…').click())
      const openCopy = visibleLeafText(view.container)
      expect(openCopy).toContain('Recommendation: Named manager')
      expect(openCopy).toContain('Assign Named manager')
      expect(openCopy).toContain('Assign Reference unavailable')
      expect(openCopy).not.toContain(knownId)
      expect(openCopy).not.toContain(unknownId)

      const unknownOption = descendants(view.container).find((node) => node.tagName === 'INPUT' && node.value === unknownId)!
      expect(unknownOption.value).toBe(unknownId)
      await act(async () => changeControl(unknownOption, unknownId))
      const originalReason = `Prefer ${unknownId} after checking ${knownId}.`
      await act(async () => changeControl(getByLabel(view.container, 'Reason (required when overriding the recommendation)', 'textarea'), originalReason))
      await act(async () => getButton(view.container, 'Submit answer').click())

      expect(api.answer.mutate).toHaveBeenCalledWith(expect.objectContaining({
        decisionId: record.id, selectedOption: unknownId, reason: originalReason,
      }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it.each(['recommendation', 'warning'] as const)('resolves %s warnings in one click and removes the row', async (severity) => {
    api.dashboard = dashboardData({ warnings: query([warning({ severity, message: 'Review this risk.' })]) })
    api.resolve.mutate.mockImplementation((_variables, options) => { options.onSuccess?.({}); options.onSettled?.() })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Acknowledge').click())
      expect(api.resolve.mutate).toHaveBeenCalledWith(expect.objectContaining({ reason: 'Acknowledged by operator' }), expect.any(Object))
      expect(api.acknowledge.mutate).not.toHaveBeenCalled()
      expect(textOf(view.container)).not.toContain('Review this risk.')
    } finally { view.cleanup() }
  })

  it.each(['blocker', 'hard_stop'] as const)('confirms %s resolution with an optional default reason', async (severity) => {
    api.dashboard = dashboardData({ warnings: query([warning({ severity })]) })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Resolve').click())
      expect(api.resolve.mutate).not.toHaveBeenCalled()
      expect(getByLabel(view.container, 'Reason (optional)', 'textarea')).toBeDefined()
      await act(async () => getButton(view.container, 'Resolve risk').click())
      expect(api.resolve.mutate).toHaveBeenCalledWith(expect.objectContaining({ reason: 'Acknowledged by operator' }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it('keeps a warning visible when acknowledgement resolution fails', async () => {
    api.dashboard = dashboardData({ warnings: query([warning({ message: 'Retry this risk.' })]) })
    api.resolve.mutate.mockImplementation((_variables, options) => { options.onError?.(new Error('Failed')); options.onSettled?.() })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Acknowledge').click())
      expect(textOf(view.container)).toContain('Retry this risk.')
    } finally { view.cleanup() }
  })

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
    api.run.mutate.mockImplementation((_variables, options) => { options.onSuccess?.({ process: { status: 'running' } }); options.onSettled?.() })
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

  it.each([
    ['idle', 'Nothing to run: no new evidence'],
    ['not_started', 'Nothing to run: no new evidence'],
    ['running', 'Running Goal definition.'],
    ['completed', 'Goal definition: Done.'],
    ['waiting_decision', 'Goal definition: Needs you.'],
    ['blocked', 'Goal definition: blocked.'],
    ['failed', 'Goal definition: failed.'],
  ])('announces the returned %s process status', async (status, message) => {
    const onMessage = vi.fn()
    api.run.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onSuccess?.({ process: { status, reason: 'no new evidence' } })
      callbacks.onSettled?.()
    })
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => controlElement(view.container, 'Run Goal definition').click())
      expect(onMessage).toHaveBeenCalledWith(message)
    } finally { view.cleanup() }
  })

  it('announces baseline authorization without claiming a process started', async () => {
    const onMessage = vi.fn()
    api.authorize.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onSuccess?.({ goal: goal(), run: run({ baseline_authorized: true }) })
      callbacks.onSettled?.()
    })
    const view = await mountDashboard({ onMessage, run: run({ phase: 'baseline', baseline_authorized: false }) })
    try {
      await act(async () => getButton(view.container, 'Start baseline').click())
      expect(onMessage).toHaveBeenCalledWith('Baseline authorized.')
      expect(onMessage).not.toHaveBeenCalledWith('Baseline started.')
    } finally { view.cleanup() }
  })

  it('surfaces structured authorization conflict details', async () => {
    const onMessage = vi.fn()
    api.authorize.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new ApiError(409, 'Request failed', { conflict: 'goal_not_runnable', message: 'Goal is paused' }))
      callbacks.onSettled?.()
    })
    const view = await mountDashboard({ onMessage, run: run({ phase: 'baseline', baseline_authorized: false }) })
    try {
      await act(async () => getButton(view.container, 'Start baseline').click())
      expect(onMessage).toHaveBeenCalledWith('Goal is paused')
      expect(textOf(view.container)).toContain('Goal is paused')
    } finally { view.cleanup() }
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

    const standard = withoutDetails(renderDashboard())
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

  it('SPR212 partial agent-definition answers preserve batch identity, frozen order, draft inputs, and focus', async () => {
    const focused = process({ id: 'review-process', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const original = [
      decision({ id: 'decision-a', decision_key: 'agent_definition_review:proposal:agent-1', source_process_run_id: focused.id, title: 'Review A', options: ['approve', 'reject', 'edit'] }),
      decision({ id: 'decision-b', decision_key: 'agent_definition_review:proposal:agent-2', source_process_run_id: focused.id, title: 'Review B', options: ['approve', 'reject', 'edit'], context: JSON.stringify({ original_description: '', proposed_description: 'Initial description', original_persona: '', proposed_persona: 'Initial persona' }) }),
      decision({ id: 'decision-c', decision_key: 'agent_definition_review:proposal:agent-3', source_process_run_id: focused.id, title: 'Review C', options: ['approve', 'reject', 'edit'] }),
    ]
    api.dashboard = dashboardData({
      processes: query([focused]),
      decisions: query(original),
      checkpoint: query({ goal_id: goal().id, items: original, deferred_count: 0, max_questions: 5 }),
    })

    const view = await mountDashboard()
    try {
      const reviewB = descendants(view.container).find((node) => node.tagName === 'ARTICLE' && textOf(node).includes('Review B'))!
      await act(async () => getButton(reviewB, 'Edit').click())
      const description = getByLabel(reviewB, 'Description', 'textarea')
      const persona = getByLabel(reviewB, 'Persona', 'textarea')
      await act(async () => changeControl(description, 'Keep this description'))
      await act(async () => changeControl(persona, 'Keep this persona'))
      persona.focus()

      const answered = { ...original[0], status: 'answered' as const, selected_option: 'approve', updated_at: '2026-08-05T12:00:00Z' }
      const second = { ...original[1], updated_at: '2026-08-05T13:00:00Z' }
      const third = { ...original[2], updated_at: '2026-08-05T14:00:00Z' }
      api.dashboard = dashboardData({
        processes: query([focused]),
        decisions: query([answered, second, third]),
        checkpoint: query({ goal_id: goal().id, items: [third, second], deferred_count: 0, max_questions: 5 }),
      })
      await view.rerender()

      const articles = descendants(view.container).filter((node) => node.tagName === 'ARTICLE')
      expect(articles.map((node) => textOf(node).match(/Review [ABC]/)?.[0])).toEqual(['Review A', 'Review B', 'Review C'])
      const retainedDescription = getByLabel(articles[1], 'Description', 'textarea')
      const retainedPersona = getByLabel(articles[1], 'Persona', 'textarea')
      expect(retainedDescription).toBe(description)
      expect(retainedPersona).toBe(persona)
      expect(retainedDescription.value).toBe('Keep this description')
      expect(retainedPersona.value).toBe('Keep this persona')
      expect(view.document.activeElement).toBe(retainedPersona)
    } finally {
      view.cleanup()
    }
  })

  it('SPR211 sanitizes agent-review previews while editable values and submitted payloads stay exact', async () => {
    const knownId = '30000000-0000-4000-8000-000000000003'
    const unknownId = '40000000-0000-4000-8000-000000000004'
    const reference: NeedsYouReference = {
      kind: 'agent', id: knownId, project_id: 'project-1', goal_id: goal().id, label: 'Named reviewer',
    }
    const focused = process({ id: 'safe-review-process', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const runtimeKey = 'run:raw-runtime-12345678'
    const proposedDescription = `Description for ${knownId} with ${unknownId} from ${runtimeKey}`
    const proposedPersona = `Persona for ${unknownId} reviewed by ${knownId}`
    const item = decision({
      id: 'safe-agent-review', decision_key: 'agent_definition_review:proposal:agent-1',
      source_process_run_id: focused.id, title: `Review ${knownId}`,
      question: `Approve ${knownId} after checking ${unknownId}?`,
      options: ['approve', 'reject', 'edit'], recommendation: 'approve',
      context: JSON.stringify({
        original_description: `Current ${knownId}`, proposed_description: proposedDescription,
        original_persona: `Current ${unknownId}`, proposed_persona: proposedPersona,
      }),
      needs_you_references: [reference],
    })
    api.dashboard = dashboardData({
      processes: query([focused]), decisions: query([item]),
      checkpoint: query({ goal_id: item.goal_id, items: [item], deferred_count: 0, max_questions: 5 }),
    })
    const view = await mountDashboard({ needsYouReferences: [reference] })
    try {
      const primary = visibleLeafText(view.container)
      expect(primary).toContain('Named reviewer')
      expect(primary).toContain('Reference unavailable')
      expect(primary).not.toContain(knownId)
      expect(primary).not.toContain(unknownId)
      expect(primary).not.toContain(runtimeKey)

      const article = descendants(view.container).find((node) => node.tagName === 'ARTICLE')!
      await act(async () => getButton(article, 'Edit').click())
      const description = getByLabel(article, 'Description', 'textarea')
      const persona = getByLabel(article, 'Persona', 'textarea')
      expect(description.value).toBe(proposedDescription)
      expect(persona.value).toBe(proposedPersona)
      const changedDescription = `${proposedDescription}!`
      await act(async () => changeControl(description, changedDescription))
      await act(async () => getButton(article, 'Confirm edit').click())

      expect(api.answerAgentDefinitionBatch.mutate).toHaveBeenCalledWith({
        goalId: goal().id,
        answers: [{
          decisionId: item.id, selectedOption: 'edit', reason: '',
          editedDescription: changedDescription, editedPersona: proposedPersona,
        }],
      }, expect.any(Object))
    } finally { view.cleanup() }
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

  it('shows agent names in proposed hierarchy assignments, reporting and unused agents without raw JSON', () => {
    const implementerId = '11111111-1111-4111-8111-111111111111'
    const managerId = '22222222-2222-4222-8222-222222222222'
    const unusedId = '33333333-3333-4333-8333-333333333333'
    const unknownId = '44444444-4444-4444-8444-444444444444'
    const hierarchy = process({ id: 'hierarchy-process', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const proposal = {
      assignments: [{ work_function: 'build', agent_ref: implementerId }],
      reporting_lines: [{ agent_ref: implementerId, reports_to: managerId }],
      approved_but_unused_agent_ids: [unusedId, unknownId],
      documented_gaps: [], rationale: 'Coverage checked.', self_review: 'Ready.',
    }
    const approval = decision({ decision_key: 'team_hierarchy:approval', source_process_run_id: hierarchy.id, options: [{ key: 'approve', proposal }] })
    api.dashboard = dashboardData({
      processes: query([hierarchy]), decisions: query([approval]),
      checkpoint: query({ goal_id: goal().id, items: [approval], deferred_count: 0, max_questions: 5 }),
      agentReviews: query([
        review({ agent_id: implementerId, definition_snapshot: { name: 'Ada Builder' } }),
        review({ agent_id: managerId, definition_snapshot: { name: 'Grace Manager' } }),
        review({ agent_id: unusedId, definition_snapshot: { name: 'Linus Reviewer' } }),
      ]),
    })
    const markup = renderDashboard()
    const visibleMarkup = withoutDetails(markup)
    expect(markup).toContain('build: Ada Builder')
    expect(markup).toContain('Ada Builder reports to Grace Manager')
    expect(markup).toContain('Linus Reviewer, Unknown agent')
    for (const id of [implementerId, managerId, unusedId, unknownId]) expect(visibleMarkup).not.toContain(id)
    expect(visibleMarkup).not.toContain('{')
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
    const recommended = decision({ id: 'decision-1', recommendation: 'approve', options: ['approve', { key: 'revise', label: 'Revise scope' }], runtime_identity: 'runtime-1', contract_version: 'plan:1' })
    api.dashboard = dashboardData({
      checkpoint: query({ goal_id: 'goal-00000000-0000-4000-8000-000000000001', items: [recommended], deferred_count: 0, max_questions: 5 }),
    })
    const view = await mountDashboard()
    try {
      expect(() => getButton(view.container, 'Submit answer')).toThrow()
      await act(async () => getButton(view.container, 'Accept: approve').click())
      expect(api.answer.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-00000000-0000-4000-8000-000000000001', decisionId: 'decision-1', selectedOption: 'approve', reason: '', contractVersion: 'plan:1', runtimeIdentity: 'runtime-1' },
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
      expect(textOf(view.container)).toContain('2 goal clarification questions need an answer.')
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
  it('SPR212 full checkpoint retains oldest edited draft after a new arrival; SPR212 runtime contract race returns inert 200 and retains draft', async () => {
    const runtimeQuestion = (number: number, askedAt: string) => decision({
      id: `runtime-q${number}`,
      decision_key: `runtime:question:${number}`,
      title: `Runtime question Q${number}`,
      question: `Explain the choice for Q${number}.`,
      options: [{ key: 'acknowledge' }],
      recommendation: null,
      runtime_identity: `runtime-identity-q${number}`,
      contract_version: `contract-q${number}:v1`,
      asked_at: askedAt,
      created_at: askedAt,
      updated_at: askedAt,
    })
    const q1 = runtimeQuestion(1, '2026-08-05T08:01:00Z')
    const q2 = runtimeQuestion(2, '2026-08-05T08:02:00Z')
    const q3 = runtimeQuestion(3, '2026-08-05T08:03:00Z')
    const q4 = runtimeQuestion(4, '2026-08-05T08:04:00Z')
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    let answerCallbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, callbacks) => { answerCallbacks = callbacks })
    const onMessage = vi.fn()
    const view = await mountDashboard({ onMessage })
    try {
      const q1Row = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes(q1.title))!
      await act(async () => getButton(q1Row, 'Answer').click())
      const draft = getByLabel(q1Row, 'Answer', 'textarea')
      const substantiveDraft = 'Keep the reviewed Q1 contract and this substantive explanation.'
      await act(async () => changeControl(draft, substantiveDraft))
      draft.focus()

      const authoritative = query([q4, q3, q2, q1])
      authoritative.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
      api.dashboard = dashboardData({
        decisions: authoritative,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()

      const retainedQ1Row = descendants(view.container).find((node) => node.tagName === 'LI' && textOf(node).includes(q1.title))!
      const retainedDraft = getByLabel(retainedQ1Row, 'Answer', 'textarea')
      expect(retainedDraft).toBe(draft)
      expect(retainedDraft.value).toBe(substantiveDraft)
      expect(view.document.activeElement).toBe(draft)
      const visibleRuntimeRows = descendants(view.container)
        .filter((node) => node.tagName === 'LI' && /Runtime question Q[1-4]/.test(textOf(node)))
        .map((node) => textOf(node).match(/Runtime question Q[1-4]/)?.[0])
      expect(visibleRuntimeRows).toEqual([
        'Runtime question Q4', 'Runtime question Q3', 'Runtime question Q2', 'Runtime question Q1',
      ])
      expect(textOf(view.container).match(/1 more queued/g)).toHaveLength(1)

      const submit = getButton(descendants(retainedQ1Row).find((node) => node.tagName === 'FORM')!, 'Answer')
      await act(async () => { submit.click(); await Promise.resolve() })
      expect(authoritative.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledWith({
        goalId: goal().id,
        decisionId: q1.id,
        selectedOption: 'acknowledge',
        reason: substantiveDraft,
        contractVersion: q1.contract_version,
        runtimeIdentity: q1.runtime_identity,
      }, expect.any(Object))

      const changedContract = {
        ...q1,
        contract_version: 'contract-q1:v2',
        runtime_identity: 'runtime-identity-q1-v2',
        updated_at: '2026-08-05T08:05:00Z',
      }
      await act(async () => {
        answerCallbacks?.onSuccess?.({ decision: changedContract, process: null, continuation_applied: false })
        answerCallbacks?.onSettled?.()
      })

      const staleDraft = getByLabel(view.container, 'Answer', 'textarea')
      expect(staleDraft).toBe(draft)
      expect(staleDraft.value).toBe(substantiveDraft)
      expect(view.document.activeElement).toBe(draft)
      expect(formFieldset(queueRow(view.container, q1.title)).disabled).toBe(false)
      expect(getButton(queueRow(view.container, q1.title), 'Cancel').disabled).toBe(false)
      expect(textOf(view.container)).toMatch(/review.*refresh|refresh.*review/i)
      expect(textOf(view.container)).toContain('1 more queued')
      expect(onMessage).not.toHaveBeenCalledWith('Answer received.')
      expect(toast.success).not.toHaveBeenCalledWith('Answer received.')
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.acceptRuntimeResult).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it.each(['validated success', 'explicit cancel'] as const)(
    'SPR212 retained edit releases only after %s and uses displayed-order focus fallback',
    async (branch) => {
      const q1 = task4RuntimeQuestion(1)
      const q2 = task4RuntimeQuestion(2)
      const q3 = task4RuntimeQuestion(3)
      const q4 = task4RuntimeQuestion(4)
      api.dashboard = dashboardData({
        decisions: query([q3, q2, q1]),
        checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
      })
      let answerCallbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
      api.answer.mutate.mockImplementation((_variables, callbacks) => { answerCallbacks = callbacks })
      const onMessage = vi.fn()
      const view = await mountDashboard({ onMessage })
      try {
        await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
        const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
        await act(async () => changeControl(draft, 'Original reviewed answer'))
        draft.focus()

        const authoritative = query([q4, q3, q2, q1])
        authoritative.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
        api.dashboard = dashboardData({
          decisions: authoritative,
          checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
        })
        await view.rerender()

        const retained = queueRow(view.container, q1.title)
        expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
        if (branch === 'explicit cancel') {
          await act(async () => getButton(retained, 'Cancel').click())
          expect(api.answer.mutate).not.toHaveBeenCalled()
        } else {
          await act(async () => { formSubmit(retained).click(); await Promise.resolve() })
          expect(api.answer.mutate).toHaveBeenCalledTimes(1)
          expect(api.answer.mutate.mock.calls[0][0]).toEqual({
            goalId: goal().id,
            decisionId: q1.id,
            selectedOption: 'acknowledge',
            reason: 'Original reviewed answer',
            contractVersion: q1.contract_version,
            runtimeIdentity: q1.runtime_identity,
          })
          const matchingAnswer = {
            ...q1,
            status: 'answered' as const,
            selected_option: 'acknowledge',
            reason: 'Original reviewed answer',
            decided_at: '2026-08-05T08:05:00Z',
          }
          await act(async () => {
            answerCallbacks?.onSuccess?.({
              decision: matchingAnswer, process: null, continuation_applied: false,
            })
            answerCallbacks?.onSettled?.()
          })
          expect(onMessage).toHaveBeenCalledTimes(1)
          expect(onMessage).toHaveBeenCalledWith('Answer received.')
          expect(toast.success).toHaveBeenCalledTimes(1)
          expect(toast.success).toHaveBeenCalledWith('Answer received.')
        }

        expect(() => queueRow(view.container, q1.title)).toThrow()
        expect(textOf(view.container)).toContain('1 more queued')
        expect(view.document.activeElement?.tagName).toBe('H2')
        expect(textOf(view.document.activeElement as TestElement)).toBe('Needs you')
      } finally { view.cleanup() }
    },
  )

  it.each([
    ['Answer', null],
    ['Choose differently…', 'acknowledge'],
  ] as const)(
    'SPR212 cancelled %s can reopen and preserve its new draft through later checkpoint eviction',
    async (openLabel, recommendation) => {
      const q1 = task4RuntimeQuestion(1, { recommendation })
      const q2 = task4RuntimeQuestion(2)
      const q3 = task4RuntimeQuestion(3)
      const q4 = task4RuntimeQuestion(4)
      api.dashboard = dashboardData({
        decisions: query([q3, q2, q1]),
        checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
      })
      const view = await mountDashboard()
      try {
        await act(async () => getButton(queueRow(view.container, q1.title), openLabel).click())
        await act(async () => getButton(queueRow(view.container, q1.title), 'Cancel').click())
        expect(() => getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')).toThrow()

        await act(async () => getButton(queueRow(view.container, q1.title), openLabel).click())
        const reopenedDraft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
        await act(async () => changeControl(reopenedDraft, 'This second edit must survive displacement'))
        reopenedDraft.focus()

        api.dashboard = dashboardData({
          decisions: query([q4, q3, q2, q1]),
          checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
        })
        await view.rerender()

        const retained = queueRow(view.container, q1.title)
        expect(getByLabel(retained, 'Answer', 'textarea')).toBe(reopenedDraft)
        expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('This second edit must survive displacement')
        expect(view.document.activeElement).toBe(reopenedDraft)
      } finally { view.cleanup() }
    },
  )

  it('SPR212 retained adaptive round keeps its frozen question order and active draft when displaced', async () => {
    const goalProcess = process({
      id: 'goal-definition-round-process', process_type: 'goal_definition', status: 'waiting_decision',
    })
    const first = decision({
      id: 'adaptive-round-q1', source_process_run_id: goalProcess.id,
      decision_key: 'goal_definition:adaptive:5:0:objective',
      title: 'Clarify objective', question: 'What outcome matters most?',
      options: [], recommendation: null,
    })
    const second = decision({
      id: 'adaptive-round-q2', source_process_run_id: goalProcess.id,
      decision_key: 'goal_definition:adaptive:5:1:constraints',
      title: 'Clarify constraints', question: 'Which constraint is fixed?',
      options: [], recommendation: null,
    })
    const arrivals = [task4RuntimeQuestion(4), task4RuntimeQuestion(3), task4RuntimeQuestion(2)]
    api.dashboard = dashboardData({
      processes: query([goalProcess]),
      decisions: query([first, second]),
      checkpoint: query({ goal_id: goal().id, items: [first, second], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      const draft = getByLabel(view.container, 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'The verified shipped outcome.'))
      await act(async () => focusControl(draft))

      api.dashboard = dashboardData({
        processes: query([goalProcess]),
        decisions: query([...arrivals, first, second]),
        checkpoint: query({ goal_id: goal().id, items: arrivals, deferred_count: 2, max_questions: 3 }),
      })
      await view.rerender()

      const retainedDraft = getByLabel(view.container, 'Answer', 'textarea')
      expect(retainedDraft).toBe(draft)
      expect(retainedDraft.value).toBe('The verified shipped outcome.')
      expect(view.document.activeElement).toBe(draft)
      expect(textOf(view.container)).toContain('Question 1 of 2')
      expect(textOf(view.container)).toContain('What outcome matters most?')
      expect(textOf(view.container)).not.toContain('Question 1 of 1')
      expect(textOf(view.container)).toContain('2 more queued')
    } finally { view.cleanup() }
  })

  it('SPR212 repeated arrivals keep one retained node and an interrupted edit transfers retention once', async () => {
    const questions = [1, 2, 3, 4, 5, 6].map((number) => task4RuntimeQuestion(number))
    const [q1, q2, q3, q4, q5, q6] = questions
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const q1Draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(q1Draft, 'Q1 draft across repeated arrivals'))
      q1Draft.focus()

      for (const [checkpoint, deferred] of [
        [[q4, q3, q2], 1],
        [[q5, q4, q3], 2],
      ] as const) {
        api.dashboard = dashboardData({
          decisions: query([q5, q4, q3, q2, q1]),
          checkpoint: query({ goal_id: goal().id, items: checkpoint, deferred_count: deferred, max_questions: 3 }),
        })
        await view.rerender()
        const retained = queueRow(view.container, q1.title)
        expect(getByLabel(retained, 'Answer', 'textarea')).toBe(q1Draft)
        expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('Q1 draft across repeated arrivals')
        expect(view.document.activeElement).toBe(q1Draft)
        expect(descendants(view.container).filter((node) => node.tagName === 'LI' && textOf(node).includes(q1.title))).toHaveLength(1)
      }

      await act(async () => getButton(queueRow(view.container, q5.title), 'Answer').click())
      const q5Draft = getByLabel(queueRow(view.container, q5.title), 'Answer', 'textarea')
      await act(async () => changeControl(q5Draft, 'Q5 became the one active edit'))
      q5Draft.focus()
      expect(() => queueRow(view.container, q1.title)).toThrow()
      expect(api.answer.mutate).not.toHaveBeenCalled()

      api.dashboard = dashboardData({
        decisions: query([q6, q5, q4, q3, q2, q1]),
        checkpoint: query({ goal_id: goal().id, items: [q6, q5, q4], deferred_count: 3, max_questions: 3 }),
      })
      await view.rerender()
      const retainedQ5 = queueRow(view.container, q5.title)
      expect(getByLabel(retainedQ5, 'Answer', 'textarea')).toBe(q5Draft)
      expect(getByLabel(retainedQ5, 'Answer', 'textarea').value).toBe('Q5 became the one active edit')
      expect(view.document.activeElement).toBe(q5Draft)
      expect(descendants(view.container).filter((node) => node.tagName === 'LI' && textOf(node).includes(q5.title))).toHaveLength(1)
      expect(textOf(view.container)).toContain('3 more queued')
    } finally { view.cleanup() }
  })

  it('SPR212 retained terminal question keeps read-only draft and cannot answer', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    const onMessage = vi.fn()
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'My still-visible draft'))
      draft.focus()
      const externallyAnswered = {
        ...q1,
        status: 'answered' as const,
        selected_option: 'acknowledge',
        reason: 'The other operator recorded this answer.',
        decided_at: '2026-08-05T08:05:00Z',
        updated_at: '2026-08-05T08:05:00Z',
      }
      api.dashboard = dashboardData({
        decisions: query([q4, q3, q2, externallyAnswered]),
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 0, max_questions: 3 }),
      })
      await view.rerender()

      const retained = queueRow(view.container, q1.title)
      const readOnlyDraft = getByLabel(retained, 'Answer', 'textarea')
      expect(readOnlyDraft).toBe(draft)
      expect(readOnlyDraft.value).toBe('My still-visible draft')
      expect(formFieldset(retained).disabled).toBe(false)
      expect(formSubmit(retained).disabled).toBe(true)
      expect(readOnlyDraft.disabled).toBe(false)
      expect(currentReactProp(readOnlyDraft, 'readOnly')).toBe(true)
      readOnlyDraft.focus()
      expect(view.document.activeElement).toBe(readOnlyDraft)
      expect(textOf(retained)).toContain('Answered elsewhere')
      expect(textOf(retained)).toContain('acknowledge')
      expect(textOf(retained)).toContain('The other operator recorded this answer.')
      expect(descendants(retained).some((node) => node.getAttribute('aria-live') === 'polite')).toBe(true)
      expect(textOf(view.container)).not.toContain('Queue clear.')
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(externallyAnswered.selected_option).toBe('acknowledge')
      expect(externallyAnswered.reason).toBe('The other operator recorded this answer.')

      await act(async () => getButton(queueRow(view.container, q1.title), 'Dismiss').click())
      expect(() => queueRow(view.container, q1.title)).toThrow()
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(onMessage).not.toHaveBeenCalledWith('Queue clear.')
      expect(view.document.activeElement?.tagName).toBe('H2')
    } finally { view.cleanup() }
  })

  it.each([
    ['contract version changed', (q1: OrchestrationAuthorityDecisionRecord) => query([{ ...q1, contract_version: 'contract-q1:v2' }])],
    ['runtime identity changed', (q1: OrchestrationAuthorityDecisionRecord) => query([{ ...q1, runtime_identity: 'runtime-identity-q1-v2' }])],
    ['authority record missing', (_q1: OrchestrationAuthorityDecisionRecord) => query<OrchestrationAuthorityDecisionRecord[]>([])],
    ['authority query loading', (_q1: OrchestrationAuthorityDecisionRecord) => query<OrchestrationAuthorityDecisionRecord[]>(undefined, { isLoading: true, isFetching: true })],
    ['authority query failed', (_q1: OrchestrationAuthorityDecisionRecord) => query<OrchestrationAuthorityDecisionRecord[]>(undefined, { isError: true })],
  ] as const)(
    'SPR212 retained contract change or authority failure blocks stale submission: %s',
    async (name, authorityQuery) => {
      const q1 = task4RuntimeQuestion(1)
      const q2 = task4RuntimeQuestion(2)
      const q3 = task4RuntimeQuestion(3)
      const q4 = task4RuntimeQuestion(4)
      api.dashboard = dashboardData({
        decisions: query([q3, q2, q1]),
        checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
      })
      const view = await mountDashboard()
      try {
        await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
        const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
        await act(async () => changeControl(draft, 'Do not bind this to a new contract.'))
        draft.focus()

        api.dashboard = dashboardData({
          decisions: authorityQuery(q1),
          checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
        })
        await view.rerender()

        const retained = queueRow(view.container, q1.title)
        const blockedDraft = getByLabel(retained, 'Answer', 'textarea')
        expect(blockedDraft).toBe(draft)
        expect(blockedDraft.value).toBe('Do not bind this to a new contract.')
        expect(formFieldset(retained).disabled).toBe(false)
        expect(formSubmit(retained).disabled).toBe(true)
        expect(getButton(retained, 'Cancel').disabled).toBe(false)
        expect(currentReactProp(blockedDraft, 'readOnly')).toBe(true)
        expect(textOf(retained)).toMatch(/review.*refresh|refresh.*review/i)
        if (name === 'contract version changed' || name === 'runtime identity changed') {
          expect(getButton(retained, 'Review refreshed decision')).toBeDefined()
        } else {
          expect(() => getButton(retained, 'Review refreshed decision')).toThrow()
        }
        await act(async () => formSubmit(retained).click())
        expect(api.answer.mutate).not.toHaveBeenCalled()
      } finally { view.cleanup() }
    },
  )

  it('SPR212 unchanged authority after uncertainty restores the same retained draft submission', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Recover this exact draft'))
      api.dashboard = dashboardData({
        decisions: query<OrchestrationAuthorityDecisionRecord[]>(undefined, { isLoading: true, isFetching: true }),
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      expect(formSubmit(queueRow(view.container, q1.title)).disabled).toBe(true)

      const restoredAuthority = query([q4, q3, q2, q1])
      restoredAuthority.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
      api.dashboard = dashboardData({
        decisions: restoredAuthority,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()

      const restored = queueRow(view.container, q1.title)
      expect(getByLabel(restored, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(restored, 'Answer', 'textarea').value).toBe('Recover this exact draft')
      expect(formSubmit(restored).disabled).toBe(false)
      await act(async () => { formSubmit(restored).click(); await Promise.resolve() })
      expect(restoredAuthority.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('SPR212 retained legacy answer-contract change blocks the frozen free-text submission', async () => {
    const legacy = task4RuntimeQuestion(1, {
      runtime_identity: null,
      contract_version: null,
      options: [],
      question: 'What exact legacy answer should be recorded?',
    })
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    api.dashboard = dashboardData({
      decisions: query([q3, q2, legacy]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, legacy], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, legacy.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, legacy.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Frozen legacy answer'))
      api.dashboard = dashboardData({
        decisions: query([q4, q3, q2, {
          ...legacy,
          question: 'A replacement legacy question that was never reviewed',
          options: ['replacement-option'],
        }]),
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()

      const retained = queueRow(view.container, legacy.title)
      expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('Frozen legacy answer')
      expect(formFieldset(retained).disabled).toBe(false)
      expect(formSubmit(retained, 'Submit answer').disabled).toBe(true)
      expect(getButton(retained, 'Cancel').disabled).toBe(false)
      expect(textOf(retained)).toMatch(/review.*refresh|refresh.*review/i)
      expect(api.answer.mutate).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('SPR212 submit preflight contract change blocks mutation and preserves the reviewed draft', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    const currentAuthority = query([q4, q3, q2, q1])
    currentAuthority.refetch.mockResolvedValue({ data: [q4, q3, q2, {
      ...q1,
      contract_version: 'contract-q1:v2',
      runtime_identity: 'runtime-identity-q1-v2',
    }], isSuccess: true, isError: false })
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Submit only against v1'))
      draft.focus()
      api.dashboard = dashboardData({
        decisions: currentAuthority,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()

      const retained = queueRow(view.container, q1.title)
      await act(async () => { formSubmit(retained).click(); await Promise.resolve() })

      expect(currentAuthority.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('Submit only against v1')
      expect(formFieldset(retained).disabled).toBe(false)
      expect(formSubmit(retained).disabled).toBe(true)
      expect(textOf(retained)).toMatch(/review.*refresh|refresh.*review/i)
      expect(view.document.activeElement).toBe(draft)
    } finally { view.cleanup() }
  })

  it('SPR212 submit preflight rejects cached pending data when the refetch result is an error', async () => {
    const q1 = task4RuntimeQuestion(1)
    const authority = query([q1])
    authority.refetch.mockResolvedValue({
      data: [q1],
      isError: true,
      isSuccess: false,
      error: new Error('authority refresh failed'),
    })
    api.dashboard = dashboardData({
      decisions: authority,
      checkpoint: query({ goal_id: goal().id, items: [q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Do not trust cached data after a failed refetch'))

      await act(async () => { formSubmit(queueRow(view.container, q1.title)).click(); await Promise.resolve() })

      expect(authority.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).not.toHaveBeenCalled()
      const retained = queueRow(view.container, q1.title)
      expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('Do not trust cached data after a failed refetch')
      expect(formFieldset(retained).disabled).toBe(false)
      expect(formSubmit(retained).disabled).toBe(true)
      expect(textOf(retained)).toMatch(/refresh.*review|review.*refresh/i)
    } finally { view.cleanup() }
  })

  it('SPR212 review refreshed decision explicitly remounts a fresh form on the authoritative contract', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    const refreshedQ1 = {
      ...q1,
      contract_version: 'contract-q1:v2',
      runtime_identity: 'runtime-identity-q1-v2',
      question: 'Explain the refreshed Q1 contract.',
      updated_at: '2026-08-05T08:05:00Z',
    }
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const staleDraft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(staleDraft, 'Draft tied only to v1'))
      api.dashboard = dashboardData({
        decisions: query([q4, q3, q2, refreshedQ1]),
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, q1.title)
      expect(getByLabel(retained, 'Answer', 'textarea')).toBe(staleDraft)
      expect(formFieldset(retained).disabled).toBe(false)
      expect(formSubmit(retained).disabled).toBe(true)

      await act(async () => getButton(retained, 'Review refreshed decision').click())

      const refreshed = queueRow(view.container, q1.title)
      const freshDraft = getByLabel(refreshed, 'Answer', 'textarea')
      expect(freshDraft).not.toBe(staleDraft)
      expect(freshDraft.value).toBe('')
      expect(freshDraft.disabled).toBe(false)
      expect(textOf(refreshed)).toContain('Explain the refreshed Q1 contract.')
      expect(textOf(refreshed)).not.toContain('Draft tied only to v1')
      expect(formSubmit(refreshed).disabled).toBe(true)
      expect(api.answer.mutate).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('SPR212 reviewing a refreshed hierarchy proposal focuses its proposal control', async () => {
    const hierarchy = process({ id: 'task4-hierarchy', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const definition = {
      name: 'Alpha', role: 'worker', description: 'Alpha description', provider: 'openai', model: 'gpt-5',
      system_prompt: 'Review the hierarchy.', adapter_type: 'api', cli_runtime: null, capabilities: ['build'], config: {},
    }
    const original = decision({
      id: 'task4-hierarchy-proposal', decision_key: 'team_hierarchy:agent:alpha', title: 'Review Alpha',
      source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], contract_version: 'hierarchy:v1',
      context: JSON.stringify({ proposal: { proposal_id: 'alpha', definition } }),
    })
    api.dashboard = dashboardData({
      processes: query([hierarchy]), decisions: query([original]),
      checkpoint: query({ goal_id: goal().id, items: [original], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      const originalRow = queueRow(view.container, original.title)
      await act(async () => getButton(originalRow, 'Edit').click())
      const originalName = getByLabel(originalRow, 'Name', 'input')
      originalName.focus()

      const refreshed = { ...original, contract_version: 'hierarchy:v2', updated_at: '2026-08-05T08:05:00Z' }
      const arrival = task4RuntimeQuestion(9)
      api.dashboard = dashboardData({
        processes: query([hierarchy]), decisions: query([arrival, refreshed]),
        checkpoint: query({ goal_id: goal().id, items: [arrival], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, original.title)
      await act(async () => getButton(retained, 'Review refreshed decision').click())
      await act(async () => { await Promise.resolve() })

      const reviewedRow = queueRow(view.container, original.title)
      const proposalFocus = view.document.activeElement as TestElement
      expect(proposalFocus.tagName).toBe('BUTTON')
      expect(textOf(proposalFocus)).toBe('Approve')
      expect(descendants(reviewedRow).some((node) => node === proposalFocus)).toBe(true)
    } finally { view.cleanup() }
  })

  it('SPR212 cancelling a still-selected reviewed hierarchy editor releases retention and restores local focus', async () => {
    const hierarchy = process({ id: 'task4-selected-hierarchy', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const definition = {
      name: 'Selected Alpha', role: 'worker', description: 'Original proposal', provider: 'openai', model: 'gpt-5',
      system_prompt: 'Review the hierarchy.', adapter_type: 'api', cli_runtime: null, capabilities: ['build'], config: {},
    }
    const original = decision({
      id: 'task4-selected-hierarchy-proposal', decision_key: 'team_hierarchy:agent:selected-alpha', title: 'Review Selected Alpha',
      source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], contract_version: 'selected-hierarchy:v1',
      context: JSON.stringify({ proposal: { proposal_id: 'selected-alpha', definition } }),
    })
    api.dashboard = dashboardData({
      processes: query([hierarchy]), decisions: query([original]),
      checkpoint: query({ goal_id: goal().id, items: [original], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, original.title), 'Edit').click())
      const refreshed = {
        ...original,
        contract_version: 'selected-hierarchy:v2',
        context: JSON.stringify({ proposal: { proposal_id: 'selected-alpha', definition: { ...definition, name: 'Selected Alpha v2' } } }),
        updated_at: '2026-08-05T08:15:00Z',
      }
      api.dashboard = dashboardData({
        processes: query([hierarchy]), decisions: query([refreshed]),
        checkpoint: query({ goal_id: goal().id, items: [refreshed], deferred_count: 0, max_questions: 3 }),
      })
      await view.rerender()
      await act(async () => getButton(queueRow(view.container, original.title), 'Review refreshed decision').click())

      const reviewed = queueRow(view.container, original.title)
      await act(async () => getButton(reviewed, 'Edit').click())
      expect(getByLabel(reviewed, 'Name', 'input').value).toBe('Selected Alpha v2')
      await act(async () => getButton(reviewed, 'Cancel').click())
      const surviving = queueRow(view.container, original.title)
      expect(() => getByLabel(surviving, 'Name', 'input')).toThrow()
      expect(view.document.activeElement === getButton(surviving, 'Edit')).toBe(true)

      await act(async () => getButton(surviving, 'Edit').click())
      expect(getByLabel(surviving, 'Name', 'input').value).toBe('Selected Alpha v2')
      await act(async () => getButton(surviving, 'Cancel').click())

      const nextContract = {
        ...refreshed,
        contract_version: 'selected-hierarchy:v3',
        context: JSON.stringify({ proposal: { proposal_id: 'selected-alpha', definition: { ...definition, name: 'Selected Alpha v3' } } }),
        updated_at: '2026-08-05T08:16:00Z',
      }
      api.dashboard = dashboardData({
        processes: query([hierarchy]), decisions: query([nextContract]),
        checkpoint: query({ goal_id: goal().id, items: [nextContract], deferred_count: 0, max_questions: 3 }),
      })
      await view.rerender()
      expect(textOf(queueRow(view.container, original.title))).toContain('Selected Alpha v3')
      expect(() => getButton(queueRow(view.container, original.title), 'Review refreshed decision')).toThrow()
      await act(async () => getButton(queueRow(view.container, original.title), 'Edit').click())
      expect(getByLabel(queueRow(view.container, original.title), 'Name', 'input').value).toBe('Selected Alpha v3')
    } finally { view.cleanup() }
  })

  it('SPR212 cancelling a displaced hierarchy editor releases the retained card and focuses the next displayed row', async () => {
    const hierarchy = process({ id: 'task4-hierarchy-cancel', process_type: 'team_hierarchy', status: 'waiting_decision' })
    const definition = {
      name: 'Cancel Alpha', role: 'worker', description: 'Cancel this edit', provider: 'openai', model: 'gpt-5',
      system_prompt: 'Review the hierarchy.', adapter_type: 'api', cli_runtime: null, capabilities: ['build'], config: {},
    }
    const proposal = decision({
      id: 'task4-hierarchy-cancel-proposal', decision_key: 'team_hierarchy:agent:cancel-alpha', title: 'Review Cancel Alpha',
      source_process_run_id: hierarchy.id, options: ['approve', 'edit', 'reject'], contract_version: 'hierarchy:cancel:v1',
      context: JSON.stringify({ proposal: { proposal_id: 'cancel-alpha', definition } }),
    })
    api.dashboard = dashboardData({
      processes: query([hierarchy]), decisions: query([proposal]),
      checkpoint: query({ goal_id: goal().id, items: [proposal], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      const originalRow = queueRow(view.container, proposal.title)
      await act(async () => getButton(originalRow, 'Edit').click())
      expect(getByLabel(originalRow, 'Name', 'input')).toBeDefined()

      const arrival = task4RuntimeQuestion(9)
      api.dashboard = dashboardData({
        processes: query([hierarchy]), decisions: query([arrival, proposal]),
        checkpoint: query({ goal_id: goal().id, items: [arrival], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, proposal.title)
      await act(async () => { getButton(retained, 'Cancel').click(); await Promise.resolve() })

      expect(() => queueRow(view.container, proposal.title)).toThrow()
      expect(view.document.activeElement === getButton(queueRow(view.container, arrival.title), 'Answer')).toBe(true)
      expect(api.answer.mutate).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('SPR212 pending submit preflight keeps the same focused draft until authority resolves', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    let resolveRefetch!: (value: { data: OrchestrationAuthorityDecisionRecord[]; isSuccess: true; isError: false }) => void
    const refetchResult = new Promise<{ data: OrchestrationAuthorityDecisionRecord[]; isSuccess: true; isError: false }>((resolve) => { resolveRefetch = resolve })
    const currentAuthority = query([q4, q3, q2, q1])
    currentAuthority.refetch.mockReturnValue(refetchResult)
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Wait for current authority'))
      draft.focus()
      api.dashboard = dashboardData({
        decisions: currentAuthority,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, q1.title)

      await act(async () => { formSubmit(retained).click(); await Promise.resolve() })
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(retained, 'Answer', 'textarea').value).toBe('Wait for current authority')
      expect(view.document.activeElement).toBe(draft)

      await act(async () => { resolveRefetch({ data: [q4, q3, q2, q1], isSuccess: true, isError: false }); await refetchResult })
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate.mock.calls[0][0]).toEqual(expect.objectContaining({
        decisionId: q1.id,
        contractVersion: q1.contract_version,
        runtimeIdentity: q1.runtime_identity,
      }))
    } finally { view.cleanup() }
  })

  it('SPR212 pending preflight cannot dispatch an old-goal mutation after the goal generation changes', async () => {
    const q1 = task4RuntimeQuestion(1)
    let generation = 1
    let resolveRefetch!: (value: { data: OrchestrationAuthorityDecisionRecord[]; isSuccess: true; isError: false }) => void
    const refetchResult = new Promise<{ data: OrchestrationAuthorityDecisionRecord[]; isSuccess: true; isError: false }>((resolve) => { resolveRefetch = resolve })
    const authority = query([q1])
    authority.refetch.mockReturnValue(refetchResult)
    api.dashboard = dashboardData({
      decisions: authority,
      checkpoint: query({ goal_id: goal().id, items: [q1], deferred_count: 0, max_questions: 3 }),
    })
    const view = await mountDashboard({ getMutationGeneration: () => generation })
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Must stay with the original goal'))
      draft.focus()

      await act(async () => { formSubmit(queueRow(view.container, q1.title)).click(); await Promise.resolve() })
      expect(api.answer.mutate).not.toHaveBeenCalled()
      generation += 1
      await act(async () => { resolveRefetch({ data: [q1], isSuccess: true, isError: false }); await refetchResult })

      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea').value).toBe('Must stay with the original goal')
      expect(view.document.activeElement).toBe(draft)
    } finally { view.cleanup() }
  })

  it('SPR212 non-retained adaptive group validates Q2 directly when no retained active-member callback exists', async () => {
    const goalProcess = process({ id: 'adaptive-callback-process', process_type: 'goal_definition', status: 'waiting_decision' })
    const adaptive = (number: number) => decision({
      id: `callback-q${number}`,
      source_process_run_id: goalProcess.id,
      decision_key: `goal_definition:adaptive:8:${number - 1}:question-${number}`,
      title: `Callback question ${number}`,
      question: `Callback question ${number}?`,
      options: [],
      recommendation: null,
      runtime_identity: null,
      contract_version: `callback:q${number}:v1`,
    })
    const [first, second] = [adaptive(1), adaptive(2)]
    const authority = query([first, second])
    const firstAnswered = { ...first, status: 'answered' as const, selected_option: 'First answer', reason: '' }
    authority.refetch
      .mockResolvedValueOnce({ data: [first, second], isSuccess: true, isError: false })
      .mockResolvedValueOnce({ data: [firstAnswered, second], isSuccess: true, isError: false })
    api.dashboard = dashboardData({
      processes: query([goalProcess]),
      decisions: authority,
      checkpoint: query({ goal_id: goal().id, items: [first, second], deferred_count: 0, max_questions: 3 }),
    })
    const callbacks: Array<{ onSuccess?: (result: unknown) => void; onSettled?: () => void }> = []
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks.push(nextCallbacks) })
    const view = await mountDashboard()
    try {
      const firstDraft = getByLabel(view.container, 'Answer', 'textarea')
      await act(async () => changeControl(firstDraft, 'First answer'))
      await act(async () => { getButton(view.container, 'Next').click(); await Promise.resolve() })
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      await act(async () => {
        callbacks[0]?.onSuccess?.({ decision: firstAnswered, process: null, continuation_applied: false })
        callbacks[0]?.onSettled?.()
      })

      expect(textOf(view.container)).toContain('Question 2 of 2')
      const secondDraft = getByLabel(view.container, 'Answer', 'textarea')
      await act(async () => changeControl(secondDraft, 'Second answer'))
      await act(async () => { getButton(view.container, 'Submit').click(); await Promise.resolve() })

      expect(authority.refetch).toHaveBeenCalledTimes(2)
      expect(api.answer.mutate).toHaveBeenCalledTimes(2)
      expect(api.answer.mutate.mock.calls[1][0]).toEqual(expect.objectContaining({
        decisionId: second.id,
        contractVersion: second.contract_version,
        selectedOption: 'Second answer',
      }))
    } finally { view.cleanup() }
  })

  it('SPR212 retained three-question round advances after its own success but stops on external terminalization', async () => {
    const goalProcess = process({
      id: 'goal-definition-round-process', process_type: 'goal_definition', status: 'waiting_decision',
    })
    const adaptive = (number: number, field: string) => decision({
      id: `adaptive-round-q${number}`,
      source_process_run_id: goalProcess.id,
      decision_key: `goal_definition:adaptive:7:${number - 1}:${field}`,
      title: `Clarification ${number}`,
      question: `Adaptive question ${number}?`,
      options: [],
      recommendation: null,
      runtime_identity: `adaptive-runtime-${number}`,
      contract_version: `adaptive:7:q${number}:v1`,
    })
    const [first, second, third] = [adaptive(1, 'objective'), adaptive(2, 'constraints'), adaptive(3, 'success')]
    const arrivals = [task4RuntimeQuestion(6), task4RuntimeQuestion(5), task4RuntimeQuestion(4)]
    api.dashboard = dashboardData({
      processes: query([goalProcess]),
      decisions: query([first, second, third]),
      checkpoint: query({ goal_id: goal().id, items: [first, second, third], deferred_count: 0, max_questions: 3 }),
    })
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const view = await mountDashboard()
    try {
      const groupRow = queueRow(view.container, 'Adaptive question 1?')
      const answer = getByLabel(groupRow, 'Answer', 'textarea')
      await act(async () => changeControl(answer, 'Confirmed first answer'))
      await act(async () => focusControl(answer))

      const fullAuthority = query([...arrivals, first, second, third])
      fullAuthority.refetch.mockResolvedValue({ data: [...arrivals, first, second, third], isSuccess: true, isError: false })
      api.dashboard = dashboardData({
        processes: query([goalProcess]),
        decisions: fullAuthority,
        checkpoint: query({ goal_id: goal().id, items: arrivals, deferred_count: 3, max_questions: 3 }),
      })
      await view.rerender()
      const retainedGroup = queueRow(view.container, 'Adaptive question 1?')
      expect(getByLabel(retainedGroup, 'Answer', 'textarea')).toBe(answer)
      expect(textOf(retainedGroup)).toContain('Question 1 of 3')

      await act(async () => { getButton(retainedGroup, 'Next').click(); await Promise.resolve() })
      expect(fullAuthority.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledWith(expect.objectContaining({
        decisionId: first.id,
        contractVersion: first.contract_version,
        runtimeIdentity: first.runtime_identity,
        selectedOption: 'Confirmed first answer',
      }), expect.any(Object))
      const firstAnswered = {
        ...first,
        status: 'answered' as const,
        selected_option: 'Confirmed first answer',
        reason: null,
        decided_at: '2026-08-05T08:10:00Z',
      }
      await act(async () => {
        callbacks?.onSuccess?.({ decision: firstAnswered, process: null, continuation_applied: false })
        callbacks?.onSettled?.()
      })
      api.dashboard = dashboardData({
        processes: query([goalProcess]),
        decisions: query([...arrivals, firstAnswered, second, third]),
        checkpoint: query({ goal_id: goal().id, items: arrivals, deferred_count: 2, max_questions: 3 }),
      })
      await view.rerender()

      const secondStep = queueRow(view.container, 'Adaptive question 2?')
      const secondDraft = getByLabel(secondStep, 'Answer', 'textarea')
      expect(secondStep).toBe(retainedGroup)
      expect(secondDraft).toBe(answer)
      expect(secondDraft.disabled).toBe(false)
      expect(textOf(secondStep)).toContain('Question 2 of 3')
      expect(textOf(secondStep)).toContain('Adaptive question 2?')
      expect(textOf(secondStep)).not.toContain('Answered elsewhere')
      // The custom TestElement keeps a stale raw textarea value across a
      // controlled same-node update. The disabled action reflects Q2's empty
      // controlled draft and catches accidental Q1-to-Q2 state reuse.
      expect(getButton(secondStep, 'Next').disabled).toBe(true)
      await act(async () => changeControl(secondDraft, 'Q2 stale-contract draft'))
      const refreshedSecond = {
        ...second,
        contract_version: 'adaptive:7:q2:v2',
        runtime_identity: 'adaptive-runtime-2-v2',
        question: 'Adaptive question 2 refreshed?',
      }
      api.dashboard = dashboardData({
        processes: query([goalProcess]),
        decisions: query([...arrivals, firstAnswered, refreshedSecond, third]),
        checkpoint: query({ goal_id: goal().id, items: arrivals, deferred_count: 2, max_questions: 3 }),
      })
      await view.rerender()
      const staleSecondStep = queueRow(view.container, 'Adaptive question 2?')
      expect(getByLabel(staleSecondStep, 'Answer', 'textarea')).toBe(secondDraft)
      await act(async () => getButton(staleSecondStep, 'Review refreshed decision').click())
      const reviewedSecondStep = queueRow(view.container, 'Adaptive question 2 refreshed?')
      const reviewedSecondDraft = getByLabel(reviewedSecondStep, 'Answer', 'textarea')
      expect(textOf(reviewedSecondStep)).toContain('Question 2 of 3')
      expect(visibleLeafText(reviewedSecondStep)).not.toContain('Adaptive question 1?')
      expect(reviewedSecondDraft).not.toBe(secondDraft)
      expect(reviewedSecondDraft.value).toBe('')
      expect(view.document.activeElement).toBe(reviewedSecondDraft)
      await act(async () => changeControl(reviewedSecondDraft, 'Q2 draft must not advance'))
      await act(async () => focusControl(reviewedSecondDraft))

      const externallyAnsweredSecond = {
        ...refreshedSecond,
        status: 'answered' as const,
        selected_option: 'External Q2 answer',
        reason: 'Recorded by another operator.',
        decided_at: '2026-08-05T08:11:00Z',
      }
      api.dashboard = dashboardData({
        processes: query([goalProcess]),
        decisions: query([...arrivals, firstAnswered, externallyAnsweredSecond, third]),
        checkpoint: query({ goal_id: goal().id, items: arrivals, deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()

      const terminalStep = queueRow(view.container, 'Adaptive question 2 refreshed?')
      expect(getByLabel(terminalStep, 'Answer', 'textarea')).toBe(reviewedSecondDraft)
      expect(getByLabel(terminalStep, 'Answer', 'textarea').value).toBe('Q2 draft must not advance')
      expect(getByLabel(terminalStep, 'Answer', 'textarea').disabled).toBe(false)
      expect(currentReactProp(getByLabel(terminalStep, 'Answer', 'textarea'), 'readOnly')).toBe(true)
      expect(getButton(terminalStep, 'Next').disabled).toBe(true)
      expect(textOf(terminalStep)).toContain('Answered elsewhere')
      expect(textOf(terminalStep)).toContain('External Q2 answer')
      expect(visibleLeafText(terminalStep)).not.toContain('Adaptive question 3?')
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('SPR212 actual mutation error preserves retained draft and source identity', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    const authoritative = query([q4, q3, q2, q1])
    authoritative.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    let callbacks: { onError?: (error: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Retry this after the transport recovers'))
      api.dashboard = dashboardData({
        decisions: authoritative,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, q1.title)
      await act(async () => { formSubmit(retained).click(); await Promise.resolve() })
      await act(async () => {
        callbacks?.onError?.(new Error('The answer request failed.'))
        callbacks?.onSettled?.()
      })

      const afterError = queueRow(view.container, q1.title)
      expect(getByLabel(afterError, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(afterError, 'Answer', 'textarea').value).toBe('Retry this after the transport recovers')
      expect(textOf(afterError)).toContain('The answer request failed.')
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('SPR212 missing runtime receipt stays locked on the reviewed draft until explicit review', async () => {
    const q1 = task4RuntimeQuestion(1)
    const authority = query([q1])
    authority.refetch.mockResolvedValue({ data: [q1], isSuccess: true, isError: false })
    api.dashboard = dashboardData({
      decisions: authority,
      checkpoint: query({ goal_id: goal().id, items: [q1], deferred_count: 0, max_questions: 3 }),
    })
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const onMessage = vi.fn()
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Keep after malformed HTTP 200'))
      await act(async () => { formSubmit(queueRow(view.container, q1.title)).click(); await Promise.resolve() })
      await act(async () => {
        callbacks?.onSuccess?.({ decision: null, process: null, continuation_applied: false })
        callbacks?.onSettled?.()
      })

      const locked = queueRow(view.container, q1.title)
      expect(getByLabel(locked, 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(locked, 'Answer', 'textarea').value).toBe('Keep after malformed HTTP 200')
      expect(formSubmit(locked).disabled).toBe(true)
      expect(onMessage).not.toHaveBeenCalledWith('Answer received.')
      expect(toast.success).not.toHaveBeenCalled()

      await view.rerender()
      const stillLocked = queueRow(view.container, q1.title)
      expect(getByLabel(stillLocked, 'Answer', 'textarea')).toBe(draft)
      expect(formSubmit(stillLocked).disabled).toBe(true)
      await act(async () => getButton(stillLocked, 'Review refreshed decision').click())
      const reviewedAgain = queueRow(view.container, q1.title)
      const freshDraft = getByLabel(reviewedAgain, 'Answer', 'textarea')
      expect(freshDraft).not.toBe(draft)
      expect(freshDraft.value).toBe('')
      expect(currentReactProp(freshDraft, 'readOnly')).toBe(false)
      expect(view.document.activeElement === freshDraft).toBe(true)
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it.each([
    ['answered', 'Answered elsewhere', 'acknowledge', 'Recorded by another operator.'],
    ['cancelled', 'Decision closed elsewhere', null, null],
  ] as const)(
    'SPR212 authoritative %s outcome overrides a sticky stale receipt and leaves one actionable item',
    async (status, heading, selectedOption, authoritativeReason) => {
      const q1 = task4RuntimeQuestion(1)
      const q2 = task4RuntimeQuestion(2)
      const authority = query([q2, q1])
      authority.refetch.mockResolvedValue({ data: [q2, q1], isSuccess: true, isError: false })
      api.dashboard = dashboardData({
        decisions: authority,
        checkpoint: query({ goal_id: goal().id, items: [q2, q1], deferred_count: 0, max_questions: 3 }),
      })
      let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
      api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
      const view = await mountDashboard()
      try {
        await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
        const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
        await act(async () => changeControl(draft, 'Keep this draft beside the authoritative outcome'))
        await act(async () => { formSubmit(queueRow(view.container, q1.title)).click(); await Promise.resolve() })
        await act(async () => {
          callbacks?.onSuccess?.({ decision: null, process: null, continuation_applied: false })
          callbacks?.onSettled?.()
        })
        expect(getButton(queueRow(view.container, q1.title), 'Review refreshed decision')).toBeDefined()

        const terminal = {
          ...q1,
          status,
          selected_option: selectedOption,
          reason: authoritativeReason,
          decided_at: '2026-08-05T08:30:00Z',
          updated_at: '2026-08-05T08:30:00Z',
        }
        api.dashboard = dashboardData({
          decisions: query([q2, terminal]),
          checkpoint: query({ goal_id: goal().id, items: [q2], deferred_count: 0, max_questions: 3 }),
        })
        await view.rerender()

        const retained = queueRow(view.container, q1.title)
        expect(getByLabel(retained, 'Answer', 'textarea')).toBe(draft)
        expect(currentReactProp(draft, 'readOnly')).toBe(true)
        expect(formSubmit(retained).disabled).toBe(true)
        expect(textOf(retained)).toContain(heading)
        expect(textOf(retained)).toContain(`Authoritative status: ${status}.`)
        if (selectedOption) expect(textOf(retained)).toContain(`Selected answer: ${selectedOption}`)
        if (authoritativeReason) expect(textOf(retained)).toContain(`Reason: ${authoritativeReason}`)
        expect(() => getButton(retained, 'Review refreshed decision')).toThrow()
        expect(getButton(retained, 'Dismiss')).toBeDefined()
        await act(async () => { await new Promise((resolve) => setTimeout(resolve, 550)) })
        expect(textOf(view.container)).toContain('1 item needs you.')
      } finally { view.cleanup() }
    },
  )

  it('SPR212 identical runtime replay with false continuation confirms persisted answer', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const onMessage = vi.fn()
    const view = await mountDashboard({ onMessage })
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, '  original attempt survived response loss\n'))

      const pendingPreflight = query([q4, q3, q2, q1])
      pendingPreflight.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
      const acceptedDashboard = dashboardData({
        decisions: pendingPreflight,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      api.dashboard = acceptedDashboard
      await view.rerender()
      const retained = queueRow(view.container, q1.title)
      await act(async () => { formSubmit(retained).click(); await Promise.resolve() })

      expect(pendingPreflight.refetch).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate.mock.calls[0][0]).toEqual(expect.objectContaining({
        decisionId: q1.id,
        contractVersion: q1.contract_version,
        runtimeIdentity: q1.runtime_identity,
        selectedOption: 'acknowledge',
        reason: 'original attempt survived response loss',
      }))
      const matchingAnswered = {
        ...q1,
        status: 'answered' as const,
        selected_option: 'acknowledge',
        reason: 'original attempt survived response loss',
        decided_at: '2026-08-05T08:05:00Z',
      }
      await act(async () => {
        callbacks?.onSuccess?.({
          decision: matchingAnswered,
          process: null,
          continuation_applied: false,
        })
        callbacks?.onSettled?.()
      })

      expect(() => queueRow(view.container, q1.title)).toThrow()
      expect(onMessage).toHaveBeenCalledTimes(1)
      expect(onMessage).toHaveBeenCalledWith('Answer received.')
      expect(toast.success).toHaveBeenCalledTimes(1)
      expect(toast.success).toHaveBeenCalledWith('Answer received.')
      expect(view.document.activeElement?.tagName).toBe('H2')
      expect(api.answer.acceptRuntimeResult).toHaveBeenCalledWith(goal().id, {
        decision: matchingAnswered,
        process: null,
        continuation_applied: false,
      })
      expect(pendingPreflight.refetch).toHaveBeenCalledTimes(2)
      expect(acceptedDashboard.checkpoint.refetch).toHaveBeenCalledTimes(1)
      expect(acceptedDashboard.processes.refetch).toHaveBeenCalledTimes(1)
      expect(acceptedDashboard.warnings.refetch).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('SPR212 cross-goal late answer receipt cannot clear or move the retained draft', async () => {
    const q1 = task4RuntimeQuestion(1)
    const q2 = task4RuntimeQuestion(2)
    const q3 = task4RuntimeQuestion(3)
    const q4 = task4RuntimeQuestion(4)
    let generation = 1
    const authoritative = query([q4, q3, q2, q1])
    authoritative.refetch.mockResolvedValue({ data: [q4, q3, q2, q1], isSuccess: true, isError: false })
    api.dashboard = dashboardData({
      decisions: query([q3, q2, q1]),
      checkpoint: query({ goal_id: goal().id, items: [q3, q2, q1], deferred_count: 0, max_questions: 3 }),
    })
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const onMessage = vi.fn()
    const view = await mountDashboard({ onMessage, getMutationGeneration: () => generation })
    try {
      await act(async () => getButton(queueRow(view.container, q1.title), 'Answer').click())
      const draft = getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')
      await act(async () => changeControl(draft, 'Belongs to the original goal generation'))
      draft.focus()
      api.dashboard = dashboardData({
        decisions: authoritative,
        checkpoint: query({ goal_id: goal().id, items: [q4, q3, q2], deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = queueRow(view.container, q1.title)
      await act(async () => { formSubmit(retained).click(); await Promise.resolve() })
      generation++
      await act(async () => {
        callbacks?.onSuccess?.({
          decision: {
            ...q1, status: 'answered', selected_option: 'acknowledge',
            reason: 'Belongs to the original goal generation',
          },
          process: null,
          continuation_applied: true,
        })
        callbacks?.onSettled?.()
      })

      expect(getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea')).toBe(draft)
      expect(getByLabel(queueRow(view.container, q1.title), 'Answer', 'textarea').value).toBe('Belongs to the original goal generation')
      expect(view.document.activeElement).toBe(draft)
      expect(onMessage).not.toHaveBeenCalledWith('Answer received.')
      expect(toast.success).not.toHaveBeenCalledWith('Answer received.')
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('submits normalized runtime text with sentinel and contract, retaining it after response loss', async () => {
    const runtime = decision({ id: 'runtime-question', recommendation: null, options: [{ key: 'acknowledge' }],
      runtime_identity: 'runtime-1', contract_version: 'plan:1' })
    api.dashboard = dashboardData({ decisions: query([runtime]), checkpoint: query({
      goal_id: goal().id, items: [runtime], deferred_count: 0, max_questions: 5,
    }) })
    api.answer.mutate.mockImplementation((_variables, callbacks) => {
      callbacks.onError?.(new Error('Response lost'))
      callbacks.onSettled?.()
    })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(view.container, 'Answer').click())
      const answerField = () => getByLabel(view.container, 'Answer', 'textarea')
      const submitButton = () => getButton(descendants(view.container).find((node) => node.tagName === 'FORM')!, 'Answer')
      expect(view.document.activeElement === answerField()).toBe(true)
      expect(submitButton().disabled).toBe(true)
      const entered = ' \u0085\uFEFFhello\n😀\uFEFF \n'
      await act(async () => changeControl(answerField(), entered))
      await act(async () => submitButton().click())
      expect(api.answer.mutate).toHaveBeenCalledWith({ goalId: goal().id, decisionId: runtime.id,
        selectedOption: 'acknowledge', reason: '\uFEFFhello\n😀\uFEFF', contractVersion: 'plan:1',
        runtimeIdentity: 'runtime-1' }, expect.any(Object))
      expect(answerField().value).toBe(entered)
      await act(async () => submitButton().click())
      expect(api.answer.mutate).toHaveBeenCalledTimes(2)
      expect(api.answer.mutate.mock.calls[1][0]).toEqual(api.answer.mutate.mock.calls[0][0])
    } finally { view.cleanup() }
  })

  function form(
    options: OrchestrationDecisionOption[],
    selectedOption: string | null,
    recommendation: string | null,
    overrides: Partial<OrchestrationAuthorityDecisionRecord> = {},
    reason = '',
  ) {
    return renderToStaticMarkup(
      <BaselineActionForm
        target={{ kind: 'answer', action: { kind: 'answer', id: 'decision-1' }, decision: decision({ options, recommendation, ...overrides }) }}
        mutationBusy={false} submitting={false} selectedOption={selectedOption} reason={reason} error={null}
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

  it.each([
    ['blank', ' \n\u0085', true],
    ['4000 multiline astral', 'a\n😀'.repeat(1333) + 'z', false],
    ['4001 multiline astral', 'a\n😀'.repeat(1333) + 'zz', true],
    ['0085 strips to4000', '\u0085' + 'x'.repeat(4000), false],
    ['FEFF remains4001', '\uFEFF' + 'x'.repeat(4000), true],
    ['accepted FEFF ends', '\uFEFFhello\uFEFF', false],
  ])('runtime sentinel renders one Answer control and enforces %s', (_name, reason, disabled) => {
    const markup = form([{ key: 'acknowledge' }], null, null,
      { runtime_identity: 'runtime-1', contract_version: 'plan:1' }, reason)
    expect(markup.match(/<textarea\b/g)).toHaveLength(1)
    expect(markup).toContain('>Answer</label>')
    expect(markup).not.toContain('type="radio"')
    expect(markup.match(/>Answer<\/button>/g)).toHaveLength(1)
    expect(/<button[^>]*disabled=""[^>]*>Answer<\/button>/.test(markup)).toBe(disabled)
    if (reason.includes('zz') || reason.startsWith('\uFEFFx')) expect(markup).toContain('4,000')
  })

  it.each([
    ['ordinary acknowledge', [{ key: 'acknowledge' }], {}],
    ['meaningful singleton', [{ key: 'approve' }], { runtime_identity: 'runtime-1' }],
    ['multi choice', [{ key: 'acknowledge' }, { key: 'reject' }], { runtime_identity: 'runtime-1' }],
    ['invalid sibling', [{ key: 'acknowledge' }, { label: 'invalid' }], { runtime_identity: 'runtime-1' }],
  ])('keeps %s as choices', (_name, options, overrides) => {
    const markup = form(options as OrchestrationDecisionOption[], null, null, overrides)
    expect(markup).toContain('type="radio"')
    expect(markup).toContain('Reason (optional)')
    expect(markup).not.toContain('>Answer</label>')
  })

  it('caps optional reasons on genuine runtime choices without requiring them', () => {
    const runtime = { runtime_identity: 'runtime-1' }
    const empty = form(['approve'], 'approve', null, runtime)
    expect(empty).not.toMatch(/<button[^>]*disabled=""[^>]*>Submit answer<\/button>/)
    const boundary = form(['approve'], 'approve', null, runtime, '😀'.repeat(4000))
    expect(boundary).not.toMatch(/<button[^>]*disabled=""[^>]*>Submit answer<\/button>/)
    const over = form(['approve'], 'approve', null, runtime, '😀'.repeat(4001))
    expect(over).toMatch(/<button[^>]*disabled=""[^>]*>Submit answer<\/button>/)
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

 describe('baseline operation generations', () => {
  it('announces progress before the provider responds and ignores callbacks after reset', async () => {
    api.dashboard = dashboardData({ processes: query([]) })
    let generation = 1
    const onMessage = vi.fn()
    const releaseMutation = vi.fn()
    const view = await mountDashboard({ onMessage, releaseMutation, getMutationGeneration: () => generation })
    try {
      await act(async () => controlElement(view.container, 'Run Goal definition').click())
      expect(onMessage).toHaveBeenCalledWith('Running Goal definition.')
      const callbacks = api.run.mutate.mock.calls[0][1]
      onMessage.mockClear()
      generation++
      await act(async () => {
        callbacks.onSuccess({ process: process({ status: 'complete' }) })
        callbacks.onError(new Error('late failure'))
        callbacks.onSettled()
      })
      expect(onMessage).not.toHaveBeenCalled()
      expect(releaseMutation).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })
})

describe('same baseline owner replacement', () => {
  it('old settled callback cannot release a newer run with the same owner', async () => {
    api.dashboard = dashboardData({ processes: query([]) })
    const onMessage = vi.fn()
    const releaseMutation = vi.fn()
    const view = await mountDashboard({ onMessage, releaseMutation })
    try {
      await act(async () => controlElement(view.container, 'Run Goal definition').click())
      const oldCallbacks = api.run.mutate.mock.calls[0][1]
      await act(async () => controlElement(view.container, 'Run Goal definition').click())
      const newCallbacks = api.run.mutate.mock.calls[1][1]
      onMessage.mockClear()
      await act(async () => { oldCallbacks.onError(new Error('old failure')); oldCallbacks.onSettled() })
      expect(onMessage).not.toHaveBeenCalled()
      expect(releaseMutation).not.toHaveBeenCalled()
      await act(async () => newCallbacks.onSettled())
      expect(releaseMutation).toHaveBeenCalledOnce()
    } finally { view.cleanup() }
  })
})

// recovery FIX PROPOSAL wiring. Answered/non-pending recovery decisions come from the
// existing `decisions` query (GET /decisions); pending ones also arrive in `checkpoint.items`.
describe('Baseline dashboard recovery FIX PROPOSAL card', () => {
  const GOAL_ID = 'goal-00000000-0000-4000-8000-000000000001'
  function recoveryDecision(id: string, status: string, extra: Record<string, unknown> = {}, overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}) {
    return decision({
      id, decision_key: `blocker_recovery:${id}`, title: 'Fix proposal', context: null, recommendation: null,
      options: ['approve_fix', 'reject'], status: status === 'pending_approval' ? 'pending' : 'answered',
      ...overrides,
      recovery: {
        kind: 'recovery_proposal_v1', proposal_id: `proposal-${id}`, version: 1, status,
        problem: 'No agent can review the finished work.', general_updates: ['Add a code reviewer'],
        after_approval: 'The blocked review step restarts automatically.', global_effect: true,
        exact_details: null, evidence: null, error: null,
        source_blocker_id: 'blocker-1', source_action_id: null, source_decision_id: id, ...extra,
      },
    } as unknown as Partial<OrchestrationAuthorityDecisionRecord>)
  }
  const withDecisions = (items: OrchestrationAuthorityDecisionRecord[]) => dashboardData({
    decisions: query(items),
    checkpoint: query({ goal_id: GOAL_ID, items: items.filter((item) => item.status === 'pending'), deferred_count: 0, max_questions: 5 }),
  })
  const cardOf = (root: Parameters<typeof descendants>[0]) => {
    const heading = descendants(root).find((node) => node.tagName === 'H3' && ['FIX PROPOSAL', 'CAPABILITY NEEDED'].includes(textOf(node)))
    if (!heading) throw new Error('FIX PROPOSAL card not found')
    let node: TestElement | null = heading
    while (node && node.tagName !== 'LI') node = node.parentNode as TestElement | null
    return node!
  }

  it('renders_a_simple_recovery_proposal_and_submits_approval_once', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval')])
    const view = await mountDashboard()
    try {
      expect(descendants(view.container).some((node) => node.tagName === 'H3' && textOf(node) === 'FIX PROPOSAL')).toBe(true)
      const li = cardOf(view.container)
      await act(async () => getButton(li, 'Approve fix').click())
      await act(async () => getButton(li, 'Approve fix').click())
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledWith(
        expect.objectContaining({ goalId: GOAL_ID, decisionId: 'rec-1', selectedOption: 'approve_fix' }),
        expect.any(Object),
      )
      expect(api.answer.mutate.mock.calls[0][0].reason ?? '').toBe('')
      expect(getButton(li, 'Approve fix').disabled).toBe(true)
      expect(getButton(li, 'Reject').disabled).toBe(true)
    } finally { view.cleanup() }
  })

  it('SPR212 refreshed retained recovery approval keeps focus on the same applying card heading', async () => {
    const original = recoveryDecision('rec-refreshed-focus', 'pending_approval', {}, {
      runtime_identity: 'recovery-runtime-v1',
      contract_version: 'recovery-contract-v1',
    })
    const refreshed = recoveryDecision('rec-refreshed-focus', 'pending_approval', {}, {
      runtime_identity: 'recovery-runtime-v2',
      contract_version: 'recovery-contract-v2',
      updated_at: '2026-08-05T08:15:00Z',
    })
    const arrivals = [task4RuntimeQuestion(4), task4RuntimeQuestion(3), task4RuntimeQuestion(2)]
    api.dashboard = withDecisions([original])
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const view = await mountDashboard()
    try {
      await act(async () => getButton(cardOf(view.container), 'Reject').click())
      api.dashboard = dashboardData({
        decisions: query([...arrivals, refreshed]),
        checkpoint: query({ goal_id: GOAL_ID, items: arrivals, deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = cardOf(view.container)
      expect(getByLabel(retained, 'Reason', 'textarea')).toBeDefined()
      await act(async () => getButton(retained, 'Review refreshed decision').click())

      const reviewed = cardOf(view.container)
      expect(() => getByLabel(reviewed, 'Reason', 'textarea')).toThrow()
      await act(async () => { getButton(reviewed, 'Approve fix').click(); await Promise.resolve() })
      expect(api.answer.mutate).toHaveBeenCalledWith(expect.objectContaining({
        decisionId: refreshed.id,
        contractVersion: refreshed.contract_version,
        runtimeIdentity: refreshed.runtime_identity,
        selectedOption: 'approve_fix',
      }), expect.any(Object))

      const applying = recoveryDecision('rec-refreshed-focus', 'applying', {}, {
        runtime_identity: refreshed.runtime_identity,
        contract_version: refreshed.contract_version,
        selected_option: 'approve_fix',
        reason: null,
        decided_at: '2026-08-05T08:16:00Z',
        updated_at: '2026-08-05T08:16:00Z',
      })
      await act(async () => {
        callbacks?.onSuccess?.({ decision: applying, process: null, continuation_applied: false })
        callbacks?.onSettled?.()
      })
      api.dashboard = dashboardData({
        decisions: query([...arrivals, applying]),
        checkpoint: query({ goal_id: GOAL_ID, items: arrivals, deferred_count: 0, max_questions: 3 }),
      })
      await view.rerender()

      const applyingCard = cardOf(view.container)
      expect(textOf(applyingCard)).toContain('Applying approved fix…')
      const heading = descendants(applyingCard).find((node) => node.tagName === 'H3' && textOf(node) === 'FIX PROPOSAL')!
      expect(view.document.activeElement).toBe(heading)
    } finally { view.cleanup() }
  })

  it('SPR212 reviewed capability-gap acceptance stays visible while re-diagnosis is pending', async () => {
    const original = recoveryDecision('gap-reviewed-acceptance', 'unsupported', {}, {
      status: 'pending',
      options: ['capability_provided'],
      runtime_identity: 'gap-reviewed-runtime-v1',
      contract_version: 'gap-reviewed-contract-v1',
    })
    const refreshed = recoveryDecision('gap-reviewed-acceptance', 'unsupported', {}, {
      status: 'pending',
      options: ['capability_provided'],
      runtime_identity: 'gap-reviewed-runtime-v2',
      contract_version: 'gap-reviewed-contract-v2',
      updated_at: '2026-08-05T08:15:00Z',
    })
    api.dashboard = withDecisions([original])
    let callbacks: { onSuccess?: (result: unknown) => void; onSettled?: () => void } | undefined
    api.answer.mutate.mockImplementation((_variables, nextCallbacks) => { callbacks = nextCallbacks })
    const view = await mountDashboard()
    try {
      await act(async () => cardOf(view.container).click())
      api.dashboard = withDecisions([refreshed])
      await view.rerender()
      await act(async () => getButton(cardOf(view.container), 'Review refreshed decision').click())
      await act(async () => { getButton(cardOf(view.container), "I've provided it").click(); await Promise.resolve() })
      expect(api.answer.mutate).toHaveBeenCalledWith(expect.objectContaining({
        decisionId: refreshed.id,
        contractVersion: refreshed.contract_version,
        runtimeIdentity: refreshed.runtime_identity,
        selectedOption: 'capability_provided',
      }), expect.any(Object))

      const answered = {
        ...refreshed,
        status: 'answered' as const,
        selected_option: 'capability_provided',
        reason: null,
        decided_at: '2026-08-05T08:16:00Z',
        updated_at: '2026-08-05T08:16:00Z',
      }
      await act(async () => {
        callbacks?.onSuccess?.({ decision: answered, process: null, continuation_applied: false })
        callbacks?.onSettled?.()
      })

      const pendingRediagnosis = cardOf(view.container)
      expect(textOf(pendingRediagnosis)).toContain('Marked as provided. Re-diagnosing the blocker…')
      expect(getButton(pendingRediagnosis, "I've provided it").disabled).toBe(true)
      expect(toast.success).not.toHaveBeenCalledWith('Blocked work is moving again.')
      const details = descendants(pendingRediagnosis).find((node) => node.tagName === 'SUMMARY' && textOf(node) === 'Details')!
      await act(async () => details.click())
      expect(textOf(cardOf(view.container))).toContain('Marked as provided. Re-diagnosing the blocker…')
      expect(getButton(cardOf(view.container), "I've provided it").disabled).toBe(true)
    } finally { view.cleanup() }
  })

  it('marks a pending capability gap done through the answer mutation (capability_provided)', async () => {
    api.dashboard = withDecisions([recoveryDecision('gap-1', 'unsupported', {}, { status: 'pending', options: ['capability_provided'] } as Partial<OrchestrationAuthorityDecisionRecord>)])
    const view = await mountDashboard()
    try {
      const li = cardOf(view.container)
      expect(textOf(li)).not.toContain('Approve fix')
      await act(async () => getButton(li, "I've provided it").click())
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledWith(
        expect.objectContaining({ goalId: GOAL_ID, decisionId: 'gap-1', selectedOption: 'capability_provided' }),
        expect.any(Object),
      )
    } finally { view.cleanup() }
  })

  it('answers with the decision record id, not recovery.source_decision_id', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval', { source_decision_id: 'other-id' })])
    const view = await mountDashboard()
    try {
      await act(async () => getButton(cardOf(view.container), 'Approve fix').click())
      expect(api.answer.mutate.mock.calls[0][0].decisionId).toBe('rec-1')
    } finally { view.cleanup() }
  })

  it('rejects through the answer mutation with the required, trimmed reason', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval')])
    const view = await mountDashboard()
    try {
      const li = cardOf(view.container)
      await act(async () => getButton(li, 'Reject').click())
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(getButton(li, 'Confirm reject').disabled).toBe(true)
      await act(async () => changeControl(getByLabel(li, 'Reason', 'textarea'), '  Too broad.  '))
      await act(async () => getButton(li, 'Confirm reject').click())
      expect(api.answer.mutate).toHaveBeenCalledTimes(1)
      expect(api.answer.mutate).toHaveBeenCalledWith(
        expect.objectContaining({ decisionId: 'rec-1', selectedOption: 'reject', reason: 'Too broad.' }),
        expect.any(Object),
      )
    } finally { view.cleanup() }
  })

  it('SPR212 cancelling a displaced recovery rejection releases retention and focuses the surviving card', async () => {
    const recovery = recoveryDecision('rec-cancel-displaced', 'pending_approval')
    const arrivals = [task4RuntimeQuestion(4), task4RuntimeQuestion(3), task4RuntimeQuestion(2)]
    api.dashboard = withDecisions([recovery])
    const view = await mountDashboard()
    try {
      await act(async () => getButton(cardOf(view.container), 'Reject').click())
      const reason = getByLabel(cardOf(view.container), 'Reason', 'textarea')
      await act(async () => changeControl(reason, 'Do not keep this abandoned rejection draft'))

      api.dashboard = dashboardData({
        decisions: query([...arrivals, recovery]),
        checkpoint: query({ goal_id: GOAL_ID, items: arrivals, deferred_count: 1, max_questions: 3 }),
      })
      await view.rerender()
      const retained = cardOf(view.container)
      expect(getByLabel(retained, 'Reason', 'textarea')).toBe(reason)
      await act(async () => { getButton(retained, 'Cancel').click(); await Promise.resolve() })

      expect(descendants(cardOf(view.container)).some((node) => node.tagName === 'TEXTAREA')).toBe(false)
      expect(view.document.activeElement === getButton(cardOf(view.container), 'Reject')).toBe(true)
      await act(async () => getButton(cardOf(view.container), 'Reject').click())
      expect(getByLabel(cardOf(view.container), 'Reason', 'textarea').value).toBe('')
      await act(async () => getButton(cardOf(view.container), 'Cancel').click())
      const refreshed = recoveryDecision('rec-cancel-displaced', 'pending_approval', {
        problem: 'Use the authoritative proposal after the abandoned rejection.',
      }, { contract_version: 'recovery-cancel:v2' })
      api.dashboard = dashboardData({
        decisions: query([...arrivals, refreshed]),
        checkpoint: query({ goal_id: GOAL_ID, items: arrivals, deferred_count: 0, max_questions: 3 }),
      })
      await view.rerender()
      expect(textOf(cardOf(view.container))).toContain('Use the authoritative proposal after the abandoned rejection.')
      expect(textOf(cardOf(view.container))).not.toContain('Do not keep this abandoned rejection draft')
      await act(async () => getButton(cardOf(view.container), 'Reject').click())
      expect(getByLabel(cardOf(view.container), 'Reason', 'textarea').value).toBe('')
      expect(api.answer.mutate).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('does not show the generic radio form or agent-definition batch for a recovery decision', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval')])
    const view = await mountDashboard()
    try {
      expect(descendants(view.container).some((node) => node.tagName === 'INPUT' && node.type === 'radio')).toBe(false)
      expect(textOf(view.container)).not.toContain('Submit all decisions')
    } finally { view.cleanup() }
  })

  it('shows the answer error and re-enables the card when the request fails', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval')])
    api.answer.mutate.mockImplementation((_v, options) => { options.onError?.(new Error('Backend said no.')); options.onSettled?.() })
    const view = await mountDashboard()
    try {
      const li = cardOf(view.container)
      await act(async () => getButton(li, 'Approve fix').click())
      expect(textOf(li)).toContain('Backend said no.')
      expect(getButton(li, 'Approve fix').disabled).toBe(false)
    } finally { view.cleanup() }
  })

  it('shows only the newest recovery decision per blocker', async () => {
    const older = recoveryDecision('older', 'failed', {}, { created_at: ISO.asked, asked_at: ISO.asked })
    const newer = recoveryDecision('newer', 'pending_approval', {}, { created_at: ISO.answered, asked_at: ISO.answered })
    api.dashboard = withDecisions([older, newer])
    const view = await mountDashboard()
    try {
      expect(descendants(view.container).filter((node) => node.tagName === 'H3' && textOf(node) === 'FIX PROPOSAL')).toHaveLength(1)
      expect(textOf(view.container)).not.toContain('The fix did not complete.')
      expect(textOf(view.container)).toContain('Approve fix')
    } finally { view.cleanup() }
  })

  it('hides a resolved recovery proposal', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'resolved')])
    const view = await mountDashboard()
    try { expect(textOf(view.container)).not.toContain('FIX PROPOSAL') } finally { view.cleanup() }
  })

  it.each([
    ['rejected', 'Fix rejected. Nothing was changed.'],
    ['failed', 'The fix did not complete.'],
    ['unsupported', 'No executable fix is available with the current tools.'],
    ['applying', 'Applying approved fix…'],
    ['verifying', 'Verifying update…'],
    ['retrying', 'Update verified. Retrying blocked work…'],
  ])('failed_recovery_prevents_on_track_empty_state (%s)', async (status, copy) => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', status)])
    const view = await mountDashboard()
    try {
      expect(textOf(view.container)).not.toMatch(/goal is on track/i)
      expect(textOf(view.container)).toContain(copy)
      const labels = descendants(cardOf(view.container)).filter((node) => node.tagName === 'BUTTON').map(textOf)
      expect(labels).not.toContain('Approve fix')
    } finally { view.cleanup() }
  })

  it('an unsupported (capability-gap) proposal never offers Approve fix even if the decision is still pending', async () => {
    api.dashboard = withDecisions([recoveryDecision('gap', 'unsupported', {}, { status: 'pending' })])
    const view = await mountDashboard()
    try { expect(textOf(cardOf(view.container))).not.toContain('Approve fix') } finally { view.cleanup() }
  })

  it('re-enables the card when the mutation owner cannot be acquired', async () => {
    api.dashboard = withDecisions([recoveryDecision('rec-1', 'pending_approval')])
    const view = await mountDashboard({ acquireMutation: () => false })
    try {
      const li = cardOf(view.container)
      await act(async () => getButton(li, 'Approve fix').click())
      expect(api.answer.mutate).not.toHaveBeenCalled()
      expect(getButton(li, 'Approve fix').getAttribute('disabled')).toBeNull()
    } finally { view.cleanup() }
  })
})


it('threads current live meeting commitments into Needs You', async () => {
  const view = await mountDashboard({ meetingCommitments: [{
    id: 'item-1', meeting_id: 'meeting-1', meeting_title: 'Release review', description: 'Live meeting obligation',
    state: 'task_in_flight', task_id: 'task-1', graph_run_id: null, reason: null,
  }] })
  try {
    expect(textOf(view.container)).toContain('Live meeting obligation')
    expect(() => getButton(view.container, 'Waive')).not.toThrow()
  } finally { view.cleanup() }
})
