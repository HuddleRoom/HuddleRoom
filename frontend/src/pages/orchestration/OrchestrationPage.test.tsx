import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type { OrchestrationGoalDetail, Task } from '@/lib/types'

const state = vi.hoisted(() => ({
  activeProjectId: null as string | null,
  goalId: undefined as string | undefined,
  goals: [] as Array<Record<string, unknown>>,
  detail: undefined as OrchestrationGoalDetail | undefined,
  detailError: false,
  detailErrorObj: null as unknown,
  dataUpdatedAt: 0 as number,
  tasks: [] as Task[],
  taskLoading: false,
  taskError: false,
  taskHasNextPage: false,
  taskFetchingNextPage: false,
  taskRefetch: vi.fn(),
  startMutation: { isPending: false, mutate: vi.fn() },
  commandMutation: { isPending: false, mutate: vi.fn() },
  healthDebugEnabled: false,
  agents: [] as Array<Record<string, unknown>>,
  // Keyed by meeting id (matches meetingKeys.detail's third element).
  meetingsById: new Map<string, Record<string, unknown>>(),
  meetingDecisions: undefined as Array<Record<string, unknown>> | undefined,
  baselineLoading: false,
  location: { hash: '', pathname: '', search: '', state: null, key: 'default' },
}))

vi.mock('react-router-dom', () => ({
  Link: ({ to, children, ...props }: { to: string; children: React.ReactNode }) => (
    <a href={to} {...props}>{children}</a>
  ),
  useNavigate: () => vi.fn(),
  useParams: () => ({ goalId: state.goalId }),
  useLocation: () => state.location,
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (value: { activeProjectId: string | null }) => unknown) =>
    selector({ activeProjectId: state.activeProjectId }),
}))

vi.mock('@/api/orchestration', () => ({
  useCreateOrchestrationGoal: () => ({ isPending: false, mutate: vi.fn() }),
  useOrchestrationGoals: () => ({
    data: { pages: [{ items: state.goals, next_cursor: null }] },
    isLoading: false,
    isError: false,
    isFetchingNextPage: false,
    hasNextPage: false,
    fetchNextPage: vi.fn(),
    refetch: vi.fn(),
  }),
  useOrchestrationGoal: () => ({
    data: state.detail,
    isLoading: false,
    isError: state.detailError,
    error: state.detailErrorObj,
    refetch: vi.fn(),
    dataUpdatedAt: state.dataUpdatedAt,
    isFetching: false,
  }),
  useOrchestrationGoalCommand: () => state.commandMutation,
  useOverrideOrchestrationGate: () => ({ isPending: false, mutate: vi.fn() }),
  useResetOrchestrationGoal: () => ({ isPending: false, mutate: vi.fn() }),
  useStartOrchestrationGoal: () => state.startMutation,
  useRecoverGoalDefinition: () => ({ isPending: false, mutate: vi.fn() }),
  useBaselineDashboard: () => ({
    memory: { data: undefined, isLoading: false, isError: false },
    processes: { data: undefined, isLoading: state.baselineLoading, isError: false },
    agentReviews: { data: undefined, isLoading: false, isError: false },
    warnings: { data: undefined, isLoading: state.baselineLoading, isError: false },
    decisions: { data: undefined, isLoading: state.baselineLoading, isError: false },
    checkpoint: { data: undefined, isLoading: false, isError: false },
  }),
  useOrchestrationHealth: () => ({
    data: { debug_enabled: state.healthDebugEnabled },
    isLoading: false,
    isError: false,
  }),
  // Real DetailsTabs (not mocked, see below) fetches a memory section on
  // toc-entry open; unused when nothing is opened in these static-render tests.
  useOrchestrationMemorySection: () => ({ data: undefined, isLoading: false, isError: false }),
  useOrchestrationConversation: () => ({
    data: { items: [], total: 0, omitted: 0, allowance: { enabled: true, limit: 100, used: 0, remaining: 100 }, steering: { enabled: false, eligibility: 'unstarted', eligibility_reason: null, inbox_version: 0, direction_version: 0, requests: [], proposals: [] } },
    isLoading: false, isError: false, refetch: vi.fn(),
  }),
  useSubmitOrchestrationConversation: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  // Only reached when the Debug tab mounts (BaselineDebugPanel, DetailsTabs.tsx).
  useStepOrchestrationBaseline: () => ({ isPending: false, mutate: vi.fn() }),
  useRerunLastOrchestrationBaseline: () => ({ isPending: false, mutate: vi.fn() }),
  useRecordOrchestrationConversationFeedback: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useSubmitOrchestrationSteering: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useWithdrawOrchestrationSteering: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useDismissOrchestrationSteeringProposal: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
}))

vi.mock('@/api/tasks', () => ({
  useAllTasks: () => ({
    items: state.tasks,
    isLoading: state.taskLoading,
    isError: state.taskError,
    hasNextPage: state.taskHasNextPage,
    isFetchingNextPage: state.taskFetchingNextPage,
    refetch: state.taskRefetch,
  }),
}))

vi.mock('@/api/agents', () => ({
  useAgents: () => ({ data: { items: state.agents }, isLoading: false, isError: false }),
}))

vi.mock('@/api/meetings', () => ({
  meetingKeys: {
    detail: (projectId: string | null, meetingId: string | undefined) => ['meeting', projectId, meetingId],
  },
  useMeetingDecisions: () => ({ data: state.meetingDecisions, isLoading: false, isError: false }),
}))

// RoomPanel's meeting lookup fans out via useQueries (real react-query
// requires a QueryClientProvider these static/mount tests don't set up) —
// resolved synchronously from state.meetingsById, keyed by the queryKey's
// meetingId (matches the mocked meetingKeys.detail shape above).
vi.mock('@tanstack/react-query', () => ({
  useQueries: ({ queries }: { queries: { queryKey: readonly unknown[] }[] }) =>
    queries.map((query) => {
      const meetingId = query.queryKey[2] as string
      return { data: state.meetingsById.get(meetingId), isLoading: false, isError: false }
    }),
}))

// BaselineDashboard (Z2-Z4: stepper/queue/step-focus) is mocked out — it has
// its own test file. DetailsTabs (Z5) is NOT mocked: its content (Goal & Plan,
// Gates, Delegations, Suggestions) is owned by GoalDetailView itself and is
// exercised directly by these tests, same as before this phase's restructure.
vi.mock('./BaselineDashboard', () => ({
  BaselineDashboard: ({ goalId, run }: { goalId: string; run: { id: string } | null }) => (
    <div data-testid="baseline-dashboard" data-goal-id={goalId} data-run-id={run?.id ?? 'none'} />
  ),
  currentProcesses: (processes: unknown[] | null | undefined) => (processes ?? []).filter((process) => (process as { superseded_by_id: string | null }).superseded_by_id === null),
}))

vi.mock('./StartWorkDialog', () => ({
  StartWorkDialog: ({ open, isPending, error, onConfirm }: {
    open: boolean
    isPending: boolean
    error: Error | null
    onConfirm: () => void
  }) => open ? (
    <div data-testid="start-work-dialog">
      {error && <p role="alert">{error.message}</p>}
      <button type="button" disabled={isPending} onClick={onConfirm}>Confirm start</button>
    </div>
  ) : null,
}))

const richDetail: OrchestrationGoalDetail = {
  goal: {
    id: 'goal-1',
    project_id: 'project-1',
    objective: 'Ship an evidence-backed release',
    success_criteria: [{ key: 'release-ready', description: 'Release is independently validated.' }],
    orchestrator_context: {},
    constraints: { owned_files: ['frontend/src'] },
    budget: { max_tokens: 20000 },
    goal_type: 'outcome',
    supersedes_goal_id: null,
    status: 'blocked',
    weight: 'standard',
    weight_overridden_by: null,
    manager_agent_id: null,
    manager_user_id: null,
    authority_model: null,
    created_by_user_id: null,
    created_at: '2026-07-15T09:00:00Z',
    updated_at: '2026-07-15T10:00:00Z',
    needs_you_count: 2,
  },
  run: {
    id: 'run-1',
    goal_id: 'goal-1',
    status: 'blocked',
    phase: 'ready',
    condition: 'baseline_complete',
    event_cursor: 42,
    plan_state: {
      status: 'accepted',
      planning_task_id: 'task-plan',
      accepted_artifact_id: 'artifact-plan',
      revision_requests: [{ revision_request: 'Split validation from implementation.' }],
      expanded_items: [{
        plan_item_id: 'validate-release',
        work_function: 'validation',
        task_id: 'task-1',
        gate_id: 'gate-1',
      }],
    },
    active_blockers: [{
      kind: 'task_blocked',
      reason: 'Missing reviewer requirements',
      task_id: 'task-1',
      gate_id: 'gate-1',
    }],
    budget_state: {},
    retry_state: {},
    started_at: '2026-07-15T09:00:00Z',
    completed_at: null,
    created_at: '2026-07-15T09:00:00Z',
    updated_at: '2026-07-15T10:00:00Z',
  },
  decisions_count: 1,
  decisions: [{
    id: 'decision-1',
    run_id: 'run-1',
    decision_type: 'request_verification',
    input_snapshot: {},
    llm_output: { action_type: 'request_verification' },
    parsed_decision: { action_type: 'request_verification', gate_id: 'gate-1' },
    validator_status: 'accepted',
    rejection_reason: null,
    reason: 'Request independent validation',
    created_at: '2026-07-15T09:30:00Z',
    updated_at: '2026-07-15T09:30:00Z',
  }],
  actions_count: 2,
  actions: [{
    id: 'action-1',
    run_id: 'run-1',
    decision_id: 'decision-1',
    idempotency_key: 'run:run-1:kind:request_verification',
    action_type: 'request_verification',
    request: { gate_id: 'gate-1', reason: 'Collect fresh proof.' },
    target_type: 'task',
    target_id: 'task-1',
    status: 'completed',
    error: null,
    created_at: '2026-07-15T09:31:00Z',
    updated_at: '2026-07-15T09:31:00Z',
  }, {
    id: 'action-2',
    run_id: 'run-1',
    decision_id: 'decision-1',
    idempotency_key: 'run:run-1:kind:schedule_meeting',
    action_type: 'schedule_meeting',
    request: { reason: 'Resolve contradictory validation outputs.' },
    target_type: 'meeting',
    target_id: 'meeting-1',
    status: 'completed',
    error: null,
    created_at: '2026-07-15T09:32:00Z',
    updated_at: '2026-07-15T09:32:00Z',
  }],
  gates_count: 1,
  gates: [{
    id: 'gate-1',
    run_id: 'run-1',
    success_criterion_key: 'release-ready',
    gate_type: 'validation_passed',
    required_evidence: { required_source_types: ['review'], min_count: 1 },
    status: 'open',
    failure_reason: null,
    created_at: '2026-07-15T09:10:00Z',
    updated_at: '2026-07-15T09:10:00Z',
    accepted_at: null,
    failed_at: null,
  }],
  evidence_count: 0,
  evidence: [],
  agent_suggestions_count: 1,
  agent_suggestions: [{
    id: 'suggestion-1',
    run_id: 'run-1',
    missing_work_function: 'validation',
    reason: 'No independent validator is active.',
    suggested_role: 'validator',
    suggested_capabilities: ['validation', 'testing'],
    suggested_adapter_type: 'api',
    suggested_model: 'gpt-4o-mini',
    suggested_system_prompt_outline: 'Validate work and report evidence.',
    status: 'open',
    created_at: '2026-07-15T09:40:00Z',
    updated_at: '2026-07-15T09:40:00Z',
  }],
  timeline: [],
}

beforeEach(() => {
  state.activeProjectId = null
  state.goalId = undefined
  state.goals = []
  state.detail = undefined
  state.detailError = false
  state.detailErrorObj = null
  state.dataUpdatedAt = 0
  state.tasks = []
  state.taskLoading = false
  state.taskError = false
  state.taskHasNextPage = false
  state.taskFetchingNextPage = false
  state.taskRefetch.mockReset()
  state.startMutation.isPending = false
  state.startMutation.mutate.mockReset()
  state.commandMutation.isPending = false
  state.commandMutation.mutate.mockReset()
  state.healthDebugEnabled = false
  state.agents = []
  state.meetingsById = new Map()
  state.meetingDecisions = undefined
  state.baselineLoading = false
  state.location = { hash: '', pathname: '', search: '', state: null, key: 'default' }
})

describe('OrchestrationPage', () => {
  it('shows project-selection guidance without an active project', async () => {
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Select a project to inspect orchestration')
    expect(markup).toContain('New project')
  })

  it('renders goal rows from the selected project', async () => {
    state.activeProjectId = 'project-1'
    state.goals = [richDetail.goal]
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Ship an evidence-backed release')
    expect(markup).toContain('/orchestration/goal-1')
    expect(markup).toContain('1 success criterion')
    expect(markup).toContain('<ul aria-label="Orchestration goals">')
    expect(markup).toContain('<li class="border-b border-huddleroom-depth last:border-b-0">')
    expect(markup).toContain('Objective:')
    expect(markup).toContain('Status:')
    expect(markup).toContain('Success criteria:')
    expect(markup).toContain('Updated:')
    expect(markup).toContain('Needs attention:')
    expect(markup).toContain('2 need you')
  })

  it('renders empty cell when needs_you_count is 0', async () => {
    state.activeProjectId = 'project-1'
    state.goals = [{ ...richDetail.goal, needs_you_count: 0 }]
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).not.toContain('needs you')
  })

  it('renders plan timeline gates evidence blockers task-delegations and suggestions', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.tasks = [{
      id: 'task-1',
      project_id: 'project-1',
      title: 'Validate the release',
      priority: 1,
      status: 'in_progress',
      metadata: { orchestration: { run_id: 'run-1', work_function: 'validation' } },
      created_at: '2026-07-15T09:00:00Z',
      updated_at: '2026-07-15T09:20:00Z',
    }]

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Accepted plan')
    expect(markup).toContain('Request independent validation')
    expect(markup).toContain('release-ready')
    expect(markup).toContain('No evidence recorded')
    expect(markup).toContain('Validate the release')
    expect(markup).toContain('meeting:meeting-1')
    expect(markup).toContain('validator')
    expect(markup).toContain('in_progress')
    expect(markup).toContain('data-testid="orchestration-goal-header"')
    expect(markup).toContain('data-testid="baseline-dashboard"')
    expect(markup).toContain('data-goal-id="goal-1"')
    expect(markup).toContain('data-run-id="run-1"')
    expect(markup).toContain('tabindex="-1"')
    expect(markup).toContain('<summary class="flex min-h-11')
    expect(markup).toContain('href="#decision-decision-1"')
    expect(markup).toContain('href="/agents"')
    // Details tabs render in a fixed order (Ledger, Plan, Gates, Delegations, Memory);
    // all panels stay mounted (hidden via the `hidden` attribute) so this still holds.
    expect(markup.indexOf('Goal context')).toBeLessThan(markup.indexOf('Agent suggestions'))
    expect(markup.indexOf('request_verification')).toBeLessThan(
      markup.indexOf('Request independent validation'),
    )
    // Each action surfaces its linked decision, not just time proximity.
    expect(markup).toContain('Caused by decision')
    expect(markup).toContain('#decision-decision-1')
    // The goal title now lives in RoomPanel's <h2>, not a standalone header <h1>.
    expect(markup).toMatch(/<h2 id="room-panel-title" class="[^"]*">Ship an evidence-backed release/)
    // Raw key/type are tucked into a title attribute once a human description leads.
    expect(markup).toContain('Release is independently validated.')
    expect(markup).toMatch(/title="release-ready[^"]*"/)
  })

  it('renders delegation load errors with a 44px retry target', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.taskError = true

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Failed to load active delegations.')
    expect(markup).toMatch(/<button[^>]*class="[^"]*min-h-11[^"]*"[^>]*>Retry<\/button>/)
    expect(markup).not.toContain('No active delegations.')
  })

  it.each([
    'taskLoading',
    'taskHasNextPage',
    'taskFetchingNextPage',
  ] as const)('does not show a false-empty delegation state while %s', async (field) => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state[field] = true

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('aria-busy')
    expect(markup).not.toContain('No active delegations.')
  })

  it('links every action to its causing decision', async () => {
    const { buildOrchestrationTimeline } = await import('./ActivityLog')
    const actions = buildOrchestrationTimeline(richDetail).filter((item) => item.kind === 'action')

    expect(actions).toHaveLength(2)
    for (const action of actions) {
      expect(action.decisionRef?.id).toBe('decision-1')
      expect(action.decisionRef?.label).toBe('request_verification')
      expect(action.decisionRef?.reason).toBe('Request independent validation')
    }
  })

  it('focuses a cold decision hash after the short-ledger page-size reset and ignores a data refresh', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.baselineLoading = true
    state.location = { hash: '#decision-decision-1', pathname: '/orchestration/goal-1', search: '', state: null, key: 'decision-link' }

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
    try {
      const focus = vi.fn()
      const scrollIntoView = vi.fn()
      ;(view.document as unknown as { getElementById: (id: string) => HTMLElement | null }).getElementById = (id) => {
        const target = descendants(view.container).find((node) => node.getAttribute('id') === id)
        if (target) Object.assign(target, { focus, scrollIntoView })
        return target as unknown as HTMLElement ?? null
      }

      expect(focus).not.toHaveBeenCalled()
      expect(scrollIntoView).not.toHaveBeenCalled()

      state.baselineLoading = false
      await view.rerender()
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)

      state.detail = { ...richDetail }
      await view.rerender()
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('excludes gates with an unrecognized status from the gate-count header without producing NaN', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = {
      ...richDetail,
      gates_count: 4,
      gates: [
        { ...richDetail.gates[0], id: 'gate-open', status: 'open' },
        { ...richDetail.gates[0], id: 'gate-accepted', status: 'accepted', accepted_at: '2026-07-15T09:15:00Z' },
        { ...richDetail.gates[0], id: 'gate-failed', status: 'failed', failure_reason: 'Missing evidence.', failed_at: '2026-07-15T09:20:00Z' },
        { ...richDetail.gates[0], id: 'gate-unknown', status: 'expired' as unknown as 'open' },
      ],
    }

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).not.toContain('NaN')
    // Gate counts are grouped under one "Gates" label; failed emphasized only when > 0.
    expect(markup).toContain('1 open')
    expect(markup).toMatch(/<span class="font-semibold text-huddleroom-status-red">1 failed<\/span>/)
    expect(markup).toContain('1 accepted')
  })

  it('handles a goal with no active run and keeps goal-level controls', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = {
      ...richDetail,
      goal: { ...richDetail.goal, status: 'active' },
      run: null,
      decisions_count: 0,
      decisions: [],
      actions_count: 0,
      actions: [],
      gates_count: 0,
      gates: [],
      evidence_count: 0,
      evidence: [],
      agent_suggestions_count: 0,
      agent_suggestions: [],
    }

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('No orchestration run is attached to this goal.')
    expect(markup).toContain('Conversation')
    expect(markup).toContain('data-testid="baseline-dashboard"')
    expect(markup).toContain('data-run-id="none"')
    expect(markup).toContain('Ship an evidence-backed release')
    expect(markup).toContain('Pause goal')
    expect(markup).toContain('Cancel goal')
    expect(markup).not.toContain('Resume goal')
  })

  it('shows controls allowed by the current goal status', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = {
      ...richDetail,
      goal: { ...richDetail.goal, status: 'paused' },
      run: { ...richDetail.run!, status: 'paused' },
    }

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Resume goal')
    expect(markup).toContain('Cancel goal')
    expect(markup).not.toContain('Pause goal')
  })

  it('only offers Start work to ready outcome goals that are active or blocked', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    const { OrchestrationPage } = await import('./OrchestrationPage')

    state.detail = richDetail
    expect(renderToStaticMarkup(<OrchestrationPage />)).toContain('Start work')

    state.detail = { ...richDetail, run: { ...richDetail.run!, phase: 'authorized' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Start work')

    state.detail = { ...richDetail, goal: { ...richDetail.goal, goal_type: 'roadmap' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Start work')

    state.detail = { ...richDetail, goal: { ...richDetail.goal, status: 'paused' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Start work')
  })

  it('starts work once and reflects the returned ready-run replacement', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    const started = { ...richDetail, run: { ...richDetail.run!, phase: 'authorized', condition: 'waiting_work' } }
    state.startMutation.mutate.mockImplementation((variables, callbacks) => {
      state.detail = started
      callbacks.onSuccess(started)
      callbacks.onSettled()
      return undefined
    })
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
    try {
      await React.act(async () => getButton(view.container, 'Start work').click())
      await React.act(async () => getButton(view.container, 'Confirm start').click())

      expect(state.startMutation.mutate).toHaveBeenCalledTimes(1)
      expect(state.startMutation.mutate).toHaveBeenCalledWith(
        { goalId: 'goal-1' }, expect.any(Object),
      )
      expect(textOf(view.container)).toContain('Work started.')
      expect(() => getButton(view.container, 'Start work')).toThrow('Button not found')
    } finally {
      view.cleanup()
    }
  })

  it('keeps detail error recovery targets at least 44px high', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detailError = true
    state.detailErrorObj = new ApiError(500, 'Internal server error', 'server_error')

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toMatch(/<button[^>]*class="[^"]*min-h-11[^"]*"[^>]*>Retry<\/button>/)
    expect(markup).toMatch(/<button[^>]*class="[^"]*min-h-11[^"]*"[^>]*>Back to goals<\/button>/)
  })

  it('renders goal-unavailable with 404 hiding Retry button', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detailError = true
    state.detailErrorObj = new ApiError(404, 'Goal not found', 'goal_not_found')

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Orchestration goal unavailable')
    expect(markup).toContain('Back to goals')
    // 404 errors should not show the Retry button, only 5xx and other retryable errors
    expect(markup).not.toMatch(/>\s*Retry\s*<\/button>/)
  })

  it('renders goal-unavailable with 500 showing both Retry and Back to goals with server error messaging', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detailError = true
    state.detailErrorObj = new ApiError(500, 'Internal server error', 'server_error')

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('Back to goals')
    expect(markup).toContain('The orchestrator could not answer.')
    expect(markup).toContain('>Retry<')
  })

  it('requires a decision and a nonblank override reason', async () => {
    const { canSubmitGateOverride } = await import('./OrchestrationPage')

    expect(canSubmitGateOverride(null, 'Operator reviewed it.')).toBe(false)
    expect(canSubmitGateOverride('accept', '   ')).toBe(false)
    expect(canSubmitGateOverride('reject', 'Missing independent proof.')).toBe(true)
  })

  it('uses the muted text token for the gate override legend, not teal', async () => {
    const { GateOverrideForm } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(
      <GateOverrideForm
        gateId="gate-1"
        pending={false}
        error={null}
        onCancel={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(markup).toContain('text-huddleroom-text-muted')
    expect(markup).not.toContain('teal')
  })

  it('shows freshness cue in amber with "not updating" when query has error and cached data', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.detailError = true
    state.dataUpdatedAt = new Date('2026-09-16T10:30:00Z').getTime()

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('text-huddleroom-status-amber')
    expect(markup).toContain('· not updating')
    expect(markup).not.toContain('· refreshing…')
  })

  const minimalSupervision = {
    condition: 'working' as const,
    operation: 'Validating release evidence',
    next_action: 'Wait for reviewer output',
    rationale: 'Independent verification is required before completion.',
    criterion: null,
    verified_progress: [],
    useful_learning: [],
    accepted_evidence: [],
    workers: [],
    waits: [],
    recovery_history: [],
    pending_direction: null,
    budget: {},
    transition: null,
  }

  describe('room panel wiring', () => {
    it('still renders Pause and Cancel goal action strings, with no inline confirm before clicking', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const markup = renderToStaticMarkup(<OrchestrationPage />)

      expect(markup).toContain('Pause goal')
      expect(markup).toContain('Cancel goal')
      expect(markup).not.toContain('Confirm pause')
      expect(markup).not.toContain('Running agents finish their current step, then stop.')
    })

    it('clicking Pause shows an inline confirm with two buttons and fires no mutation until confirmed', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
      try {
        await React.act(async () => getButton(view.container, 'Pause goal').click())

        expect(textOf(view.container)).toContain('Pause goal? Running agents finish their current step, then stop.')
        expect(() => getButton(view.container, 'Confirm pause')).not.toThrow()
        expect(() => getButton(view.container, 'Cancel')).not.toThrow()
        expect(state.commandMutation.mutate).not.toHaveBeenCalled()
      } finally {
        view.cleanup()
      }
    })

    it('clicking Confirm pause calls the command mutation with pause', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
      try {
        await React.act(async () => getButton(view.container, 'Pause goal').click())
        await React.act(async () => getButton(view.container, 'Confirm pause').click())

        expect(state.commandMutation.mutate).toHaveBeenCalledTimes(1)
        expect(state.commandMutation.mutate).toHaveBeenCalledWith(
          { goalId: 'goal-1', command: 'pause' }, expect.any(Object),
        )
      } finally {
        view.cleanup()
      }
    })

    it('clicking Cancel on the inline pause confirm dismisses it without firing a mutation', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
      try {
        await React.act(async () => getButton(view.container, 'Pause goal').click())
        await React.act(async () => getButton(view.container, 'Cancel').click())

        expect(textOf(view.container)).not.toContain('Running agents finish their current step, then stop.')
        expect(() => getButton(view.container, 'Pause goal')).not.toThrow()
        expect(state.commandMutation.mutate).not.toHaveBeenCalled()
      } finally {
        view.cleanup()
      }
    })

    it('hides the Reset button from the main markup and shows it only in the Debug tabpanel when debug is enabled', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail
      state.healthDebugEnabled = false

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const withoutDebug = renderToStaticMarkup(<OrchestrationPage />)
      expect(withoutDebug).not.toContain('Reset goal')
      expect(withoutDebug).not.toContain('Danger zone')

      state.healthDebugEnabled = true
      const withDebug = renderToStaticMarkup(<OrchestrationPage />)
      // All tabpanels stay mounted (hidden attribute), so a static render still
      // contains the Debug tabpanel's markup.
      expect(withDebug).toContain('Reset goal — deletes all history')
      const debugPanelStart = withDebug.indexOf('id="details-panel-debug"')
      const resetIndex = withDebug.indexOf('Reset goal — deletes all history')
      expect(debugPanelStart).toBeGreaterThan(-1)
      expect(resetIndex).toBeGreaterThan(debugPanelStart)
    })

    it('hides ExecutionSupervision "Current operation" from the main body and shows it only in the Debug tabpanel when debug is enabled', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = { ...richDetail, supervision: minimalSupervision }
      state.healthDebugEnabled = false

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const withoutDebug = renderToStaticMarkup(<OrchestrationPage />)
      expect(withoutDebug).not.toContain('Current operation')

      state.healthDebugEnabled = true
      const withDebug = renderToStaticMarkup(<OrchestrationPage />)
      expect(withDebug).toContain('Current operation')
      const debugPanelStart = withDebug.indexOf('id="details-panel-debug"')
      const supervisionIndex = withDebug.indexOf('Current operation')
      expect(debugPanelStart).toBeGreaterThan(-1)
      expect(supervisionIndex).toBeGreaterThan(debugPanelStart)
    })
  })
})
