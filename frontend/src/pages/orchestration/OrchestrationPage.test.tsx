import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { changeControl, descendants, getButton, getByLabel, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { NeedsYouQueue as RealQueue } from './NeedsYouQueue'
import type { OrchestrationGoalDetail, OrchestrationProcessRunRecord, Task } from '@/lib/types'

const state = vi.hoisted(() => ({
  activeProjectId: null as string | null,
  goalId: undefined as string | undefined,
  goals: [] as Array<Record<string, unknown>>,
  detail: undefined as OrchestrationGoalDetail | undefined,
  detailError: false,
  detailFetching: false,
  realQueueForwarding: false,
  queueDecisions: [] as import('@/lib/types').OrchestrationAuthorityDecisionRecord[],
  detailErrorObj: null as unknown,
  dataUpdatedAt: 0 as number,
  tasks: [] as Task[],
  taskLoading: false,
  taskError: false,
  taskHasNextPage: false,
  taskFetchingNextPage: false,
  taskRefetch: vi.fn(),
  createMutation: { isPending: false, mutate: vi.fn() },
  announceError: vi.fn(),
  startMutation: { isPending: false, mutate: vi.fn() },
  previewMutation: { isPending: false, mutate: vi.fn() },
  applyMutation: { isPending: false, mutate: vi.fn() },
  commandMutation: { isPending: false, mutate: vi.fn() },
  resetMutation: { isPending: false, mutate: vi.fn() },
  healthDebugEnabled: false,
  agents: [] as Array<Record<string, unknown>>,
  // Keyed by meeting id (matches meetingKeys.detail's third element).
  meetingsById: new Map<string, Record<string, unknown>>(),
  meetingDecisions: undefined as Array<Record<string, unknown>> | undefined,
  baselineLoading: false,
  baselineProcesses: [] as OrchestrationProcessRunRecord[],
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

// Root shares onOpenChange so the footer Close (Cancel) can dismiss the dialog.
const dialogClose = vi.hoisted(() => ({ current: undefined as undefined | ((open: boolean) => void) }))
vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ open, onOpenChange, children }: { open?: boolean; onOpenChange?: (open: boolean) => void; children: React.ReactNode }) => {
    dialogClose.current = onOpenChange
    return open ? <>{children}</> : null
  },
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />,
  Content: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  Title: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => <h2 {...props}>{children}</h2>,
  Description: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props}>{children}</p>,
  Close: ({ children }: { children: React.ReactElement<{ onClick?: () => void }> }) =>
    React.cloneElement(children, { onClick: () => dialogClose.current?.(false) }),
}))

vi.mock('./goalAnnouncer', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./goalAnnouncer')>()
  return {
    ...actual,
    useGoalAnnouncer: () => ({ ...actual.useGoalAnnouncer(), announceError: state.announceError }),
  }
})

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (value: { activeProjectId: string | null }) => unknown) =>
    selector({ activeProjectId: state.activeProjectId }),
}))

vi.mock('@/api/orchestration', () => ({
  useCreateOrchestrationGoal: () => state.createMutation,
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
    isFetching: state.detailFetching,
  }),
  useOrchestrationGoalCommand: () => state.commandMutation,
  useOverrideOrchestrationGate: () => ({ isPending: false, mutate: vi.fn() }),
  useResetOrchestrationGoal: () => state.resetMutation,
  useStartOrchestrationGoal: () => state.startMutation,
  usePreviewGoalTypeCorrection: () => state.previewMutation,
  useApplyGoalTypeCorrection: () => state.applyMutation,
  useRecoverGoalDefinition: () => ({ isPending: false, mutate: vi.fn() }),
  useBaselineDashboard: () => ({
    memory: { data: undefined, isLoading: false, isError: false },
    processes: { data: state.baselineProcesses, isLoading: state.baselineLoading, isError: false },
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
    data: { items: [], total: 0, omitted: 0, allowance: { enabled: true, unlimited: false, limit: 100, used: 0, remaining: 100 }, steering: { enabled: false, eligibility: 'unstarted', eligibility_reason: null, inbox_version: 0, direction_version: 0, requests: [], proposals: [] } },
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
  isManagerSelectionDecision: () => false,
  decisionOptionKey: (option: unknown) => typeof option === 'string' ? option : '',
  normalizeDecisionOption: (option: unknown) => ({ key: typeof option === 'string' ? option : '', label: typeof option === 'string' ? option : '' }),
  parseManagerComparison: () => null,
  BaselineDashboard: ({ goalId, run, acquireMutation, onRevealSource, onSelectedProcessTypeChange, onViewActivity, suppressQueueAnnouncement, queueAnnouncementScope }: { goalId: string; run: { id: string } | null; acquireMutation: (owner: string) => boolean; onRevealSource?: (target: { kind: 'gate' | 'process' | 'action'; id: string; goalId: string }) => void; onSelectedProcessTypeChange?: (type: 'team_hierarchy') => void; onViewActivity?: () => void; suppressQueueAnnouncement?: boolean; queueAnnouncementScope?: 'all' | 'human' | 'none' }) => (
    <div data-testid="baseline-dashboard" data-goal-id={goalId} data-run-id={run?.id ?? 'none'}><button onClick={() => acquireMutation("baseline:run:goal_definition")}>Mock run</button><button onClick={() => acquireMutation("baseline:answer:decision-1")}>Mock answer</button><button onClick={() => onRevealSource?.({ kind: 'gate', id: 'gate-1', goalId })}>Reveal source gate</button><button onClick={() => onRevealSource?.({ kind: 'process', id: 'process-old', goalId })}>Reveal superseded process</button><button onClick={() => { onSelectedProcessTypeChange?.('team_hierarchy'); onViewActivity?.() }}>View current step activity</button>{state.realQueueForwarding && state.detail && <RealQueue
      goal={state.detail.goal} run={state.detail.run} gates={[]} processes={[]} warnings={[]}
      checkpointItems={state.queueDecisions} allDecisions={state.queueDecisions} authorityReady checkpointLoaded
      queriesErrored={false} queriesUnready={false} deferredCount={0} isWorking={false} mutationBusy={false}
      openAction={null} onOpenAction={() => undefined} onActionTrigger={() => undefined} onSelectProcess={() => undefined}
      renderActionForm={() => null} renderAgentDefinitionReviewBatch={() => null} renderTeamHierarchyProposal={() => null}
      onAcceptDecision={() => undefined} onAnswerGoalDefinitionQuestion={() => undefined}
      suppressAnnouncement={suppressQueueAnnouncement} announcementScope={queueAnnouncementScope}
    />}</div>
  ),
  currentProcesses: (processes: unknown[] | null | undefined) => (processes ?? []).filter((process) => (process as { superseded_by_id: string | null }).superseded_by_id === null),
}))

vi.mock('./StartWorkDialog', () => ({
  StartWorkDialog: ({ open, isPending, error, onConfirm, authorization, restartReason }: {
    open: boolean
    isPending: boolean
    error: Error | null
    onConfirm: () => void
    authorization?: { model_spend_remaining: Record<string, string>; workspace_mutation: string } | null
    restartReason?: string | null
  }) => open ? (
    <div data-testid="start-work-dialog">
      <p data-testid="start-authorization">{authorization ? `${JSON.stringify(authorization.model_spend_remaining)} ${authorization.workspace_mutation}` : 'no authorization'}</p>
      <p data-testid="start-restart-reason">{restartReason ?? 'first start'}</p>
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
  state.detailFetching = false
  state.realQueueForwarding = false
  state.queueDecisions = []
  state.detailErrorObj = null
  state.dataUpdatedAt = 0
  state.tasks = []
  state.taskLoading = false
  state.taskError = false
  state.taskHasNextPage = false
  state.taskFetchingNextPage = false
  state.taskRefetch.mockReset()
  state.createMutation.isPending = false
  state.createMutation.mutate.mockReset()
  state.announceError.mockReset()
  state.startMutation.isPending = false
  state.startMutation.mutate.mockReset()
  state.resetMutation.isPending = false
  state.resetMutation.mutate.mockReset()
  state.commandMutation.isPending = false
  state.commandMutation.mutate.mockReset()
  state.previewMutation.mutate.mockReset()
  state.applyMutation.mutate.mockReset()
  state.healthDebugEnabled = false
  state.agents = []
  state.meetingsById = new Map()
  state.meetingDecisions = undefined
  state.baselineLoading = false
  state.baselineProcesses = []
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

    const runsHeading = markup.match(/<h3 id="([^"]+-runs)"[^>]*>Runs<\/h3>/)
    expect(runsHeading).not.toBeNull()
    expect(markup).toContain(`aria-labelledby="${runsHeading![1]}"`)
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

  it('SPR210 wires source navigation to the actual record in the active goal', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
    try {
      await React.act(async () => getButton(view.container, 'Reveal source gate').click())
      const gatePanel = descendants(view.container).find((node) => node.getAttribute('id') === 'details-panel-gates')
      expect(gatePanel?.getAttribute('hidden')).toBeNull()
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'gate-gate-1')).toBeDefined()
    } finally { view.cleanup() }
  })

  it('SPR210 fixround2 keeps ordinary step activity current while revealing an exact superseded source', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    const processRecord = (id: string, supersededById: string | null): OrchestrationProcessRunRecord => ({
      id, goal_id: 'goal-1', run_id: 'run-1', process_type: 'team_hierarchy', process_version: 1,
      status: 'completed', trigger_reason: 'test', input_snapshot: {}, outputs: {}, skipped_by: null,
      override_reason: null, superseded_by_id: supersededById, started_at: '2026-07-15T09:00:00Z',
      completed_at: '2026-07-15T09:05:00Z', created_at: '2026-07-15T09:00:00Z', updated_at: '2026-07-15T09:05:00Z',
    })
    state.baselineProcesses = [processRecord('process-old', 'process-new'), processRecord('process-new', null)]

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

      await React.act(async () => getButton(view.container, 'View current step activity').click())
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-new')).toBeDefined()
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-old')).toBeUndefined()

      await React.act(async () => getButton(view.container, 'Reveal superseded process').click())

      const target = descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-old')
      expect(target).toBeDefined()
      expect(target?.getAttribute('tabindex')).toBe('-1')
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)

      await React.act(async () => getButton(view.container, 'Whole goal').click())
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-new')).toBeDefined()
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-old')).toBeUndefined()

      await React.act(async () => getButton(view.container, 'Reveal superseded process').click())
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-old')).toBeDefined()
      expect(focus).toHaveBeenCalledTimes(2)
      expect(scrollIntoView).toHaveBeenCalledTimes(2)

      await React.act(async () => getButton(view.container, 'View current step activity').click())
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-new')).toBeDefined()
      expect(descendants(view.container).find((node) => node.getAttribute('id') === 'process-process-old')).toBeUndefined()
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

  it('uses muted text for waiting gates and red only for failed gates', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = {
      ...richDetail,
      gates_count: 2,
      gates: [
        { ...richDetail.gates[0], id: 'gate-waiting', status: 'open', failure_reason: 'Waiting for Start' },
        { ...richDetail.gates[0], id: 'gate-failed', status: 'failed', failure_reason: 'Missing evidence.' },
      ],
    }
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toMatch(/<p class="mt-1 break-words text-sm text-huddleroom-text-muted">Waiting for Start<\/p>/)
    expect(markup).toMatch(/<p class="mt-1 break-words text-sm text-huddleroom-status-red">Missing evidence\.<\/p>/)
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
    expect(markup).toContain('Open conversation')
    expect(markup).not.toContain('textarea')
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

  it('only offers Start work to ready goals of any type that are active or blocked', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    const { OrchestrationPage } = await import('./OrchestrationPage')

    state.detail = richDetail
    expect(renderToStaticMarkup(<OrchestrationPage />)).toContain('Start work')

    state.detail = { ...richDetail, run: { ...richDetail.run!, phase: 'authorized' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Start work')

    state.detail = { ...richDetail, goal: { ...richDetail.goal, goal_type: 'roadmap' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).toContain('Start work')

    state.detail = { ...richDetail, goal: { ...richDetail.goal, status: 'paused' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Start work')
  })

  describe('paused authorization prompts', () => {
    const authorized = (auth: Partial<NonNullable<OrchestrationGoalDetail['effective_authorization']>> | null, wait: OrchestrationGoalDetail['authorization_wait'] = null): OrchestrationGoalDetail => ({
      ...richDetail,
      run: { ...richDetail.run!, phase: 'authorized' },
      effective_authorization: auth && {
        allowed: false, reason_code: null, required_decision: null, authority_source: 'root_start',
        run_id: 'run-1', contract_fingerprint: 'fp', model_spend_remaining: {}, external_write_targets: [],
        live_send_channels: [], workspace_mutation: 'direct', tool_constraints: [], ...auth,
      },
      authorization_wait: wait,
    })

    it('asks for a re-Start and keeps Start work when the decision is start', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      const { OrchestrationPage } = await import('./OrchestrationPage')
      state.detail = authorized({ reason_code: 'contract_changed_since_start', required_decision: 'start' })
      const html = renderToStaticMarkup(<OrchestrationPage />)
      expect(html).toContain('Work is paused until you start again: the goal changed after you started it.')
      expect(html).toContain('Start work')
    })

    it('passes the effective authorization and the restart reason into the Start dialog', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      const { OrchestrationPage } = await import('./OrchestrationPage')
      state.detail = authorized({
        reason_code: 'contract_changed_since_start', required_decision: 'start',
        model_spend_remaining: { max_tokens: '7000' }, workspace_mutation: 'staged',
      })
      const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
      try {
        await React.act(async () => getButton(view.container, 'Start work').click())
        expect(textOf(view.container)).toContain('{"max_tokens":"7000"} staged')
        expect(textOf(view.container)).toContain('the goal changed after you started it')
      } finally { view.cleanup() }
    })

    it('passes no restart reason for a first Start', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = richDetail
      const { OrchestrationPage } = await import('./OrchestrationPage')
      const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
      try {
        await React.act(async () => getButton(view.container, 'Start work').click())
        expect(textOf(view.container)).toContain('first start')
      } finally { view.cleanup() }
    })

    it('falls back to authorization_wait and shows the reason without Start for other decisions', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      const { OrchestrationPage } = await import('./OrchestrationPage')
      state.detail = authorized(null, { task_ids: ['task-1'], reason_code: 'budget_exhausted', required_decision: 'increase_budget' })
      const html = renderToStaticMarkup(<OrchestrationPage />)
      expect(html).toContain('Work is paused: the budget is used up. Needed: A budget increase.')
      expect(html).not.toContain('Start work')
    })

    it('shows no prompt when the run is allowed', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      const { OrchestrationPage } = await import('./OrchestrationPage')
      state.detail = authorized({ allowed: true })
      expect(renderToStaticMarkup(<OrchestrationPage />)).not.toContain('Work is paused')
    })
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
    state.detail = { ...richDetail, supervision: minimalSupervision }
    state.detailError = true
    state.dataUpdatedAt = new Date('2026-09-16T10:30:00Z').getTime()

    const { OrchestrationPage } = await import('./OrchestrationPage')
    const markup = renderToStaticMarkup(<OrchestrationPage />)

    expect(markup).toContain('text-huddleroom-status-amber')
    expect(markup).toContain('· not updating')
    expect((markup.match(/Data as of/g) ?? [])).toHaveLength(1)
    expect(markup).not.toContain('· refreshing…')
  })

  const minimalSupervision = {
    condition: 'working' as const,
    status_line: { current_state: { kind: 'unknown' as const, summary: 'No active worker confirmed.', activities: [] }, attention_summary: null, last_material_event: null },
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

    it('shows compact Current operation in the normal page while full supervision remains Debug-only', async () => {
      state.activeProjectId = 'project-1'
      state.goalId = 'goal-1'
      state.detail = { ...richDetail, supervision: minimalSupervision }
      state.healthDebugEnabled = false

      const { OrchestrationPage } = await import('./OrchestrationPage')
      const withoutDebug = renderToStaticMarkup(<OrchestrationPage />)
      expect(withoutDebug).toContain('Current operation')
      expect(withoutDebug).not.toContain('Recovery linkage')
      expect((withoutDebug.match(/Data as of/g) ?? [])).toHaveLength(1)

      state.healthDebugEnabled = true
      const withDebug = renderToStaticMarkup(<OrchestrationPage />)
      expect(withDebug).toContain('Current operation')
      const debugPanelStart = withDebug.indexOf('id="details-panel-debug"')
      const supervisionIndex = withDebug.indexOf('Current operation')
      expect(debugPanelStart).toBeGreaterThan(-1)
      expect(supervisionIndex).toBeLessThan(debugPanelStart)
      expect(withDebug.indexOf('Recovery linkage')).toBeGreaterThan(debugPanelStart)
    })
  })
})

describe('goal header shape provenance', () => {
  const shaped = {
    ...richDetail,
    goal: {
      ...richDetail.goal,
      goal_type_source: 'legacy',
      goal_type_rationale: 'One bounded result.',
      planning_mode: 'delivery',
      planning_rationale: 'Small enough for one plan level.',
    },
  }

  it('renders type, source, planning and rationales without raw JSON; successor line only when superseding', async () => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    const { OrchestrationPage } = await import('./OrchestrationPage')
    state.detail = shaped
    const html = renderToStaticMarkup(<OrchestrationPage />)
    expect(html).toContain('Outcome')
    expect(html).toContain('Existing goal (set before automatic selection)')
    expect(html).toContain('Delivery plan')
    expect(html).toContain('Why Outcome: One bounded result.')
    expect(html).toContain('▸ Why this plan depth')
    // Planning mode and plan status share one "Plan" column.
    expect(html).not.toContain('>Planning<')
    expect(html).toMatch(/>Plan<\/span><span[^>]*>Delivery plan<\/span>.*Accepted/)
    expect(html).toContain('Small enough for one plan level.')
    expect(html).not.toContain('replaces an earlier run')
    expect(html).not.toContain('{&quot;')

    state.detail = { ...shaped, run: { ...shaped.run!, supersedes_run_id: 'run-0' } }
    expect(renderToStaticMarkup(<OrchestrationPage />)).toContain('Current run replaces an earlier run')
  })
})

describe('CreateGoalModal goal type selector', () => {
  async function mountModal() {
    const { CreateGoalModal } = await import('./OrchestrationPage')
    return mountWithTestDom(() => <CreateGoalModal projectId="project-1" open onClose={vi.fn()} />, React.act)
  }
  const radios = (root: Parameters<typeof descendants>[0]) =>
    descendants(root).filter((n) => n.tagName === 'INPUT' && n.getAttribute('name') === 'goal-type')
  async function fill(view: Awaited<ReturnType<typeof mountModal>>) {
    const [objective, criteria] = descendants(view.document.body).filter((n) => n.tagName === 'TEXTAREA')
    await React.act(async () => {
      changeControl(objective, 'Do it')
      changeControl(criteria, 'It works')
    })
  }
  async function submit(view: Awaited<ReturnType<typeof mountModal>>) {
    await React.act(async () => {
      const form = descendants(view.document.body).find((n) => n.tagName === 'FORM')!
      const key = Object.keys(form).find((k) => k.startsWith('__reactProps$'))!
      ;(form as unknown as Record<string, { onSubmit: (e: { preventDefault: () => void }) => void }>)[key].onSubmit({ preventDefault() {} })
    })
  }

  it('defaults to Auto and describes all four choices', async () => {
    const view = await mountModal()
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('Goal type')
      for (const copy of [
        'Auto (recommended)', 'HuddleRoom picks the shape from your objective and criteria. You can correct it later.',
        'One bounded result, even if it is large.',
        'A finite sequence of outcomes, planned one level at a time.',
        'Ongoing, recurring work with no natural end. Needs an accepted recurring policy before it can start.',
      ]) expect(text).toContain(copy)
      expect(radios(view.document.body).map((r) => r.checked)).toEqual([true, false, false, false])
    } finally { view.cleanup() }
  })

  it('sends auto by default and the picked type otherwise', async () => {
    const view = await mountModal()
    try {
      await fill(view)
      await submit(view)
      expect(state.createMutation.mutate.mock.calls[0][0]).toMatchObject({ goal_type: 'auto' })
      await React.act(async () => changeControl(radios(view.document.body)[2], true))
      await submit(view)
      expect(state.createMutation.mutate.mock.calls[1][0]).toMatchObject({ goal_type: 'roadmap' })
    } finally { view.cleanup() }
  })

  it('explains a classifier failure, announces it, and focuses Auto', async () => {
    const { ApiError } = await import('@/lib/api-client')
    const view = await mountModal()
    try {
      await fill(view)
      await React.act(async () => changeControl(radios(view.document.body)[1], true))
      await submit(view)
      const copy = "We couldn't choose a goal type automatically. Pick Outcome, Roadmap, or Continuous and try again."
      await React.act(async () => {
        state.createMutation.mutate.mock.calls[0][1].onError(
          new ApiError(503, 'HTTP 503', { code: 'goal_shape_unresolved', reason: 'x', message: 'y' }))
      })
      expect(textOf(view.document.body)).toContain(copy)
      expect(state.announceError).toHaveBeenCalledWith(copy)
      expect(view.activeElement).toBe(radios(view.document.body)[0])
    } finally { view.cleanup() }
  })

  it('restores Auto after a successful create', async () => {
    const view = await mountModal()
    try {
      await fill(view)
      await React.act(async () => changeControl(radios(view.document.body)[2], true))
      await submit(view)
      await React.act(async () => state.createMutation.mutate.mock.calls[0][1].onSuccess({ goal: { id: 'g' } }))
      expect(radios(view.document.body).map((r) => r.checked)).toEqual([true, false, false, false])
    } finally { view.cleanup() }
  })

  it('disables the type inputs while creating', async () => {
    state.createMutation.isPending = true
    const view = await mountModal()
    try {
      const fieldset = descendants(view.document.body).find((n) => n.tagName === 'FIELDSET')!
      expect(fieldset.getAttribute('disabled')).not.toBeNull()
      const cards = descendants(view.document.body).filter((n) => n.tagName === 'LABEL' && n.childNodes.some((c) => 'tagName' in c && c.tagName === 'INPUT'))
      expect(cards).toHaveLength(4)
      expect(cards.every((card) => card.getAttribute('class')?.includes('focus-within:ring-huddleroom-primary'))).toBe(true)
    } finally { view.cleanup() }
  })
})

describe('CorrectGoalTypeDialog', () => {
  const previewFixture = (over: Record<string, unknown> = {}) => ({
    goal_id: 'goal-1', current_run_id: 'run-1',
    current_goal_type: 'outcome', proposed_goal_type: 'roadmap',
    current_planning_mode: 'delivery', proposed_planning_mode: 'roadmap',
    source: 'user', rationale: 'Too big', blockers: [],
    retained_evidence_manifest: [{
      source_run_id: 'run-1', source_task_id: 't1', source_gate_id: 'g1', artifact_id: 'a1b2c3d4-9999',
      source_contract_fingerprint: 'fp', revalidation_required: true, reason: 'Readiness proof from the first run',
    }],
    remaining_budget: { tokens: '120000' }, missing_recurring_policy: false,
    requires_new_start: true, required_authority: ['baseline_rerun', 'start'],
    preview_fingerprint: 'pf-1', ...over,
  })
  const onApplied = vi.fn()
  const onOpenChange = vi.fn()
  const acquireMutation = vi.fn(() => true)
  const refreshDetail = vi.fn()
  const freshDetail = { run: { id: 'run-2' }, contract_fingerprint: 'fp-2' } as unknown as OrchestrationGoalDetail

  async function mountDialog() {
    const { CorrectGoalTypeDialog } = await import('./OrchestrationPage')
    return mountWithTestDom(() => (
      <CorrectGoalTypeDialog
        projectId="project-1" open goalId="goal-1" currentType="outcome" runId="run-1"
        contractFingerprint="contract-fp" acquireMutation={acquireMutation} releaseMutation={() => {}}
        refreshDetail={refreshDetail} onOpenChange={onOpenChange} onApplied={onApplied}
      />
    ), React.act)
  }
  type View = Awaited<ReturnType<typeof mountDialog>>
  const radios = (view: View) =>
    descendants(view.document.body).filter((n) => n.tagName === 'INPUT' && n.getAttribute('name') === 'corrected-goal-type')
  const rationale = (view: View) => getByLabel(view.container, 'Why the current type is wrong', 'textarea')
  const press = (view: View, label: string) => React.act(async () => getButton(view.container, label).click())
  async function fillAndPreview(view: View, text = 'Too big') {
    await React.act(async () => {
      changeControl(radios(view)[1], true)
      changeControl(rationale(view), text)
    })
    await press(view, 'Preview change')
  }

  beforeEach(() => {
    onApplied.mockReset()
    onOpenChange.mockReset()
    acquireMutation.mockReset()
    acquireMutation.mockReturnValue(true)
    refreshDetail.mockReset()
    refreshDetail.mockResolvedValue(freshDetail)
    state.previewMutation.mutate.mockImplementation((_input: unknown, opts: { onSuccess: (d: unknown) => void }) => opts.onSuccess(previewFixture()))
  })

  it('disables the current type and gates Preview on a choice and a rationale', async () => {
    const view = await mountDialog()
    try {
      expect(radios(view).map((r) => r.getAttribute('disabled') !== null)).toEqual([true, false, false])
      expect(textOf(view.container)).toContain('Outcome (current)')
      expect(getButton(view.container, 'Preview change').getAttribute('disabled')).not.toBeNull()
      await React.act(async () => changeControl(radios(view)[1], true))
      expect(getButton(view.container, 'Preview change').getAttribute('disabled')).not.toBeNull()
      await React.act(async () => changeControl(rationale(view), '  '))
      expect(getButton(view.container, 'Preview change').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('previews with the run, contract fingerprint and rationale, and shows the result', async () => {
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      expect(state.previewMutation.mutate.mock.calls[0][0]).toEqual({
        goalId: 'goal-1', requestedGoalType: 'roadmap', expectedRunId: 'run-1',
        expectedContractFingerprint: 'contract-fp', rationale: 'Too big',
      })
      const text = textOf(view.container)
      for (const copy of [
        'Outcome → Roadmap', 'Delivery plan → Roadmap plan', 'A new baseline run, Start', 'tokens 120,000',
        'No blockers.', 'Evidence carried over (1)', 'Needs revalidation', 'a1b2c3d4', 'Also requires',
        'Readiness proof from the first run', 'Carried-over evidence is never accepted automatically.',
      ]) expect(text).toContain(copy)
      expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).toBeNull()
      const summary = descendants(view.container).find((node) => node.getAttribute('aria-live') === 'polite' && node.getAttribute('class') === 'sr-only')!
      expect(summary.getAttribute('aria-atomic')).toBe('true')
      expect(textOf(summary)).toBe('Preview: Outcome to Roadmap. 0 blockers. 1 evidence items carried over. A new Start is required.')
      const cards = descendants(view.container).filter((n) => n.tagName === 'LABEL' && n.childNodes.some((c) => 'tagName' in c && c.tagName === 'INPUT'))
      expect(cards).toHaveLength(3)
      expect(cards.every((card) => card.getAttribute('class')?.includes('focus-within:ring-huddleroom-primary'))).toBe(true)
    } finally { view.cleanup() }
  })

  it('shows blockers with their fix, the kept-evidence copy and disables Apply', async () => {
    state.previewMutation.mutate.mockImplementation((_i: unknown, opts: { onSuccess: (d: unknown) => void }) => opts.onSuccess(previewFixture({
      blockers: [{ code: 'open_work', entity_type: 'task', entity_id: 't1', message: 'A task is running.', resolution: 'Pause the goal first.' }],
      retained_evidence_manifest: [{ source_run_id: 'r', source_task_id: 't', source_gate_id: 'g', artifact_id: null, source_contract_fingerprint: 'f', revalidation_required: false, reason: 'Kept proof' }],
      missing_recurring_policy: true,
    })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      const text = textOf(view.container)
      expect(text).toContain('Resolve before applying (1)')
      expect(text).toContain('A task is running.')
      expect(text).toContain('How to fix: Pause the goal first.')
      expect(text).toContain('Kept, checked again by the new gate')
      expect(text).toContain('A recurring policy must be accepted before this goal can start.')
      expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('disables Apply once the inputs change after a preview', async () => {
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await React.act(async () => changeControl(rationale(view), 'Different reason'))
      expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).not.toBeNull()
      expect(textOf(view.container)).toContain('Inputs changed. Preview again before applying.')
      await React.act(async () => changeControl(radios(view)[2], true))
      await React.act(async () => changeControl(rationale(view), 'Too big'))
      expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('refreshes the goal on a stale 409, re-previews with the fresh run and contract, then focuses Apply', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError(new ApiError(409, 'HTTP 409', { conflict: 'stale_contract', message: 'out of date' })))
    state.previewMutation.mutate.mockImplementation((_i: unknown, opts: { onSuccess: (d: unknown) => void }) =>
      opts.onSuccess(previewFixture({ preview_fingerprint: `pf-${state.previewMutation.mutate.mock.calls.length}` })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await press(view, 'Apply correction')
      expect(state.applyMutation.mutate.mock.calls[0][0]).toMatchObject({
        previewFingerprint: 'pf-1', requestedGoalType: 'roadmap', rationale: 'Too big',
        expectedRunId: 'run-1', expectedContractFingerprint: 'contract-fp',
      })
      expect(refreshDetail).toHaveBeenCalledTimes(1)
      expect(state.previewMutation.mutate).toHaveBeenCalledTimes(2)
      expect(state.previewMutation.mutate.mock.calls[1][0]).toMatchObject({
        requestedGoalType: 'roadmap', rationale: 'Too big', expectedRunId: 'run-2', expectedContractFingerprint: 'fp-2',
      })
      expect(radios(view)[1].checked).toBe(true)
      expect(rationale(view).value).toBe('Too big')
      const stale = 'The goal changed since your preview. We refreshed it. Review and apply again.'
      expect(textOf(view.container)).toContain(stale)
      expect(state.announceError).toHaveBeenCalledWith(stale)
      expect(view.activeElement).toBe(getButton(view.container, 'Apply correction'))
      // The refreshed preview carries the new fingerprint and the refreshed run/contract.
      await press(view, 'Apply correction')
      expect(state.applyMutation.mutate.mock.calls[1][0]).toMatchObject({
        previewFingerprint: 'pf-2', expectedRunId: 'run-2', expectedContractFingerprint: 'fp-2',
      })
    } finally { view.cleanup() }
  })

  it('explains when the goal can no longer be corrected after a stale 409', async () => {
    const { ApiError } = await import('@/lib/api-client')
    const expectedCopy = 'This goal changed and can no longer be corrected here. Close and reopen the dialog.'
    for (const failure of [() => refreshDetail.mockResolvedValue(undefined), () => refreshDetail.mockRejectedValue(new Error('offline'))]) {
      state.applyMutation.mutate.mockReset()
      state.previewMutation.mutate.mockClear()
      state.announceError.mockReset()
      failure()
      state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onError: (e: unknown) => void }) =>
        opts.onError(new ApiError(409, 'HTTP 409', { conflict: 'stale_run', message: 'out of date' })))
      const view = await mountDialog()
      try {
        await fillAndPreview(view)
        await press(view, 'Apply correction')
        expect(state.previewMutation.mutate).toHaveBeenCalledTimes(1)
        expect(textOf(view.container)).toContain(expectedCopy)
        expect(state.announceError).toHaveBeenCalledWith(expectedCopy)
        expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).not.toBeNull()
      } finally { view.cleanup() }
    }
  })

  it('tells the user when another action holds the mutation lock', async () => {
    acquireMutation.mockReturnValue(false)
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      expect(state.previewMutation.mutate).not.toHaveBeenCalled()
      expect(textOf(view.container)).toContain('Another action is in progress. Try again in a moment.')
    } finally { view.cleanup() }
  })

  it('moves focus to the Preview heading after a manual preview', async () => {
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      expect(view.activeElement?.textContent).toBe('Preview')
      expect(view.activeElement?.tagName).toBe('H3')
    } finally { view.cleanup() }
  })

  it('shows a preview error and returns focus to the Preview button', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.previewMutation.mutate.mockImplementation((_i: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError(new ApiError(422, 'HTTP 422', { message: 'Cannot preview this change.' })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      expect(textOf(view.container)).toContain('Cannot preview this change.')
      expect(view.activeElement).toBe(getButton(view.container, 'Preview change'))
    } finally { view.cleanup() }
  })

  it('cancel while previewing closes the dialog and a late preview response is ignored', async () => {
    let late: { onSuccess: (d: unknown) => void } | null = null
    state.previewMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onSuccess: (d: unknown) => void }) => { late = opts })
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await press(view, 'Cancel')
      expect(onOpenChange).toHaveBeenCalledWith(false)
      await React.act(async () => { late?.onSuccess(previewFixture()) })
      expect(textOf(view.container)).not.toContain('Retained evidence')
      expect(state.previewMutation.mutate).toHaveBeenCalledTimes(1)
      // Reopened: the form starts clean and Preview runs again.
      await fillAndPreview(view)
      expect(state.previewMutation.mutate).toHaveBeenCalledTimes(2)
    } finally { view.cleanup() }
  })

  it('failed stale re-preview returns focus to the Preview button', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError(new ApiError(409, 'HTTP 409', { conflict: 'stale_contract', message: 'out of date' })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      state.previewMutation.mutate.mockImplementation((_i: unknown, opts: { onError: (e: unknown) => void }) =>
        opts.onError(new ApiError(422, 'HTTP 422', { message: 'Cannot preview this change.' })))
      await press(view, 'Apply correction')
      expect(textOf(view.container)).toContain('Cannot preview this change.')
      expect(view.activeElement).toBe(getButton(view.container, 'Preview change'))
    } finally { view.cleanup() }
  })

  it('keeps the blockers from a correction_blocked 409 and disables Apply', async () => {
    const { ApiError } = await import('@/lib/api-client')
    const blocked = previewFixture({
      blockers: [{ code: 'open_work', entity_type: 'task', entity_id: 't1', message: 'A task started meanwhile.', resolution: 'Pause the goal first.' }],
    })
    state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError(new ApiError(409, 'HTTP 409', { conflict: 'correction_blocked', message: 'Resolve the blockers first.', preview: blocked })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await press(view, 'Apply correction')
      const text = textOf(view.container)
      expect(text).toContain('Resolve before applying (1)')
      expect(text).toContain('A task started meanwhile.')
      expect(text).toContain('Resolve the blockers first.')
      expect(getButton(view.container, 'Apply correction').getAttribute('disabled')).not.toBeNull()
      expect(refreshDetail).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it("says 'Nothing extra' when no further authority is required", async () => {
    state.previewMutation.mutate.mockImplementation((_i: unknown, opts: { onSuccess: (d: unknown) => void }) =>
      opts.onSuccess(previewFixture({ required_authority: [] })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      expect(textOf(view.container)).toContain('Nothing extra')
    } finally { view.cleanup() }
  })

  it('shows other apply errors inline, announces them and keeps Apply focused', async () => {
    const { ApiError } = await import('@/lib/api-client')
    state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onError: (e: unknown) => void }) =>
      opts.onError(new ApiError(403, 'HTTP 403', { conflict: 'explicit_user_request_required', message: 'Only a user can correct a goal type.' })))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await press(view, 'Apply correction')
      expect(state.previewMutation.mutate).toHaveBeenCalledTimes(1)
      expect(textOf(view.container)).toContain('Only a user can correct a goal type.')
      expect(state.announceError).toHaveBeenCalledWith('Only a user can correct a goal type.')
      expect(view.activeElement).toBe(getButton(view.container, 'Apply correction'))
    } finally { view.cleanup() }
  })

  it('reports success to the page', async () => {
    const detail = { goal: { goal_type: 'roadmap' } }
    state.applyMutation.mutate.mockImplementationOnce((_i: unknown, opts: { onSuccess: (d: unknown) => void }) => opts.onSuccess(detail))
    const view = await mountDialog()
    try {
      await fillAndPreview(view)
      await press(view, 'Apply correction')
      expect(onApplied).toHaveBeenCalledWith(detail)
    } finally { view.cleanup() }
  })
})

describe('Correct goal type trigger', () => {
  async function mountPage(patch: (detail: OrchestrationGoalDetail) => OrchestrationGoalDetail = (d) => d) {
    const { OrchestrationPage } = await import('./OrchestrationPage')
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = patch({ ...richDetail, contract_fingerprint: 'contract-fp' })
    return mountWithTestDom(() => <OrchestrationPage />, React.act)
  }

  it('shows the trigger for a live goal with a run', async () => {
    const view = await mountPage()
    try {
      expect(getButton(view.container, 'Correct goal type')).toBeDefined()
    } finally { view.cleanup() }
  })

  it('disables the Correct and Start triggers while another mutation is pending', async () => {
    state.commandMutation.isPending = true
    const view = await mountPage()
    try {
      expect(getButton(view.container, 'Correct goal type').getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.container, 'Start work').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('hides the trigger when the goal is terminal or has no run', async () => {
    const terminal = await mountPage((d) => ({ ...d, goal: { ...d.goal, status: 'completed' } }))
    try {
      expect(() => getButton(terminal.container, 'Correct goal type')).toThrow()
    } finally { terminal.cleanup() }
    const noRun = await mountPage((d) => ({ ...d, run: null }))
    try {
      expect(() => getButton(noRun.container, 'Correct goal type')).toThrow()
    } finally { noRun.cleanup() }
  })
})

describe('baseline reset interruption', () => {
  it.each([['Mock run', false], ['Mock answer', true]])('Reset availability while %s owns the action', async (label, disabled) => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.healthDebugEnabled = true
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
    try {
      await React.act(async () => getButton(view.container, label).click())
      const reset = getButton(view.container, 'Reset goal — deletes all history')
      expect(reset.getAttribute('disabled') !== null).toBe(disabled)
      if (!disabled) {
        await React.act(async () => reset.click())
        expect(textOf(view.container)).toContain('In-flight baseline results will be discarded')
        const confirm = getButton(view.container, 'Confirm reset')
        await React.act(async () => { confirm.click(); confirm.click() })
        expect(state.resetMutation.mutate).toHaveBeenCalledOnce()
      }
    } finally { view.cleanup() }
  })
})

describe('authoritative baseline run replacement', () => {
  it.each([['Mock run', false], ['Mock answer', false], ['reset', true]])('releases replaced %s ownership while preserving reset', async (owner, stillBusy) => {
    state.activeProjectId = 'project-1'
    state.goalId = 'goal-1'
    state.detail = richDetail
    state.healthDebugEnabled = true
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
    try {
      if (owner === 'reset') {
        await React.act(async () => getButton(view.container, 'Reset goal — deletes all history').click())
        await React.act(async () => getButton(view.container, 'Confirm reset').click())
      } else await React.act(async () => getButton(view.container, owner).click())
      state.detail = { ...richDetail, run: { ...richDetail.run!, id: 'replacement-run' } }
      await view.rerender(() => <OrchestrationPage />)
      expect(getButton(view.container, 'Pause goal').getAttribute('disabled') !== null).toBe(stillBusy)
    } finally { view.cleanup() }
  })
})


describe('SPR188 query freshness integration', () => {
  it.each([{ fetching: true, error: false, cue: 'refreshing' }, { fetching: false, error: true, cue: 'not updating' }, { fetching: false, error: false, cue: 'Data as of' }])('uses one last-success cue for $cue', async ({ fetching, error, cue }) => {
    state.activeProjectId = 'project-1'; state.goalId = 'goal-1'
    state.dataUpdatedAt = Date.parse('2026-10-14T12:00:00Z')
    state.detailFetching = fetching; state.detailError = error
    state.detail = { ...richDetail, supervision: {
      condition: 'unknown', operation: 'No active worker confirmed.', next_action: 'No action', rationale: 'Unknown',
      criterion: null, verified_progress: [], useful_learning: [], accepted_evidence: [], workers: [], waits: [], recovery_history: [], pending_direction: null, budget: {}, transition: null,
      status_line: { current_state: { kind: 'unknown', summary: 'No active worker confirmed.', activities: [] }, attention_summary: null, last_material_event: null },
    } }
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const html = renderToStaticMarkup(<OrchestrationPage />)
    expect(html).toContain('No active worker confirmed.')
    expect(html).toContain(cue)
    expect((html.match(/Data as of/g) ?? [])).toHaveLength(1)
  })
  it('keeps initial failure in the existing error state without fabricated status', async () => {
    state.activeProjectId = 'project-1'; state.goalId = 'goal-1'; state.detail = undefined; state.detailError = true
    const { OrchestrationPage } = await import('./OrchestrationPage')
    const html = renderToStaticMarkup(<OrchestrationPage />)
    expect(html).not.toContain('Current operation')
    expect(html).not.toContain('No active worker confirmed.')
  })
})


it('SPR188 ordinary page forwards a new generic human question to exactly one real queue owner', async () => {
  vi.useFakeTimers()
  state.activeProjectId = 'project-1'; state.goalId = 'goal-1'; state.realQueueForwarding = true
  const statusLine: import('@/lib/types').OrchestrationStatusLine = { current_state: { kind: 'unknown', summary: 'No active worker confirmed.', activities: [] }, attention_summary: null, last_material_event: null }
  state.detail = { ...richDetail, supervision: { condition: 'unknown', operation: 'Unknown', next_action: 'No action', rationale: 'Unknown', criterion: null, verified_progress: [], useful_learning: [], accepted_evidence: [], workers: [], waits: [], recovery_history: [], pending_direction: null, budget: {}, transition: null, status_line: statusLine } }
  const { OrchestrationPage } = await import('./OrchestrationPage')
  const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
  const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
  try {
    await React.act(async () => { vi.advanceTimersByTime(500) })
    expect(textOf(live())).toBe('')
    state.queueDecisions = [pageHumanQuestion()]
    state.detail = { ...state.detail!, supervision: { ...state.detail!.supervision!, status_line: { ...statusLine, last_material_event: { key: 'question:1', kind: 'human_question', summary: 'COMPACT HUMAN SILENT', subject: null, occurred_at: '2026-10-14T12:00:00Z' } } } }
    await view.rerender(() => <OrchestrationPage />)
    await React.act(async () => { vi.advanceTimersByTime(500) })
    expect(textOf(live())).toBe('1 new question needs you.')
    const once = live().textContent
    await view.rerender(() => <OrchestrationPage />)
    await React.act(async () => { vi.advanceTimersByTime(500) })
    expect(live().textContent).toBe(once)
    const originalQuestion = state.queueDecisions[0]
    state.queueDecisions = [{ ...originalQuestion, id: 'human:2', asked_at: '2026-10-14T12:05:00Z' }]
    await view.rerender(() => <OrchestrationPage />)
    expect(textOf(live())).toBe('1 new question needs you.')
    expect(live().textContent).not.toBe(once)
    const replacementOnce = live().textContent
    await view.rerender(() => <OrchestrationPage />)
    expect(live().textContent).toBe(replacementOnce)
    state.queueDecisions = [{ ...state.queueDecisions[0], status: 'answered', decided_at: '2026-10-14T12:10:00Z' }]
    await view.rerender(() => <OrchestrationPage />)
    expect(textOf(live())).toBe('1 question answered.')
    const externalAnswerOnce = live().textContent
    await view.rerender(() => <OrchestrationPage />)
    expect(live().textContent).toBe(externalAnswerOnce)
    state.queueDecisions = [originalQuestion]
    await view.rerender(() => <OrchestrationPage />)
    expect(live().textContent).toBe(externalAnswerOnce)
    state.detail = { ...state.detail!, supervision: { ...state.detail!.supervision!, status_line: { ...statusLine, last_material_event: { key: 'evidence:2', kind: 'accepted_evidence', summary: 'New verification accepted.', subject: null, occurred_at: '2026-10-14T13:00:00Z' } } } }
    await view.rerender(() => <OrchestrationPage />)
    expect(textOf(live())).toBe('New verification accepted.')
    const evidenceOnce = live().textContent
    await view.rerender(() => <OrchestrationPage />)
    expect(live().textContent).toBe(evidenceOnce)
  } finally { view.cleanup(); vi.useRealTimers() }
})

function pageHumanQuestion(): import('@/lib/types').OrchestrationAuthorityDecisionRecord {
  return { id: 'human:1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'runtime:question', title: 'Choose the deployment region', status: 'pending', authority: 'human', authority_agent_id: null, source_process_run_id: null, question: 'Where should Checkout deploy?', context: null, options: [], recommendation: null, consequences: null, selected_option: null, reason: null, decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false, created_warning_id: null, related_gate_id: null, related_action_id: null, asked_at: '2026-10-14T12:00:00Z', decided_at: null, created_at: '2026-10-14T12:00:00Z', updated_at: '2026-10-14T12:00:00Z' }
}


it('SPR188 ordinary page silently establishes initial and goal-switch human baselines', async () => {
  state.activeProjectId = 'project-1'; state.goalId = 'goal-1'; state.realQueueForwarding = true
  state.queueDecisions = [pageHumanQuestion()]; state.detail = richDetail
  const { OrchestrationPage } = await import('./OrchestrationPage')
  const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
  const live = () => descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
  try {
    expect(textOf(live())).toBe('')
    state.goalId = 'goal-2'; state.detail = { ...richDetail, goal: { ...richDetail.goal, id: 'goal-2' }, run: { ...richDetail.run!, goal_id: 'goal-2', id: 'run-2' } }
    state.queueDecisions = [{ ...pageHumanQuestion(), id: 'goal2-question', goal_id: 'goal-2', run_id: 'run-2' }]
    await view.rerender(() => <OrchestrationPage />)
    expect(textOf(live())).toBe('')
  } finally { view.cleanup() }
})


it('SPR188 ordinary page delivers a simultaneous human question and material event without overwriting either owner', async () => {
  state.activeProjectId = 'project-1'; state.goalId = 'goal-1'; state.realQueueForwarding = true
  const statusLine: import('@/lib/types').OrchestrationStatusLine = { current_state: { kind: 'unknown', summary: 'No active worker confirmed.', activities: [] }, attention_summary: null, last_material_event: null }
  state.detail = { ...richDetail, supervision: { condition: 'unknown', operation: 'Unknown', next_action: 'No action', rationale: 'Unknown', criterion: null, verified_progress: [], useful_learning: [], accepted_evidence: [], workers: [], waits: [], recovery_history: [], pending_direction: null, budget: {}, transition: null, status_line: statusLine } }
  const { OrchestrationPage } = await import('./OrchestrationPage')
  const view = await mountWithTestDom(() => <OrchestrationPage />, React.act)
  try {
    state.queueDecisions = [pageHumanQuestion()]
    state.detail = { ...state.detail!, supervision: { ...state.detail!.supervision!, status_line: { ...statusLine, last_material_event: { key: 'material:simultaneous', kind: 'accepted_evidence', summary: 'Checkout verification accepted.', subject: null, occurred_at: '2026-10-14T13:00:00Z' } } } }
    await view.rerender(() => <OrchestrationPage />)
    const live = descendants(view.container).find((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')!
    expect(textOf(live)).toContain('1 new question needs you.')
    expect(textOf(live)).toContain('Checkout verification accepted.')
    const once = live.textContent
    await view.rerender(() => <OrchestrationPage />)
    expect(live.textContent).toBe(once)
  } finally { view.cleanup() }
})
