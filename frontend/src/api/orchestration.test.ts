import { QueryClient, useQueries, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearToken } from '@/lib/api-client'
import type { OrchestrationGoalDetail } from '@/lib/types'
import { useWSStore } from '@/stores/ws'
import * as orchestrationApi from './orchestration'
import {
  acknowledgeOrchestrationWarning,
  answerAgentDefinitionReviewBatch,
  answerOrchestrationDecision,
  authorizeOrchestrationBaseline,
  fetchOrchestrationAgentReviews,
  fetchOrchestrationGoal,
  fetchOrchestrationGoals,
  fetchOrchestrationCheckpoint,
  fetchOrchestrationDecisions,
  fetchOrchestrationHealth,
  fetchOrchestrationMemoryOverview,
  fetchOrchestrationMemorySection,
  fetchOrchestrationProcesses,
  fetchOrchestrationWarnings,
  fetchOrchestrationConversation,
  postOrchestrationGoalCommand,
  putOrchestrationConversationFeedback,
  startOrchestrationGoal,
  postOrchestrationGateOverride,
  createOrchestrationGoal,
  previewGoalTypeCorrection,
  applyGoalTypeCorrection,
  orchestrationBaselineKey,
  orchestrationConversationKey,
  rerunOrchestrationBaselineStep,
  rerunLastOrchestrationBaseline,
  runOrchestrationBaselineStep,
  retryOrchestrationBaselineStep,
  resolveOrchestrationWarning,
  skipOrchestrationProcess,
  submitOrchestrationConversation,
  submitOrchestrationSteering,
  withdrawOrchestrationSteering,
  dismissOrchestrationSteeringProposal,
  stepOrchestrationBaseline,
  syncOrchestrationGoalCache,
  useAcknowledgeOrchestrationWarning,
  useAnswerOrchestrationDecision,
  useAuthorizeOrchestrationBaseline,
  useRerunLastOrchestrationBaseline,
  useResetOrchestrationGoal,
  useStartOrchestrationGoal,
  usePreviewGoalTypeCorrection,
  useApplyGoalTypeCorrection,
  useRerunOrchestrationBaselineStep,
  useRunOrchestrationBaselineStep,
  useRetryOrchestrationBaselineStep,
  useResolveOrchestrationWarning,
  useSkipOrchestrationProcess,
  useOrchestrationConversation,
  useRecordOrchestrationConversationFeedback,
  useSubmitOrchestrationConversation,
  useSubmitOrchestrationSteering,
  useWithdrawOrchestrationSteering,
  useDismissOrchestrationSteeringProposal,
  useStepOrchestrationBaseline,
  useOrchestrationGoal,
  useBaselineDashboard,
  fetchProjectChat,
  submitProjectChat,
  projectChatKey,
  useSubmitProjectChat,
  createOrchestrationAnswerSubmission,
  validateSavedAnswerReceipt,
} from './orchestration'

describe('unified project conversation', () => {
  it('keeps one project query key across navigation and posts explicit typed action binding', async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(okJson({ items: [], actions: [], receipts: [] })))
    vi.stubGlobal('fetch', fetchMock)
    await fetchProjectChat('project-1')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects/project-1/orchestration/chat')
    await submitProjectChat('project-1', { clientRequestId: 'request-1', content: 'yes', contextGoalId: 'goal-alpha', actionId: 'action-alpha' })
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({ client_request_id: 'request-1', content: 'yes', context_goal_id: 'goal-alpha', action_id: 'action-alpha' })
    expect(projectChatKey('project-1')).toEqual(['project-chat', 'project-1'])
  })

  it('refreshes unified history and affected goal state after an action without retrying mutation', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
    const mutation = useSubmitProjectChat('project-1') as unknown as {
      retry: boolean; onSuccess: (data: unknown, input: unknown) => Promise<void>
    }
    expect(mutation.retry).toBe(false)
    await mutation.onSuccess({}, { contextGoalId: 'goal-alpha' })
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: projectChatKey('project-1') })
    expect(invalidateQueries.mock.calls.length).toBeGreaterThanOrEqual(2)
  })
})

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useMutation: vi.fn((options) => options),
  useQuery: vi.fn((options) => options),
  useQueries: vi.fn(({ queries }) => queries),
  useQueryClient: vi.fn(),
}))

vi.mock('@/stores/ws', () => ({
  useWSStore: {
    getState: vi.fn(() => ({ connected: false })),
  },
}))

const detail: OrchestrationGoalDetail = {
  goal: {
    id: 'goal-1',
    project_id: 'project-1',
    objective: 'Ship an evidence-backed release',
    success_criteria: [{ key: 'done', description: 'Done' }],
    orchestrator_context: {},
    constraints: {},
    budget: {},
    goal_type: 'outcome',
    goal_type_source: 'auto',
    goal_type_rationale: 'Single deliverable.',
    planning_mode: 'delivery',
    planning_rationale: 'Plan then deliver.',
    supersedes_goal_id: null,
    parent_goal_id: null,
    status: 'active',
    weight: 'standard',
    weight_overridden_by: null,
    manager_agent_id: null,
    manager_user_id: null,
    authority_model: null,
    created_by_user_id: null,
    created_at: '2026-07-15T00:00:00Z',
    updated_at: '2026-07-15T00:00:00Z',
  },
  run: {
    id: 'run-1',
    goal_id: 'goal-1',
    status: 'running',
    phase: 'baseline',
    condition: 'running',
    event_cursor: null,
    plan_state: {},
    active_blockers: [],
    budget_state: {},
    retry_state: {},
    started_at: '2026-07-15T00:00:00Z',
    completed_at: null,
    created_at: '2026-07-15T00:00:00Z',
    updated_at: '2026-07-15T00:00:00Z',
  },
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
  timeline: [],
  supervision: null,
  contract_fingerprint: null,
  runs: [],
  child_goals: [],
  task_reports: [],
  plan: null,
  effective_authorization: null,
  authorization_wait: null,
}

function okJson(value: unknown) {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => {
  clearToken()
  vi.unstubAllGlobals()
  vi.mocked(useMutation).mockClear()
  vi.mocked(useQuery).mockClear()
  vi.mocked(useQueryClient).mockReset()
})

describe('orchestration API', () => {
  it('SPR230 posts lightweight Pause with its frozen run/version/request identity', async () => {
    const receipt = {
      project_id: 'project-1', goal_id: 'goal-1', run_id: 'run-1', request_id: 'pause-1',
      accepted_at: '2026-10-16T12:00:00Z', accepted_control_version: 4, current_control_version: 4,
      state: 'applied', goal_status: 'paused', run_status: 'paused',
    }
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(okJson(receipt)))
    vi.stubGlobal('fetch', fetchMock)
    const api = orchestrationApi as unknown as {
      pauseOrchestrationControl: (project: string, goal: string, input: {
        runId: string; expectedControlVersion: number; requestId: string
      }) => Promise<unknown>
    }
    const input = { runId: 'run-1', expectedControlVersion: 3, requestId: 'pause-1' }
    await expect(api.pauseOrchestrationControl('project-1', 'goal-1', input)).resolves.toEqual(receipt)
    await api.pauseOrchestrationControl('project-1', 'goal-1', input)
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/projects/project-1/orchestration/goals/goal-1/pause-control',
      '/api/v1/projects/project-1/orchestration/goals/goal-1/pause-control',
    ])
    for (const [, request] of fetchMock.mock.calls) {
      expect(request.method).toBe('POST')
      expect(JSON.parse(request.body)).toEqual({ run_id: 'run-1', expected_control_version: 3, request_id: 'pause-1' })
    }
  })

  it('SPR230 reconciles only the exact Pause receipt, preserving resumed or missing outcomes', async () => {
    const receipt = {
      project_id: 'project-1', goal_id: 'goal-1', run_id: 'run-1', request_id: 'pause-1',
      accepted_at: '2026-10-16T12:00:00Z', accepted_control_version: 4, current_control_version: 5,
      state: 'resumed', goal_status: 'active', run_status: 'running',
    }
    const fetchMock = vi.fn().mockResolvedValueOnce(okJson(receipt)).mockResolvedValueOnce(okJson(null))
    vi.stubGlobal('fetch', fetchMock)
    const api = orchestrationApi as unknown as {
      fetchOrchestrationPauseControl: (project: string, goal: string, requestId: string) => Promise<unknown>
    }
    await expect(api.fetchOrchestrationPauseControl('project-1', 'goal-1', 'pause-1')).resolves.toEqual(receipt)
    await expect(api.fetchOrchestrationPauseControl('project-1', 'goal-1', 'pause-missing')).resolves.toBeNull()
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/projects/project-1/orchestration/goals/goal-1/pause-control/pause-1',
      '/api/v1/projects/project-1/orchestration/goals/goal-1/pause-control/pause-missing',
    ])
  })

  it('SPR231 reads the frozen decision identity directly, including Saved/application receipt', async () => {
    const result = { decision: { id: 'reviewed-decision', status: 'answered' }, process: null,
      answer_receipt: { action_id: 'application-1', decision_ids: ['reviewed-decision'], application_state: 'pending' } }
    const fetchMock = vi.fn().mockResolvedValue(okJson(result))
    vi.stubGlobal('fetch', fetchMock)
    const api = orchestrationApi as unknown as {
      fetchOrchestrationDecision: (project: string, goal: string, decisionId: string) => Promise<unknown>
    }
    await expect(api.fetchOrchestrationDecision('project-1', 'goal-1', 'reviewed-decision')).resolves.toEqual(result)
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/decisions/reviewed-decision')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('SPR230 preserves unknown Pause outcome on transport failure without an automatic second POST', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('response disconnected'))
    vi.stubGlobal('fetch', fetchMock)
    const api = orchestrationApi as unknown as {
      pauseOrchestrationControl: (project: string, goal: string, input: {
        runId: string; expectedControlVersion: number; requestId: string
      }) => Promise<unknown>
    }
    await expect(api.pauseOrchestrationControl('project-1', 'goal-1', {
      runId: 'run-1', expectedControlVersion: 3, requestId: 'pause-1',
    })).rejects.toThrow('response disconnected')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('SPR230/SPR231 propagates exact receipt readback failures instead of inventing confirmed state', async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(new Response(
      JSON.stringify({ detail: 'Readback unavailable' }), { status: 502, headers: { 'Content-Type': 'application/json' } },
    )))
    vi.stubGlobal('fetch', fetchMock)
    const api = orchestrationApi as unknown as {
      fetchOrchestrationPauseControl: (project: string, goal: string, requestId: string) => Promise<unknown>
      fetchOrchestrationDecision: (project: string, goal: string, decisionId: string) => Promise<unknown>
    }
    await expect(api.fetchOrchestrationPauseControl('project-1', 'goal-1', 'pause-1')).rejects.toThrow()
    await expect(api.fetchOrchestrationDecision('project-1', 'goal-1', 'reviewed-decision')).rejects.toThrow()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('preserves form-normalized runtime FEFF text through actual JSON and identical retry', async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(okJson({})))
    vi.stubGlobal('fetch', fetchMock)
    const input = ['project-1', 'goal-1', 'decision-1', 'acknowledge', '\uFEFFhello\uFEFF', undefined,
      'plan:1', 'runtime-1'] as const
    await answerOrchestrationDecision(...input)
    await answerOrchestrationDecision(...input)
    const bodies = fetchMock.mock.calls.map(([, request]) => JSON.parse(request.body))
    expect(bodies).toEqual([{
      selected_option: 'acknowledge', reason: '\uFEFFhello\uFEFF', contract_version: 'plan:1',
    }, {
      selected_option: 'acknowledge', reason: '\uFEFFhello\uFEFF', contract_version: 'plan:1',
    }])
  })

  it('POSTs the answer-application retry route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    await orchestrationApi.retryAnswerApplication('project-1', 'goal-1', 'action-1')
    expect(fetchMock.mock.calls[0][0]).toContain('/projects/project-1/orchestration/goals/goal-1/answer-applications/action-1/retry')
    expect(fetchMock.mock.calls[0][1].method).toBe('POST')
  })

  it('keeps ordinary reason JavaScript trimming unchanged', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    await answerOrchestrationDecision('project-1', 'goal-1', 'decision-1', 'approve', ' \uFEFFhello\uFEFF ')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ selected_option: 'approve', reason: 'hello' })
  })

  it('forwards runtime identity from the mutation input while keeping it out of the backend body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries: vi.fn() } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      mutationFn: (input: { goalId: string; decisionId: string; selectedOption: string; reason: string;
        runtimeIdentity: string; contractVersion: string }) => Promise<unknown>
    }
    await mutation.mutationFn({ goalId: 'goal-1', decisionId: 'decision-1', selectedOption: 'approve',
      reason: '\uFEFFhello\uFEFF', runtimeIdentity: 'runtime-1', contractVersion: 'plan:1' })
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      selected_option: 'approve', reason: '\uFEFFhello\uFEFF', contract_version: 'plan:1',
    })
  })

  it('puts typed answer feedback through the exact response-scoped endpoint', async () => {
    const feedback = {
      feedback_id: 'feedback-1', rating: 'not_helpful' as const, reason: 'missing_context' as const,
      created_at: '2026-09-16T10:00:00Z',
    }
    const fetchMock = vi.fn().mockResolvedValue(okJson(feedback))
    vi.stubGlobal('fetch', fetchMock)

    await expect(putOrchestrationConversationFeedback('project-1', 'goal-1', 'response-1', {
      rating: 'not_helpful', reason: 'missing_context',
    })).resolves.toEqual(feedback)

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation/response-1/feedback',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({ rating: 'not_helpful', reason: 'missing_context' }),
      }),
    )
  })

  it('records feedback without awaiting its conversation refetch', () => {
    const invalidateQueries = vi.fn().mockReturnValue(new Promise(() => undefined))
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
    const mutation = useRecordOrchestrationConversationFeedback('project-1') as unknown as {
      retry: boolean
      onSuccess: (data: unknown, variables: { goalId: string }) => void
    }

    expect(mutation.retry).toBe(false)
    expect(mutation.onSuccess({}, { goalId: 'goal-1' })).toBeUndefined()
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationConversationKey('project-1', 'goal-1'), exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledTimes(1)
  })

  it('loads and submits a conversation through the goal-scoped endpoint', async () => {
    const history = { items: [], total: 0, omitted: 0, allowance: { enabled: true, unlimited: false, limit: 1000, used: 0, remaining: 1000 } }
    const turn = { status: 'completed' }
    const fetchMock = vi.fn().mockResolvedValueOnce(okJson(history)).mockResolvedValueOnce(okJson(turn))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchOrchestrationConversation('project-1', 'goal-1')).resolves.toEqual(history)
    await expect(submitOrchestrationConversation('project-1', 'goal-1', {
      clientRequestId: 'request-1', content: 'Question?',
    })).resolves.toEqual(turn)

    expect(fetchMock).toHaveBeenNthCalledWith(1,
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation', expect.any(Object))
    expect(fetchMock).toHaveBeenNthCalledWith(2,
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation', expect.objectContaining({
        method: 'POST', body: JSON.stringify({ client_request_id: 'request-1', content: 'Question?' }),
      }))
  })

  it('goal detail polling respects WS connection state', () => {
    const goal = useOrchestrationGoal('project-1', 'goal-1') as unknown as {
      refetchInterval: (query: { state: { data?: { run?: Record<string, unknown> } } }) => number | false
      staleTime: number
    }
    const idle = { state: { data: undefined } }
    const baseline = { state: { data: { run: { phase: 'baseline', baseline_authorized: true, status: 'running' } } } }

    // Test disconnected state (default mock)
    expect(goal.refetchInterval(idle)).toBe(10_000)
    expect(goal.staleTime).toBe(10_000)

    // Test connected state
    vi.mocked(useWSStore.getState).mockReturnValue({ connected: true })
    expect(goal.refetchInterval(idle)).toBe(false)
    expect(goal.refetchInterval(baseline)).toBe(2000)
  })

  it('uses a disabled, goal-scoped conversation query that polls only active turns', () => {
    const disabled = useOrchestrationConversation(null, undefined) as unknown as {
      queryKey: unknown[]
      enabled: boolean
      refetchInterval: (query: { state: { data?: { items: Array<{
        status: string
        investigation?: { status: string } | null
      }> } } }) => number | false
    }
    const enabled = useOrchestrationConversation('project-1', 'goal-1') as typeof disabled

    expect(orchestrationConversationKey('project-1', 'goal-1')).toEqual([
      'orchestration-goal', 'project-1', 'goal-1', 'conversation',
    ])
    expect(disabled.enabled).toBe(false)
    expect(disabled.queryKey).toEqual(['orchestration-goal', null, undefined, 'conversation'])
    expect(enabled.enabled).toBe(true)
    for (const status of ['pending', 'running']) {
      expect(enabled.refetchInterval({ state: { data: { items: [{
        status, investigation: { status: 'completed' },
      }] } } })).toBe(2000)
      expect(enabled.refetchInterval({ state: { data: { items: [{
        status: 'completed', investigation: { status },
      }] } } })).toBe(2000)
    }
    expect(enabled.refetchInterval({ state: { data: { items: [
      { status: 'completed', investigation: { status: 'completed' } },
      { status: 'failed', investigation: { status: 'running' } },
    ] } } })).toBe(2000)
    expect(enabled.refetchInterval({ state: { data: { items: [{ status: 'completed' }, { status: 'failed' }] } } })).toBe(false)
    for (const status of ['completed', 'limited', 'failed', 'cancelled', 'unavailable', 'interrupted_unknown']) {
      expect(enabled.refetchInterval({ state: { data: { items: [{
        status: 'completed', investigation: { status },
      }] } } })).toBe(false)
    }
    expect(enabled.refetchInterval({ state: { data: { items: [{ status: 'completed', investigation: null }] } } })).toBe(false)
    expect(enabled.refetchInterval({ state: { data: { items: [] } } })).toBe(false)
    expect(enabled.refetchInterval({ state: {} })).toBe(false)
  })

  it('does not retry submits and invalidates only the matching conversation', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
    const mutation = useSubmitOrchestrationConversation('project-1') as unknown as {
      retry: boolean
      onSuccess: (data: unknown, variables: { goalId: string }) => Promise<void>
    }

    expect(mutation.retry).toBe(false)
    await mutation.onSuccess({}, { goalId: 'goal-1' })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationConversationKey('project-1', 'goal-1'), exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledTimes(1)
  })

  it('posts steering mutations with the exact conversation routes and bodies', async () => {
    const request = { request_id: 'request-1', status: 'pending' }
    const proposal = { proposal_id: 'proposal-1', status: 'dismissed' }
    const fetchMock = vi.fn().mockResolvedValueOnce(okJson(request)).mockResolvedValueOnce(okJson(request)).mockResolvedValueOnce(okJson(proposal))
    vi.stubGlobal('fetch', fetchMock)
    const draft = {
      clientRequestId: 'client-1', directive: 'Prioritize validation.', targetType: 'goal' as const,
      targetId: 'goal-1', scope: 'run' as const, lifetime: 'remaining_current_run' as const,
      impactSummary: 'Applies to unstarted work.', sourceProposalId: null, supersedesRequestId: null,
    }

    await expect(submitOrchestrationSteering('project-1', 'goal-1', draft)).resolves.toEqual(request)
    await expect(withdrawOrchestrationSteering('project-1', 'goal-1', 'request-1')).resolves.toEqual(request)
    await expect(dismissOrchestrationSteeringProposal('project-1', 'goal-1', 'proposal-1')).resolves.toEqual(proposal)

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation/steering',
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation/steering/request-1/withdraw',
      '/api/v1/projects/project-1/orchestration/goals/goal-1/conversation/steering/proposals/proposal-1/dismiss',
    ])
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({
      method: 'POST', body: JSON.stringify({
        client_request_id: 'client-1', directive: 'Prioritize validation.', target_type: 'goal', target_id: 'goal-1',
        scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Applies to unstarted work.',
        source_proposal_id: null, supersedes_request_id: null,
      }),
    }))
  })

  it.each([
    ['submit', useSubmitOrchestrationSteering, { goalId: 'goal-1' }],
    ['withdraw', useWithdrawOrchestrationSteering, { goalId: 'goal-1' }],
    ['dismiss', useDismissOrchestrationSteeringProposal, { goalId: 'goal-1' }],
  ])('does not retry and invalidates only the matching conversation after steering %s', async (_label, useAction, variables) => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
    const mutation = useAction('project-1') as unknown as { retry: boolean; onSuccess: (data: unknown, variables: { goalId: string }) => Promise<void> }

    expect(mutation.retry).toBe(false)
    await mutation.onSuccess({}, variables)
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: orchestrationConversationKey('project-1', 'goal-1'), exact: true })
    expect(invalidateQueries).toHaveBeenCalledTimes(1)
  })

  it('loads project-scoped goal detail', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchOrchestrationGoal('project-1', 'goal-1')).resolves.toEqual(detail)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1',
      expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
    )
  })

  it('loads a cursor page with the fixed page size', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ items: [], next_cursor: null }))
    vi.stubGlobal('fetch', fetchMock)

    await fetchOrchestrationGoals('project-1', 'cursor-2')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals?limit=50&cursor=cursor-2',
      expect.any(Object),
    )
  })

  it('posts lifecycle controls to the project-scoped endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await postOrchestrationGoalCommand('project-1', 'goal-1', 'pause')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/pause',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('starts an orchestration goal through its project-scoped endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await startOrchestrationGoal('project-1', 'goal-1')

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/start',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('creates goals with goal_type auto unless one is given', async () => {
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(okJson(detail)))
    vi.stubGlobal('fetch', fetchMock)
    const base = { objective: 'Ship', success_criteria: [{ description: 'Done' }] }

    await createOrchestrationGoal('project-1', base)
    await createOrchestrationGoal('project-1', { ...base, goal_type: 'roadmap' })

    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects/project-1/orchestration/goals')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).goal_type).toBe('auto')
    expect(JSON.parse(fetchMock.mock.calls[1][1].body).goal_type).toBe('roadmap')
  })

  const fp = 'a'.repeat(64)
  const correction = {
    goalId: 'goal-1',
    requestedGoalType: 'roadmap' as const,
    expectedRunId: 'run-1',
    expectedContractFingerprint: fp,
    rationale: '  Needs a roadmap.  ',
  }
  const correctionBody = {
    requested_goal_type: 'roadmap',
    expected_run_id: 'run-1',
    expected_contract_fingerprint: fp,
    rationale: 'Needs a roadmap.',
  }

  it('previews a goal type correction with a snake_case trimmed body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ preview_fingerprint: 'b'.repeat(64) }))
    vi.stubGlobal('fetch', fetchMock)

    await previewGoalTypeCorrection('project-1', correction)

    const [url, request] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/projects/project-1/orchestration/goals/goal-1/type-preview')
    expect(request).toEqual(expect.objectContaining({ method: 'POST', body: JSON.stringify(correctionBody) }))
  })

  it('applies a goal type correction including the preview fingerprint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await applyGoalTypeCorrection('project-1', { ...correction, previewFingerprint: 'b'.repeat(64) })

    const [url, request] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/v1/projects/project-1/orchestration/goals/goal-1/type')
    expect(JSON.parse(request.body)).toEqual({ ...correctionBody, preview_fingerprint: 'b'.repeat(64) })
  })

  it('rejects a stale correction with the 409 status and detail', async () => {
    const staleDetail = { conflict: 'stale_preview', message: 'Preview is stale', preview: {} }
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: staleDetail }), {
      status: 409,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(
      applyGoalTypeCorrection('project-1', { ...correction, previewFingerprint: 'b'.repeat(64) }),
    ).rejects.toMatchObject({ status: 409, detail: staleDetail })
  })

  it('does not retry the correction preview', () => {
    const mutation = usePreviewGoalTypeCorrection('project-1') as unknown as { retry: boolean }
    expect(mutation.retry).toBe(false)
  })

  it('apply correction syncs detail and refreshes list and Baseline without refetching detail', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)

    const mutation = useApplyGoalTypeCorrection('project-1') as unknown as {
      onSuccess: (detail: OrchestrationGoalDetail, variables: { goalId: string }) => Promise<void>
    }
    await mutation.onSuccess(detail, { goalId: 'goal-1' })

    expect(setQueryData).toHaveBeenCalledWith(['orchestration-goal', 'project-1', 'goal-1'], detail)
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goals', 'project-1'],
    })
    expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: orchestrationBaselineKey('project-1', 'goal-1') })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
  })

  it('posts the gate override decision reason and metadata', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await postOrchestrationGateOverride('project-1', {
      goalId: 'goal-1',
      gateId: 'gate-1',
      decision: 'accept',
      reason: 'Operator reviewed the release evidence.',
    })

    const [, request] = fetchMock.mock.calls[0]
    expect(request).toEqual(expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        gate_id: 'gate-1',
        decision: 'accept',
        reason: 'Operator reviewed the release evidence.',
        evidence_metadata: {},
      }),
    }))
  })

  it.each([
    ['memory overview', () => fetchOrchestrationMemoryOverview('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/memory'],
    ['memory section', () => fetchOrchestrationMemorySection('project-1', 'goal-1', 'agent-review'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/memory/agent-review'],
    ['processes', () => fetchOrchestrationProcesses('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/processes'],
    ['agent reviews', () => fetchOrchestrationAgentReviews('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/agent-reviews'],
    ['warnings', () => fetchOrchestrationWarnings('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/warnings'],
    ['decisions', () => fetchOrchestrationDecisions('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/decisions'],
    ['checkpoint', () => fetchOrchestrationCheckpoint('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/decisions/checkpoint'],
  ])('loads Baseline %s', async (_label, fetchResource, expectedUrl) => {
    const fetchMock = vi.fn().mockResolvedValue(okJson([]))
    vi.stubGlobal('fetch', fetchMock)

    await fetchResource()

    expect(fetchMock).toHaveBeenCalledWith(
      expectedUrl,
      expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
    )
  })

  it('loads orchestration health with the debug flag', async () => {
    const health = {
      consumers: { events: { running: true, active_connections: 3 } },
      active_sessions: 2,
      event_bus_mode: 'in_process',
      event_log_total: 42,
      debug_enabled: true,
    }
    const fetchMock = vi.fn().mockResolvedValue(okJson(health))
    vi.stubGlobal('fetch', fetchMock)

    await expect(fetchOrchestrationHealth()).resolves.toEqual(
      expect.objectContaining({ debug_enabled: true }),
    )
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/orchestration/health',
      expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
    )
  })

  it.each([
    ['skip process',
      () => skipOrchestrationProcess('project-1', 'goal-1', 'agent_definition_review', '  Accepted operational risk.  '),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/processes/agent_definition_review/skip',
      { reason: 'Accepted operational risk.' }],
    ['acknowledge warning',
      () => acknowledgeOrchestrationWarning('project-1', 'goal-1', 'warning-1', '  Accepted operational risk.  '),
      '/api/v1/projects/project-1/orchestration/warnings/warning-1/acknowledge',
      { reason: 'Accepted operational risk.' }],
    ['resolve warning',
      () => resolveOrchestrationWarning('project-1', 'goal-1', 'warning-1', '  Accepted operational risk.  '),
      '/api/v1/projects/project-1/orchestration/warnings/warning-1/resolve',
      { reason: 'Accepted operational risk.' }],
    ['answer decision',
      () => answerOrchestrationDecision(
        'project-1', 'goal-1', 'decision-1', 'approve_with_documented_gaps',
        '  The human will validate independently.  ',
      ),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/decisions/decision-1/answer',
      {
        selected_option: 'approve_with_documented_gaps',
        reason: 'The human will validate independently.',
      }],
    ['step baseline process',
      () => stepOrchestrationBaseline('project-1', 'goal-1', 'goal_definition'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/debug/baseline/step',
      { process_type: 'goal_definition' }],
    ['rerun last baseline process',
      () => rerunLastOrchestrationBaseline('project-1', 'goal-1'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/debug/baseline/rerun-last',
      {}],
    ['rerun selected baseline process',
      () => rerunLastOrchestrationBaseline('project-1', 'goal-1', 'agent_definition_review'),
      '/api/v1/projects/project-1/orchestration/goals/goal-1/debug/baseline/rerun-last',
      { process_type: 'agent_definition_review' }],
  ])('posts Baseline %s with its exact route and payload', async (_label, postResource, expectedUrl, body) => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)

    await postResource()

    const [url, request] = fetchMock.mock.calls[0]
    expect(url).toBe(expectedUrl)
    expect(request).toEqual(expect.objectContaining({
      method: 'POST',
      body: JSON.stringify(body),
    }))
  })

  it('sends the runtime contract version through the answer mutation', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    vi.mocked(useQueryClient).mockReturnValue({} as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      mutationFn: (variables: {
        goalId: string; decisionId: string; selectedOption: string; reason: string; contractVersion: string
      }) => Promise<unknown>
    }
    await mutation.mutationFn({
      goalId: 'goal-1', decisionId: 'decision-1', selectedOption: 'approve', reason: '', contractVersion: 'plan:1',
    })
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      selected_option: 'approve', reason: null, contract_version: 'plan:1',
    })
  })

  it('sends null for a blank optional decision-answer reason', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)

    await answerOrchestrationDecision('project-1', 'goal-1', 'decision-1', 'approve', '  ')

    const [, request] = fetchMock.mock.calls[0]
    expect(request).toEqual(expect.objectContaining({
      body: JSON.stringify({ selected_option: 'approve', reason: null }),
    }))
  })

  it('posts all agent-definition review answers to the batch route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)

    await answerAgentDefinitionReviewBatch('project-1', 'goal-1', [
      { decisionId: 'decision-a', selectedOption: 'approve', reason: '' },
      { decisionId: 'decision-b', selectedOption: 'edit', reason: '', editedDescription: 'Description', editedPersona: 'Persona' },
    ])

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/decisions/agent-definition-review/batch-answer',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ answers: [
        { decision_id: 'decision-a', selected_option: 'approve', reason: null },
        { decision_id: 'decision-b', selected_option: 'edit', reason: null,
          edited_description: 'Description', edited_persona: 'Persona' },
      ] }) }),
    )
  })

  it.each([
    ['runs a baseline process', runOrchestrationBaselineStep,
      '/api/v1/projects/project-1/orchestration/goals/goal-1/baseline/step', 'step'],
    ['reruns a baseline process', rerunOrchestrationBaselineStep,
      '/api/v1/projects/project-1/orchestration/goals/goal-1/baseline/rerun', 'rerun_last'],
  ])('%s with its exact route, payload, and response', async (_label, run, expectedUrl, action) => {
    const result = {
      action,
      goal_id: 'goal-1',
      run_id: 'run-1',
      process_type: 'goal_definition',
      process: { id: 'process-1' },
    } as const
    const fetchMock = vi.fn().mockResolvedValue(okJson(result))
    vi.stubGlobal('fetch', fetchMock)

    await expect(run('project-1', 'goal-1', 'goal_definition')).resolves.toEqual(result)

    expect(fetchMock).toHaveBeenCalledWith(
      expectedUrl,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ process_type: 'goal_definition' }),
      }),
    )
  })

  it('authorizes the baseline with its exact route and no body payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(detail))
    vi.stubGlobal('fetch', fetchMock)

    await expect(authorizeOrchestrationBaseline('project-1', 'goal-1')).resolves.toEqual(detail)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/baseline/authorize',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('retries a failed baseline LM request with its exact route and payload', async () => {
    const result = { action: 'retry', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition', process: { id: 'process-1' } } as const
    const fetchMock = vi.fn().mockResolvedValue(okJson(result))
    vi.stubGlobal('fetch', fetchMock)

    await expect(retryOrchestrationBaselineStep('project-1', 'goal-1', 'goal_definition')).resolves.toEqual(result)
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/projects/project-1/orchestration/goals/goal-1/baseline/retry',
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ process_type: 'goal_definition' }) }),
    )
  })

  it.each([
    ['skip', useSkipOrchestrationProcess, { goalId: 'goal-1', processType: 'agent_definition_review', reason: 'reason' }],
    ['acknowledge', useAcknowledgeOrchestrationWarning, { goalId: 'goal-1', warningId: 'warning-1', reason: 'reason' }],
    ['resolve', useResolveOrchestrationWarning, { goalId: 'goal-1', warningId: 'warning-1', reason: 'reason' }],
    ['step baseline', useStepOrchestrationBaseline, { goalId: 'goal-1', processType: 'goal_definition' }],
    ['rerun last baseline', useRerunLastOrchestrationBaseline, { goalId: 'goal-1' }],
    ['rerun selected baseline', useRerunLastOrchestrationBaseline, { goalId: 'goal-1', processType: 'agent_definition_review' }],
    ['run baseline', useRunOrchestrationBaselineStep, { goalId: 'goal-1', processType: 'goal_definition' }],
    ['rerun baseline', useRerunOrchestrationBaselineStep, { goalId: 'goal-1', processType: 'goal_definition' }],
    ['retry baseline', useRetryOrchestrationBaselineStep, { goalId: 'goal-1', processType: 'goal_definition' }],
    ['authorize baseline', useAuthorizeOrchestrationBaseline, { goalId: 'goal-1' }],
  ])('invalidates exact goal detail and Baseline subtree after Baseline %s', async (_label, useAction, variables) => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)

    const mutation = useAction('project-1') as unknown as {
      onSuccess?: (data: unknown, variables: typeof variables) => Promise<void>
      onSettled?: (data: unknown, error: unknown, variables: typeof variables) => Promise<void>
    }
    if (mutation.onSettled) await mutation.onSettled({}, null, variables)
    else await mutation.onSuccess!({}, variables)

    expect(invalidateQueries).toHaveBeenNthCalledWith(1, {
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenNthCalledWith(2, {
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
  })

  it('reset goal syncs detail and refreshes list and Baseline without refetching detail', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)

    const mutation = useResetOrchestrationGoal('project-1') as unknown as {
      onSuccess: (detail: unknown, variables: { goalId: string }) => Promise<void>
    }
    const detail = { goal: { id: 'goal-1' } }
    await mutation.onSuccess(detail, { goalId: 'goal-1' })

    expect(setQueryData).toHaveBeenCalledWith(['orchestration-goal', 'project-1', 'goal-1'], detail)
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goals', 'project-1'],
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
  })

  it('start work syncs the returned detail into the exact goal cache', () => {
    const setQueryData = vi.fn()
    const invalidateQueries = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ setQueryData, invalidateQueries } as never)

    const mutation = useStartOrchestrationGoal('project-1') as unknown as {
      onSuccess: (detail: OrchestrationGoalDetail, variables: { goalId: string }) => void
    }
    mutation.onSuccess(detail, { goalId: 'goal-1' })

    expect(setQueryData).toHaveBeenCalledWith(['orchestration-goal', 'project-1', 'goal-1'], detail)
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goals', 'project-1'],
    })
  })

  it('uses the Baseline cache-key segment', () => {
    expect(orchestrationBaselineKey('project-1', 'goal-1')).toEqual([
      'orchestration-goal', 'project-1', 'goal-1', 'baseline',
    ])
  })

  it('reconciles an answered decision locally, then invalidates goal detail and Baseline subtree', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      acceptResult: (goalId: string, data: unknown, expected: unknown) => boolean
    }
    const decision = { id: 'decision-1', goal_id: 'goal-1', source_process_run_id: 'process-1', run_id: 'run-1',
      decision_key: 'approval', options: ['approve'], authority: 'human', authority_agent_id: null, contract_version: null }
    const expected = createOrchestrationAnswerSubmission([decision as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'approve', reason: 'reason',
    }])
    const result = { decision, process: null, answer_receipt: { action_id: 'application-1', decision_ids: ['decision-1'],
      actor_user_id: 'user-1', run_id: 'run-1', source_process_run_id: 'process-1', submission: expected,
      contracts: expected.contracts, application_state: 'pending', reason: null } }
    expect(mutation.acceptResult('goal-1', result, expected)).toBe(true)

    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
    expect(setQueryData).toHaveBeenCalled()
  })

  it('SPR212 leaves runtime answer cache untouched until the component validates the receipt', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      onSuccess?: (data: unknown, variables: {
        goalId: string
        decisionId: string
        selectedOption: string
        reason: string
        runtimeIdentity: string
      }) => Promise<void> | void
    }

    await mutation.onSuccess?.(
      {
        decision: {
          id: 'unrelated-or-malformed-runtime-receipt',
          source_process_run_id: null,
          status: 'answered',
        },
        process: null,
        continuation_applied: false,
      },
      {
        goalId: 'goal-1',
        decisionId: 'runtime-q1',
        selectedOption: 'approve',
        reason: 'reviewed reason',
        runtimeIdentity: 'runtime-q1:v1',
      },
    )

    expect(setQueryData).not.toHaveBeenCalled()
    expect(invalidateQueries).not.toHaveBeenCalled()
  })

  it('SPR212 commits an accepted runtime receipt through the normal goal and Baseline cache boundary', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const answer = useAnswerOrchestrationDecision('project-1') as unknown as {
      acceptRuntimeResult: (goalId: string, result: unknown) => Promise<void>
    }
    const accepted = { decision: { id: 'runtime-q1', source_process_run_id: null }, process: null, continuation_applied: false }

    await answer.acceptRuntimeResult('goal-1', accepted)

    expect(setQueryData).toHaveBeenCalledWith(
      ['orchestration-goal', 'project-1', 'goal-1', 'decision', 'runtime-q1'], accepted,
    )
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
  })

  it('SPR231 leaves malformed or stale ordinary HTTP 200 receipts inert until validated acceptance', () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      onSuccess?: (data: unknown, variables: unknown) => unknown
      acceptResult: (goalId: string, data: unknown, expected: unknown) => boolean
    }
    const expected = createOrchestrationAnswerSubmission([{ id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1',
      source_process_run_id: 'process-1', decision_key: 'approval', options: ['approve'], authority: 'human' } as never],
    'user-1', [{ decisionId: 'decision-1', selectedOption: 'approve', reason: 'Reviewed.' }])
    for (const returned of [
      { decision: { id: 'decision-1', status: 'answered' }, answer_receipt: null },
      { decision: { id: 'other-goal-question', status: 'answered' }, answer_receipt: { actor_user_id: 'other-user' } },
    ]) {
      mutation.onSuccess?.(returned, { goalId: 'goal-1', decisionId: 'decision-1', selectedOption: 'approve', reason: 'Reviewed.' })
      expect(mutation.acceptResult('goal-1', returned, expected)).toBe(false)
    }
    expect(setQueryData).not.toHaveBeenCalled()
    expect(invalidateQueries).not.toHaveBeenCalled()
  })

  it('SPR231 publishes validated Saved immediately despite a failing background refresh', async () => {
    const invalidateQueries = vi.fn().mockRejectedValue(new Error('background refresh unavailable'))
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      acceptResult: (goalId: string, data: unknown, expected: unknown) => boolean
    }
    const decision = { id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', source_process_run_id: 'process-1',
      decision_key: 'approval', options: ['approve'], authority: 'human', authority_agent_id: null,
      contract_version: null, status: 'answered', selected_option: 'approve', reason: 'Reviewed.' }
    const expected = createOrchestrationAnswerSubmission([decision as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'approve', reason: 'Reviewed.',
    }])
    const result = { decision, process: null, answer_receipt: { action_id: 'application-1', decision_ids: ['decision-1'],
      actor_user_id: 'user-1', source_process_run_id: 'process-1', run_id: 'run-1', submission: expected,
      contracts: expected.contracts, application_state: 'pending', reason: null } }
    expect(validateSavedAnswerReceipt(result.answer_receipt, expected)).toBe(true)
    expect(mutation.acceptResult('goal-1', result, expected)).toBe(true)
    expect(setQueryData).toHaveBeenCalledWith(['orchestration-goal', 'project-1', 'goal-1', 'decision', 'decision-1'], result)
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
    await Promise.resolve()
    await Promise.resolve()
    expect(setQueryData).toHaveBeenCalledTimes(3)
  })

  it('replaces exact detail state and invalidates the project list after a control', async () => {
    const queryClient = new QueryClient()
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue()

    syncOrchestrationGoalCache(queryClient, 'project-1', 'goal-1', detail)

    expect(queryClient.getQueryData(['orchestration-goal', 'project-1', 'goal-1'])).toEqual(detail)
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: ['orchestration-goals', 'project-1'],
    })
  })
})

describe('SPR231 saved answer receipt identity', () => {
  const reviewed = {
    id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', source_process_run_id: 'process-1',
    decision_key: 'goal_definition:approval', options: ['approve', 'request_changes'],
    authority: 'human', authority_agent_id: null, contract_version: 'review:1',
  }
  const expected = {
    kind: 'single' as const, actor_user_id: 'user-1', decision_ids: ['decision-1'],
    source_process_run_id: 'process-1', run_id: 'run-1',
    answers: [{ decision_id: 'decision-1', selected_option: 'request_changes', reason: 'Keep the reviewed scope.',
      contract_version: 'review:1', edited_agent: null }],
    contracts: [{ id: 'decision-1', key: 'goal_definition:approval', options: ['approve', 'request_changes'],
      authority: 'human', authority_agent_id: '', source_process_run_id: 'process-1', run_id: 'run-1',
      contract_version: 'review:1' }],
  }
  const receipt = () => ({
    action_id: 'application-1', actor_user_id: 'user-1', decision_ids: ['decision-1'],
    source_process_run_id: 'process-1', run_id: 'run-1', submission: structuredClone(expected),
    contracts: structuredClone(expected.contracts), application_state: 'pending', reason: null,
  })

  it('builds the expected payload from the prewrite actor and reviewed contract rather than returned receipt', () => {
    expect(createOrchestrationAnswerSubmission([reviewed as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'request_changes', reason: '  Keep the reviewed scope.  ',
      contractVersion: 'review:1',
    }])).toEqual(expected)
  })

  it.each(['pending', 'applied', 'blocked', 'failed'])('confirms Saved independently of application %s', (state) => {
    expect(validateSavedAnswerReceipt({ ...receipt(), application_state: state }, expected)).toBe(true)
  })

  it.each([
    ['actor', (value: ReturnType<typeof receipt>) => { value.actor_user_id = 'another-user' }],
    ['decision identity', (value: ReturnType<typeof receipt>) => { value.decision_ids = ['newest-question'] }],
    ['source', (value: ReturnType<typeof receipt>) => { value.source_process_run_id = 'replacement-process' }],
    ['run', (value: ReturnType<typeof receipt>) => { value.run_id = 'replacement-run' }],
    ['selected option', (value: ReturnType<typeof receipt>) => { value.submission.answers[0].selected_option = 'approve' }],
    ['reason', (value: ReturnType<typeof receipt>) => { value.submission.answers[0].reason = 'Different text.' }],
    ['submission actor', (value: ReturnType<typeof receipt>) => { value.submission.actor_user_id = 'another-user' }],
    ['contract', (value: ReturnType<typeof receipt>) => { value.contracts[0].contract_version = 'review:2' }],
    ['submitted contract', (value: ReturnType<typeof receipt>) => { value.submission.contracts[0].options = ['approve'] }],
  ])('rejects a mismatching %s even when HTTP succeeded', (_label, mutate) => {
    const returned = receipt()
    mutate(returned)
    expect(validateSavedAnswerReceipt(returned, expected)).toBe(false)
  })

  it('does not accept a missing receipt or infer the expected actor from returned metadata', () => {
    expect(validateSavedAnswerReceipt(null, expected)).toBe(false)
    expect(validateSavedAnswerReceipt(undefined, expected)).toBe(false)
    expect(validateSavedAnswerReceipt(receipt(), { ...expected, actor_user_id: '' })).toBe(false)
  })

  it('normalizes typed batch edits exactly while preserving frozen member order', () => {
    const second = { ...reviewed, id: 'decision-2', decision_key: 'agent_definition_review:proposal:two' }
    const submission = createOrchestrationAnswerSubmission([reviewed as never, second as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'edit', reason: '',
      editedDescription: '  Define café scope.  ', editedPersona: '  Report evidence.  ',
    }, { decisionId: 'decision-2', selectedOption: 'approve', reason: ' Reviewed. ' }], 'batch')
    expect(submission.answers).toEqual([
      { decision_id: 'decision-1', selected_option: 'edit',
        reason: '{"description": "Define caf\\u00e9 scope.", "persona": "Report evidence."}' },
      { decision_id: 'decision-2', selected_option: 'approve', reason: 'Reviewed.' },
    ])
    const returned = { ...receipt(), submission, decision_ids: ['decision-1', 'decision-2'], contracts: submission.contracts }
    expect(validateSavedAnswerReceipt(returned, submission)).toBe(true)
    expect(validateSavedAnswerReceipt({ ...returned, decision_ids: ['decision-2', 'decision-1'] }, submission)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...returned, submission: { ...submission, answers: [...submission.answers].reverse() } }, submission)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...returned, submission: { ...submission, decision_ids: ['decision-1'] } }, submission)).toBe(false)
  })

  it.each([
    ['FEFF retained', '\uFEFFscope\uFEFF', '\\ufeffscope\\ufeff'],
    ['NEL stripped', '\u0085scope\u0085', 'scope'],
    ['Python control separators stripped', '\u001c\u001d\u001escope\u001f', 'scope'],
  ])('matches backend Python strip for typed batch edits: %s', (_label, raw, normalized) => {
    const submission = createOrchestrationAnswerSubmission([reviewed as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'edit', reason: '',
      editedDescription: raw, editedPersona: raw,
    }], 'batch')
    expect(submission.answers[0].reason).toBe(`{"description": "${normalized}", "persona": "${normalized}"}`)
  })

  it('binds the entire typed hierarchy agent edit without including newly generated agent identity', () => {
    const editedAgent = { name: 'Reviewed helper', role: 'specialist', provider: 'openai', model: 'gpt-4o-mini',
      description: 'Scoped support', system_prompt: 'Report concrete evidence.', adapter_type: 'api' as const,
      cli_runtime: null, capabilities: ['implementation'], config: { temperature: 0 } }
    const submission = createOrchestrationAnswerSubmission([reviewed as never], 'user-1', [{
      decisionId: 'decision-1', selectedOption: 'edit', reason: 'Reviewed.', editedAgent,
    }])
    expect(submission.answers[0].edited_agent).toEqual(editedAgent)
    expect(JSON.stringify(submission)).not.toContain('created_agent_id')
    const returned = { ...receipt(), submission, contracts: submission.contracts }
    expect(validateSavedAnswerReceipt(returned, submission)).toBe(true)
    const changed = structuredClone(returned)
    changed.submission.answers[0].edited_agent!.name = 'Changed helper'
    expect(validateSavedAnswerReceipt(changed, submission)).toBe(false)
  })

  it('permits nullable action/run only for an exact source-free applied single answer', () => {
    const runless = createOrchestrationAnswerSubmission([
      { ...reviewed, run_id: null, source_process_run_id: null } as never,
    ], 'user-1', [{ decisionId: 'decision-1', selectedOption: 'approve', reason: '' }])
    const returned = { ...receipt(), action_id: null, run_id: null, source_process_run_id: null,
      application_state: 'applied', submission: runless, contracts: runless.contracts }
    expect(validateSavedAnswerReceipt(returned, runless)).toBe(true)
    expect(validateSavedAnswerReceipt({ ...returned, application_state: 'pending' }, runless)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...returned, action_id: 'invented-action' }, runless)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...returned, source_process_run_id: 'process-1' }, runless)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...returned, submission: { ...runless, kind: 'batch' } }, runless)).toBe(false)
    expect(validateSavedAnswerReceipt({ ...receipt(), action_id: null }, expected)).toBe(false)
  })
})

describe('baseline progress polling', () => {
  it('polls all dashboard data every two seconds during a pending action even with connected websocket', () => {
    vi.mocked(useWSStore.getState).mockReturnValue({ connected: true } as never)
    vi.mocked(useQueryClient).mockReturnValue(new QueryClient())
    const dashboard = useBaselineDashboard('project-1', 'goal-1', true)
    for (const query of Object.values(dashboard)) {
      const options = query as unknown as { refetchInterval: (query: unknown) => number | false }
      expect(options.refetchInterval({ state: { data: [] } })).toBe(2000)
    }
  })
  it('polls persisted current running processes but stops for superseded and completed ones', () => {
    const client = new QueryClient()
    vi.mocked(useQueryClient).mockReturnValue(client)
    const dashboard = useBaselineDashboard('project-1', 'goal-1')
    const interval = (dashboard.processes as unknown as { refetchInterval: (query: unknown) => number | false }).refetchInterval
    expect(interval({ state: { data: [{ status: 'running', superseded_by_id: null }] } })).toBe(2000)
    expect(interval({ state: { data: [{ status: 'running', superseded_by_id: 'new' }] } })).toBe(false)
    expect(interval({ state: { data: [{ status: 'complete', superseded_by_id: null }] } })).toBe(false)
    client.setQueryData([...orchestrationBaselineKey('project-1', 'goal-1'), 'processes'], [{ status: 'running', superseded_by_id: null }])
    expect((dashboard.memory as unknown as { refetchInterval: () => number | false }).refetchInterval()).toBe(2000)
  })
})

describe('baseline mutation progress refresh', () => {
  it.each([useRunOrchestrationBaselineStep, useRerunOrchestrationBaselineStep, useRetryOrchestrationBaselineStep, useAuthorizeOrchestrationBaseline])('refreshes on submission and every settlement for %s', async (hook) => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
    const mutation = hook('project-1') as unknown as { onMutate: (variables: { goalId: string }) => Promise<void>; onSettled: (data: unknown, error: unknown, variables: { goalId: string }) => Promise<void> }
    await mutation.onMutate({ goalId: 'goal-1' })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
    invalidateQueries.mockClear()
    await mutation.onSettled(undefined, new Error('cancelled'), { goalId: 'goal-1' })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
    invalidateQueries.mockClear()
    await mutation.onSettled({}, null, { goalId: 'goal-1' })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
  })
})

it('launches both submission refreshes without waiting for network requests', () => {
  const invalidateQueries = vi.fn().mockReturnValue(new Promise(() => undefined))
  vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries } as never)
  const mutation = useRunOrchestrationBaselineStep('project-1') as unknown as { onMutate: (variables: { goalId: string }) => unknown }
  expect(mutation.onMutate({ goalId: 'goal-1' })).toBeUndefined()
  expect(invalidateQueries).toHaveBeenCalledTimes(2)
})

describe('blocker recovery proposals', () => {
  const recovery = {
    kind: 'recovery_proposal_v1', proposal_id: 'p-1', version: 1, status: 'pending_approval',
    problem: 'Nobody can review the work.', general_updates: ['Add a reviewer'], after_approval: 'Blocked work resumes.',
    global_effect: true, exact_details: { agent: 'reviewer' }, evidence: null, error: null,
    source_blocker_id: 'b-1', source_action_id: null, source_decision_id: 'decision-1',
  }

  it('returns the typed recovery field untouched from the decisions list (answered decisions come from GET /decisions)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson([{ id: 'decision-1', status: 'answered', context: null, recovery }]))
    vi.stubGlobal('fetch', fetchMock)
    const decisions = await fetchOrchestrationDecisions('project-1', 'goal-1')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects/project-1/orchestration/goals/goal-1/decisions')
    expect(decisions[0]).toHaveProperty('recovery', recovery)
  })

  it('posts approve_fix with no reason and no edited agent', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    await answerOrchestrationDecision('project-1', 'goal-1', 'decision-1', 'approve_fix', '')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ selected_option: 'approve_fix', reason: null })
  })

  it('posts reject with the trimmed reason', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({}))
    vi.stubGlobal('fetch', fetchMock)
    await answerOrchestrationDecision('project-1', 'goal-1', 'decision-1', 'reject', '  Too broad.  ')
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ selected_option: 'reject', reason: 'Too broad.' })
  })

  it('answer cache sync replaces the decision with the server record including its new recovery status', async () => {
    const invalidateQueries = vi.fn().mockResolvedValue(undefined)
    const setQueryData = vi.fn()
    vi.mocked(useQueryClient).mockReturnValue({ invalidateQueries, setQueryData } as never)
    const mutation = useAnswerOrchestrationDecision('project-1') as unknown as {
      acceptRuntimeResult: (goalId: string, data: unknown) => void
    }
    const answered = { id: 'decision-1', source_process_run_id: null, recovery: { ...recovery, status: 'applying' } }
    mutation.acceptRuntimeResult('goal-1', { decision: answered, process: null })
    const decisionsUpdater = setQueryData.mock.calls.find(([key]) =>
      JSON.stringify(key) === JSON.stringify([...orchestrationBaselineKey('project-1', 'goal-1'), 'decisions']),
    )?.[1] as (value: unknown) => unknown
    expect(decisionsUpdater([{ id: 'decision-1', recovery }, { id: 'other' }])).toEqual([answered, { id: 'other' }])
  })
})
