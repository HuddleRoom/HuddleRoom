import { QueryClient, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { clearToken } from '@/lib/api-client'
import type { OrchestrationGoalDetail } from '@/lib/types'
import { useWSStore } from '@/stores/ws'
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
} from './orchestration'

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useMutation: vi.fn((options) => options),
  useQuery: vi.fn((options) => options),
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
    supersedes_goal_id: null,
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
    const history = { items: [], total: 0, omitted: 0, allowance: { enabled: true, limit: 1000, used: 0, remaining: 1000 } }
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
      refetchInterval: () => number | false
      staleTime: number
    }

    // Test disconnected state (default mock)
    expect(goal.refetchInterval()).toBe(10_000)
    expect(goal.staleTime).toBe(10_000)

    // Test connected state
    vi.mocked(useWSStore.getState).mockReturnValue({ connected: true })
    expect(goal.refetchInterval()).toBe(false)
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
      onSuccess: (data: unknown, variables: typeof variables) => Promise<void>
    }
    await mutation.onSuccess({}, variables)

    expect(invalidateQueries).toHaveBeenNthCalledWith(1, {
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenNthCalledWith(2, {
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
    expect(invalidateQueries).toHaveBeenCalledTimes(2)
  })

  it('reset goal syncs returned detail and invalidates the Baseline subtree', async () => {
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
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
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
      onSuccess: (data: unknown, variables: {
        goalId: string
        decisionId: string
        selectedOption: string
        reason: string
      }) => Promise<void>
    }

    await mutation.onSuccess(
      { decision: { id: 'decision-1', source_process_run_id: 'process-1' }, process: null },
      { goalId: 'goal-1', decisionId: 'decision-1', selectedOption: 'approve', reason: 'reason' },
    )

    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ['orchestration-goal', 'project-1', 'goal-1'],
      exact: true,
    })
    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: orchestrationBaselineKey('project-1', 'goal-1'),
    })
    expect(setQueryData).toHaveBeenCalled()
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
