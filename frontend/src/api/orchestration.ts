import {
  type QueryClient,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
  useQueries,
} from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'
import type { AgentCreatePayload } from '@/api/agents'
import type {
  CursorPage,
  OrchestrationGateOverrideInput,
  OrchestrationGoal,
  OrchestrationGoalCommand,
  OrchestrationGoalCreateInput,
  OrchestrationGoalDetail,
  OrchestrationMemoryOverview,
  OrchestrationMemorySection,
  OrchestrationProcessRunRecord,
  OrchestrationAgentReviewRecord,
  OrchestrationWarningRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationDecisionAnswerResult,
  OrchestrationCheckpoint,
  OrchestrationConversationHistory,
  OrchestrationConversationFeedback,
  OrchestrationConversationFeedbackInput,
  OrchestrationConversationTurn,
  OrchestrationSteeringProposal,
  OrchestrationSteeringRequest,
  OrchestrationSteeringScope,
  OrchestrationSteeringLifetime,
  OrchestrationSteeringTargetType,
  OrchestrationBaselineProcessType,
  OrchestrationHealth,
  OrchestrationDebugActionResult,
  ProjectAdvisorHistory,
  ProjectAdvisorTurn,
} from '@/lib/types'

export interface AgentDefinitionReviewBatchAnswerInput {
  decisionId: string
  selectedOption: string
  reason: string
  editedDescription?: string
  editedPersona?: string
}

export interface AgentDefinitionReviewBatchAnswerResult {
  decisions: OrchestrationAuthorityDecisionRecord[]
  process: OrchestrationProcessRunRecord
}

const root = (projectId: string) =>
  `/api/v1/projects/${projectId}/orchestration/goals`

const advisorRoot = (projectId: string) =>
  `/api/v1/projects/${projectId}/orchestration/conversation`

export const projectAdvisorKey = (projectId: string | null) => ['project-advisor', projectId] as const

export const orchestrationBaselineKey = (
  projectId: string | null,
  goalId: string | undefined,
) => ['orchestration-goal', projectId, goalId, 'baseline'] as const

export const orchestrationConversationKey = (
  projectId: string | null,
  goalId: string | undefined,
) => ['orchestration-goal', projectId, goalId, 'conversation'] as const

export function fetchOrchestrationGoals(
  projectId: string,
  cursor?: string | null,
) {
  const params = new URLSearchParams({ limit: '50' })
  if (cursor) params.set('cursor', cursor)
  return apiFetch<CursorPage<OrchestrationGoal>>(`${root(projectId)}?${params}`)
}

export function fetchOrchestrationGoal(projectId: string, goalId: string) {
  return apiFetch<OrchestrationGoalDetail>(`${root(projectId)}/${goalId}`)
}

export function fetchOrchestrationConversation(projectId: string, goalId: string) {
  return apiFetch<OrchestrationConversationHistory>(`${root(projectId)}/${goalId}/conversation`)
}

export function submitOrchestrationConversation(
  projectId: string,
  goalId: string,
  input: { clientRequestId: string; content: string },
) {
  return apiFetch<OrchestrationConversationTurn>(`${root(projectId)}/${goalId}/conversation`, {
    method: 'POST',
    body: JSON.stringify({ client_request_id: input.clientRequestId, content: input.content }),
  })
}

export function fetchProjectAdvisorConversation(projectId: string) {
  return apiFetch<ProjectAdvisorHistory>(advisorRoot(projectId))
}

export function submitProjectAdvisorTurn(projectId: string, content: string) {
  return apiFetch<ProjectAdvisorTurn>(advisorRoot(projectId), {
    method: 'POST',
    body: JSON.stringify({ content }),
  })
}

export function putOrchestrationConversationFeedback(
  projectId: string,
  goalId: string,
  responseId: string,
  input: OrchestrationConversationFeedbackInput,
) {
  return apiFetch<OrchestrationConversationFeedback>(
    `${root(projectId)}/${goalId}/conversation/${responseId}/feedback`,
    { method: 'PUT', body: JSON.stringify(input) },
  )
}

export interface OrchestrationSteeringSubmitInput {
  clientRequestId: string
  directive: string
  targetType: OrchestrationSteeringTargetType
  targetId: string
  scope: OrchestrationSteeringScope
  lifetime: OrchestrationSteeringLifetime
  impactSummary: string
  sourceProposalId: string | null
  supersedesRequestId: string | null
}

export function submitOrchestrationSteering(
  projectId: string,
  goalId: string,
  input: OrchestrationSteeringSubmitInput,
) {
  return apiFetch<OrchestrationSteeringRequest>(`${root(projectId)}/${goalId}/conversation/steering`, {
    method: 'POST',
    body: JSON.stringify({
      client_request_id: input.clientRequestId, directive: input.directive, target_type: input.targetType,
      target_id: input.targetId, scope: input.scope, lifetime: input.lifetime, impact_summary: input.impactSummary,
      source_proposal_id: input.sourceProposalId, supersedes_request_id: input.supersedesRequestId,
    }),
  })
}

export function withdrawOrchestrationSteering(projectId: string, goalId: string, requestId: string) {
  return apiFetch<OrchestrationSteeringRequest>(`${root(projectId)}/${goalId}/conversation/steering/${requestId}/withdraw`, { method: 'POST' })
}

export function dismissOrchestrationSteeringProposal(projectId: string, goalId: string, proposalId: string) {
  return apiFetch<OrchestrationSteeringProposal>(`${root(projectId)}/${goalId}/conversation/steering/proposals/${proposalId}/dismiss`, { method: 'POST' })
}

export function fetchOrchestrationMemoryOverview(projectId: string, goalId: string) {
  return apiFetch<OrchestrationMemoryOverview>(`${root(projectId)}/${goalId}/memory`)
}

export function fetchOrchestrationMemorySection(
  projectId: string,
  goalId: string,
  sectionKey: string,
) {
  return apiFetch<OrchestrationMemorySection>(`${root(projectId)}/${goalId}/memory/${sectionKey}`)
}

export function fetchOrchestrationProcesses(projectId: string, goalId: string) {
  return apiFetch<OrchestrationProcessRunRecord[]>(`${root(projectId)}/${goalId}/processes`)
}

export function fetchOrchestrationAgentReviews(projectId: string, goalId: string) {
  return apiFetch<OrchestrationAgentReviewRecord[]>(`${root(projectId)}/${goalId}/agent-reviews`)
}

export function fetchOrchestrationWarnings(projectId: string, goalId: string) {
  return apiFetch<OrchestrationWarningRecord[]>(`${root(projectId)}/${goalId}/warnings`)
}

export function fetchOrchestrationDecisions(projectId: string, goalId: string) {
  return apiFetch<OrchestrationAuthorityDecisionRecord[]>(`${root(projectId)}/${goalId}/decisions`)
}

export function fetchOrchestrationCheckpoint(projectId: string, goalId: string) {
  return apiFetch<OrchestrationCheckpoint>(`${root(projectId)}/${goalId}/decisions/checkpoint`)
}

export function fetchOrchestrationHealth() {
  return apiFetch<OrchestrationHealth>('/api/v1/orchestration/health')
}

export function stepOrchestrationBaseline(
  projectId: string,
  goalId: string,
  processType: OrchestrationBaselineProcessType,
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/debug/baseline/step`,
    { method: 'POST', body: JSON.stringify({ process_type: processType }) },
  )
}

export function runOrchestrationBaselineStep(
  projectId: string,
  goalId: string,
  processType: OrchestrationBaselineProcessType,
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/baseline/step`,
    { method: 'POST', body: JSON.stringify({ process_type: processType }) },
  )
}

export function rerunOrchestrationBaselineStep(
  projectId: string,
  goalId: string,
  processType: OrchestrationBaselineProcessType,
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/baseline/rerun`,
    { method: 'POST', body: JSON.stringify({ process_type: processType }) },
  )
}

export function authorizeOrchestrationBaseline(
  projectId: string,
  goalId: string,
) {
  return apiFetch<OrchestrationGoalDetail>(
    `${root(projectId)}/${goalId}/baseline/authorize`,
    { method: 'POST' },
  )
}

export function retryOrchestrationBaselineStep(
  projectId: string,
  goalId: string,
  processType: OrchestrationBaselineProcessType,
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/baseline/retry`,
    { method: 'POST', body: JSON.stringify({ process_type: processType }) },
  )
}

export function rerunLastOrchestrationBaseline(
  projectId: string,
  goalId: string,
  processType?: OrchestrationBaselineProcessType,
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/debug/baseline/rerun-last`,
    { method: 'POST', body: JSON.stringify(processType ? { process_type: processType } : {}) },
  )
}

export function resetOrchestrationGoal(
  projectId: string,
  goalId: string,
) {
  return apiFetch<OrchestrationGoalDetail>(
    `${root(projectId)}/${goalId}/reset`,
    { method: 'POST', body: JSON.stringify({}) },
  )
}

export function skipOrchestrationProcess(
  projectId: string,
  goalId: string,
  processType: string,
  reason: string,
) {
  return apiFetch<OrchestrationProcessRunRecord>(
    `${root(projectId)}/${goalId}/processes/${processType}/skip`,
    { method: 'POST', body: JSON.stringify({ reason: reason.trim() }) },
  )
}

export function acknowledgeOrchestrationWarning(
  projectId: string,
  _goalId: string,
  warningId: string,
  reason: string,
) {
  return apiFetch<OrchestrationWarningRecord>(
    `/api/v1/projects/${projectId}/orchestration/warnings/${warningId}/acknowledge`,
    { method: 'POST', body: JSON.stringify({ reason: reason.trim() }) },
  )
}

export function resolveOrchestrationWarning(
  projectId: string,
  _goalId: string,
  warningId: string,
  reason: string,
) {
  return apiFetch<OrchestrationWarningRecord>(
    `/api/v1/projects/${projectId}/orchestration/warnings/${warningId}/resolve`,
    { method: 'POST', body: JSON.stringify({ reason: reason.trim() }) },
  )
}

export function answerOrchestrationDecision(
  projectId: string,
  goalId: string,
  decisionId: string,
  selectedOption: string,
  reason: string,
  editedAgent?: AgentCreatePayload,
) {
  return apiFetch<OrchestrationDecisionAnswerResult>(
    `${root(projectId)}/${goalId}/decisions/${decisionId}/answer`,
    {
      method: 'POST',
      body: JSON.stringify({
        selected_option: selectedOption,
        reason: reason.trim() || null,
        ...(editedAgent ? { edited_agent: editedAgent } : {}),
      }),
    },
  )
}

export function answerAgentDefinitionReviewBatch(
  projectId: string,
  goalId: string,
  answers: readonly AgentDefinitionReviewBatchAnswerInput[],
) {
  return apiFetch<AgentDefinitionReviewBatchAnswerResult>(
    `${root(projectId)}/${goalId}/decisions/agent-definition-review/batch-answer`,
    {
      method: 'POST',
      body: JSON.stringify({ answers: answers.map((answer) => ({
        decision_id: answer.decisionId,
        selected_option: answer.selectedOption,
        reason: answer.reason.trim() || null,
        ...(answer.selectedOption === 'edit' ? {
          edited_description: answer.editedDescription,
          edited_persona: answer.editedPersona,
        } : {}),
      })) }),
    },
  )
}

export function postOrchestrationGoalCommand(
  projectId: string,
  goalId: string,
  command: OrchestrationGoalCommand,
) {
  return apiFetch<OrchestrationGoalDetail>(`${root(projectId)}/${goalId}/${command}`, {
    method: 'POST',
    body: JSON.stringify({}),
  })
}

export function startOrchestrationGoal(projectId: string, goalId: string) {
  return apiFetch<OrchestrationGoalDetail>(`${root(projectId)}/${goalId}/start`, {
    method: 'POST',
    body: JSON.stringify({}),
  })
}

export function supersedeOrchestrationGoal(
  projectId: string,
  goalId: string,
  goalType: 'outcome' | 'roadmap' | 'continuous' = 'outcome',
) {
  return apiFetch<OrchestrationGoalDetail>(`${root(projectId)}/${goalId}/supersede`, {
    method: 'POST',
    body: JSON.stringify({ goal_type: goalType }),
  })
}

export function postOrchestrationGateOverride(
  projectId: string,
  input: OrchestrationGateOverrideInput,
) {
  return apiFetch<OrchestrationGoalDetail>(`${root(projectId)}/${input.goalId}/override`, {
    method: 'POST',
    body: JSON.stringify({
      gate_id: input.gateId,
      decision: input.decision,
      reason: input.reason.trim(),
      evidence_metadata: input.evidenceMetadata ?? {},
    }),
  })
}

export function recoverGoalDefinition(
  projectId: string,
  goalId: string,
  mode: 'proceed' | 'another_round',
) {
  return apiFetch<OrchestrationDebugActionResult>(
    `${root(projectId)}/${goalId}/goal-definition/recover`,
    { method: 'POST', body: JSON.stringify({ mode }) },
  )
}

export function syncOrchestrationGoalCache(
  queryClient: QueryClient,
  projectId: string | null,
  goalId: string,
  detail: OrchestrationGoalDetail,
) {
  queryClient.setQueryData(['orchestration-goal', projectId, goalId], detail)
  void queryClient.invalidateQueries({ queryKey: ['orchestration-goals', projectId] })
}

export function useOrchestrationGoals(projectId: string | null) {
  return useInfiniteQuery({
    queryKey: ['orchestration-goals', projectId],
    queryFn: ({ pageParam }: { pageParam: string | null }) =>
      fetchOrchestrationGoals(projectId!, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    enabled: !!projectId,
  })
}

export function useOrchestrationGoal(
  projectId: string | null,
  goalId: string | undefined,
) {
  return useQuery({
    queryKey: ['orchestration-goal', projectId, goalId],
    queryFn: () => fetchOrchestrationGoal(projectId!, goalId!),
    enabled: !!projectId && !!goalId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 10_000),
    staleTime: 10_000,
  })
}

export function isConversationTurnActive(turn: OrchestrationConversationTurn) {
  return turn.status === 'pending'
    || turn.status === 'running'
    || turn.investigation?.status === 'pending'
    || turn.investigation?.status === 'running'
}

export function isConversationHistoryActive(history: OrchestrationConversationHistory) {
  return history.items.some(isConversationTurnActive)
    || history.steering?.requests.some((request) => request.status === 'pending' || request.status === 'being_considered') === true
}

export function useOrchestrationConversation(projectId: string | null, goalId: string | undefined) {
  return useQuery({
    queryKey: orchestrationConversationKey(projectId, goalId),
    queryFn: () => fetchOrchestrationConversation(projectId!, goalId!),
    enabled: !!projectId && !!goalId,
    refetchInterval: (query) => query.state.data && isConversationHistoryActive(query.state.data) ? 2000 : false,
  })
}

// ponytail: request/response advisor, no polling — submit invalidates the
// query directly; add a refetchInterval if idle-drift (other actors asking
// questions) turns out to matter in practice.
export function useProjectAdvisorConversation(projectId: string | null) {
  return useQuery({
    queryKey: projectAdvisorKey(projectId),
    queryFn: () => fetchProjectAdvisorConversation(projectId!),
    enabled: !!projectId,
  })
}

export function useSubmitProjectAdvisorTurn(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (content: string) => submitProjectAdvisorTurn(projectId!, content),
    retry: false,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: projectAdvisorKey(projectId), exact: true }),
  })
}

export function useBaselineDashboard(projectId: string | null, goalId: string | undefined) {
  const enabled = !!projectId && !!goalId
  const [memory, processes, agentReviews, warnings, decisions, checkpoint] = useQueries({
    queries: [
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'memory'],
        queryFn: () => fetchOrchestrationMemoryOverview(projectId!, goalId!),
        enabled,
      },
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'processes'],
        queryFn: () => fetchOrchestrationProcesses(projectId!, goalId!),
        enabled,
      },
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'agent-reviews'],
        queryFn: () => fetchOrchestrationAgentReviews(projectId!, goalId!),
        enabled,
      },
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'warnings'],
        queryFn: () => fetchOrchestrationWarnings(projectId!, goalId!),
        enabled,
      },
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'decisions'],
        queryFn: () => fetchOrchestrationDecisions(projectId!, goalId!),
        enabled,
      },
      {
        queryKey: [...orchestrationBaselineKey(projectId, goalId), 'checkpoint'],
        queryFn: () => fetchOrchestrationCheckpoint(projectId!, goalId!),
        enabled,
      },
    ],
  })

  return { memory, processes, agentReviews, warnings, decisions, checkpoint }
}

export function useOrchestrationHealth() {
  return useQuery({
    queryKey: ['orchestration-health'],
    queryFn: fetchOrchestrationHealth,
  })
}

export function useOrchestrationMemorySection(
  projectId: string | null,
  goalId: string | undefined,
  sectionKey: string | undefined,
) {
  return useQuery({
    queryKey: [...orchestrationBaselineKey(projectId, goalId), 'memory', 'sections', sectionKey],
    queryFn: () => fetchOrchestrationMemorySection(projectId!, goalId!, sectionKey!),
    enabled: !!projectId && !!goalId && !!sectionKey,
  })
}

async function invalidateGoalAndBaseline(queryClient: QueryClient, projectId: string | null, goalId: string) {
  await queryClient.invalidateQueries({
    queryKey: ['orchestration-goal', projectId, goalId],
    exact: true,
  })
  await queryClient.invalidateQueries({
    queryKey: orchestrationBaselineKey(projectId, goalId),
  })
}

function syncAnsweredDecisionCache(
  queryClient: QueryClient,
  projectId: string | null,
  goalId: string,
  result: OrchestrationDecisionAnswerResult,
) {
  const updateDecision = (decision: OrchestrationAuthorityDecisionRecord) =>
    decision.id === result.decision.id ? result.decision : decision
  const key = orchestrationBaselineKey(projectId, goalId)

  queryClient.setQueryData<OrchestrationAuthorityDecisionRecord[]>([...key, 'decisions'], (decisions) =>
    decisions?.map(updateDecision),
  )
  queryClient.setQueryData<OrchestrationCheckpoint>([...key, 'checkpoint'], (checkpoint) => checkpoint && {
    ...checkpoint,
    items: checkpoint.items.map(updateDecision),
  })

  const process = result.process
  if (process?.id !== result.decision.source_process_run_id) return
  queryClient.setQueryData<OrchestrationProcessRunRecord[]>([...key, 'processes'], (processes) =>
    processes?.map((item) => item.id === process.id ? process : item),
  )
}

export function useSkipOrchestrationProcess(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType, reason }: {
      goalId: string
      processType: string
      reason: string
    }) => skipOrchestrationProcess(projectId!, goalId, processType, reason),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useAcknowledgeOrchestrationWarning(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, warningId, reason }: {
      goalId: string
      warningId: string
      reason: string
    }) => acknowledgeOrchestrationWarning(projectId!, goalId, warningId, reason),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useResolveOrchestrationWarning(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, warningId, reason }: {
      goalId: string
      warningId: string
      reason: string
    }) => resolveOrchestrationWarning(projectId!, goalId, warningId, reason),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useAnswerOrchestrationDecision(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, decisionId, selectedOption, reason, editedAgent }: {
      goalId: string
      decisionId: string
      selectedOption: string
      reason: string
      editedAgent?: AgentCreatePayload
    }) => answerOrchestrationDecision(
      projectId!, goalId, decisionId, selectedOption, reason, editedAgent,
    ),
    onSuccess: (data, { goalId }) => {
      syncAnsweredDecisionCache(queryClient, projectId, goalId, data)
      return invalidateGoalAndBaseline(queryClient, projectId, goalId)
    },
  })
}

export function useAnswerAgentDefinitionReviewBatch(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, answers }: {
      goalId: string
      answers: readonly AgentDefinitionReviewBatchAnswerInput[]
    }) => answerAgentDefinitionReviewBatch(projectId!, goalId, answers),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useStepOrchestrationBaseline(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType }: {
      goalId: string
      processType: OrchestrationBaselineProcessType
    }) => stepOrchestrationBaseline(projectId!, goalId, processType),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useRunOrchestrationBaselineStep(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType }: {
      goalId: string
      processType: OrchestrationBaselineProcessType
    }) => runOrchestrationBaselineStep(projectId!, goalId, processType),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useRerunOrchestrationBaselineStep(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType }: {
      goalId: string
      processType: OrchestrationBaselineProcessType
    }) => rerunOrchestrationBaselineStep(projectId!, goalId, processType),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useAuthorizeOrchestrationBaseline(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId }: { goalId: string }) => authorizeOrchestrationBaseline(projectId!, goalId),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useRetryOrchestrationBaselineStep(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType }: {
      goalId: string
      processType: OrchestrationBaselineProcessType
    }) => retryOrchestrationBaselineStep(projectId!, goalId, processType),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useRerunLastOrchestrationBaseline(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, processType }: {
      goalId: string
      processType?: OrchestrationBaselineProcessType
    }) => rerunLastOrchestrationBaseline(projectId!, goalId, processType),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function useResetOrchestrationGoal(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId }: { goalId: string }) => resetOrchestrationGoal(projectId!, goalId),
    onSuccess: (detail, { goalId }) => {
      syncOrchestrationGoalCache(queryClient, projectId, goalId, detail)
      return invalidateGoalAndBaseline(queryClient, projectId, goalId)
    },
  })
}

export function useOrchestrationGoalCommand(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, command }: { goalId: string; command: OrchestrationGoalCommand }) =>
      postOrchestrationGoalCommand(projectId!, goalId, command),
    onSuccess: (detail, { goalId }) => {
      syncOrchestrationGoalCache(queryClient, projectId, goalId, detail)
    },
  })
}

export function useStartOrchestrationGoal(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId }: { goalId: string }) => startOrchestrationGoal(projectId!, goalId),
    onSuccess: (detail, { goalId }) => syncOrchestrationGoalCache(queryClient, projectId, goalId, detail),
  })
}

export function useOverrideOrchestrationGate(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (input: OrchestrationGateOverrideInput) =>
      postOrchestrationGateOverride(projectId!, input),
    onSuccess: (detail, { goalId }) => {
      syncOrchestrationGoalCache(queryClient, projectId, goalId, detail)
    },
  })
}

export function useRecoverGoalDefinition(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, mode }: {
      goalId: string
      mode: 'proceed' | 'another_round'
    }) => recoverGoalDefinition(projectId!, goalId, mode),
    onSuccess: (_data, { goalId }) => invalidateGoalAndBaseline(queryClient, projectId, goalId),
  })
}

export function createOrchestrationGoal(
  projectId: string,
  input: OrchestrationGoalCreateInput,
) {
  return apiFetch<OrchestrationGoalDetail>(root(projectId), {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function useCreateOrchestrationGoal(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (input: OrchestrationGoalCreateInput) =>
      createOrchestrationGoal(projectId!, input),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['orchestration-goals', projectId] }),
  })
}

export function useSubmitOrchestrationConversation(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, clientRequestId, content }: { goalId: string; clientRequestId: string; content: string }) =>
      submitOrchestrationConversation(projectId!, goalId, { clientRequestId, content }),
    retry: false,
    onSuccess: (_data, { goalId }) => queryClient.invalidateQueries({
      queryKey: orchestrationConversationKey(projectId, goalId), exact: true,
    }),
  })
}

export function useRecordOrchestrationConversationFeedback(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, responseId, rating, reason }: {
      goalId: string
      responseId: string
      rating: OrchestrationConversationFeedbackInput['rating']
      reason: OrchestrationConversationFeedbackInput['reason']
    }) => putOrchestrationConversationFeedback(projectId!, goalId, responseId, { rating, reason }),
    retry: false,
    onSuccess: (_data, { goalId }) => { void queryClient.invalidateQueries({
      queryKey: orchestrationConversationKey(projectId, goalId), exact: true,
    }) },
  })
}

async function invalidateConversation(queryClient: QueryClient, projectId: string | null, goalId: string) {
  await queryClient.invalidateQueries({ queryKey: orchestrationConversationKey(projectId, goalId), exact: true })
}

export function useSubmitOrchestrationSteering(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, ...input }: OrchestrationSteeringSubmitInput & { goalId: string }) =>
      submitOrchestrationSteering(projectId!, goalId, input),
    retry: false,
    onSuccess: (_data, { goalId }) => invalidateConversation(queryClient, projectId, goalId),
  })
}

export function useWithdrawOrchestrationSteering(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, requestId }: { goalId: string; requestId: string }) =>
      withdrawOrchestrationSteering(projectId!, goalId, requestId),
    retry: false,
    onSuccess: (_data, { goalId }) => invalidateConversation(queryClient, projectId, goalId),
  })
}

export function useDismissOrchestrationSteeringProposal(projectId: string | null) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ goalId, proposalId }: { goalId: string; proposalId: string }) =>
      dismissOrchestrationSteeringProposal(projectId!, goalId, proposalId),
    retry: false,
    onSuccess: (_data, { goalId }) => invalidateConversation(queryClient, projectId, goalId),
  })
}
