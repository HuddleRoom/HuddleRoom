import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { QueryClient } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import type {
  Meeting,
  MeetingTurn,
  MeetingDecision,
  MeetingActionItem,
  MeetingFinalReview,
  MeetingFinalReviewSubmit,
  AgendaItem,
  MeetingSignal,
  MeetingStatus,
  CursorPage,
} from '@/lib/types'

export const meetingKeys = {
  detail: (projectId: string | null, meetingId: string | undefined) => ['meeting', projectId, meetingId] as const,
  turns: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'turns', projectId, meetingId] as const,
  decisions: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'decisions', projectId, meetingId] as const,
  actionItems: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'action-items', projectId, meetingId] as const,
  finalReview: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'final-review', projectId, meetingId] as const,
  agenda: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'agenda', projectId, meetingId] as const,
  signals: (projectId: string | null, meetingId: string | undefined) => ['meeting', 'signals', projectId, meetingId] as const,
}

export function fetchMeetingFinalReview(meetingId: string) {
  return apiFetch<MeetingFinalReview | null>(`/api/v1/meetings/${meetingId}/final-review`)
}

export function completeMeetingFinalReview(meetingId: string, data: MeetingFinalReviewSubmit) {
  return apiFetch<void>(`/api/v1/meetings/${meetingId}/final-review`, {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function invalidateMeetingFinalReviewQueries(
  queryClient: Pick<QueryClient, 'invalidateQueries'>,
  projectId: string | null,
  meetingId: string,
) {
  await queryClient.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
  await queryClient.invalidateQueries({ queryKey: meetingKeys.finalReview(projectId, meetingId) })
  await queryClient.invalidateQueries({ queryKey: meetingKeys.actionItems(projectId, meetingId) })
}

export function useMeetings(projectId: string | null, statusFilter?: MeetingStatus) {
  return useInfiniteQuery({
    queryKey: ['meetings', projectId, statusFilter],
    queryFn: async ({ pageParam }: { pageParam: string | null }) => {
      const params = new URLSearchParams({ limit: '50' })
      if (statusFilter) params.set('status_filter', statusFilter)
      if (pageParam) params.set('cursor', pageParam)
      return apiFetch<CursorPage<Meeting>>(
        `/api/v1/projects/${projectId}/meetings?${params}`
      )
    },
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: CursorPage<Meeting>) => lastPage.next_cursor ?? undefined,
    enabled: !!projectId,
  })
}

export function useMeeting(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.detail(projectId, meetingId),
    queryFn: () => apiFetch<Meeting>(`/api/v1/meetings/${meetingId}`),
    enabled: !!projectId && !!meetingId,
  })
}

export function useMeetingTurns(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.turns(projectId, meetingId),
    queryFn: () => apiFetch<MeetingTurn[]>(`/api/v1/meetings/${meetingId}/turns`),
    enabled: !!projectId && !!meetingId,
  })
}

export function useMeetingDecisions(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.decisions(projectId, meetingId),
    queryFn: () => apiFetch<MeetingDecision[]>(`/api/v1/meetings/${meetingId}/decisions`),
    enabled: !!projectId && !!meetingId,
  })
}

export function useMeetingActionItems(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.actionItems(projectId, meetingId),
    queryFn: () =>
      apiFetch<MeetingActionItem[]>(`/api/v1/meetings/${meetingId}/action-items`),
    enabled: !!projectId && !!meetingId,
  })
}

export function useMeetingFinalReview(projectId: string | null, meetingId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: meetingKeys.finalReview(projectId, meetingId),
    queryFn: () => fetchMeetingFinalReview(meetingId!),
    enabled: !!projectId && !!meetingId && enabled,
    refetchInterval: enabled ? 5_000 : false,
  })
}

export function useCompleteMeetingFinalReview(projectId: string | null, meetingId: string | undefined) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: MeetingFinalReviewSubmit) => completeMeetingFinalReview(meetingId!, data),
    onSuccess: () => invalidateMeetingFinalReviewQueries(queryClient, projectId, meetingId!),
  })
}

export function useMeetingAgenda(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.agenda(projectId, meetingId),
    queryFn: () => apiFetch<AgendaItem[]>(`/api/v1/meetings/${meetingId}/agenda`),
    enabled: !!projectId && !!meetingId,
  })
}

export function useMeetingSignals(projectId: string | null, meetingId: string | undefined) {
  return useQuery({
    queryKey: meetingKeys.signals(projectId, meetingId),
    queryFn: () => apiFetch<MeetingSignal[]>(`/api/v1/meetings/${meetingId}/signals`),
    enabled: !!projectId && !!meetingId,
  })
}

export type CreateMeetingRequest = {
  title: string
  meeting_type: string
  participant_agent_ids: string[]
  agenda_items?: Array<{
    title: string
    order: number
    description?: string | null
    question?: string | null
  }>
  max_duration_minutes?: number
  turn_strategy?: string
  deadlock_strategy?: string
  auto_start?: boolean
  signal_check_enabled?: boolean
  organizer_agent_id?: string
  planner_agent_id?: string
  scheduled_at?: string
}

export function useCreateMeeting(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: CreateMeetingRequest) =>
      apiFetch<Meeting>(`/api/v1/projects/${projectId}/meetings`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['meetings', projectId] })
      qc.invalidateQueries({ queryKey: ['meetings', 'count', projectId] })
    },
  })
}

export function useCopyMeeting() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (meetingId: string) =>
      apiFetch<Meeting>(`/api/v1/meetings/${meetingId}/copy`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (meeting) => {
      qc.setQueryData(meetingKeys.detail(meeting.project_id, meeting.id), meeting)
      qc.invalidateQueries({ queryKey: ['meetings'] })
      qc.invalidateQueries({ queryKey: ['meetings', 'count'] })
    },
  })
}

export function useSubmitHumanTurn(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (content: string) =>
      apiFetch<MeetingTurn>(`/api/v1/meetings/${meetingId}/human-turn`, {
        method: 'POST',
        body: JSON.stringify({ content }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.turns(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useEndMeeting() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (meetingId: string) =>
      apiFetch<Meeting>(`/api/v1/meetings/${meetingId}/end`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (meeting) => {
      qc.setQueryData(meetingKeys.detail(meeting.project_id, meeting.id), meeting)
      qc.invalidateQueries({ queryKey: ['meetings'] })
      qc.invalidateQueries({ queryKey: ['meetings', 'count'] })
    },
  })
}

export function useCancelMeeting() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (meetingId: string) =>
      apiFetch<Meeting>(`/api/v1/meetings/${meetingId}`, {
        method: 'DELETE',
      }),
    onSuccess: (meeting) => {
      qc.setQueryData(meetingKeys.detail(meeting.project_id, meeting.id), meeting)
      qc.invalidateQueries({ queryKey: ['meetings'] })
      qc.invalidateQueries({ queryKey: ['meetings', 'count'] })
    },
  })
}

export function useResumeMeeting() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (meetingId: string) =>
      apiFetch<Meeting>(`/api/v1/meetings/${meetingId}/resume`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (meeting) => {
      // ponytail: cache-only marker resets on reload; persist it if retries need cross-client visibility.
      qc.setQueryData(meetingKeys.detail(meeting.project_id, meeting.id), {
        ...meeting,
        resume_state: { ...meeting.resume_state, resuming: true },
      })
      qc.invalidateQueries({ queryKey: ['meetings'] })
      qc.invalidateQueries({ queryKey: ['meetings', 'count'] })
    },
  })
}

export function useGrantTurn(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (participant_agent_id: string) =>
      apiFetch<MeetingTurn>(`/api/v1/meetings/${meetingId}/grant-turn`, {
        method: 'POST',
        body: JSON.stringify({ participant_agent_id }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.turns(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useAdvanceAgenda(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (agendaItemId: string) =>
      apiFetch<AgendaItem>(
        `/api/v1/meetings/${meetingId}/agenda/${agendaItemId}/advance`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.agenda(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useVetoDecision(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (decisionId: string) =>
      apiFetch<MeetingDecision>(
        `/api/v1/meetings/${meetingId}/decisions/${decisionId}/veto`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.decisions(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useAddAgendaItem(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: Partial<AgendaItem>) =>
      apiFetch<AgendaItem>(`/api/v1/meetings/${meetingId}/agenda`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.agenda(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useUpdateActionItem(projectId: string | null, meetingId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      actionItemId,
      data,
    }: {
      actionItemId: string
      data: Partial<MeetingActionItem>
    }) =>
      apiFetch<MeetingActionItem>(
        `/api/v1/meetings/${meetingId}/action-items/${actionItemId}`,
        { method: 'PATCH', body: JSON.stringify(data) }
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: meetingKeys.actionItems(projectId, meetingId) })
      qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
    },
  })
}

export function useMeetingCount(projectId: string | null, status?: MeetingStatus) {
  return useQuery({
    queryKey: ['meetings', 'count', projectId, status],
    queryFn: () =>
      apiFetch<{ count: number }>(
        `/api/v1/projects/${projectId}/meetings/count${status ? `?status=${status}` : ''}`
      ),
    enabled: !!projectId,
    refetchInterval: 15_000,
  })
}
