import { useMutation, useQueryClient } from '@tanstack/react-query'
import type { Query, QueryClient } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import type { Project } from '@/lib/types'
import { useWSStore } from '@/stores/ws'

export type ProjectResetRequest = { confirm_name: string }

export type ProjectResetResponse = {
  cancelled_sessions: number
  cancelled_meeting_tasks: number
  deletions: Record<string, number>
}

export function resetProject(projectId: string, data: ProjectResetRequest) {
  return apiFetch<ProjectResetResponse>(`/api/v1/projects/${projectId}/reset`, {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export function fetchProject(projectId: string) {
  return apiFetch<Project>(`/api/v1/projects/${projectId}`)
}

function isProjectQuery(query: Query, projectId: string) {
  const key = query.queryKey
  const [prefix, second, third] = key

  if (prefix === 'meeting') {
    return (second === 'turns' || second === 'decisions' || second === 'action-items'
      || second === 'final-review' || second === 'agenda' || second === 'signals' ? third : second) === projectId
  }
  if (prefix === 'knowledge-item') return second === projectId
  if (prefix === 'memory') return second === 'project' && third === projectId
  if (prefix === 'tasks' || prefix === 'meetings' || prefix === 'protocol-instances' || prefix === 'sessions') {
    return (second === 'count' ? third : second) === projectId
  }
  if (prefix === 'task' || prefix === 'protocol-instance') {
    return (second === 'subtasks' || second === 'sessions' || second === 'transitions' ? third : second) === projectId
  }
  return [
    'project', 'events', 'hooks', 'knowledge', 'optimizations', 'patterns', 'pattern',
    'orchestration-goals', 'orchestration-goal', 'protocols', 'protocol', 'rules',
  ].includes(String(prefix)) && second === projectId
}

export function removeProjectQueries(queryClient: Pick<QueryClient, 'removeQueries'>, projectId: string) {
  queryClient.removeQueries({ predicate: (query) => isProjectQuery(query, projectId) })
}

export async function resetProjectDataBoundary(
  queryClient: Pick<QueryClient, 'removeQueries' | 'fetchQuery' | 'refetchQueries'>,
  projectId: string,
  reportRefreshError: (phase: 'project' | 'projects', error: unknown) => void =
    (phase, error) => console.error(`[projects] reset ${phase} refresh failed`, error),
) {
  removeProjectQueries(queryClient, projectId)
  useWSStore.getState().resetReplayCursor(projectId)
  try {
    await queryClient.fetchQuery({ queryKey: ['project', projectId], queryFn: () => fetchProject(projectId) })
  } catch (error) {
    reportRefreshError('project', error)
  }
  try {
    await queryClient.refetchQueries({ queryKey: ['projects'] }, { throwOnError: true })
  } catch (error) {
    reportRefreshError('projects', error)
  }
}

export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; workspace_path: string; description?: string }) =>
      apiFetch<Project>('/api/v1/projects', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
    },
  })
}

export function useResetProject(projectId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: ProjectResetRequest) => resetProject(projectId, data),
    onSuccess: () => resetProjectDataBoundary(queryClient, projectId),
  })
}
