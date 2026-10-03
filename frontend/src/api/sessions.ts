import { useQuery } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'

export function useSessionCount(projectId: string | null, status?: string) {
  return useQuery({
    queryKey: ['sessions', 'count', projectId, status],
    queryFn: () =>
      apiFetch<{ count: number }>(
        `/api/v1/sessions/count?project_id=${projectId}${status ? `&status=${status}` : ''}`
      ),
    enabled: !!projectId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 15_000),
    staleTime: 15_000,
  })
}
