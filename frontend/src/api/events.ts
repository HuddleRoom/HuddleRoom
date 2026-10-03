import { useQuery } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'
import type { HuddleRoomEvent } from '@/lib/types'

export function useRecentEvents(projectId: string | null, limit = 20) {
  return useQuery({
    queryKey: ['events', projectId, limit],
    queryFn: () =>
      apiFetch<{ items: HuddleRoomEvent[]; next_cursor: string | null }>(
        `/api/v1/events?project_id=${projectId}&limit=${limit}`
      ).then((r) => r.items),
    enabled: !!projectId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 10_000),
    staleTime: 10_000,
  })
}
