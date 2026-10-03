import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch, fetchAllPages } from '@/lib/api-client'
import type { MemoryItem } from '@/lib/types'

// List project memory (includes global shared)
export function useProjectMemory(projectId: string | null, params?: { agent_id?: string; shared?: boolean }) {
  const extra: Record<string, string> = {}
  if (params?.agent_id) extra['agent_id'] = params.agent_id
  if (params?.shared !== undefined) extra['shared'] = String(params.shared)
  return useQuery({
    queryKey: ['memory', 'project', projectId, params],
    queryFn: () => fetchAllPages<MemoryItem>(`/api/v1/projects/${projectId}/memory`, extra),
    enabled: !!projectId,
  })
}

// Semantic search in project memory
export function useSearchProjectMemory(projectId: string | null) {
  return useMutation({
    mutationFn: async ({ query, tags }: { query: string; tags?: string[] }): Promise<MemoryItem[]> => {
      const response = await apiFetch<{ results: MemoryItem[]; count?: number }>(
        `/api/v1/projects/${projectId}/memory/search`,
        {
          method: 'POST',
          body: JSON.stringify({ query, tags }),
        }
      )
      return response.results
    },
  })
}

// Delete project-scoped memory
export function useDeleteProjectMemory(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/projects/${projectId}/memory/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['memory', 'project', projectId] }),
  })
}

// List global memory
export function useGlobalMemory() {
  return useQuery({
    queryKey: ['memory', 'global'],
    queryFn: () => fetchAllPages<MemoryItem>('/api/v1/memory', { scope: 'global' }),
  })
}

// Delete global memory (admin only)
export function useDeleteGlobalMemory() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/memory/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['memory', 'global'] }),
  })
}
