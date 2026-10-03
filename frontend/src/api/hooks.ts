import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch, fetchAllPages } from '@/lib/api-client'
import type { Hook, HookStatus } from '@/lib/types'

export function useHooksList(projectId: string | null, statusFilter?: HookStatus) {
  const extra: Record<string, string> = {}
  if (statusFilter) extra['status'] = statusFilter
  return useQuery({
    queryKey: ['hooks', projectId, statusFilter],
    queryFn: () => fetchAllPages<Hook>(`/api/v1/projects/${projectId}/hooks`, extra),
    enabled: !!projectId,
  })
}

export function useCreateHook(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; trigger_event: string; code: string; description?: string }) =>
      apiFetch<Hook>(`/api/v1/projects/${projectId}/hooks`, { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hooks', projectId] }),
  })
}

export function useUpdateHook(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<{ name: string; trigger_event: string; code: string; description: string; status: HookStatus }> }) =>
      apiFetch<Hook>(`/api/v1/projects/${projectId}/hooks/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hooks', projectId] }),
  })
}

export function useDeleteHook(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/projects/${projectId}/hooks/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['hooks', projectId] }),
  })
}
