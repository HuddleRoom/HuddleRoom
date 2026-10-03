import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch, fetchAllPages } from '@/lib/api-client'
import type { RoutingRule } from '@/lib/types'

export function useRules(projectId: string | null) {
  return useQuery({
    queryKey: ['rules', projectId],
    queryFn: () => fetchAllPages<RoutingRule>(`/api/v1/projects/${projectId}/rules`),
    enabled: !!projectId,
  })
}

export function useCreateRule(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { name: string; on_event: string; conditions: Record<string, unknown>; actions: Record<string, unknown>; description?: string; enabled?: boolean }) =>
      apiFetch<RoutingRule>(`/api/v1/projects/${projectId}/rules`, { method: 'POST', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rules', projectId] }),
  })
}

export function useUpdateRule(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<{ name: string; on_event: string; conditions: Record<string, unknown>; actions: Record<string, unknown>; description: string; enabled: boolean; priority: number }> }) =>
      apiFetch<RoutingRule>(`/api/v1/projects/${projectId}/rules/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rules', projectId] }),
  })
}

export function useDeleteRule(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/projects/${projectId}/rules/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rules', projectId] }),
  })
}

export function useReorderRules(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (rule_ids: string[]) =>
      apiFetch<void>(`/api/v1/projects/${projectId}/rules/reorder`, { method: 'POST', body: JSON.stringify({ rule_ids }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['rules', projectId] }),
  })
}
