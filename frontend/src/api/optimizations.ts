import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch, fetchAllPages } from '@/lib/api-client'
import type { Optimization, OptimizationStatus, Pattern } from '@/lib/types'

export function useOptimizations(projectId: string | null, status?: OptimizationStatus) {
  const extra: Record<string, string> = {}
  if (status) extra['status'] = status
  return useQuery({
    queryKey: ['optimizations', projectId, status],
    queryFn: () => fetchAllPages<Optimization>(`/api/v1/projects/${projectId}/optimizations`, extra),
    enabled: !!projectId,
  })
}

export function useUpdateOptimization(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: { status?: OptimizationStatus; generated_code?: string } }) =>
      apiFetch<Optimization>(`/api/v1/projects/${projectId}/optimizations/${id}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['optimizations', projectId] }),
  })
}

export function useDeleteOptimization(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/projects/${projectId}/optimizations/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['optimizations', projectId] }),
  })
}

export function usePatterns(projectId: string | null) {
  return useQuery({
    queryKey: ['patterns', projectId],
    queryFn: () => fetchAllPages<Pattern>(`/api/v1/projects/${projectId}/patterns`),
    enabled: !!projectId,
  })
}

export function usePattern(projectId: string | null, patternId: string | undefined) {
  return useQuery({
    queryKey: ['pattern', projectId, patternId],
    queryFn: () => apiFetch<Pattern>(`/api/v1/projects/${projectId}/patterns/${patternId}`),
    enabled: !!projectId && !!patternId,
  })
}
