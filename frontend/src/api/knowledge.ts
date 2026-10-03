import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch, fetchAllPages } from '@/lib/api-client'
import type { KnowledgeItem, KnowledgeSearchResult, ContentType } from '@/lib/types'

export const knowledgeItemKey = (projectId: string | null, id: string | undefined) =>
  ['knowledge-item', projectId, id] as const

// List knowledge items (all pages)
export function useKnowledge(projectId: string | null) {
  return useQuery({
    queryKey: ['knowledge', projectId],
    queryFn: () => fetchAllPages<KnowledgeItem>(`/api/v1/projects/${projectId}/knowledge`),
    enabled: !!projectId,
  })
}

// Semantic search
export function useSearchKnowledge(projectId: string | null) {
  return useMutation({
    mutationFn: async (query: string): Promise<KnowledgeSearchResult[]> => {
      const response = await apiFetch<{ results: KnowledgeSearchResult[] }>(
        `/api/v1/projects/${projectId}/knowledge/search`,
        {
          method: 'POST',
          body: JSON.stringify({ query }),
        }
      )
      return response.results
    },
  })
}

// Get single item (not project-scoped per spec)
export function useKnowledgeItem(projectId: string | null, id: string | undefined) {
  return useQuery({
    queryKey: knowledgeItemKey(projectId, id),
    queryFn: () => apiFetch<KnowledgeItem>(`/api/v1/knowledge/${id}`),
    enabled: !!projectId && !!id,
  })
}

// Create item
export function useCreateKnowledge(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      content: string
      content_type: ContentType
      title?: string
      tags?: string[]
      provenance?: string
    }) =>
      apiFetch<KnowledgeItem>(
        `/api/v1/projects/${projectId}/knowledge`,
        {
          method: 'POST',
          body: JSON.stringify(data),
        }
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['knowledge', projectId] }),
  })
}

// Update item
export function useUpdateKnowledge(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string
      data: Partial<{
        content: string
        content_type: ContentType
        title: string
        tags: string[]
      }>
    }) =>
      apiFetch<KnowledgeItem>(`/api/v1/knowledge/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: (item) => {
      qc.setQueryData(knowledgeItemKey(projectId, item.id), item)
      qc.invalidateQueries({ queryKey: ['knowledge', projectId] })
    },
  })
}

// Delete item
export function useDeleteKnowledge(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/knowledge/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['knowledge', projectId] }),
  })
}
