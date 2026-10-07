import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'
import type { Graph, GraphRun, CursorPage, GraphRunStep, GraphRunDetail } from '@/lib/types'

// Normalise whatever the backend returns (flat array OR CursorPage) to Graph[]
function normaliseGraphList(data: Graph[] | CursorPage<Graph>): Graph[] {
  if (Array.isArray(data)) return data
  return (data.items ?? []).filter(Boolean)
}

export function useGraphs(projectId: string | null, includeInactive?: boolean) {
  return useQuery({
    queryKey: ['graphs', projectId, { includeInactive }],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (includeInactive) params.set('include_inactive', 'true')
      const data = await apiFetch<Graph[] | CursorPage<Graph>>(
        `/api/v1/projects/${projectId}/graphs${params.toString() ? `?${params}` : ''}`
      )
      return normaliseGraphList(data)
    },
    enabled: !!projectId,
  })
}

export function useGraph(projectId: string | null, graphId: string | undefined) {
  return useQuery({
    queryKey: ['graph', projectId, graphId],
    queryFn: () =>
      apiFetch<Graph>(`/api/v1/projects/${projectId}/graphs/${graphId}`),
    enabled: !!projectId && !!graphId,
  })
}

// useGraphsAll: same as useGraphs but aliased for components that expect { items, isLoading }
export function useGraphsAll(projectId: string | null, includeInactive?: boolean) {
  const query = useGraphs(projectId, includeInactive)
  const items = useMemo(() => query.data ?? [], [query.data])
  return { ...query, items }
}

export function useCreateGraph(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      name: string
      version: string
      description?: string
      definition: Record<string, unknown>
      triggers?: string[]
    }) =>
      apiFetch<Graph>(`/api/v1/projects/${projectId}/graphs`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['graphs', projectId] })
    },
  })
}

export function useUpdateGraph(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      graphId,
      data,
    }: {
      graphId: string
      data: Partial<{
        name: string
        version: string
        description: string
        definition: Record<string, unknown>
        triggers: string[]
      }>
    }) =>
      apiFetch<Graph>(`/api/v1/projects/${projectId}/graphs/${graphId}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: (graph, { graphId }) => {
      qc.setQueryData(['graph', projectId, graphId], graph)
      qc.invalidateQueries({ queryKey: ['graphs', projectId] })
    },
  })
}

export function useDeactivateGraph(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (graphId: string) =>
      apiFetch<Graph>(`/api/v1/projects/${projectId}/graphs/${graphId}`, {
        method: 'DELETE',
      }),
    onSuccess: (_, graphId) => {
      qc.invalidateQueries({ queryKey: ['graphs', projectId] })
      qc.invalidateQueries({ queryKey: ['graph', projectId, graphId] })
    },
  })
}

export function useActivateGraph(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (graphId: string) =>
      apiFetch<Graph>(
        `/api/v1/projects/${projectId}/graphs/${graphId}/activate`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (graph, graphId) => {
      qc.setQueryData(['graph', projectId, graphId], graph)
      qc.invalidateQueries({ queryKey: ['graphs', projectId] })
    },
  })
}

export function useGraphRuns(
  projectId: string | null,
  filters?: { graph_id?: string; status?: string }
) {
  const qs = new URLSearchParams()
  if (filters?.graph_id) qs.set('graph_id', filters.graph_id)
  if (filters?.status) qs.set('status', filters.status)

  return useQuery({
    queryKey: ['graph-runs', projectId, filters],
    queryFn: () =>
      apiFetch<CursorPage<GraphRun>>(
        `/api/v1/projects/${projectId}/graph-runs${qs.toString() ? `?${qs}` : ''}`
      ),
    enabled: !!projectId,
  })
}

export function useGraphRunsAll(
  projectId: string | null,
  filters?: { graph_id?: string; status?: string }
) {
  const qs = new URLSearchParams({ limit: '500' })
  if (filters?.graph_id) qs.set('graph_id', filters.graph_id)
  if (filters?.status) qs.set('status', filters.status)

  const query = useInfiniteQuery({
    queryKey: ['graph-runs', projectId, 'all', filters],
    queryFn: ({ pageParam }: { pageParam: string | null }) => {
      const p = new URLSearchParams(qs)
      if (pageParam) p.set('cursor', pageParam)
      return apiFetch<CursorPage<GraphRun>>(
        `/api/v1/projects/${projectId}/graph-runs?${p}`
      )
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled: !!projectId,
  })

  const { hasNextPage, isFetchingNextPage, fetchNextPage } = query
  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage()
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])

  const items = useMemo(
    () => (query.data?.pages.flatMap((p) => p.items ?? []) ?? []).filter(Boolean),
    [query.data]
  )

  return { ...query, items }
}

export function useGraphRunDetail(
  projectId: string | null,
  runId: string | undefined
) {
  return useQuery({
    queryKey: ['graph-run', projectId, runId],
    queryFn: () =>
      apiFetch<GraphRunDetail>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/detail`
      ),
    enabled: !!projectId && !!runId,
  })
}

export function useGraphRunSteps(
  projectId: string | null,
  runId: string | undefined
) {
  return useQuery({
    queryKey: ['graph-run', 'steps', projectId, runId],
    queryFn: () =>
      apiFetch<GraphRunStep[]>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/steps`
      ),
    enabled: !!projectId && !!runId,
  })
}

export function useAdvanceRun(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<GraphRun>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/advance`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (run, runId) => {
      qc.setQueryData(['graph-run', projectId, runId], run)
      qc.invalidateQueries({ queryKey: ['graph-runs', projectId] })
      qc.invalidateQueries({
        queryKey: ['graph-run', 'steps', projectId, runId],
      })
    },
  })
}

export function usePauseRun(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<GraphRun>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/pause`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (run, runId) => {
      qc.setQueryData(['graph-run', projectId, runId], run)
      qc.invalidateQueries({ queryKey: ['graph-runs', projectId] })
    },
  })
}

export function useResumeRun(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<GraphRun>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/resume`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (run, runId) => {
      qc.setQueryData(['graph-run', projectId, runId], run)
      qc.invalidateQueries({ queryKey: ['graph-runs', projectId] })
    },
  })
}

export function useAbandonRun(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      apiFetch<GraphRun>(
        `/api/v1/projects/${projectId}/graph-runs/${runId}/abandon`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (run, runId) => {
      qc.setQueryData(['graph-run', projectId, runId], run)
      qc.invalidateQueries({ queryKey: ['graph-runs', projectId] })
    },
  })
}

export function useGraphRunCount(projectId: string | null, status?: string) {
  return useQuery({
    queryKey: ['graph-runs', 'count', projectId, status],
    queryFn: () =>
      apiFetch<{ count: number }>(
        `/api/v1/projects/${projectId}/graph-runs/count${status ? `?status=${status}` : ''}`
      ),
    enabled: !!projectId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 15_000),
    staleTime: 15_000,
  })
}
