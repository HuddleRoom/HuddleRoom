import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'
import type { Protocol, ProtocolInstance, CursorPage, ProtocolTransition, ProtocolInstanceDetail } from '@/lib/types'

// Normalise whatever the backend returns (flat array OR CursorPage) to Protocol[]
function normaliseProtocolList(data: Protocol[] | CursorPage<Protocol>): Protocol[] {
  if (Array.isArray(data)) return data
  return (data.items ?? []).filter(Boolean)
}

export function useProtocols(projectId: string | null, includeInactive?: boolean) {
  return useQuery({
    queryKey: ['protocols', projectId, { includeInactive }],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (includeInactive) params.set('include_inactive', 'true')
      const data = await apiFetch<Protocol[] | CursorPage<Protocol>>(
        `/api/v1/projects/${projectId}/protocols${params.toString() ? `?${params}` : ''}`
      )
      return normaliseProtocolList(data)
    },
    enabled: !!projectId,
  })
}

export function useProtocol(projectId: string | null, protocolId: string | undefined) {
  return useQuery({
    queryKey: ['protocol', projectId, protocolId],
    queryFn: () =>
      apiFetch<Protocol>(`/api/v1/projects/${projectId}/protocols/${protocolId}`),
    enabled: !!projectId && !!protocolId,
  })
}

// useProtocolsAll: same as useProtocols but aliased for components that expect { items, isLoading }
export function useProtocolsAll(projectId: string | null, includeInactive?: boolean) {
  const query = useProtocols(projectId, includeInactive)
  const items = useMemo(() => query.data ?? [], [query.data])
  return { ...query, items }
}

export function useCreateProtocol(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      name: string
      version: string
      description?: string
      definition: Record<string, unknown>
      triggers?: string[]
    }) =>
      apiFetch<Protocol>(`/api/v1/projects/${projectId}/protocols`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['protocols', projectId] })
    },
  })
}

export function useUpdateProtocol(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({
      protocolId,
      data,
    }: {
      protocolId: string
      data: Partial<{
        name: string
        version: string
        description: string
        definition: Record<string, unknown>
        triggers: string[]
      }>
    }) =>
      apiFetch<Protocol>(`/api/v1/projects/${projectId}/protocols/${protocolId}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: (protocol, { protocolId }) => {
      qc.setQueryData(['protocol', projectId, protocolId], protocol)
      qc.invalidateQueries({ queryKey: ['protocols', projectId] })
    },
  })
}

export function useDeactivateProtocol(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (protocolId: string) =>
      apiFetch<Protocol>(`/api/v1/projects/${projectId}/protocols/${protocolId}`, {
        method: 'DELETE',
      }),
    onSuccess: (_, protocolId) => {
      qc.invalidateQueries({ queryKey: ['protocols', projectId] })
      qc.invalidateQueries({ queryKey: ['protocol', projectId, protocolId] })
    },
  })
}

export function useActivateProtocol(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (protocolId: string) =>
      apiFetch<Protocol>(
        `/api/v1/projects/${projectId}/protocols/${protocolId}/activate`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (protocol, protocolId) => {
      qc.setQueryData(['protocol', projectId, protocolId], protocol)
      qc.invalidateQueries({ queryKey: ['protocols', projectId] })
    },
  })
}

export function useProtocolInstances(
  projectId: string | null,
  filters?: { protocol_id?: string; status?: string }
) {
  const qs = new URLSearchParams()
  if (filters?.protocol_id) qs.set('protocol_id', filters.protocol_id)
  if (filters?.status) qs.set('status', filters.status)

  return useQuery({
    queryKey: ['protocol-instances', projectId, filters],
    queryFn: () =>
      apiFetch<CursorPage<ProtocolInstance>>(
        `/api/v1/projects/${projectId}/protocol-instances${qs.toString() ? `?${qs}` : ''}`
      ),
    enabled: !!projectId,
  })
}

export function useProtocolInstancesAll(
  projectId: string | null,
  filters?: { protocol_id?: string; status?: string }
) {
  const qs = new URLSearchParams({ limit: '500' })
  if (filters?.protocol_id) qs.set('protocol_id', filters.protocol_id)
  if (filters?.status) qs.set('status', filters.status)

  const query = useInfiniteQuery({
    queryKey: ['protocol-instances', projectId, 'all', filters],
    queryFn: ({ pageParam }: { pageParam: string | null }) => {
      const p = new URLSearchParams(qs)
      if (pageParam) p.set('cursor', pageParam)
      return apiFetch<CursorPage<ProtocolInstance>>(
        `/api/v1/projects/${projectId}/protocol-instances?${p}`
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

export function useProtocolInstanceDetail(
  projectId: string | null,
  instanceId: string | undefined
) {
  return useQuery({
    queryKey: ['protocol-instance', projectId, instanceId],
    queryFn: () =>
      apiFetch<ProtocolInstanceDetail>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/detail`
      ),
    enabled: !!projectId && !!instanceId,
  })
}

export function useProtocolInstanceTransitions(
  projectId: string | null,
  instanceId: string | undefined
) {
  return useQuery({
    queryKey: ['protocol-instance', 'transitions', projectId, instanceId],
    queryFn: () =>
      apiFetch<ProtocolTransition[]>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/transitions`
      ),
    enabled: !!projectId && !!instanceId,
  })
}

export function useAdvanceInstance(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (instanceId: string) =>
      apiFetch<ProtocolInstance>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/advance`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (instance, instanceId) => {
      qc.setQueryData(['protocol-instance', projectId, instanceId], instance)
      qc.invalidateQueries({ queryKey: ['protocol-instances', projectId] })
      qc.invalidateQueries({
        queryKey: ['protocol-instance', 'transitions', projectId, instanceId],
      })
    },
  })
}

export function usePauseInstance(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (instanceId: string) =>
      apiFetch<ProtocolInstance>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/pause`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (instance, instanceId) => {
      qc.setQueryData(['protocol-instance', projectId, instanceId], instance)
      qc.invalidateQueries({ queryKey: ['protocol-instances', projectId] })
    },
  })
}

export function useResumeInstance(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (instanceId: string) =>
      apiFetch<ProtocolInstance>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/resume`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (instance, instanceId) => {
      qc.setQueryData(['protocol-instance', projectId, instanceId], instance)
      qc.invalidateQueries({ queryKey: ['protocol-instances', projectId] })
    },
  })
}

export function useAbandonInstance(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (instanceId: string) =>
      apiFetch<ProtocolInstance>(
        `/api/v1/projects/${projectId}/protocol-instances/${instanceId}/abandon`,
        { method: 'POST', body: JSON.stringify({}) }
      ),
    onSuccess: (instance, instanceId) => {
      qc.setQueryData(['protocol-instance', projectId, instanceId], instance)
      qc.invalidateQueries({ queryKey: ['protocol-instances', projectId] })
    },
  })
}

export function useProtocolInstanceCount(projectId: string | null, status?: string) {
  return useQuery({
    queryKey: ['protocol-instances', 'count', projectId, status],
    queryFn: () =>
      apiFetch<{ count: number }>(
        `/api/v1/projects/${projectId}/protocol-instances/count${status ? `?status=${status}` : ''}`
      ),
    enabled: !!projectId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 15_000),
    staleTime: 15_000,
  })
}
