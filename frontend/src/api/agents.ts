import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { apiFetch } from '@/lib/api-client'
import type { Agent, AdapterType, Session, Task, CursorPage } from '@/lib/types'

const MAX_PAGES = 5

export interface AgentCreatePayload {
  name: string
  role: string
  provider: string
  model: string
  description?: string | null
  system_prompt?: string | null
  adapter_type?: AdapterType
  cli_runtime?: string | null
  capabilities?: string[]
  config?: Record<string, unknown>
}

export interface AgentUpdatePayload {
  name?: string
  role?: string
  description?: string | null
  provider?: string
  model?: string
  system_prompt?: string | null
  adapter_type?: AdapterType
  cli_runtime?: string | null
  capabilities?: string[]
  config?: Record<string, unknown>
  is_active?: boolean
}

export function useAgents() {
  return useQuery({
    queryKey: ['agents'],
    queryFn: () => apiFetch<CursorPage<Agent>>('/api/v1/agents?limit=500'),
  })
}

export function useActiveAgents() {
  return useQuery({
    queryKey: ['agents', 'active'],
    queryFn: () => apiFetch<CursorPage<Agent>>('/api/v1/agents?is_active=true&limit=500'),
  })
}

export function useAllActiveAgents() {
  const query = useInfiniteQuery({
    queryKey: ['agents', 'active', 'all'],
    queryFn: ({ pageParam }: { pageParam: string | null }) => {
      const p = new URLSearchParams({ is_active: 'true', limit: '100' })
      if (pageParam) p.set('cursor', pageParam)
      return apiFetch<CursorPage<Agent>>(`/api/v1/agents?${p}`)
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  })

  const { hasNextPage, isFetchingNextPage, fetchNextPage, data } = query
  useEffect(() => {
    const pageCount = data?.pages.length ?? 0
    if (hasNextPage && !isFetchingNextPage && pageCount < MAX_PAGES) {
      fetchNextPage()
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, data?.pages.length])

  const items = useMemo(
    () => query.data?.pages.flatMap((p) => p.items) ?? [],
    [query.data?.pages]
  )

  const isCapped = (query.data?.pages.length ?? 0) >= MAX_PAGES && query.hasNextPage

  return { ...query, items, isCapped }
}

export function useAllAgents() {
  const query = useInfiniteQuery({
    queryKey: ['agents', 'all'],
    queryFn: ({ pageParam }: { pageParam: string | null }) => {
      const p = new URLSearchParams({ limit: '100' })
      if (pageParam) p.set('cursor', pageParam)
      return apiFetch<CursorPage<Agent>>(`/api/v1/agents?${p}`)
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  })

  const { hasNextPage, isFetchingNextPage, fetchNextPage, data } = query
  useEffect(() => {
    const pageCount = data?.pages.length ?? 0
    if (hasNextPage && !isFetchingNextPage && pageCount < MAX_PAGES) {
      fetchNextPage()
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, data?.pages.length])

  const items = useMemo(
    () => query.data?.pages.flatMap((p) => p.items) ?? [],
    [query.data?.pages]
  )

  const isCapped = (query.data?.pages.length ?? 0) >= MAX_PAGES && query.hasNextPage

  return { ...query, items, isCapped }
}

export function useAgent(agentId: string | undefined) {
  return useQuery({
    queryKey: ['agents', agentId],
    queryFn: () => apiFetch<Agent>(`/api/v1/agents/${agentId}`),
    enabled: !!agentId,
  })
}

export function useAgentSessions(agentId: string | undefined) {
  return useQuery({
    queryKey: ['agents', 'sessions', agentId],
    queryFn: () =>
      apiFetch<CursorPage<Session>>(`/api/v1/agents/${agentId}/sessions`),
    enabled: !!agentId,
  })
}

export function useAgentTasks(projectId: string | null, agentId: string | undefined) {
  return useQuery({
    queryKey: ['tasks', projectId, { assigned_to: agentId }],
    queryFn: () =>
      apiFetch<CursorPage<Task>>(
        `/api/v1/projects/${projectId}/tasks?assigned_to=${agentId}`
      ),
    enabled: !!projectId && !!agentId,
  })
}

export function useCreateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: AgentCreatePayload) =>
      apiFetch<Agent>('/api/v1/agents', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}

export function useUpdateAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: AgentUpdatePayload }) =>
      apiFetch<Agent>(`/api/v1/agents/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    onSuccess: (_, { id }) => {
      qc.invalidateQueries({ queryKey: ['agents'] })
      qc.invalidateQueries({ queryKey: ['agents', id] })
    },
  })
}

export function useDeleteAgent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/api/v1/agents/${id}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['agents'] }),
  })
}
