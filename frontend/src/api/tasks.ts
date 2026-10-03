import { useQuery, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo } from 'react'
import { apiFetch } from '@/lib/api-client'
import { useWSStore } from '@/stores/ws'
import type { Task, Session, CursorPage, TaskStatus } from '@/lib/types'

export function useTaskCount(projectId: string | null, status?: string) {
  return useQuery({
    queryKey: ['tasks', 'count', projectId, status],
    queryFn: () =>
      apiFetch<{ count: number }>(
        `/api/v1/projects/${projectId}/tasks/count${status ? `?status=${status}` : ''}`
      ),
    enabled: !!projectId,
    refetchInterval: () => (useWSStore.getState().connected ? false : 15_000),
    staleTime: 15_000,
  })
}

export function useTasks(
  projectId: string | null,
  params?: { assigned_to?: string; status?: string; limit?: number }
) {
  const qs = new URLSearchParams()
  if (params?.assigned_to) qs.set('assigned_to', params.assigned_to)
  if (params?.status) qs.set('status', params.status)
  if (params?.limit) qs.set('limit', String(params.limit))
  const q = qs.toString()

  return useQuery({
    queryKey: ['tasks', projectId, params],
    queryFn: () =>
      apiFetch<CursorPage<Task>>(
        `/api/v1/projects/${projectId}/tasks${q ? `?${q}` : ''}`
      ),
    enabled: !!projectId,
  })
}

export function useAllTasks(
  projectId: string | null,
  params?: { assigned_to?: string; status?: string }
) {
  const qs = new URLSearchParams({ limit: '500' })
  if (params?.assigned_to) qs.set('assigned_to', params.assigned_to)
  if (params?.status) qs.set('status', params.status)

  const query = useInfiniteQuery({
    queryKey: ['tasks', projectId, 'all', params],
    queryFn: ({ pageParam }: { pageParam: string | null }) => {
      const p = new URLSearchParams(qs)
      if (pageParam) p.set('cursor', pageParam)
      return apiFetch<CursorPage<Task>>(`/api/v1/projects/${projectId}/tasks?${p}`)
    },
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled: !!projectId,
  })

  // Auto-fetch all remaining pages
  const { hasNextPage, isFetchingNextPage, fetchNextPage } = query
  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage()
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])

  const items = useMemo(
    () => query.data?.pages.flatMap((p) => p.items) ?? [],
    [query.data]
  )

  return { ...query, items }
}

export function useTask(projectId: string | null, taskId: string | undefined) {
  return useQuery({
    queryKey: ['task', projectId, taskId],
    queryFn: () => apiFetch<Task>(`/api/v1/projects/${projectId}/tasks/${taskId}`),
    enabled: !!projectId && !!taskId,
  })
}

export function useTaskSubtasks(projectId: string | null, taskId: string | undefined) {
  return useQuery({
    queryKey: ['task', 'subtasks', projectId, taskId],
    queryFn: () =>
      apiFetch<Task[]>(`/api/v1/projects/${projectId}/tasks/${taskId}/subtasks`),
    enabled: !!projectId && !!taskId,
  })
}

export function useTaskSessions(projectId: string | null, taskId: string | undefined) {
  return useQuery({
    queryKey: ['task', 'sessions', projectId, taskId],
    queryFn: () =>
      apiFetch<CursorPage<Session>>(
        `/api/v1/projects/${projectId}/tasks/${taskId}/sessions`
      ),
    enabled: !!projectId && !!taskId,
  })
}

export function useCreateTask(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: {
      title: string
      description?: string
      priority: number
      assigned_to?: string
    }) =>
      apiFetch<Task>(`/api/v1/projects/${projectId}/tasks`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tasks', projectId] })
      qc.invalidateQueries({ queryKey: ['tasks', 'count', projectId] })
    },
  })
}

export function useCopyTask(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (taskId: string) =>
      apiFetch<Task>(`/api/v1/projects/${projectId}/tasks/${taskId}/copy`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (task) => {
      qc.setQueryData(['task', projectId, task.id], task)
      qc.invalidateQueries({ queryKey: ['tasks', projectId] })
      qc.invalidateQueries({ queryKey: ['tasks', 'count', projectId] })
    },
  })
}

export function usePatchTaskStatus(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ taskId, status }: { taskId: string; status: TaskStatus }) =>
      apiFetch<Task>(`/api/v1/projects/${projectId}/tasks/${taskId}/status`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      }),
    onSuccess: (task, { taskId }) => {
      qc.setQueryData(['task', projectId, taskId], task)
      qc.invalidateQueries({ queryKey: ['tasks', projectId] })
      qc.invalidateQueries({ queryKey: ['tasks', 'count', projectId] })
    },
  })
}

interface RunTaskResponse {
  task: Task
  session_id: string
}

export function useRunTask(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (taskId: string) =>
      apiFetch<RunTaskResponse>(`/api/v1/projects/${projectId}/tasks/${taskId}/run`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: (data, taskId) => {
      qc.setQueryData(['task', projectId, taskId], data.task)
      qc.invalidateQueries({ queryKey: ['tasks', projectId] })
      qc.invalidateQueries({ queryKey: ['tasks', 'count', projectId] })
      qc.invalidateQueries({ queryKey: ['task', 'sessions', projectId, taskId] })
      qc.invalidateQueries({ queryKey: ['sessions', 'count', projectId] })
    },
  })
}

export function useResumeSession(projectId: string | null) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiFetch<Session>(`/api/v1/sessions/${sessionId}/resume`, {
        method: 'POST',
        body: JSON.stringify({}),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['task', 'sessions', projectId] })
    },
  })
}
