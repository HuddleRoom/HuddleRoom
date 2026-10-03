import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import {
  removeProjectQueries,
  resetProject,
  resetProjectDataBoundary,
  type ProjectResetResponse,
  useResetProject,
} from './projects'

const { resetReplayCursor } = vi.hoisted(() => ({
  resetReplayCursor: vi.fn(),
}))

vi.mock('@/stores/ws', () => ({
  useWSStore: { getState: () => ({ resetReplayCursor }) },
}))

vi.mock('@/lib/api-client', () => ({ apiFetch: vi.fn() }))

const apiFetchMock = vi.mocked(apiFetch)
const response: ProjectResetResponse = {
  cancelled_sessions: 2,
  cancelled_meeting_tasks: 1,
  deletions: { tasks: 3 },
}

describe('resetProject', () => {
  beforeEach(() => apiFetchMock.mockReset())

  it('posts the exact confirmation name and returns reset counts', async () => {
    apiFetchMock.mockResolvedValue(response)

    await expect(resetProject('project-1', { confirm_name: 'Alpha' })).resolves.toEqual(response)

    expect(apiFetchMock).toHaveBeenCalledWith('/api/v1/projects/project-1/reset', {
      method: 'POST',
      body: JSON.stringify({ confirm_name: 'Alpha' }),
    })
  })
})

describe('removeProjectQueries', () => {
  it('removes target caches across key shapes but preserves other project and global caches', () => {
    const queryClient = new QueryClient()
    queryClient.setQueryData(['tasks', 'project-1'], [])
    queryClient.setQueryData(['tasks', 'count', 'project-1', 'ready'], { count: 1 })
    queryClient.setQueryData(['task', 'subtasks', 'project-1', 'task-1'], [])
    queryClient.setQueryData(['protocol-instance', 'transitions', 'project-1', 'instance-1'], [])
    queryClient.getQueryCache().build(queryClient, {
      queryKey: ['meeting', 'turns', 'project-1', 'meeting-1'],
      queryFn: async () => undefined,
    })
    queryClient.getQueryCache().build(queryClient, {
      queryKey: ['knowledge-item', 'project-1', 'knowledge-1'],
      queryFn: async () => undefined,
    })
    queryClient.setQueryData(['tasks', 'project-2'], [])
    queryClient.getQueryCache().build(queryClient, {
      queryKey: ['meeting', 'turns', 'project-2', 'meeting-2'],
      queryFn: async () => undefined,
    })
    queryClient.getQueryCache().build(queryClient, {
      queryKey: ['knowledge-item', 'project-2', 'knowledge-2'],
      queryFn: async () => undefined,
    })
    queryClient.setQueryData(['config'], { auth_enabled: true })

    removeProjectQueries(queryClient, 'project-1')

    expect(queryClient.getQueryState(['tasks', 'project-1'])).toBeUndefined()
    expect(queryClient.getQueryState(['tasks', 'count', 'project-1', 'ready'])).toBeUndefined()
    expect(queryClient.getQueryState(['task', 'subtasks', 'project-1', 'task-1'])).toBeUndefined()
    expect(queryClient.getQueryState(['protocol-instance', 'transitions', 'project-1', 'instance-1'])).toBeUndefined()
    expect(queryClient.getQueryState(['meeting', 'turns', 'project-1', 'meeting-1'])).toBeUndefined()
    expect(queryClient.getQueryState(['knowledge-item', 'project-1', 'knowledge-1'])).toBeUndefined()
    expect(queryClient.getQueryState(['tasks', 'project-2'])).toBeDefined()
    expect(queryClient.getQueryState(['meeting', 'turns', 'project-2', 'meeting-2'])).toBeDefined()
    expect(queryClient.getQueryState(['knowledge-item', 'project-2', 'knowledge-2'])).toBeDefined()
    expect(queryClient.getQueryState(['config'])).toBeDefined()
  })
})

describe('resetProjectDataBoundary', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    resetReplayCursor.mockReset()
  })

  it('uses a real query client and reports refresh errors in strict order', async () => {
    const order: string[] = []
    const errors: Array<[string, unknown]> = []
    const projectError = new Error('project refresh failed')
    const listError = new Error('list refresh failed')
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    let seedList = true
    await queryClient.prefetchQuery({
      queryKey: ['projects'],
      queryFn: async () => {
        if (seedList) return []
        order.push('projects')
        throw listError
      },
    })
    seedList = false
    queryClient.setQueryData(['tasks', 'project-1'], [])
    const removeQueries = queryClient.removeQueries.bind(queryClient)
    vi.spyOn(queryClient, 'removeQueries').mockImplementation((filters) => {
      order.push('remove')
      removeQueries(filters)
    })
    apiFetchMock.mockImplementation(async () => {
      order.push('project')
      throw projectError
    })
    resetReplayCursor.mockImplementation(() => order.push('cursor'))

    await expect(resetProjectDataBoundary(
      queryClient,
      'project-1',
      (phase, error) => errors.push([phase, error]),
    )).resolves.toBeUndefined()

    expect(order).toEqual(['remove', 'cursor', 'project', 'projects'])
    expect(errors).toEqual([
      ['project', projectError],
      ['projects', listError],
    ])
    expect(queryClient.getQueryState(['tasks', 'project-1'])).toBeUndefined()
  })
})

describe('useResetProject', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    resetReplayCursor.mockReset()
  })

  function renderMutation(queryClient: QueryClient) {
    let mutation: ReturnType<typeof useResetProject> | undefined
    function Harness() {
      mutation = useResetProject('project-1')
      return null
    }
    renderToString(createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(Harness),
    ))
    return mutation!
  }

  it('keeps mutateAsync successful when post-reset refresh fails', async () => {
    const refreshError = new Error('refresh failed')
    apiFetchMock
      .mockResolvedValueOnce(response)
      .mockRejectedValueOnce(refreshError)
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const mutation = renderMutation(new QueryClient({
      defaultOptions: { queries: { retry: false } },
    }))

    await expect(mutation.mutateAsync({ confirm_name: 'Alpha' })).resolves.toEqual(response)
    expect(consoleError).toHaveBeenCalledWith(
      '[projects] reset project refresh failed',
      refreshError,
    )
    consoleError.mockRestore()
  })

  it('rejects mutateAsync when the reset request fails', async () => {
    const resetError = new Error('reset failed')
    apiFetchMock.mockRejectedValueOnce(resetError)
    const mutation = renderMutation(new QueryClient({
      defaultOptions: { queries: { retry: false } },
    }))

    await expect(mutation.mutateAsync({ confirm_name: 'Alpha' })).rejects.toBe(resetError)
    expect(resetReplayCursor).not.toHaveBeenCalled()
  })
})
