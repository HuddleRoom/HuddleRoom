import React, { act } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mountWithTestDom } from '../../tests/support/dom'

describe('useOrchestrationGoal polling', () => {
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

  it('makes one detail GET at ten seconds and no tick or mutation request', async () => {
    vi.useFakeTimers()
    vi.resetModules()
    vi.stubGlobal('window', { document: {} })
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({}), { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const { QueryClient, QueryClientProvider } = await import('@tanstack/react-query')
    const { useOrchestrationGoal } = await import('./orchestration')
    function Probe() {
      useOrchestrationGoal('project-1', 'goal-1')
      return null
    }
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = await mountWithTestDom(() => <QueryClientProvider client={client}><Probe /></QueryClientProvider>, act)
    try {
      await act(async () => { await vi.advanceTimersByTimeAsync(0) })
      fetchMock.mockClear()
      await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
      expect(fetchMock).toHaveBeenCalledTimes(1)
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/v1/projects/project-1/orchestration/goals/goal-1',
        expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
      )
      expect(fetchMock.mock.calls[0][1]).not.toHaveProperty('method')
    } finally {
      view.cleanup()
      client.clear()
    }
  })
})
