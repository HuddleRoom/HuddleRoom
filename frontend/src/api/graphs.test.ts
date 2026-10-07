import { useQuery } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useGraphRunCount } from './graphs'
import { useWSStore } from '@/stores/ws'

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQuery: vi.fn((options) => options),
}))

vi.mock('@/stores/ws', () => ({
  useWSStore: {
    getState: vi.fn(() => ({ connected: false })),
  },
}))

afterEach(() => {
  vi.mocked(useQuery).mockClear()
})

describe('graphs API', () => {
  it('graph run count polling respects WS connection state', () => {
    const query = useGraphRunCount('project-1') as unknown as {
      refetchInterval: () => number | false
      staleTime: number
    }

    // Test disconnected state (default mock)
    expect(query.refetchInterval()).toBe(15_000)
    expect(query.staleTime).toBe(15_000)

    // Test connected state
    vi.mocked(useWSStore.getState).mockReturnValue({ connected: true })
    expect(query.refetchInterval()).toBe(false)
  })
})
