import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiFetch } from '@/lib/api-client'
import { useSearchProjectMemory } from './memory'

const { useMutationMock } = vi.hoisted(() => ({
  useMutationMock: vi.fn((options) => ({
    mutate: options.mutationFn,
    mutateAsync: options.mutationFn,
  })),
}))

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useMutation: useMutationMock,
}))

vi.mock('@/lib/api-client', () => ({ apiFetch: vi.fn() }))

const apiFetchMock = vi.mocked(apiFetch)

describe('useSearchProjectMemory', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    useMutationMock.mockClear()
  })

  it('unwraps { results, count } response and returns array', async () => {
    const mockItem = { id: 'm1', content: 'test memory', created_at: '2024-01-01', scope: 'project' as const }
    apiFetchMock.mockResolvedValue({ results: [mockItem], count: 1 })

    const mutation = useSearchProjectMemory('proj-1')
    const result = await mutation.mutateAsync({ query: 'test' })

    expect(result).toEqual([mockItem])
    expect(result).toHaveLength(1)
    expect(Array.isArray(result)).toBe(true)
  })
})
