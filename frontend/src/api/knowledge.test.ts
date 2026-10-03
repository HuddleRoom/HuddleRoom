import { QueryClient, QueryObserver } from '@tanstack/react-query'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiFetch } from '@/lib/api-client'
import { knowledgeItemKey, useKnowledgeItem, useSearchKnowledge } from './knowledge'

const { useQueryMock, useMutationMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn((options) => options),
  useMutationMock: vi.fn((options) => ({
    mutate: options.mutationFn,
    mutateAsync: options.mutationFn,
  })),
}))

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQuery: useQueryMock,
  useMutation: useMutationMock,
}))

vi.mock('@/lib/api-client', () => ({ apiFetch: vi.fn() }))

const apiFetchMock = vi.mocked(apiFetch)

describe('knowledge query keys', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    useQueryMock.mockClear()
    useMutationMock.mockClear()
  })

  it('includes the project in child item keys', () => {
    expect(knowledgeItemKey('project-1', 'knowledge-1')).toEqual([
      'knowledge-item',
      'project-1',
      'knowledge-1',
    ])
  })

  it('does not fetch or cache a knowledge child without a project', async () => {
    apiFetchMock.mockResolvedValue({ id: 'knowledge-1' })
    useKnowledgeItem(null, 'knowledge-1')

    const options = useQueryMock.mock.calls[0][0]
    const queryClient = new QueryClient()
    const observer = new QueryObserver(queryClient, options)
    const unsubscribe = observer.subscribe(() => undefined)
    await Promise.resolve()
    unsubscribe()

    expect(apiFetchMock).not.toHaveBeenCalled()
    expect(queryClient.getQueryData(options.queryKey)).toBeUndefined()
  })
})

describe('useSearchKnowledge', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    useMutationMock.mockClear()
  })

  it('unwraps { results } response and returns array', async () => {
    const mockItem = { id: 'k1', content: 'test', content_type: 'text' as const, relevance_score: 0.95 }
    apiFetchMock.mockResolvedValue({ results: [mockItem] })

    const mutation = useSearchKnowledge('proj-1')
    const result = await mutation.mutateAsync('test query')

    expect(result).toEqual([mockItem])
    expect(result).toHaveLength(1)
    expect(Array.isArray(result)).toBe(true)
  })
})
