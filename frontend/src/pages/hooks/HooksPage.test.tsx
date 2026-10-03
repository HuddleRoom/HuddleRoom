import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Hook } from '@/lib/types'

const uiState = { activeProjectId: 'project-1' as string | null }
let hooksMock: Hook[] = []

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: { activeProjectId: string | null }) => unknown) => selector(uiState),
}))

vi.mock('@/api/hooks', () => ({
  useHooksList: () => ({
    data: { items: hooksMock },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
  useCreateHook: () => ({
    isPending: false,
    mutateAsync: vi.fn(),
  }),
  useUpdateHook: () => ({
    isPending: false,
    mutateAsync: vi.fn(),
  }),
  useDeleteHook: () => ({
    isPending: false,
    mutateAsync: vi.fn(),
  }),
}))

vi.mock('@/hooks/useDocumentTitle', () => ({
  useDocumentTitle: vi.fn(),
}))

describe('HooksPage', () => {
  beforeEach(() => {
    hooksMock = []
  })

  it('renders a single empty state when all groups are empty', async () => {
    const { HooksPage } = await import('./HooksPage')

    const markup = renderToStaticMarkup(<HooksPage />)

    expect(markup).toContain('No hooks yet')
    expect(markup).toContain('Hooks appear here once generated or added.')
    // Verify only one empty state block
    const emptyStateCount = (markup.match(/No hooks yet/g) || []).length
    expect(emptyStateCount).toBe(1)
  })

  it('renders per-section empty states when at least one group has hooks', async () => {
    hooksMock = [
      {
        id: 'hook-1',
        name: 'Test Hook',
        trigger_event: 'test.event',
        description: 'A test hook',
        code: 'print("test")',
        status: 'active',
        execution_count: 0,
        error_count: 0,
        created_at: '2026-09-16T00:00:00Z',
        updated_at: '2026-09-16T00:00:00Z',
      },
    ]

    const { HooksPage } = await import('./HooksPage')

    const markup = renderToStaticMarkup(<HooksPage />)

    // Should show section headers
    expect(markup).toContain('proposed')
    expect(markup).toContain('active')
    expect(markup).toContain('shadow')
    expect(markup).toContain('disabled')
    // Should show per-section empty states
    expect(markup).toContain('No proposed hooks')
    expect(markup).toContain('No shadow hooks')
  })
})
