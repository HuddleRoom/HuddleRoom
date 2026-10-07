import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi, beforeEach } from 'vitest'

// ---------------------------------------------------------------------------
// Mutable mock state — must be declared via vi.hoisted so it is available
// inside vi.mock factory closures (which are hoisted before top-level code).
// ---------------------------------------------------------------------------
const mockCfg = vi.hoisted(() => ({
  activeProjectId: null as string | null,
  eventsData: [] as unknown[] | undefined,
  eventsIsLoading: false,
  eventsIsError: false,
  goalsData: undefined as unknown[] | undefined,
}))

// ---------------------------------------------------------------------------
// Module mocks
// ---------------------------------------------------------------------------

vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
  Link: ({ to, children, ...rest }: { to: string; children?: React.ReactNode }) =>
    React.createElement('a', { href: to, ...rest }, children),
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: () => ({ activeProjectId: mockCfg.activeProjectId }),
}))

vi.mock('@/api/tasks', () => ({
  useTaskCount: () => ({ data: undefined, isLoading: false, isError: false }),
}))

vi.mock('@/api/sessions', () => ({
  useSessionCount: () => ({ data: undefined, isLoading: false, isError: false }),
}))

vi.mock('@/api/meetings', () => ({
  useMeetingCount: () => ({ data: undefined, isLoading: false, isError: false }),
}))

vi.mock('@/api/graphs', () => ({
  useGraphRunCount: () => ({ data: undefined, isLoading: false, isError: false }),
}))

vi.mock('@/api/events', () => ({
  useRecentEvents: () => ({
    data: mockCfg.eventsData,
    isLoading: mockCfg.eventsIsLoading,
    isError: mockCfg.eventsIsError,
    refetch: vi.fn(),
  }),
}))

vi.mock('@/api/orchestration', () => ({
  useOrchestrationGoals: () => ({
    data: mockCfg.goalsData ? { pages: [{ items: mockCfg.goalsData, next_cursor: null }] } : undefined,
    isLoading: false,
    isError: false,
  }),
}))

// ---------------------------------------------------------------------------
// Reset mock state before each test so tests are independent.
// ---------------------------------------------------------------------------
beforeEach(() => {
  mockCfg.activeProjectId = null
  mockCfg.eventsData = []
  mockCfg.eventsIsLoading = false
  mockCfg.eventsIsError = false
  mockCfg.goalsData = undefined
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('DashboardPage', () => {
  it('turns the no-project state into guided onboarding copy', async () => {
    const { DashboardPage } = await import('./DashboardPage')

    const markup = renderToStaticMarkup(<DashboardPage />)

    expect(markup).toContain('Dashboard')
    expect(markup).toContain('Select a project to see what needs attention')
    expect(markup).toContain('New project')
  })

  /**
   * Regression: before QueryState migration, a failed events fetch was silently
   * rendered as the empty "No recent activity" state. The dashboard must now
   * surface the error state (error label + Retry) instead of masking it.
   */
  it('shows error state in event feed when events query fails — not "No recent activity"', async () => {
    mockCfg.activeProjectId = 'proj-test-001'
    mockCfg.eventsData = undefined
    mockCfg.eventsIsError = true

    const { DashboardPage } = await import('./DashboardPage')
    const markup = renderToStaticMarkup(<DashboardPage />)

    // Error label and Retry button must appear.
    expect(markup).toContain('Failed to load recent events')
    expect(markup).toContain('Retry')

    // The silent empty-state copy must NOT appear — that was the original bug.
    expect(markup).not.toContain('No recent activity')
  })

  it('humanizes feed event labels and keeps raw code as secondary mono', async () => {
    const { FeedRow } = await import('./DashboardPage')

    const markup = renderToStaticMarkup(
      <FeedRow
        event={{
          id: 'evt-1',
          project_id: 'proj-test-001',
          event_type: 'meeting.turn_complete',
          payload: {},
          source: 'meeting-runner',
          emitted_at: new Date().toISOString(),
        }}
        count={1}
        isLast
      />
    )

    expect(markup).toContain('Agent turn completed')
    expect(markup).toContain('meeting.turn_complete')
    // Raw code stays secondary mono — primary/secondary rule guard.
    expect(markup).toContain('font-mono')
    expect(markup).toContain('text-xs')
  })

  it('links each stat tile to its section', async () => {
    mockCfg.activeProjectId = 'proj-test-001'

    const { DashboardPage } = await import('./DashboardPage')
    const markup = renderToStaticMarkup(<DashboardPage />)

    expect(markup).toContain('href="/agents"')
  })

  it('collapses consecutive trace events into one row with count', async () => {
    mockCfg.activeProjectId = 'proj-test-001'
    mockCfg.eventsData = [
      { id: 'e1', project_id: 'proj-test-001', event_type: 'meeting.trace', payload: {}, source: 'meeting-runner', emitted_at: new Date().toISOString() },
      { id: 'e2', project_id: 'proj-test-001', event_type: 'meeting.trace', payload: {}, source: 'meeting-runner', emitted_at: new Date().toISOString() },
      { id: 'e3', project_id: 'proj-test-001', event_type: 'meeting.trace', payload: {}, source: 'meeting-runner', emitted_at: new Date().toISOString() },
      { id: 'e4', project_id: 'proj-test-001', event_type: 'meeting.concluded', payload: {}, source: 'meeting-runner', emitted_at: new Date().toISOString() },
    ]

    const { DashboardPage } = await import('./DashboardPage')
    const markup = renderToStaticMarkup(<DashboardPage />)

    // Collapsed into 2 rows total (3x trace -> 1 row, 1x concluded -> 1 row).
    const rowMatches = markup.match(/queue-row-in/g)
    expect(rowMatches).toHaveLength(2)

    expect(markup).toContain('×3')
    expect(markup).toContain('Meeting trace')
    expect(markup).toContain('Meeting concluded')
  })

  it('shows blocked goals chip and hides the healthy check when a goal is blocked', async () => {
    mockCfg.activeProjectId = 'proj-test-001'
    mockCfg.goalsData = [{ id: 'goal-1', status: 'blocked', needs_you_count: 0 }]

    const { DashboardPage } = await import('./DashboardPage')
    const markup = renderToStaticMarkup(<DashboardPage />)

    expect(markup).toContain('blocked goals')
    expect(markup).not.toContain('System healthy')
  })
})

describe('buildAttentionSummary', () => {
  it('reports steady state when nothing is going on', async () => {
    const { buildAttentionSummary } = await import('./DashboardPage')

    expect(buildAttentionSummary({})).toEqual({
      title: 'System steady',
      detail: 'No active work in this project right now.',
    })
  })

  it('reports unavailable status when any query errored', async () => {
    const { buildAttentionSummary } = await import('./DashboardPage')

    expect(buildAttentionSummary({ hasAnyError: true })).toEqual({
      title: 'Status unavailable',
      detail: 'One or more status queries failed. Check connectivity.',
    })
  })

  it('summarizes ready tasks', async () => {
    const { buildAttentionSummary } = await import('./DashboardPage')

    const result = buildAttentionSummary({ tasks: 3 })
    expect(result.title).toBe('Project activity')
    expect(result.detail).toContain('3 ready tasks')
  })
})
