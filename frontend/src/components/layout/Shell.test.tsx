import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  descendants,
  mountWithTestDom,
  textOf,
} from '../../../tests/support/dom'
import type { Project } from '@/lib/types'

const projectState = vi.hoisted(() => ({
  projects: [] as Project[],
  isSuccess: true,
}))

const wsState = vi.hoisted(() => ({
  connect: vi.fn(),
  disconnect: vi.fn(),
  manualReconnect: vi.fn(),
  connected: false,
  retrying: false,
  connecting: false,
  hasConnectedOnce: false,
  gaveUp: false,
  events: [] as any[],
}))

vi.mock('@tanstack/react-query', () => ({
  // ponytail: keyed by queryKey[0] so the Shell's projects query keeps its
  // fixture while other useQuery callers mounted under Shell (e.g. the
  // project-advisor rail query) get an inert idle result instead of the
  // projects array — mirrors AgentActivityPanel.test.tsx's mock.
  useQuery: (opts: { queryKey?: unknown[] }) =>
    Array.isArray(opts?.queryKey) && opts.queryKey[0] === 'projects'
      ? { data: projectState.projects, isSuccess: projectState.isSuccess }
      : { data: undefined, isLoading: false, isError: false, refetch: () => {} },
  useMutation: () => ({ mutate: () => {}, isPending: false, isError: false, error: null }),
  useQueryClient: () => ({ invalidateQueries: () => Promise.resolve() }),
}))

vi.mock('react-router-dom', () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  ),
  Outlet: () => null,
}))

vi.mock('sonner', () => ({
  Toaster: () => null,
}))

vi.mock('./Sidebar', () => ({
  Sidebar: () => null,
}))

vi.mock('./TopBar', () => ({
  TopBar: () => null,
}))

vi.mock('@/hooks/useWSQuerySync', () => ({
  useWSQuerySync: () => undefined,
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: (selector: (state: { token: null }) => unknown) => selector({ token: null }),
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) => selector({
    activeProjectId: 'project-1',
    sidebarCollapsed: false,
    sidebarOpen: false,
    closeSidebar: vi.fn(),
  }),
}))

vi.mock('@/stores/ws', () => {
  const useWSStore = Object.assign(
    (selector?: (state: typeof wsState) => unknown) => {
      if (!selector) return wsState
      return selector(wsState)
    },
    {
      getState: () => wsState,
      setState: (updates: any) => {
        if (typeof updates === 'function') {
          Object.assign(wsState, updates(wsState))
        } else {
          Object.assign(wsState, updates)
        }
      },
    },
  )
  return { useWSStore }
})

vi.mock('@/lib/api-client', () => ({
  apiFetch: vi.fn(),
  isAuthDisabled: () => false,
}))

const project: Project = {
  id: 'project-1',
  name: 'Project One',
  workspace_path: null,
  config: {},
  created_at: '2026-07-29T00:00:00Z',
  updated_at: '2026-07-29T00:00:00Z',
}

describe('Shell workspace warning', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    projectState.isSuccess = true
  })

  it('does not show the workspace warning while the projects query is still loading', async () => {
    projectState.isSuccess = false
    projectState.projects = []
    const { Shell } = await import('./Shell')
    const view = await mountWithTestDom(() => <Shell />, React.act)

    try {
      const warning = descendants(view.container).find((node) =>
        textOf(node).includes('Server directory required'))
      expect(warning).toBeUndefined()
    } finally {
      view.cleanup()
    }
  })

  it.each([null, ''])(
    'shows the Settings link for workspace_path %j and removes it after query refresh',
    async (workspacePath) => {
      projectState.projects = [{ ...project, workspace_path: workspacePath }]
      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)

      try {
        const warning = descendants(view.container).find((node) =>
          textOf(node).includes('Server directory required'))
        const settingsLink = descendants(view.container).find((node) =>
          node.tagName === 'A' && textOf(node) === 'Open project settings')

        expect(warning).toBeDefined()
        expect(textOf(warning!)).toContain(
          'Execution is not enabled for this project until a server directory is configured.',
        )
        expect(settingsLink?.getAttribute('href')).toBe('/settings')

        projectState.projects = [{ ...project, workspace_path: '/srv/project-one' }]
        await view.rerender()

        expect(descendants(view.container).some((node) =>
          textOf(node).includes('Server directory required'))).toBe(false)
        expect(descendants(view.container).some((node) =>
          node.tagName === 'A' && textOf(node) === 'Open project settings')).toBe(false)
      } finally {
        view.cleanup()
      }
    },
  )
})

describe('Shell gaveUp banner', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    projectState.isSuccess = true
    projectState.projects = [project]
    Object.assign(wsState, { gaveUp: false, connected: false, retrying: false, connecting: false, hasConnectedOnce: false })
  })

  it('renders the offline Banner with a Reconnect action when gaveUp is true', async () => {
    wsState.gaveUp = true
    const { Shell } = await import('./Shell')
    const view = await mountWithTestDom(() => <Shell />, React.act)

    try {
      const banner = descendants(view.container).find((node) =>
        textOf(node).includes('Live updates stopped'))
      expect(banner).toBeDefined()

      const reconnectButton = descendants(view.container).find((node) =>
        node.tagName === 'BUTTON' && textOf(node) === 'Reconnect')
      expect(reconnectButton).toBeDefined()

      reconnectButton!.click()
      expect(wsState.manualReconnect).toHaveBeenCalled()
    } finally {
      view.cleanup()
    }
  })

  it('does not render the Banner when gaveUp is false', async () => {
    wsState.gaveUp = false
    const { Shell } = await import('./Shell')
    const view = await mountWithTestDom(() => <Shell />, React.act)

    try {
      const banner = descendants(view.container).find((node) =>
        textOf(node).includes('Live updates stopped'))
      expect(banner).toBeUndefined()
    } finally {
      view.cleanup()
    }
  })
})
