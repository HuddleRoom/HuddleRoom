import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  descendants,
  mountWithTestDom,
  textOf,
} from '../tests/support/dom'
import type { Project } from '@/lib/types'

const projectState = vi.hoisted(() => ({
  projects: [] as Project[],
  isSuccess: true,
}))

const wsState = vi.hoisted(() => ({
  connect: vi.fn(),
  disconnect: vi.fn(),
  connected: false,
  retrying: false,
  connecting: false,
  hasConnectedOnce: false,
}))

const configState = vi.hoisted(() => ({
  promise: null as Promise<void> | null,
  resolve: (() => {}) as () => void,
}))

vi.mock('@tanstack/react-query', () => ({
  useQuery: () => ({ data: projectState.projects, isSuccess: projectState.isSuccess }),
}))

vi.mock('react-router-dom', () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => (
    <a href={to}>{children}</a>
  ),
  Navigate: ({ to }: { to: string }) => <div data-testid="navigate" data-to={to} />,
  Outlet: () => null,
}))

vi.mock('sonner', () => ({
  Toaster: () => null,
}))

vi.mock('@/components/layout/Sidebar', () => ({
  Sidebar: () => <nav data-testid="sidebar">sidebar</nav>,
}))

vi.mock('@/components/layout/TopBar', () => ({
  TopBar: () => <header data-testid="topbar">topbar</header>,
}))

vi.mock('@/hooks/useWSQuerySync', () => ({
  useWSQuerySync: () => undefined,
}))

vi.mock('@/stores/auth', () => ({
  useAuthStore: (selector: (state: { token: string | null }) => unknown) => selector({ token: null }),
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) => selector({
    activeProjectId: null,
    sidebarCollapsed: false,
    sidebarOpen: false,
    closeSidebar: vi.fn(),
  }),
}))

vi.mock('@/stores/ws', () => {
  const useWSStore = Object.assign(
    (selector?: (state: typeof wsState) => unknown) => (selector ? selector(wsState) : wsState),
    { getState: () => wsState },
  )
  return { useWSStore }
})

vi.mock('@/lib/api-client', () => ({
  apiFetch: vi.fn().mockResolvedValue({ items: [] }),
  isAuthDisabled: () => true,
  loadRuntimeConfig: () => {
    if (!configState.promise) {
      configState.promise = new Promise<void>((resolve) => { configState.resolve = resolve })
    }
    return configState.promise
  },
}))

describe('RequireAuth', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    configState.promise = null
    configState.resolve = () => {}
  })

  it('renders the shell chrome synchronously and shows a connecting notice until config resolves, then removes it', async () => {
    const { RequireAuth } = await import('./App')
    const view = await mountWithTestDom(() => <RequireAuth />, React.act)

    try {
      // Chrome is present immediately, before /api/v1/config resolves.
      expect(descendants(view.container).some((n) => n.getAttribute('data-testid') === 'sidebar')).toBe(true)
      expect(descendants(view.container).some((n) => n.getAttribute('data-testid') === 'topbar')).toBe(true)

      const connectingNotice = descendants(view.container).find((n) => textOf(n).includes('Connecting to HuddleRoom'))
      expect(connectingNotice).toBeDefined()

      await React.act(async () => {
        configState.resolve()
        await configState.promise
      })
      await view.rerender()

      expect(descendants(view.container).some((n) => textOf(n).includes('Connecting to HuddleRoom'))).toBe(false)
      // Chrome remains present after resolution.
      expect(descendants(view.container).some((n) => n.getAttribute('data-testid') === 'sidebar')).toBe(true)
    } finally {
      view.cleanup()
    }
  })
})
