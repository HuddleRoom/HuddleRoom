import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  changeControl,
  descendants,
  findElement,
  getButton,
  getByLabel,
  mountWithTestDom,
  TestEvent,
  textOf,
} from '../../../tests/support/dom'

const createProjectMutate = vi.hoisted(() => vi.fn())
const queryClientStub = {
  cancelQueries: vi.fn(),
  clear: vi.fn(),
}

vi.mock('@tanstack/react-query', () => ({
  useQuery: () => ({
    data: [{ id: 'project-1', name: 'Project One' }],
  }),
  useQueryClient: () => queryClientStub,
}))

vi.mock('@/api/projects', () => ({
  useCreateProject: () => ({
    isPending: false,
    mutate: createProjectMutate,
  }),
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
}))

vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: () => null,
  Content: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Title: ({ children }: { children: React.ReactNode }) => <h1>{children}</h1>,
  Description: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  Close: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      activeProjectId: 'project-1',
      setActiveProject: vi.fn(),
      openSidebar: vi.fn(),
      sidebarOpen: false,
    }),
}))

vi.mock('@/stores/ws', () => ({
  useWSStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      disconnect: vi.fn(),
      connected: true,
    }),
}))

vi.mock('@/lib/api-client', () => ({
  ApiError: class ApiError extends Error {
    constructor(public status: number, message: string, public detail?: unknown) {
      super(message)
    }
  },
  apiFetch: vi.fn(),
  clearToken: vi.fn(),
}))

describe('TopBar', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0)
      return 0
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  it.skip('keeps an accessible name on the logout control', async () => {
    const { TopBar } = await import('./TopBar')
    const view = await mountWithTestDom(() => <TopBar />, React.act)

    try {
      expect(getButton(view.container, 'Log out').getAttribute('aria-label')).toBe('Log out')
    } finally {
      view.cleanup()
    }
  })

  it('focuses required fields and submits a trimmed server directory', async () => {
    const { CreateProjectModal } = await import('./TopBar')
    const view = await mountWithTestDom(
      () => <CreateProjectModal open onClose={vi.fn()} />,
      React.act,
    )

    try {
      const name = getByLabel(view.container, 'Name *', 'input')
      const workspace = getByLabel(view.container, 'Server directory *', 'input')
      const submit = getButton(view.container, 'Create project')
      const form = findElement(view.container, 'form')

      expect(submit.getAttribute('disabled')).toBeNull()

      workspace.focus()
      await React.act(async () => form?.dispatchEvent(new TestEvent('submit')))
      expect(view.activeElement).toBe(name)

      await React.act(async () => changeControl(name, '  Project One  '))
      await React.act(async () => form?.dispatchEvent(new TestEvent('submit')))
      expect(view.activeElement).toBe(workspace)

      await React.act(async () => changeControl(workspace, '  /srv/huddleroom/project-one  '))
      await React.act(async () => form?.dispatchEvent(new TestEvent('submit')))

      expect(createProjectMutate).toHaveBeenCalledWith(
        {
          name: 'Project One',
          workspace_path: '/srv/huddleroom/project-one',
          description: undefined,
        },
        expect.any(Object),
      )
    } finally {
      view.cleanup()
    }
  })

  it('preserves and focuses the server directory after backend validation fails', async () => {
    const { CreateProjectModal } = await import('./TopBar')
    const { ApiError } = await import('@/lib/api-client')
    const view = await mountWithTestDom(
      () => <CreateProjectModal open onClose={vi.fn()} />,
      React.act,
    )

    try {
      const name = getByLabel(view.container, 'Name *', 'input')
      const workspace = getByLabel(view.container, 'Server directory *', 'input')

      await React.act(async () => {
        changeControl(name, 'Project One')
        changeControl(workspace, '/srv/operator-entry')
      })
      await React.act(async () => findElement(view.container, 'form')?.dispatchEvent(new TestEvent('submit')))

      const callbacks = createProjectMutate.mock.calls[0]?.[1] as {
        onError: (error: Error) => void
      }
      name.focus()
      await React.act(async () => callbacks.onError(new ApiError(422, 'HTTP 422', [{
        loc: ['body', 'workspace_path'],
        msg: 'Directory is not writable',
      }])))

      expect(workspace.value).toBe('/srv/operator-entry')
      expect(view.activeElement).toBe(workspace)
      expect(descendants(view.container).some((node) =>
        node.getAttribute('role') === 'alert' && textOf(node) === 'Directory is not writable')).toBe(true)
    } finally {
      view.cleanup()
    }
  })
})
