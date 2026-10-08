import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  changeControl,
  descendants,
  getButton,
  getByLabel,
  mountWithTestDom,
  TestEvent,
  textOf,
} from '../../../tests/support/dom'
import type { Project } from '@/lib/types'

const apiFetchMock = vi.hoisted(() => vi.fn())
const toastSuccess = vi.hoisted(() => vi.fn())
const toastError = vi.hoisted(() => vi.fn())
const routeState = vi.hoisted(() => ({ hash: '', activeProjectId: 'project-1' as string | null }))

vi.mock('react-router-dom', () => ({
  useLocation: () => ({ pathname: '/settings', search: '', hash: routeState.hash }),
}))

vi.mock('@/api/auth', () => ({
  useApiKeys: () => ({ data: [], isLoading: false, isError: false, refetch: vi.fn() }),
  useCreateApiKey: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useRevokeApiKey: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCurrentUser: () => ({
    data: { id: 'user-1', email: 'operator@example.com', display_name: 'Operator', role: 'admin' },
    isLoading: false,
    isError: false,
    refetch: vi.fn(),
  }),
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: () => ({ activeProjectId: routeState.activeProjectId }),
}))

vi.mock('@/stores/ws', () => ({
  useWSStore: { getState: () => ({ resetReplayCursor: vi.fn() }) },
}))

vi.mock('@monaco-editor/react', () => ({
  default: () => <textarea aria-label="Configuration JSON" />,
}))

vi.mock('sonner', () => ({
  toast: { error: toastError, success: toastSuccess },
}))

vi.mock('@/lib/api-client', () => {
  class ApiError extends Error {
    constructor(public status: number, message: string, public detail?: unknown) {
      super(message)
    }
  }
  return { ApiError, apiFetch: apiFetchMock }
})

const project: Project = {
  id: 'project-1',
  name: 'Project One',
  workspace_path: '/srv/original',
  config: {},
  created_at: '2026-07-29T00:00:00Z',
  updated_at: '2026-07-29T00:00:00Z',
}

function createClient() {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Infinity },
      mutations: { retry: false },
    },
  })
  client.setQueryData(['project', project.id], project)
  return client
}

async function mountSettings(client: QueryClient) {
  const { SettingsPage } = await import('./SettingsPage')
  return mountWithTestDom(
    () => (
      <QueryClientProvider client={client}>
        <SettingsPage />
      </QueryClientProvider>
    ),
    React.act,
  )
}

function submit(button: ReturnType<typeof getButton>) {
  return button.closest('form')?.dispatchEvent(new TestEvent('submit'))
}

function byTestId(root: Parameters<typeof descendants>[0], testId: string) {
  const node = descendants(root).find((el) => el.getAttribute('data-testid') === testId)
  if (!node) throw new Error(`No element with data-testid="${testId}"`)
  return node
}

describe('ProjectConfigSection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0)
      return 0
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  it('saves the trimmed path, announces success, refreshes caches, and leaves other controls usable', async () => {
    let resolveSave!: (value: Project) => void
    apiFetchMock.mockImplementation((_url: string, options?: RequestInit) => {
      if (options?.method === 'PUT') {
        return new Promise<Project>((resolve) => { resolveSave = resolve })
      }
      return Promise.resolve({ ...project, workspace_path: '/srv/canonical' })
    })
    const client = createClient()
    const invalidateQueries = vi.spyOn(client, 'invalidateQueries')
    const view = await mountSettings(client)

    try {
      const workspace = getByLabel(view.container, 'Server directory', 'input')
      const saveWorkspace = getButton(view.container, 'Save server directory')
      const saveConfig = getButton(view.container, 'Save config')
      const archive = getButton(view.container, 'Archive project')

      expect(workspace.value).toBe('/srv/original')
      await React.act(async () => changeControl(workspace, '  /srv/canonical/../canonical  '))
      await React.act(async () => { submit(saveWorkspace); await Promise.resolve() })

      expect(saveConfig.getAttribute('disabled')).toBeNull()
      expect(archive.getAttribute('disabled')).toBeNull()

      await React.act(async () => {
        resolveSave({ ...project, workspace_path: '/srv/canonical' })
        await Promise.resolve()
      })

      expect(apiFetchMock).toHaveBeenCalledWith(
        '/api/v1/projects/project-1',
        {
          method: 'PUT',
          body: JSON.stringify({ workspace_path: '/srv/canonical/../canonical' }),
        },
      )
      expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ['project', 'project-1'] })
      expect(invalidateQueries).toHaveBeenCalledWith({ queryKey: ['projects'] })
      expect(workspace.value).toBe('/srv/canonical')
      const status = descendants(view.container).find((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Server directory saved')
      expect(status?.getAttribute('aria-live')).toBe('polite')
      expect(toastSuccess).toHaveBeenCalledWith('Server directory saved')
    } finally {
      view.cleanup()
      client.clear()
    }
  })

  it.each([
    {
      label: '422',
      error: () => {
        const detail = [{ loc: ['body', 'workspace_path'], msg: 'Directory is not writable' }]
        return import('@/lib/api-client').then(({ ApiError }) => new ApiError(422, 'HTTP 422', detail))
      },
      message: 'Directory is not writable',
    },
    {
      label: '409',
      error: () => import('@/lib/api-client').then(({ ApiError }) => new ApiError(409, 'active work')),
      message: 'Cannot change the server directory while project work is active.',
    },
  ])('preserves and focuses the workspace field after a $label response', async ({ error, message }) => {
    apiFetchMock.mockRejectedValue(await error())
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const workspace = getByLabel(view.container, 'Server directory', 'input')
      await React.act(async () => changeControl(workspace, '/srv/operator-entry'))
      await React.act(async () => {
        submit(getButton(view.container, 'Save server directory'))
        await Promise.resolve()
      })

      expect(workspace.value).toBe('/srv/operator-entry')
      expect(view.activeElement).toBe(workspace)
      expect(descendants(view.container).some((node) =>
        node.getAttribute('role') === 'alert' && textOf(node) === message)).toBe(true)
    } finally {
      view.cleanup()
      client.clear()
    }
  })
})

describe('reset danger zone', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0)
      return 0
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  function mockResetApi(resetImpl: (body: unknown) => unknown) {
    apiFetchMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.endsWith('/reset')) {
        return resetImpl(options?.body ? JSON.parse(options.body as string) : undefined)
      }
      return Promise.resolve(project)
    })
  }

  // The success path chains through resetProjectDataBoundary's own async work
  // (removeQueries -> fetchQuery -> refetchQueries), so a couple of microtask ticks
  // isn't reliably enough — flush a real macrotask turn instead.
  function flush() {
    return new Promise((resolve) => setTimeout(resolve, 0))
  }

  it('disables the submit button when empty or a near-miss', async () => {
    mockResetApi(() => Promise.resolve({}))
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const input = byTestId(view.container, 'settings-reset-confirm-input')
      const button = byTestId(view.container, 'settings-reset-submit')

      expect(button.getAttribute('disabled')).not.toBeNull()

      await React.act(async () => changeControl(input, 'project one'))
      expect(button.getAttribute('disabled')).not.toBeNull()

      await React.act(async () => changeControl(input, 'Project One '))
      expect(button.getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
      client.clear()
    }
  })

  it('enables submit on exact match, posts confirm_name, and reports success', async () => {
    mockResetApi(() => Promise.resolve({
      cancelled_sessions: 2,
      cancelled_meeting_tasks: 1,
      deletions: { tasks: 3, meetings: 1 },
    }))
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const input = byTestId(view.container, 'settings-reset-confirm-input')
      const button = byTestId(view.container, 'settings-reset-submit')

      await React.act(async () => changeControl(input, 'Project One'))
      expect(button.getAttribute('disabled')).toBeNull()

      await React.act(async () => {
        submit(button)
        await flush()
      })

      expect(apiFetchMock).toHaveBeenCalledWith(
        '/api/v1/projects/project-1/reset',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ confirm_name: 'Project One' }),
        }),
      )

      const status = byTestId(view.container, 'settings-reset-status')
      expect(status && textOf(status)).toContain('removed 4')
      expect(input.value).toBe('')
      expect(toastSuccess).toHaveBeenCalledWith('Project reset — operational data cleared')
    } finally {
      view.cleanup()
      client.clear()
    }
  })

  it('shows pending state and disables the input while resetting', async () => {
    mockResetApi(() => new Promise(() => {}))
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const input = byTestId(view.container, 'settings-reset-confirm-input')
      const button = byTestId(view.container, 'settings-reset-submit')

      await React.act(async () => changeControl(input, 'Project One'))
      await React.act(async () => {
        submit(button)
        await Promise.resolve()
      })

      expect(textOf(button)).toBe('Resetting…')
      expect(button.getAttribute('disabled')).not.toBeNull()
      expect(input.getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
      client.clear()
    }
  })

  it('reports a mismatch error on 400 and focuses the input', async () => {
    const { ApiError } = await import('@/lib/api-client')
    mockResetApi(() => Promise.reject(new ApiError(400, 'Project name confirmation does not match')))
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const input = byTestId(view.container, 'settings-reset-confirm-input')
      const button = byTestId(view.container, 'settings-reset-submit')

      await React.act(async () => changeControl(input, 'Project One'))
      await React.act(async () => {
        submit(button)
        await flush()
      })

      expect(descendants(view.container).some((node) =>
        node.getAttribute('role') === 'alert'
        && textOf(node) === 'Confirmation did not match the project name. Check the exact spelling, case, and spaces, then try again.')).toBe(true)
      expect(input.value).toBe('Project One')
      expect(view.activeElement).toBe(input)
    } finally {
      view.cleanup()
      client.clear()
    }
  })

  it('reports a conflict error on 409', async () => {
    const { ApiError } = await import('@/lib/api-client')
    mockResetApi(() => Promise.reject(new ApiError(409, 'in progress')))
    const client = createClient()
    const view = await mountSettings(client)

    try {
      const input = byTestId(view.container, 'settings-reset-confirm-input')
      const button = byTestId(view.container, 'settings-reset-submit')

      await React.act(async () => changeControl(input, 'Project One'))
      await React.act(async () => {
        submit(button)
        await flush()
      })

      expect(descendants(view.container).some((node) =>
        node.getAttribute('role') === 'alert'
        && textOf(node) === 'This project can’t be reset — it may have been archived. Reload and check the project status.')).toBe(true)
    } finally {
      view.cleanup()
      client.clear()
    }
  })
})

describe('ProjectDetailsSection', () => {
  const withDescription = { ...project, description: 'Build the thing.' }

  beforeEach(() => {
    vi.clearAllMocks()
    routeState.hash = ''
    routeState.activeProjectId = 'project-1'
    apiFetchMock.mockImplementation(() => Promise.resolve(withDescription))
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      callback(0)
      return 0
    })
  })

  afterEach(() => vi.unstubAllGlobals())

  async function setup(data: Record<string, unknown> = withDescription) {
    const client = createClient()
    client.setQueryData(['project', project.id], data)
    const view = await mountSettings(client)
    return {
      client,
      view,
      name: getByLabel(view.container, 'Project name', 'input'),
      description: getByLabel(view.container, 'Description', 'textarea'),
      save: getButton(view.container, 'Save'),
      discard: getButton(view.container, 'Discard changes'),
      done() { view.cleanup(); client.clear() },
    }
  }

  const putCalls = () => apiFetchMock.mock.calls.filter(([, o]) => o?.method === 'PUT')

  it('shows project name and description', async () => {
    const t = await setup()
    try {
      expect(t.name.value).toBe('Project One')
      expect(t.description.value).toBe('Build the thing.')
    } finally { t.done() }
  })

  it('null description renders empty textarea', async () => {
    const t = await setup({ ...project, description: null })
    try {
      expect(t.description.value).toBe('')
      expect(t.save.getAttribute('disabled')).not.toBeNull()
    } finally { t.done() }
  })

  it('save disabled until dirty and discard resets', async () => {
    const t = await setup()
    try {
      expect(t.save.getAttribute('disabled')).not.toBeNull()
      expect(t.discard.getAttribute('disabled')).not.toBeNull()
      await React.act(async () => changeControl(t.description, 'Changed'))
      expect(t.save.getAttribute('disabled')).toBeNull()
      expect(t.discard.getAttribute('disabled')).toBeNull()
      await React.act(async () => t.discard.click())
      expect(t.description.value).toBe('Build the thing.')
      expect(t.save.getAttribute('disabled')).not.toBeNull()
    } finally { t.done() }
  })

  it('save sends PUT with changed fields only', async () => {
    const t = await setup()
    try {
      await React.act(async () => changeControl(t.description, '  New text  '))
      await React.act(async () => { submit(t.save); await new Promise((r) => setTimeout(r, 0)) })
      expect(putCalls()).toEqual([[
        '/api/v1/projects/project-1',
        { method: 'PUT', body: JSON.stringify({ description: 'New text' }) },
      ]])
    } finally { t.done() }
  })

  it('success shows toast/status and invalidates projects', async () => {
    const t = await setup()
    const invalidate = vi.spyOn(t.client, 'invalidateQueries')
    apiFetchMock.mockImplementation(() => Promise.resolve({ ...withDescription, name: 'Renamed' }))
    try {
      await React.act(async () => changeControl(t.name, 'Renamed'))
      await React.act(async () => { submit(t.save); await new Promise((r) => setTimeout(r, 0)) })
      // react-query delivers the setQueryData notification on its own timer tick
      await React.act(async () => { await new Promise((r) => setTimeout(r, 0)) })
      expect(toastSuccess).toHaveBeenCalledWith('Project saved')
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ['projects'] })
      const status = descendants(t.view.container).find((n) => n.getAttribute('role') === 'status' && textOf(n) === 'Project saved')
      expect(status?.getAttribute('aria-live')).toBe('polite')
      expect(t.save.getAttribute('disabled')).not.toBeNull()
    } finally { t.done() }
  })

  it('422 on description shows inline error and focuses field', async () => {
    const { ApiError } = await import('@/lib/api-client')
    apiFetchMock.mockRejectedValue(new ApiError(422, 'HTTP 422', [{ loc: ['body', 'description'], msg: 'Description too long' }]))
    const t = await setup()
    try {
      await React.act(async () => changeControl(t.description, 'x'))
      await React.act(async () => { submit(t.save); await new Promise((r) => setTimeout(r, 0)) })
      expect(t.view.activeElement).toBe(t.description)
      expect(textOf(t.view.container)).toContain('Description too long')
      expect(t.description.value).toBe('x')
    } finally { t.done() }
  })

  it('other errors toast', async () => {
    apiFetchMock.mockRejectedValue(new Error('boom'))
    const t = await setup()
    try {
      await React.act(async () => changeControl(t.description, 'x'))
      await React.act(async () => { submit(t.save); await new Promise((r) => setTimeout(r, 0)) })
      expect(toastError).toHaveBeenCalledWith('Could not save the project. Try again.')
    } finally { t.done() }
  })

  it('empty name blocks save', async () => {
    const t = await setup()
    try {
      await React.act(async () => changeControl(t.name, '   '))
      await React.act(async () => { submit(t.save); await new Promise((r) => setTimeout(r, 0)) })
      expect(putCalls()).toHaveLength(0)
      expect(textOf(t.view.container)).toContain('Enter a project name.')
      expect(t.view.activeElement).toBe(t.name)
    } finally { t.done() }
  })

  it('no project selected shows muted message', async () => {
    routeState.activeProjectId = null
    const client = createClient()
    const view = await mountSettings(client)
    try {
      expect(textOf(view.container)).toContain('Select a project to edit its name and description.')
    } finally { view.cleanup(); client.clear() }
  })

  it('#project-details hash scrolls and focuses description', async () => {
    routeState.hash = '#project-details'
    const scrollIntoView = vi.fn()
    const getElementById = vi.fn(() => ({ scrollIntoView }))
    const client = createClient()
    client.setQueryData(['project', project.id], withDescription)
    const { SettingsPage } = await import('./SettingsPage')
    const view = await mountWithTestDom(
      () => (
        <QueryClientProvider client={client}>
          <SettingsPage />
        </QueryClientProvider>
      ),
      async (cb) => {
        ;(globalThis.document as unknown as { getElementById: unknown }).getElementById = getElementById
        return React.act(cb)
      },
    )
    try {
      expect(getElementById).toHaveBeenCalledWith('project-details')
      expect(scrollIntoView).toHaveBeenCalledWith({ block: 'start' })
      expect(view.activeElement).toBe(getByLabel(view.container, 'Description', 'textarea'))
    } finally { view.cleanup(); client.clear() }
  })

  it('helper text mentions the orchestrator', async () => {
    const t = await setup()
    try {
      expect(textOf(t.view.container)).toContain('The orchestrator reads this description in every decision it makes.')
    } finally { t.done() }
  })

  it('project config section no longer shows a duplicate name field', async () => {
    const t = await setup()
    try {
      expect(descendants(t.view.container).some((n) => textOf(n) === 'Project Name')).toBe(false)
    } finally { t.done() }
  })
})
