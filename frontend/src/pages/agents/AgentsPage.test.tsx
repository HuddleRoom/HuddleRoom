import React from 'react'
import { describe, expect, it, vi } from 'vitest'
import { STATUS_COLORS } from '@/lib/statusColors'
import { changeControl, descendants, findElement, getButton, getByLabel, mountWithTestDom, TestEvent, textOf } from '../../../tests/support/dom'

const mocks = vi.hoisted(() => ({
  agents: [] as any[],
  createMutate: vi.fn(),
}))

vi.mock('react-router-dom', () => ({
  useNavigate: () => vi.fn(),
  useParams: () => ({}),
}))

vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ open, children }: { open: boolean, children: React.ReactNode }) => open ? <div>{children}</div> : null,
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: () => null,
  Content: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Title: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Description: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Close: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('@monaco-editor/react', () => ({ default: () => <textarea /> }))
vi.mock('sonner', () => ({ toast: { error: vi.fn(), success: vi.fn() } }))
vi.mock('@/api/agents', () => ({
  useAllAgents: () => ({ items: mocks.agents, isLoading: false, isError: false, refetch: vi.fn(), isCapped: false }),
  useAgent: vi.fn(), useAgentSessions: vi.fn(), useAgentTasks: vi.fn(),
  useCreateAgent: () => ({ isPending: false, mutate: mocks.createMutate }),
  useUpdateAgent: () => ({ isPending: false, mutate: vi.fn() }),
  useDeleteAgent: () => ({ isPending: false, mutate: vi.fn() }),
}))
vi.mock('@/stores/ui', () => ({ useUIStore: vi.fn() }))

const API_WARNING = 'For code development or document editing, use a CLI agent. API agents do not have a workspace toolchain.'

describe('AgentsPage CLI runtime selector', () => {
  it('offers the supported runtimes with accessible guidance', async () => {
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)

    try {
      await React.act(async () => getButton(view.container, 'New agent').click())
      const adapter = getByLabel(view.container, 'Adapter type', 'select')
      await React.act(async () => changeControl(adapter, 'cli'))
      const runtime = getByLabel(view.container, 'CLI runtime', 'select')
      const notice = () => descendants(view.container).find((node) => node.id === 'agent-cli-runtime-guidance')

      expect(Array.from(runtime.options).map((option) => [option.getAttribute('value'), textOf(option)])).toEqual([
        ['', 'Select a runtime'],
        ['claude_code', 'Claude Code'], ['codex', 'Codex'], ['aider', 'Aider'],
        ['copilot', 'GitHub Copilot CLI'], ['opencode', 'OpenCode'], ['pi', 'pi (pi-agent.dev)'],
        ['custom', 'Custom script'],
      ])

      await React.act(async () => changeControl(runtime, 'claude_code'))
      expect(textOf(notice()!)).toContain('permissions bypassed')
      expect(textOf(notice()!)).toContain('saved Claude sessions resume')

      await React.act(async () => changeControl(runtime, 'copilot'))
      expect(textOf(notice()!)).toContain('sign in to GitHub')
      expect(textOf(notice()!)).toContain('without an enforced sandbox')
      expect(textOf(notice()!)).toContain('saved-session resume')
      expect(descendants(notice()!).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)

      await React.act(async () => changeControl(runtime, 'opencode'))
      expect(textOf(notice()!)).toContain('Install OpenCode')
      expect(textOf(notice()!)).toContain('without an enforced sandbox')
      expect(descendants(notice()!).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)

      await React.act(async () => changeControl(runtime, 'pi'))
      expect(textOf(notice()!)).toContain('Install pi')
      expect(textOf(notice()!)).toContain('without an enforced sandbox')
      expect(descendants(notice()!).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)

      await React.act(async () => changeControl(runtime, 'aider'))
      expect(descendants(notice()!).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)

      await React.act(async () => changeControl(runtime, 'custom'))
      expect(descendants(notice()!).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)

      expect(runtime.getAttribute('aria-describedby')).toContain('agent-cli-runtime-guidance')
      expect(notice()!.getAttribute('role')).toBeNull()
      expect(notice()!.getAttribute('aria-live')).toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('requires a runtime before creating a CLI agent', async () => {
    mocks.createMutate.mockReset()
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)

    try {
      await React.act(async () => getButton(view.container, 'New agent').click())
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'CLI agent')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'developer')
        changeControl(getByLabel(view.container, 'Provider *', 'input'), 'openrouter')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'model')
        changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli')
      })

      const runtime = getByLabel(view.container, 'CLI runtime', 'select')
      expect(runtime.getAttribute('required')).not.toBeNull()
      expect(getButton(view.container, 'Create agent').getAttribute('disabled')).not.toBeNull()
      await React.act(async () => findElement(view.container, 'form')?.dispatchEvent(new TestEvent('submit')))
      expect(mocks.createMutate).not.toHaveBeenCalled()

      await React.act(async () => changeControl(runtime, 'opencode'))
      expect(getButton(view.container, 'Create agent').getAttribute('disabled')).toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('shows an unknown saved runtime without changing its value', async () => {
    mocks.agents = [{
      id: 'agent-unknown', name: 'Legacy', role: 'developer', provider: 'local', model: 'local-model',
      adapter_type: 'cli', cli_runtime: 'legacy-cli', system_prompt: null, description: null,
      capabilities: [], is_active: true, config: {}, created_at: '', updated_at: '',
    }]
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)

    try {
      await React.act(async () => getButton(view.container, 'Edit').click())
      const runtime = getByLabel(view.container, 'CLI runtime', 'select')
      expect(Array.from((runtime as HTMLSelectElement).options).find((option) => option.getAttribute('value') === 'legacy-cli')).toBeDefined()
      expect(textOf(Array.from((runtime as HTMLSelectElement).options).find((option) => option.getAttribute('value') === 'legacy-cli')!)).toBe('Unknown runtime: legacy-cli')
      const notice = descendants(view.container).find((node) => node.id === 'agent-cli-runtime-guidance')!
      expect(textOf(notice)).toContain('Unknown runtime: legacy-cli')
      expect(descendants(notice).find((node) => node.getAttribute('data-testid') === 'banner')!.style.borderColor).toBe(STATUS_COLORS.amber)
    } finally {
      mocks.agents = []
      view.cleanup()
    }
  })
})

describe('AgentsPage API adapter guidance', () => {
  it('shows the warning for API agents and hides it for CLI and routine agents', async () => {
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)

    try {
      await React.act(async () => getButton(view.container, 'New agent').click())
      const adapter = getByLabel(view.container, 'Adapter type', 'select')
      const warning = () => descendants(view.container).find((node) => node.id === 'agent-api-adapter-warning')

      expect(textOf(warning()!)).toContain(API_WARNING)
      expect(adapter.getAttribute('aria-describedby')).toBe('agent-api-adapter-warning')

      for (const value of ['cli', 'routine']) {
        await React.act(async () => changeControl(adapter, value))
        expect(warning()).toBeUndefined()
        expect(adapter.getAttribute('aria-describedby')).toBeNull()
      }
    } finally {
      view.cleanup()
    }
  })

  it('shows the warning in edit and submits a valid API agent', async () => {
    mocks.agents = [{
      id: 'agent-1', name: 'Existing', role: 'developer', provider: 'openai', model: 'gpt-5',
      adapter_type: 'cli', cli_runtime: 'codex', system_prompt: null, description: null,
      capabilities: [], is_active: true, config: {}, created_at: '', updated_at: '',
    }]
    mocks.createMutate.mockReset()
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)

    try {
      await React.act(async () => getButton(view.container, 'Edit').click())
      const adapter = getByLabel(view.container, 'Adapter type', 'select')
      expect(descendants(view.container).find((node) => node.id === 'agent-api-adapter-warning')).toBeUndefined()
      await React.act(async () => changeControl(adapter, 'api'))
      expect(textOf(descendants(view.container).find((node) => node.id === 'agent-api-adapter-warning')!)).toContain(API_WARNING)

      await React.act(async () => getButton(view.container, 'Cancel').click())
      await React.act(async () => getButton(view.container, 'New agent').click())
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'API agent')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'developer')
        changeControl(getByLabel(view.container, 'Provider *', 'input'), 'openai')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'gpt-5')
      })
      await React.act(async () => findElement(view.container, 'form')?.dispatchEvent(new TestEvent('submit')))

      expect(mocks.createMutate).toHaveBeenCalledWith(expect.objectContaining({
        name: 'API agent', adapter_type: 'api', cli_runtime: null,
      }), expect.any(Object))
    } finally {
      mocks.agents = []
      view.cleanup()
    }
  })
})
