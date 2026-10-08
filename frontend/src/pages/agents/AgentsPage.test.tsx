import React from 'react'
import { describe, expect, it, vi } from 'vitest'
import { STATUS_COLORS } from '@/lib/statusColors'
import { changeControl, descendants, findElement, getButton, getByLabel, mountWithTestDom, TestEvent, textOf } from '../../../tests/support/dom'

const mocks = vi.hoisted(() => ({
  agents: [] as any[],
  createMutate: vi.fn(),
  updateMutate: vi.fn(),
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
  useUpdateAgent: () => ({ isPending: false, mutate: mocks.updateMutate }),
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

describe('AgentsPage create form adapter/provider/effort', () => {
  async function open() {
    mocks.createMutate.mockReset()
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)
    await React.act(async () => getButton(view.container, 'New agent').click())
    return view
  }
  const hasEffort = (v: any) => descendants(v.container).some((n) => n.id === 'agent-effort')
  const submit = (v: any) => React.act(async () => findElement(v.container, 'form')?.dispatchEvent(new TestEvent('submit')))

  it('places adapter type before provider and model', async () => {
    const view = await open()
    try {
      const ids = descendants(view.container).map((n) => n.id)
      expect(ids.indexOf('agent-adapter-type')).toBeLessThan(ids.indexOf('agent-provider'))
      expect(ids.indexOf('agent-adapter-type')).toBeLessThan(ids.indexOf('agent-model'))
    } finally { view.cleanup() }
  })

  it('creates a CLI agent with provider set to the runtime', async () => {
    const view = await open()
    try {
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'CLI')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'dev')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'm')
        changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli')
      })
      expect(() => getByLabel(view.container, 'Provider', 'input')).toThrow()
      await React.act(async () => changeControl(getByLabel(view.container, 'CLI runtime', 'select'), 'aider'))
      await submit(view)
      expect(mocks.createMutate).toHaveBeenCalledWith(expect.objectContaining({ provider: 'aider', cli_runtime: 'aider' }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it('cli create payload provider == runtime even if form had a provider from switching api→cli', async () => {
    const view = await open()
    try {
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'Agent')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'dev')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'm')
        changeControl(getByLabel(view.container, 'Provider *', 'input'), 'openai')
      })
      await React.act(async () => changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli'))
      expect(() => getByLabel(view.container, 'Provider', 'input')).toThrow()
      await React.act(async () => changeControl(getByLabel(view.container, 'CLI runtime', 'select'), 'codex'))
      await submit(view)
      expect(mocks.createMutate).toHaveBeenCalledWith(expect.objectContaining({ provider: 'codex', cli_runtime: 'codex' }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it('shows effort only for claude_code and codex CLI agents', async () => {
    const view = await open()
    try {
      expect(hasEffort(view)).toBe(false)
      await React.act(async () => changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli'))
      const runtime = getByLabel(view.container, 'CLI runtime', 'select')
      for (const [r, shown] of [['aider', false], ['custom', false], ['copilot', false], ['opencode', false], ['pi', false], ['claude_code', true], ['codex', true]] as const) {
        await React.act(async () => changeControl(runtime, r))
        expect(hasEffort(view)).toBe(shown)
      }
      await React.act(async () => changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'api'))
      expect(hasEffort(view)).toBe(false)
    } finally { view.cleanup() }
  })

  it('sends reasoning_effort when chosen and omits it on Default', async () => {
    const view = await open()
    try {
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'CLI')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'dev')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'm')
        changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli')
      })
      await React.act(async () => changeControl(getByLabel(view.container, 'CLI runtime', 'select'), 'codex'))
      await React.act(async () => changeControl(getByLabel(view.container, 'Effort', 'select'), 'high'))
      await submit(view)
      expect(mocks.createMutate.mock.calls[0][0].config).toEqual({ memory_enabled: false, reasoning_effort: 'high' })
      await React.act(async () => changeControl(getByLabel(view.container, 'Effort', 'select'), ''))
      await submit(view)
      expect(mocks.createMutate.mock.calls[1][0].config).toEqual({ memory_enabled: false })
    } finally { view.cleanup() }
  })
})

describe('AgentsPage edit form effort/config', () => {
  const agent = (over: any) => ({
    id: 'a1', name: 'Ed', role: 'dev', provider: 'openai', model: 'm', adapter_type: 'cli', cli_runtime: 'codex',
    system_prompt: null, description: null, capabilities: [], is_active: true, config: {}, created_at: '', updated_at: '', ...over,
  })
  async function edit(a: any) {
    mocks.agents = [a]
    mocks.updateMutate.mockReset()
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)
    await React.act(async () => getButton(view.container, 'Edit').click())
    const submit = () => React.act(async () => findElement(view.container, 'form')?.dispatchEvent(new TestEvent('submit')))
    return { view, submit, sent: () => mocks.updateMutate.mock.calls[0][0].data }
  }

  it('loads reasoning_effort, keeps unrelated config keys, and lists effort options', async () => {
    const { view, submit, sent } = await edit(agent({ config: { foo: 1, reasoning_effort: 'high' } }))
    try {
      const effort = getByLabel(view.container, 'Effort', 'select') as HTMLSelectElement
      expect(Array.from(effort.options).map((o) => o.getAttribute('value'))).toEqual(['', 'low', 'medium', 'high', 'xhigh'])
      await submit()
      expect(sent().config).toEqual({ foo: 1, memory_enabled: false, reasoning_effort: 'high' })
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('removes a legacy config runtime while preserving other config on edit', async () => {
    const { view, submit, sent } = await edit(agent({
      provider: 'claude_code', cli_runtime: 'claude_code', model: 'claude-sonnet-5-5',
      config: { cli_runtime: 'codex', foo: 1 },
    }))
    try {
      await submit()
      expect(sent()).toEqual(expect.objectContaining({ provider: 'claude_code', cli_runtime: 'claude_code' }))
      expect(sent().config).toEqual({ foo: 1, memory_enabled: false })
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('drops reasoning_effort when switching to a runtime without effort', async () => {
    const { view, submit, sent } = await edit(agent({ config: { reasoning_effort: 'high' } }))
    try {
      await React.act(async () => changeControl(getByLabel(view.container, 'CLI runtime', 'select'), 'aider'))
      await submit()
      expect(sent().config).toEqual({ memory_enabled: false })
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('omits max for codex and drops a stored max when the runtime is codex', async () => {
    const { view, submit, sent } = await edit(agent({ cli_runtime: 'claude_code', config: { reasoning_effort: 'max' } }))
    try {
      const values = () => Array.from((getByLabel(view.container, 'Effort', 'select') as HTMLSelectElement).options).map((o) => o.getAttribute('value'))
      expect(values()).toContain('max')
      await React.act(async () => changeControl(getByLabel(view.container, 'CLI runtime', 'select'), 'codex'))
      expect(values()).toEqual(['', 'low', 'medium', 'high', 'xhigh'])
      const effort = getByLabel(view.container, 'Effort', 'select') as HTMLSelectElement
      expect(effort.value).toBe('')
      await submit()
      expect(sent().config).toEqual({ memory_enabled: false })
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('leaves an API agent payload without reasoning_effort', async () => {
    const { view, submit, sent } = await edit(agent({ adapter_type: 'api', cli_runtime: null, config: { foo: 1 } }))
    try {
      await submit()
      expect(sent()).toEqual(expect.objectContaining({ provider: 'openai', cli_runtime: null, config: { foo: 1, memory_enabled: false } }))
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('editing an existing cli agent saves provider == current runtime', async () => {
    const { view, submit, sent } = await edit(agent({ adapter_type: 'cli', cli_runtime: 'codex' }))
    try {
      expect(() => getByLabel(view.container, 'Provider', 'input')).toThrow()
      await submit()
      expect(sent()).toEqual(expect.objectContaining({ provider: 'codex', cli_runtime: 'codex' }))
    } finally { mocks.agents = []; view.cleanup() }
  })

  it('keeps Create disabled for a routine agent with blank provider', async () => {
    mocks.agents = []
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)
    try {
      await React.act(async () => getButton(view.container, 'New agent').click())
      await React.act(async () => {
        changeControl(getByLabel(view.container, 'Name *', 'input'), 'R')
        changeControl(getByLabel(view.container, 'Role *', 'input'), 'r')
        changeControl(getByLabel(view.container, 'Model *', 'input'), 'm')
        changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'routine')
      })
      expect(textOf(descendants(view.container).find((n) => n.id === 'agent-provider')!.parentNode as any)).not.toContain('Optional')
      expect(getButton(view.container, 'Create agent').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('does not show Provider field for CLI agents', async () => {
    mocks.agents = []
    const { AgentsPage } = await import('./AgentsPage')
    const view = await mountWithTestDom(() => <AgentsPage />, React.act)
    try {
      await React.act(async () => getButton(view.container, 'New agent').click())
      await React.act(async () => changeControl(getByLabel(view.container, 'Adapter type', 'select'), 'cli'))
      expect(() => getByLabel(view.container, 'Provider', 'input')).toThrow()
    } finally { view.cleanup() }
  })
})
