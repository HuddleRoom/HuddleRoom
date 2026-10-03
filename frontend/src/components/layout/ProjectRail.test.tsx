import { describe, it, expect, beforeEach, vi } from 'vitest'
import React from 'react'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { useAgentResponseStore } from '@/stores/agent-response'
import type { ProjectAdvisorHistory } from '@/lib/types'

const advisor = vi.hoisted(() => ({
  query: { data: undefined as ProjectAdvisorHistory | undefined, isLoading: false, isError: false, refetch: vi.fn() },
}))

vi.mock('@/api/orchestration', () => ({
  useProjectAdvisorConversation: () => advisor.query,
}))

vi.mock('./AgentActivityPanel', () => ({
  AgentActivityPanel: () => <div data-testid="calls-panel">Calls panel</div>,
}))

vi.mock('./ProjectAdvisorPanel', () => ({
  ProjectAdvisorPanel: () => <div data-testid="ask-panel">Ask panel</div>,
}))

import { ProjectRail } from './ProjectRail'

const enabledHistory = (): ProjectAdvisorHistory => ({ items: [], allowance: { enabled: true, unlimited: false, limit: 1000, remaining: 1000 } })

function callRecord(terminal: boolean) {
  return { terminal: terminal ? { status: 'completed', error: null } : null } as never
}

beforeEach(() => {
  advisor.query = { data: enabledHistory(), isLoading: false, isError: false, refetch: vi.fn() }
  useAgentResponseStore.setState({
    projectId: null,
    projectName: null,
    rawEvents: [],
    calls: {},
    seenEventIds: new Set(),
    candidateRevision: 0,
    lifecycleAnnouncement: '',
  })
})

async function mount() {
  return mountWithTestDom(() => <ProjectRail projectId="project-1" />, React.act)
}

describe('ProjectRail', () => {
  it('defaults to the Ask tab when there are no non-terminal calls', async () => {
    const view = await mount()
    try {
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'ask-panel')).toBeDefined()
      const askTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node) === 'Ask')
      expect(askTab?.getAttribute('aria-selected')).toBe('true')
    } finally { view.cleanup() }
  })

  it('defaults to the Calls tab when non-terminal calls exist', async () => {
    useAgentResponseStore.setState({ calls: { 'call-1': callRecord(false) } })
    const view = await mount()
    try {
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'calls-panel')).toBeDefined()
      const callsTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node) === 'Calls')
      expect(callsTab?.getAttribute('aria-selected')).toBe('true')
    } finally { view.cleanup() }
  })

  it('treats only terminal calls as not requiring the Calls default', async () => {
    useAgentResponseStore.setState({ calls: { 'call-1': callRecord(true) } })
    const view = await mount()
    try {
      const askTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node) === 'Ask')
      expect(askTab?.getAttribute('aria-selected')).toBe('true')
    } finally { view.cleanup() }
  })

  it('keeps the selected tab sticky after selection', async () => {
    const view = await mount()
    try {
      await React.act(async () => getButton(view.container, 'Calls').click())
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'calls-panel')).toBeDefined()

      // A store update that would otherwise change the default-tab heuristic
      // must not steal the tab back once a selection was made.
      useAgentResponseStore.setState({ calls: { 'call-1': callRecord(false) } })
      await view.rerender()
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'calls-panel')).toBeDefined()
      const callsTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node) === 'Calls')
      expect(callsTab?.getAttribute('aria-selected')).toBe('true')
    } finally { view.cleanup() }
  })

  it('shows an unread dot on Calls when a new call starts while Ask is active', async () => {
    const view = await mount()
    try {
      expect(descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && /^Calls$/.test(textOf(node)))).toBeDefined()
      useAgentResponseStore.setState({ calls: { 'call-1': callRecord(false) } })
      await view.rerender()
      const callsTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node).startsWith('Calls'))
      expect(textOf(callsTab!)).toBe('Calls •')

      // Switching to Calls clears the unread marker.
      await React.act(async () => getButton(view.container, 'Calls •').click())
      const clearedTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node).startsWith('Calls'))
      expect(textOf(clearedTab!)).toBe('Calls')
    } finally { view.cleanup() }
  })

  it('does not mark Calls unread for calls that arrive while Calls is already active', async () => {
    useAgentResponseStore.setState({ calls: { 'call-1': callRecord(false) } })
    const view = await mount()
    try {
      useAgentResponseStore.setState({ calls: { 'call-1': callRecord(false), 'call-2': callRecord(false) } })
      await view.rerender()
      const callsTab = descendants(view.container).find((node) => node.getAttribute('role') === 'tab' && textOf(node).startsWith('Calls'))
      expect(textOf(callsTab!)).toBe('Calls')
    } finally { view.cleanup() }
  })

  it('renders AgentActivityPanel directly with no tab strip when advisor allowance is disabled', async () => {
    advisor.query.data = { items: [], allowance: { enabled: false, unlimited: false, limit: 0, remaining: 0 } }
    const view = await mount()
    try {
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'calls-panel')).toBeDefined()
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'tab')).toHaveLength(0)
      expect(descendants(view.container).find((node) => node.getAttribute('data-testid') === 'ask-panel')).toBeUndefined()
    } finally { view.cleanup() }
  })
})
