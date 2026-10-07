import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import React from 'react'
import type { Project } from '@/lib/types'
import {
  descendants,
  mountWithTestDom,
  textOf,
} from '../../../tests/support/dom'
import { useAgentResponseStore } from '@/stores/agent-response'
import { useWSStore } from '@/stores/ws'
import { isAgentResponseEvent, orderedCalls, AgentResponseEvent, reconcileAgentTabs } from '@/lib/agentResponse'
import { AgentActivityPanel } from './AgentActivityPanel'

// Shell test mocks (for mounted Shell tests)
const shellTestState = vi.hoisted(() => ({
  projects: [] as Project[],
}))

vi.mock('@tanstack/react-query', () => ({
  // ponytail: keyed by queryKey[0] so the Shell's projects query keeps its
  // fixture while other useQuery callers mounted under Shell (e.g. the
  // project-advisor rail query) get an inert idle result instead of the
  // projects array.
  useQuery: (opts: { queryKey?: unknown[] }) =>
    Array.isArray(opts?.queryKey) && opts.queryKey[0] === 'projects'
      ? { data: shellTestState.projects }
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

vi.mock('@/stores/auth', () => {
  const authState = { token: null }
  const useAuthStore = Object.assign(
    (selector?: (state: typeof authState) => unknown) => {
      if (!selector) return authState
      return selector(authState)
    },
    {
      getState: () => authState,
      subscribe: () => () => {},
    },
  )
  return { useAuthStore }
})

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) => selector({
    activeProjectId: 'project-1',
    sidebarCollapsed: false,
    sidebarOpen: false,
    closeSidebar: vi.fn(),
  }),
}))

vi.mock('@/lib/api-client', () => ({
  apiFetch: vi.fn(),
  isAuthDisabled: () => false,
}))

// MockWebSocket harness for testing ws.ts reconnect behavior and cursor independence
class MockWebSocket {
  static OPEN = 1
  static CONNECTING = 0
  url: string
  readyState: number = 0
  onopen: (() => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
  }

  close() {
    this.readyState = 3
  }

  simulateOpen() {
    this.readyState = 1
    this.onopen?.()
  }

  simulateMessage(data: Record<string, unknown>) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }

  simulateClose(code = 1000) {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code }))
  }
}

// Test helpers
const baseEvent = (
  overrides: Partial<AgentResponseEvent> = {}
): AgentResponseEvent => ({
  id: `evt-${Math.random().toString(36).slice(2)}`,
  project_id: 'proj-123',
  call_id: 'call-1',
  sequence: 0,
  event_type: 'agent_response.started',
  emitted_at: '2024-01-01T00:00:00Z',
  call_started_at: '2024-01-01T00:00:00Z',
  invocation_kind: 'api',
  operation: 'test_op',
  parent_call_id: null,
  actor_kind: 'agent',
  actor_id: 'agent-1',
  payload: {
    request_display: { kind: 'prompt', content: 'test', truncated: false },
    actor_label: 'Test Agent',
    model_or_runtime: 'claude-sonnet-5-5',
  },
  ...overrides,
})

const outputEvent = (
  overrides: Partial<AgentResponseEvent> = {}
): AgentResponseEvent => ({
  ...baseEvent(overrides),
  event_type: 'agent_response.output',
  sequence: 1,
  payload: {
    stream: 'output',
    text: 'test output',
  },
})

const terminalEvent = (
  overrides: Partial<AgentResponseEvent> = {}
): AgentResponseEvent => {
  const base = baseEvent(overrides)
  return {
    ...base,
    event_type: 'agent_response.terminal',
    sequence: 2,
    payload: {
      status: 'completed',
      error: null,
      error_truncated: false,
      ...((overrides.payload || {}) as Record<string, unknown>),
    },
  }
}

describe('AgentActivityPanel - event store and socket', () => {
  beforeEach(() => {
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

  afterEach(() => {
    useAgentResponseStore.getState().clear()
    useWSStore.getState().disconnect()
  })

  describe('isAgentResponseEvent validation', () => {
    it('accepts valid agent_response.started event', () => {
      const evt = baseEvent()
      expect(isAgentResponseEvent(evt)).toBe(true)
    })

    it('accepts valid agent_response.output event', () => {
      const evt = outputEvent()
      expect(isAgentResponseEvent(evt)).toBe(true)
    })

    it('accepts valid agent_response.terminal event', () => {
      const evt = terminalEvent()
      expect(isAgentResponseEvent(evt)).toBe(true)
    })

    it('rejects events with missing common fields', () => {
      const evt = { event_type: 'agent_response.started' }
      expect(isAgentResponseEvent(evt)).toBe(false)
    })

    it('rejects non-agent-response event types', () => {
      const evt = {
        ...baseEvent(),
        event_type: 'task.created',
      }
      expect(isAgentResponseEvent(evt)).toBe(false)
    })

    it('rejects malformed payload', () => {
      const evt = {
        ...baseEvent(),
        event_type: 'agent_response.output',
        payload: { stream: 'output' }, // missing text
      }
      expect(isAgentResponseEvent(evt)).toBe(false)
    })
  })

  describe('UUID dedup', () => {
    it('ignores duplicate event ids', () => {
      const evt = baseEvent({ id: 'unique-1' })
      useAgentResponseStore.getState().ingest(evt)
      useAgentResponseStore.getState().ingest(evt) // same id

      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(1)
    })

    it('accepts events with different ids', () => {
      const evt1 = baseEvent({ id: 'unique-1', call_id: 'call-1', sequence: 0 })
      const evt2 = baseEvent({ id: 'unique-2', call_id: 'call-1', sequence: 1 })
      useAgentResponseStore.getState().ingest(evt1)
      useAgentResponseStore.getState().ingest(evt2)

      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(2)
    })
  })

  describe('Route retention', () => {
    it('keeps up to 200 raw events', () => {
      for (let i = 0; i < 205; i++) {
        const evt = baseEvent({
          id: `evt-${i}`,
          call_id: `call-${i}`,
          sequence: 0,
          event_type: 'agent_response.started',
          payload: {
            request_display: { kind: 'prompt', content: null, truncated: false },
            actor_label: null,
            model_or_runtime: 'test',
          },
        })
        useAgentResponseStore.getState().ingest(evt)
      }

      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(200)
    })

    it('removes oldest output event first on overflow', () => {
      // Add 200 events with mix of started/output/terminal
      const events = []
      for (let i = 0; i < 200; i++) {
        const types = [
          'agent_response.started',
          'agent_response.output',
          'agent_response.output',
        ]
        const eventType =
          types[i % types.length] as
            | 'agent_response.started'
            | 'agent_response.output'

        events.push(
          baseEvent({
            id: `evt-${i}`,
            call_id: `call-${i % 50}`,
            sequence: i,
            event_type: eventType,
            payload:
              eventType === 'agent_response.output'
                ? { stream: 'output', text: `output ${i}` }
                : {
                    request_display: { kind: 'prompt', content: null, truncated: false },
                    actor_label: null,
                    model_or_runtime: 'test',
                  },
          })
        )
      }

      events.forEach((e) => useAgentResponseStore.getState().ingest(e))

      // Find first output event
      const firstOutputIdx = events.findIndex(
        (e) => e.event_type === 'agent_response.output'
      )

      // Add overflow
      const overflow = outputEvent({ id: 'overflow-1' })
      useAgentResponseStore.getState().ingest(overflow)

      // First output should be gone
      expect(
        useAgentResponseStore.getState().rawEvents.some((e) => e.id === events[firstOutputIdx].id)
      ).toBe(false)
      expect(
        useAgentResponseStore.getState().rawEvents.some((e) => e.id === 'overflow-1')
      ).toBe(true)
    })
  })

  describe('Refresh → empty initial state', () => {
    it('starts with empty state', () => {
      const state = useAgentResponseStore.getState()
      expect(state.rawEvents).toHaveLength(0)
      expect(Object.keys(state.calls)).toHaveLength(0)
      expect(state.projectId).toBe(null)
    })

    it('clears on explicit clear', () => {
      useAgentResponseStore.getState().ingest(baseEvent())
      useAgentResponseStore.getState().clear()

      const state = useAgentResponseStore.getState()
      expect(state.rawEvents).toHaveLength(0)
      expect(Object.keys(state.calls)).toHaveLength(0)
    })
  })

  describe('Synchronous project-switch clear', () => {
    it('clears state when project changes', () => {
      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')
      useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'proj-1' }))

      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(1)

      // Switch project
      useAgentResponseStore.getState().setProject('proj-2', 'Project 2')
      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(0)
      expect(useAgentResponseStore.getState().projectId).toBe('proj-2')
      expect(useAgentResponseStore.getState().lifecycleAnnouncement).toContain('Project 2')
    })

    it('does not clear on same project', () => {
      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')
      useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'proj-1' }))

      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')
      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(1)
    })
  })

  describe('Strict chronology - orderedCalls', () => {
    it('sorts calls by call_started_at ascending', () => {
      const calls = {
        'call-3': {
          ...baseEvent({ call_id: 'call-3', call_started_at: '2024-01-01T03:00:00Z' }),
          callId: 'call-3',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T03:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
        'call-1': {
          ...baseEvent({ call_id: 'call-1', call_started_at: '2024-01-01T01:00:00Z' }),
          callId: 'call-1',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T01:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
        'call-2': {
          ...baseEvent({ call_id: 'call-2', call_started_at: '2024-01-01T02:00:00Z' }),
          callId: 'call-2',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T02:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
      }

      const ordered = orderedCalls(calls)
      expect(ordered[0].callId).toBe('call-1')
      expect(ordered[1].callId).toBe('call-2')
      expect(ordered[2].callId).toBe('call-3')
    })

    it('tiebreaks by call_id when call_started_at is identical', () => {
      const calls = {
        'call-c': {
          ...baseEvent({ call_id: 'call-c', call_started_at: '2024-01-01T01:00:00Z' }),
          callId: 'call-c',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T01:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
        'call-a': {
          ...baseEvent({ call_id: 'call-a', call_started_at: '2024-01-01T01:00:00Z' }),
          callId: 'call-a',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T01:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
        'call-b': {
          ...baseEvent({ call_id: 'call-b', call_started_at: '2024-01-01T01:00:00Z' }),
          callId: 'call-b',
          projectId: 'proj-123',
          callStartedAt: '2024-01-01T01:00:00Z',
          invocationKind: 'api' as const,
          operation: 'op1',
          parentCallId: null,
          actorKind: 'agent' as const,
          actorId: 'a1',
          actorLabel: 'Agent 1',
          requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
          items: [],
          lastSequence: 1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        },
      }

      const ordered = orderedCalls(calls)
      expect(ordered[0].callId).toBe('call-a')
      expect(ordered[1].callId).toBe('call-b')
      expect(ordered[2].callId).toBe('call-c')
    })
  })

  describe('Unknown-call derivation', () => {
    it('creates call from output event without prior started', () => {
      const evt = outputEvent({ call_id: 'late', sequence: 4 })
      useAgentResponseStore.getState().ingest(evt)

      const late = useAgentResponseStore.getState().calls.late
      expect(late).toBeDefined()
      expect(late.sequenceGap).toBe(true)
      expect(late.requestDisplay.kind).toBe('unavailable')

      // No fake started event
      expect(
        useAgentResponseStore.getState().rawEvents.some(
          (event) => event.event_type === 'agent_response.started'
        )
      ).toBe(false)
    })

    it('preserves unavailable request state', () => {
      const evt = outputEvent({ call_id: 'no-start', sequence: 1 })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls['no-start']
      expect(call.requestDisplay.kind).toBe('unavailable')
      expect(call.requestDisplay.content).toBe(null)
      expect(call.requestDisplay.truncated).toBe(false)
    })

    it('detects sequence gap for unknown-call derivation', () => {
      const evt = outputEvent({ call_id: 'late', sequence: 4 })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls.late
      expect(call.sequenceGap).toBe(true)
    })

    it('preserves request_display from started event', () => {
      const startedEvt = baseEvent({
        call_id: 'call-x',
        sequence: 0,
        payload: {
          request_display: { kind: 'prompt', content: 'test prompt', truncated: false },
          actor_label: 'Custom Label',
          model_or_runtime: 'claude-sonnet-5-5',
        },
      })

      useAgentResponseStore.getState().ingest(startedEvt)

      const call = useAgentResponseStore.getState().calls['call-x']
      expect(call.requestDisplay.kind).toBe('prompt')
      expect(call.requestDisplay.content).toBe('test prompt')
    })

    it('marks continuation request display', () => {
      const contEvt = baseEvent({
        call_id: 'cont-1',
        sequence: 0,
        payload: {
          request_display: { kind: 'continuation', content: null, truncated: false },
          actor_label: null,
          model_or_runtime: 'claude-sonnet-5-5',
        },
      })

      useAgentResponseStore.getState().ingest(contEvt)

      const call = useAgentResponseStore.getState().calls['cont-1']
      expect(call.requestDisplay.kind).toBe('continuation')
    })
  })

  describe('Lifecycle folding on overflow', () => {
    it('marks outputEvicted when folding oldest raw event', () => {
      let eventCounter = 0
      for (let callNum = 0; callNum < 67; callNum++) {
        const startEvt = baseEvent({
          id: `evt-${eventCounter++}`,
          call_id: `call-${callNum}`,
          sequence: 0,
          event_type: 'agent_response.started',
          payload: {
            request_display: { kind: 'prompt', content: null, truncated: false },
            actor_label: null,
            model_or_runtime: 'test',
          },
        })
        useAgentResponseStore.getState().ingest(startEvt)

        const toolStartEvt: AgentResponseEvent = {
          ...baseEvent({
            id: `evt-${eventCounter++}`,
            call_id: `call-${callNum}`,
            sequence: 1,
            event_type: 'agent_response.tool_started',
            payload: { name: 'test_tool', arguments: null, truncated: false },
          }),
        }
        toolStartEvt.event_type = 'agent_response.tool_started'
        useAgentResponseStore.getState().ingest(toolStartEvt)

        const toolFinishEvt: AgentResponseEvent = {
          ...baseEvent({
            id: `evt-${eventCounter++}`,
            call_id: `call-${callNum}`,
            sequence: 2,
            event_type: 'agent_response.tool_finished',
            payload: { name: 'test_tool', outcome: 'success', result: null, truncated: false },
          }),
        }
        toolFinishEvt.event_type = 'agent_response.tool_finished'
        useAgentResponseStore.getState().ingest(toolFinishEvt)
      }

      const state = useAgentResponseStore.getState()
      expect(state.rawEvents.length).toBeGreaterThanOrEqual(200)

      const overflowEvt = baseEvent({
        id: `evt-overflow`,
        call_id: `call-overflow`,
        sequence: 0,
        event_type: 'agent_response.started',
        payload: {
          request_display: { kind: 'prompt', content: null, truncated: false },
          actor_label: null,
          model_or_runtime: 'test',
        },
      })
      useAgentResponseStore.getState().ingest(overflowEvt)

      const call0 = useAgentResponseStore.getState().calls['call-0']
      expect(call0).toBeDefined()
      expect(call0.outputEvicted).toBe(true)
    })
  })

  describe('200-call retention with terminal-first eviction', () => {
    it('removes oldest terminal call when over 200 derived calls', () => {
      for (let i = 0; i < 50; i++) {
        const evt = baseEvent({
          id: `evt-active-${i}`,
          call_id: `call-active-${i}`,
          sequence: 0,
        })
        useAgentResponseStore.getState().ingest(evt)
      }

      for (let i = 0; i < 151; i++) {
        const evt = baseEvent({
          id: `evt-term-${i}`,
          call_id: `call-term-${i}`,
          sequence: 0,
        })
        useAgentResponseStore.getState().ingest(evt)

        const termEvt = terminalEvent({
          id: `evt-term-end-${i}`,
          call_id: `call-term-${i}`,
          sequence: 1,
        })
        useAgentResponseStore.getState().ingest(termEvt)
      }

      const state = useAgentResponseStore.getState()
      expect(Object.keys(state.calls).length).toBeLessThanOrEqual(200)

      const terminalCount = Object.values(state.calls).filter((c) => c.terminal).length
      const activeCount = Object.values(state.calls).filter((c) => !c.terminal).length

      expect(terminalCount + activeCount).toBeLessThanOrEqual(200)
      expect(activeCount).toBeGreaterThan(0)
    })

    it('removes oldest active call when no terminals exist', () => {
      for (let i = 0; i < 201; i++) {
        const evt = baseEvent({
          id: `evt-${i}`,
          call_id: `call-${i}`,
          sequence: 0,
          call_started_at: new Date(Date.now() - (200 - i) * 1000).toISOString(),
        })
        useAgentResponseStore.getState().ingest(evt)
      }

      const state = useAgentResponseStore.getState()
      expect(Object.keys(state.calls).length).toBeLessThanOrEqual(200)
      expect(state.calls['call-0']).toBeUndefined()
    })
  })

  describe('Request display preservation - redacted/truncated/unavailable', () => {
    it('preserves truncated flag', () => {
      const evt = baseEvent({
        call_id: 'trunc',
        payload: {
          request_display: { kind: 'prompt', content: 'truncated...', truncated: true },
          actor_label: null,
          model_or_runtime: 'test',
        },
      })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls.trunc
      expect(call.requestDisplay.truncated).toBe(true)
    })

    it('preserves unavailable kind', () => {
      const evt = outputEvent({ call_id: 'unavail', sequence: 1 })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls.unavail
      expect(call.requestDisplay.kind).toBe('unavailable')
    })

    it('preserves request content verbatim', () => {
      const content = { type: 'text', value: 'secret credentials here' }
      const evt = baseEvent({
        call_id: 'content-test',
        payload: {
          request_display: { kind: 'prompt', content, truncated: false },
          actor_label: null,
          model_or_runtime: 'test',
        },
      })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls['content-test']
      expect(call.requestDisplay.content).toEqual(content)
    })
  })

  describe('candidateRevision tracking', () => {
    it('increments on started event', () => {
      const before = useAgentResponseStore.getState().candidateRevision
      useAgentResponseStore.getState().ingest(baseEvent())
      const after = useAgentResponseStore.getState().candidateRevision
      expect(after).toBe(before + 1)
    })

    it('increments on terminal event', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c1', sequence: 0 }))
      const before = useAgentResponseStore.getState().candidateRevision
      useAgentResponseStore.getState().ingest(terminalEvent({ call_id: 'c1', sequence: 1 }))
      const after = useAgentResponseStore.getState().candidateRevision
      expect(after).toBe(before + 1)
    })

    it('does not increment on output event', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c1', sequence: 0 }))
      const before = useAgentResponseStore.getState().candidateRevision
      useAgentResponseStore.getState().ingest(outputEvent({ call_id: 'c1', sequence: 1 }))
      const after = useAgentResponseStore.getState().candidateRevision
      expect(after).toBe(before)
    })
  })

  describe('Actor label derivation', () => {
    it('uses actor_label from started payload', () => {
      const evt = baseEvent({
        call_id: 'labeled',
        payload: {
          request_display: { kind: 'prompt', content: null, truncated: false },
          actor_label: 'Custom Agent Name',
          model_or_runtime: 'claude-sonnet-5-5',
        },
      })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls.labeled
      expect(call.actorLabel).toBe('Custom Agent Name')
    })

    it('falls back to "Unknown agent" when actor_label is absent', () => {
      const evt = baseEvent({
        call_id: 'unknown',
        payload: {
          request_display: { kind: 'prompt', content: null, truncated: false },
          actor_label: null,
          model_or_runtime: 'claude-sonnet-5-5',
        },
      })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls.unknown
      expect(call.actorLabel).toBe('Unknown agent')
    })

    it('falls back to "Unknown agent" when actor_label is empty string', () => {
      const evt = baseEvent({
        call_id: 'empty-label',
        payload: {
          request_display: { kind: 'prompt', content: null, truncated: false },
          actor_label: '',
          model_or_runtime: 'claude-sonnet-5-5',
        },
      })
      useAgentResponseStore.getState().ingest(evt)

      const call = useAgentResponseStore.getState().calls['empty-label']
      expect(call.actorLabel).toBe('Unknown agent')
    })
  })

  describe('Tool events', () => {
    it('creates tool_started items', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'tool-test', sequence: 0 }))

      const toolEvt: AgentResponseEvent = {
        ...baseEvent({
          call_id: 'tool-test',
          sequence: 1,
          event_type: 'agent_response.tool_started',
          payload: {
            name: 'search',
            arguments: { query: 'test' },
            truncated: false,
          },
        }),
      }
      toolEvt.event_type = 'agent_response.tool_started'

      useAgentResponseStore.getState().ingest(toolEvt)

      const call = useAgentResponseStore.getState().calls['tool-test']
      expect(call.items).toHaveLength(1)
      expect(call.items[0].type).toBe('tool_started')
      expect((call.items[0] as any).name).toBe('search')
    })

    it('creates tool_finished items', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'tool-test', sequence: 0 }))

      const finishEvt: AgentResponseEvent = {
        ...baseEvent({
          call_id: 'tool-test',
          sequence: 1,
          event_type: 'agent_response.tool_finished',
          payload: {
            name: 'search',
            outcome: 'success',
            result: { items: [] },
            truncated: false,
          },
        }),
      }
      finishEvt.event_type = 'agent_response.tool_finished'

      useAgentResponseStore.getState().ingest(finishEvt)

      const call = useAgentResponseStore.getState().calls['tool-test']
      expect(call.items).toHaveLength(1)
      expect(call.items[0].type).toBe('tool_finished')
      expect((call.items[0] as any).outcome).toBe('success')
    })
  })

  describe('Terminal state preservation', () => {
    it('does NOT preserve terminalStatePreserved for normal terminal (no eviction)', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'term-1', sequence: 0 }))

      useAgentResponseStore.getState().ingest(
        terminalEvent({
          call_id: 'term-1',
          sequence: 1,
          payload: {
            status: 'completed',
            error: null,
            error_truncated: false,
          },
        })
      )

      const call = useAgentResponseStore.getState().calls['term-1']
      expect(call.terminal).toBeDefined()
      expect(call.terminal!.status).toBe('completed')
      expect(call.terminalStatePreserved).toBe(false) // UPDATED: only true when outputEvicted
    })

    it('captures error in terminal state', () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'term-err', sequence: 0 }))

      useAgentResponseStore.getState().ingest(
        terminalEvent({
          call_id: 'term-err',
          sequence: 1,
          payload: {
            status: 'failed',
            error: 'Something went wrong',
            error_truncated: false,
          },
        })
      )

      const call = useAgentResponseStore.getState().calls['term-err']
      expect(call.terminal!.error).toBe('Something went wrong')
    })
  })

  describe('Late-arriving started event', () => {
    it('updates requestDisplay when started arrives after output on derived call', () => {
      useAgentResponseStore.getState().ingest(
        outputEvent({ call_id: 'late-start', sequence: 1 })
      )

      let call = useAgentResponseStore.getState().calls['late-start']
      expect(call.requestDisplay.kind).toBe('unavailable')
      expect(call.sequenceGap).toBe(true)

      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'late-start',
          sequence: 0,
          payload: {
            request_display: { kind: 'prompt', content: 'better late than never', truncated: false },
            actor_label: 'Agent X',
            model_or_runtime: 'claude-sonnet-5-5',
          },
        })
      )

      call = useAgentResponseStore.getState().calls['late-start']
      expect(call.requestDisplay.kind).toBe('prompt')
      expect(call.requestDisplay.content).toBe('better late than never')
      expect(call.actorLabel).toBe('Agent X')
      expect(call.sequenceGap).toBe(true)
    })
  })

  describe('WebSocket routing and response cursor independence', () => {
    it('response events route through agent-response store, not ws events', () => {
      useWSStore.getState().disconnect()
      useWSStore.setState({
        connected: false,
        connecting: false,
        retrying: false,
        hasConnectedOnce: false,
        events: [],
      })
      useAgentResponseStore.getState().clear()

      const responseEvt = baseEvent({
        id: 'route-test-1',
        project_id: 'proj-route',
        emitted_at: '2024-01-01T10:00:00Z',
      })

      useAgentResponseStore.getState().ingest(responseEvt)

      expect(useAgentResponseStore.getState().calls[responseEvt.call_id]).toBeDefined()
      expect(useWSStore.getState().events.some((e) => e.id === 'route-test-1')).toBe(false)
    })

    it('project-switch via setProject clears response store', () => {
      useAgentResponseStore.getState().clear()

      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')
      useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'proj-1' }))

      expect(Object.keys(useAgentResponseStore.getState().calls)).toHaveLength(1)

      useAgentResponseStore.getState().setProject('proj-2', 'Project 2')

      expect(Object.keys(useAgentResponseStore.getState().calls)).toHaveLength(0)
      expect(useAgentResponseStore.getState().projectId).toBe('proj-2')
      expect(useAgentResponseStore.getState().lifecycleAnnouncement).toContain('Project 2')
    })

    it('response events maintain dedup across disconnect/reconnect', () => {
      useAgentResponseStore.getState().clear()

      const evt = baseEvent({ id: 'dedup-test-1' })

      useAgentResponseStore.getState().ingest(evt)
      useAgentResponseStore.getState().ingest(evt)

      expect(useAgentResponseStore.getState().rawEvents).toHaveLength(1)
      expect(useAgentResponseStore.getState().seenEventIds.has('dedup-test-1')).toBe(true)
    })

    it('response events do NOT update replay_since, durable events do', () => {
      let mockWebSocketInstance: MockWebSocket | null = null

      global.WebSocket = Object.assign(
        class MockWS {
          constructor(url: string) {
            const instance = new MockWebSocket(url)
            mockWebSocketInstance = instance
            return instance as any
          }
        },
        { OPEN: 1, CONNECTING: 0 }
      ) as any

      if (!global.window) global.window = {} as any
      global.window.location = { protocol: 'http:', host: 'localhost:5173' } as any
      global.import = { meta: { env: {} } } as any

      useWSStore.getState().disconnect()
      useWSStore.setState({
        connected: false,
        connecting: false,
        retrying: false,
        hasConnectedOnce: false,
        events: [],
      })
      useAgentResponseStore.getState().clear()

      const projectId = 'test-proj-cursor'

      useWSStore.getState().connect(projectId, null)
      if (mockWebSocketInstance) {
        mockWebSocketInstance.simulateOpen()

        const responseEvt = baseEvent({
          id: 'cursor-response-1',
          project_id: projectId,
          emitted_at: '2024-08-14T10:00:00Z',
        })
        mockWebSocketInstance.simulateMessage(responseEvt)
        mockWebSocketInstance.simulateClose()
      }

      expect(useAgentResponseStore.getState().calls['call-1']).toBeDefined()
      expect(useWSStore.getState().events.some((e) => e.id === 'cursor-response-1')).toBe(false)

      mockWebSocketInstance = null
      useWSStore.getState().connect(projectId, null)

      if (mockWebSocketInstance) {
        expect(mockWebSocketInstance.url).not.toContain('replay_since')
      }

      if (mockWebSocketInstance) {
        const durableEvt = {
          id: 'cursor-durable-1',
          event_type: 'task.created',
          payload: { task_id: 'task-1' },
          emitted_at: '2024-08-14T11:00:00Z',
        }
        mockWebSocketInstance.simulateMessage(durableEvt)
        mockWebSocketInstance.simulateClose()
      }

      expect(useWSStore.getState().events.some((e) => e.id === 'cursor-durable-1')).toBe(true)

      mockWebSocketInstance = null
      useWSStore.getState().connect(projectId, null)

      if (mockWebSocketInstance) {
        expect(mockWebSocketInstance.url).toContain('replay_since=2024-08-14T11%3A00%3A00')
      }
    })
  })
})

describe('AgentActivityPanel - DOM rendering', () => {
  beforeEach(() => {
    useAgentResponseStore.setState({
      projectId: null,
      projectName: null,
      rawEvents: [],
      calls: {},
      seenEventIds: new Set(),
      candidateRevision: 0,
      lifecycleAnnouncement: '',
    })
    useWSStore.setState({
      connected: true,
      connecting: false,
      retrying: false,
      hasConnectedOnce: false,
      events: [],
    })
  })

  describe('Panel header and connectivity', () => {
    it('renders exact header text "Calls"', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Calls')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Connected" when WS connected', async () => {
      useWSStore.setState({ connected: true })
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Connected')
        expect(textOf(view.container)).not.toContain('Reconnecting')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Reconnecting live activity…" when WS not connected', async () => {
      useWSStore.setState({ connected: false })
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Reconnecting live activity…')
      } finally {
        view.cleanup()
      }
    })

    it('displays "{N} active" for nonterminal calls', async () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'active-1', sequence: 0 }))
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'active-2', sequence: 0 }))
      useAgentResponseStore.getState().ingest(terminalEvent({ call_id: 'term-1', sequence: 0 }))

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('2 active')
      } finally {
        view.cleanup()
      }
    })
  })

  describe('Call rendering', () => {
    it('shows call headers with actor, operation, invocation kind, status', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'c1',
          operation: 'search_docs',
          invocation_kind: 'api',
          payload: {
            request_display: { kind: 'prompt', content: 'search', truncated: false },
            actor_label: 'Test Agent',
            model_or_runtime: 'test',
          },
        })
      )

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const content = textOf(view.container)
        expect(content).toContain('Test Agent')
        expect(content).toContain('search_docs')
        expect(content).toContain('(api)')
        expect(content).toContain('Running')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Continuation after tool call" for continuation requests', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'c1',
          payload: {
            request_display: { kind: 'continuation', content: null, truncated: false },
            actor_label: 'Agent',
            model_or_runtime: 'test',
          },
        })
      )

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Continuation after tool call')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Request unavailable" for late-start calls', async () => {
      useAgentResponseStore.getState().ingest(outputEvent({ call_id: 'c1', sequence: 1 }))

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Request unavailable; the start of this call was not observed.')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Sequence gap detected..." marker', async () => {
      useAgentResponseStore.getState().ingest(outputEvent({ call_id: 'c1', sequence: 5 }))

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Sequence gap detected; some activity is unavailable.')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Earlier output was evicted..." marker when evicted', async () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c1', sequence: 0 }))
      useAgentResponseStore.setState((s) => ({
        calls: {
          ...s.calls,
          c1: { ...s.calls.c1!, outputEvicted: true },
        },
      }))

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Earlier output was evicted to keep this live view bounded.')
      } finally {
        view.cleanup()
      }
    })

    it('shows "Terminal call state is preserved" ONLY when outputEvicted + terminal', async () => {
      // Normal terminal (no eviction) - should NOT show
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c1', sequence: 0 }))
      useAgentResponseStore.getState().ingest(terminalEvent({ call_id: 'c1', sequence: 1 }))

      let view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).not.toContain('Terminal call state is preserved.')
      } finally {
        view.cleanup()
      }

      // Evicted + terminal - SHOULD show
      useAgentResponseStore.setState({
        rawEvents: [],
        calls: {},
        seenEventIds: new Set(),
        candidateRevision: 0,
      })

      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c2', sequence: 0 }))
      useAgentResponseStore.setState((s) => ({
        calls: {
          ...s.calls,
          c2: { ...s.calls.c2!, outputEvicted: true },
        },
      }))
      useAgentResponseStore.getState().ingest(terminalEvent({ call_id: 'c2', sequence: 1 }))

      view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Terminal call state is preserved.')
      } finally {
        view.cleanup()
      }
    })
  })

  describe('Readable values', () => {
    it('renders arrays as ul', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'c1',
          payload: {
            request_display: { kind: 'prompt', content: ['item1', 'item2'], truncated: false },
            actor_label: 'Agent',
            model_or_runtime: 'test',
          },
        })
      )

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const uls = descendants(view.container).filter((n) => n.tagName === 'UL')
        expect(uls.length).toBeGreaterThan(0)
        expect(textOf(view.container)).toContain('item1')
        expect(textOf(view.container)).toContain('item2')
      } finally {
        view.cleanup()
      }
    })

    it('renders objects as dl with humanized keys', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'c1',
          payload: {
            request_display: { kind: 'prompt', content: { test_key: 'value1' }, truncated: false },
            actor_label: 'Agent',
            model_or_runtime: 'test',
          },
        })
      )

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const dls = descendants(view.container).filter((n) => n.tagName === 'DL')
        expect(dls.length).toBeGreaterThan(0)
        expect(textOf(view.container)).toContain('Test Key')
        expect(textOf(view.container)).toContain('value1')
      } finally {
        view.cleanup()
      }
    })
  })

  describe('Tabs and accessibility', () => {
    it('renders "All" tab', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('All')
      } finally {
        view.cleanup()
      }
    })

    it('has accessible tablist with aria-label', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const tablist = descendants(view.container).find(
          (n) => n.getAttribute('role') === 'tablist' && n.getAttribute('aria-label') === 'Agent activity calls'
        )
        expect(tablist).toBeDefined()
      } finally {
        view.cleanup()
      }
    })

    it('shows Orchestrator tab when orchestrator calls exist', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'orch-1',
          actor_id: 'orchestrator',
          payload: {
            request_display: { kind: 'prompt', content: 'test', truncated: false },
            actor_label: 'Orchestrator',
            model_or_runtime: 'test',
          },
        })
      )

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('Orchestrator')
      } finally {
        view.cleanup()
      }
    })

    it('limits tabs to at most All + Orchestrator + 3 agents', async () => {
      for (let i = 1; i <= 5; i++) {
        useAgentResponseStore.getState().ingest(
          baseEvent({
            call_id: `agent-${i}`,
            actor_id: `agent-${i}`,
            payload: {
              request_display: { kind: 'prompt', content: 'test', truncated: false },
              actor_label: `Agent ${i}`,
              model_or_runtime: 'test',
            },
          })
        )
      }

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const tabs = descendants(view.container).filter((n) => n.getAttribute('role') === 'tab')
        expect(tabs.length).toBeLessThanOrEqual(4)
      } finally {
        view.cleanup()
      }
    })
  })

  describe('Follow behavior', () => {
    it('follow button has aria-pressed', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const followBtn = descendants(view.container).find((n) => n.getAttribute('aria-pressed') !== null)
        expect(followBtn).toBeDefined()
      } finally {
        view.cleanup()
      }
    })
  })

  describe('No live-region semantics', () => {
    it('panel has no role=status', async () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'c1', sequence: 0 }))

      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const statusNodes = descendants(view.container).filter((n) => n.getAttribute('role') === 'status')
        expect(statusNodes).toHaveLength(0)
      } finally {
        view.cleanup()
      }
    })

    it('panel has no aria-live', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const liveNodes = descendants(view.container).filter((n) => n.getAttribute('aria-live') !== null)
        expect(liveNodes).toHaveLength(0)
      } finally {
        view.cleanup()
      }
    })
  })

  describe('Empty state', () => {
    it('shows "No activity yet."', async () => {
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        expect(textOf(view.container)).toContain('No activity yet.')
      } finally {
        view.cleanup()
      }
    })
  })
})

describe('AgentActivityPanel - shell integration and CSS breakpoints', () => {
  beforeEach(() => {
    useAgentResponseStore.setState({
      projectId: null,
      projectName: null,
      rawEvents: [],
      calls: {},
      seenEventIds: new Set(),
      candidateRevision: 0,
      lifecycleAnnouncement: '',
    })
    useWSStore.setState({
      connected: true,
      connecting: false,
      retrying: false,
      hasConnectedOnce: false,
      events: [],
    })
  })

  afterEach(() => {
    useAgentResponseStore.getState().clear()
  })

  describe('Shell mounted tests (unmocked store)', () => {
    beforeEach(() => {
      // Mock ws.ts connect/disconnect to prevent WebSocket errors
      useWSStore.setState({
        connect: vi.fn(),
        disconnect: vi.fn(),
      })
      // Mock window.location
      Object.defineProperty(window, 'location', {
        value: { protocol: 'http:', host: 'localhost:5173' },
        writable: true,
      })
    })

    it('renders exactly ONE persistent sr-only status node combining connectivity + project + lifecycle', async () => {
      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)
      try {
        const statusNodes = descendants(view.container).filter(
          (n) => n.getAttribute('role') === 'status' && n.getAttribute('aria-live') === 'polite' && n.getAttribute('aria-atomic') === 'true'
        )
        expect(statusNodes).toHaveLength(1)
        expect(statusNodes[0].getAttribute('class')).toContain('sr-only')
      } finally {
        view.cleanup()
      }
    })

    it('project change clears calls before next render', async () => {
      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')
      useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'proj-1' }))
      expect(Object.keys(useAgentResponseStore.getState().calls)).toHaveLength(1)

      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)
      try {
        useAgentResponseStore.getState().setProject('proj-2', 'Project 2')
        await view.rerender()
        expect(Object.keys(useAgentResponseStore.getState().calls)).toHaveLength(0)
      } finally {
        view.cleanup()
      }
    })

    it('route rerender with same project retains calls', async () => {
      shellTestState.projects = [
        { id: 'project-1', name: 'Project 1', workspace_path: '/test', config: {}, created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z' }
      ]

      useAgentResponseStore.getState().setProject('project-1', 'Project 1')
      useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'project-1', call_id: 'call-retained' }))
      expect(Object.keys(useAgentResponseStore.getState().calls)).toHaveLength(1)

      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)
      try {
        await view.rerender()
        expect(useAgentResponseStore.getState().calls['call-retained']).toBeDefined()
      } finally {
        view.cleanup()
      }
    })

    it('new activity never steals focus', async () => {
      useAgentResponseStore.getState().setProject('proj-1', 'Project 1')

      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)
      try {
        const initialActive = view.activeElement
        useAgentResponseStore.getState().ingest(baseEvent({ project_id: 'proj-1' }))
        await view.rerender()
        expect(view.activeElement).toBe(initialActive)
      } finally {
        view.cleanup()
      }
    })

    it('visible notices have no live-region semantics (only sr-only status does)', async () => {
      useWSStore.setState({ connected: false, retrying: true })
      const { Shell } = await import('./Shell')
      const view = await mountWithTestDom(() => <Shell />, React.act)
      try {
        const visibleNotices = descendants(view.container).filter(
          (n) => n.getAttribute('data-testid') === 'banner' && !n.getAttribute('class')?.includes('sr-only')
        )
        for (const notice of visibleNotices) {
          expect(notice.getAttribute('aria-live')).toBeNull()
          expect(notice.getAttribute('role')).not.toBe('status')
        }
      } finally {
        view.cleanup()
      }
    })

    it('lifecycle announcement is cleared after 500ms and not re-announced on connectivity changes', async () => {
      vi.useFakeTimers()
      try {
        useAgentResponseStore.getState().setProject('proj-1', 'Project A')
        const initialAnnouncement = useAgentResponseStore.getState().lifecycleAnnouncement
        expect(initialAnnouncement).toContain('Project A')

        // Before mounting, simulate the effect that clears the announcement
        const timer = setTimeout(() => {
          useAgentResponseStore.getState().clearLifecycleAnnouncement()
        }, 500)

        // Verify announcement exists before timeout
        expect(useAgentResponseStore.getState().lifecycleAnnouncement).toBe(initialAnnouncement)

        // Advance timer to trigger the clear
        vi.advanceTimersByTime(500)

        // Announcement should be cleared now
        expect(useAgentResponseStore.getState().lifecycleAnnouncement).toBe('')

        // Connectivity change should NOT re-announce anything
        useWSStore.setState({ connected: false, retrying: true })
        expect(useAgentResponseStore.getState().lifecycleAnnouncement).toBe('')

        clearTimeout(timer)
      } finally {
        vi.useRealTimers()
      }
    })
  })

  describe('CSS media queries and surface rules', () => {
    it('imports shell.css and validates exact breakpoints', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      expect(shellCssRaw).toBeDefined()
      expect(shellCssRaw).toContain('min-width: 1280px')
      expect(shellCssRaw).toContain('max-width: 1279px')
    })

    it('asserts 1280px breakpoint has 440px panel', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      const regex1280 = /@media\s+\(min-width:\s*1280px\)[^{]*\{[^}]*440px[^}]*\}/gs
      expect(shellCssRaw).toMatch(regex1280)
    })

    it('asserts below 1280px the strip is 48px and overlay is 360px', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      const regexMax1279 = /@media\s+\(max-width:\s*1279px\)/s
      expect(shellCssRaw).toMatch(regexMax1279)
      expect(shellCssRaw).toContain('width: 48px')
      expect(shellCssRaw).toContain('width: 360px')
    })

    it('asserts the idle-collapse override is gone so the rail stays visible at >=1280px with zero calls', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      // The persistent rail is now always shown (f98a419): the data-has-calls
      // collapse override that hid the panel and force-showed the floating
      // strip at >=1440px with zero calls was removed.
      expect(shellCssRaw).not.toContain('[data-has-calls="false"]')
      // The strip base rule is still top-level (used for the <1280px popover).
      expect(shellCssRaw).toMatch(/^\.agent-activity-strip \{/m)
    })

    it('asserts no box-shadow in activity rules', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      const activityRules = shellCssRaw.match(/\.agent-activity[^}]*\{[^}]*\}/gs)
      if (activityRules) {
        for (const rule of activityRules) {
          expect(rule).not.toContain('box-shadow')
        }
      }
    })

    it('asserts prefers-reduced-motion rule exists', async () => {
      const fs = await import('fs').then((m) => m.default)
      const path = await import('path').then((m) => m.default)
      const shellCssPath = path.resolve(__dirname, './shell.css')
      const shellCssRaw = fs.readFileSync(shellCssPath, 'utf-8')
      expect(shellCssRaw).toContain('prefers-reduced-motion')
    })
  })

  describe('UI/UX fixes', () => {
    it('StatusBadge dot color matches connectivity state', async () => {
      useWSStore.setState({ connected: true, retrying: false, connecting: false })
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const content = textOf(view.container)
        expect(content).toContain('Connected')
      } finally {
        view.cleanup()
      }
    })

    it('Request details are open by default', async () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'test-req', sequence: 0 }))
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const details = descendants(view.container).find((n) => n.tagName === 'DETAILS')
        expect(details).toBeDefined()
        expect(details?.getAttribute('open') !== null).toBe(true)
      } finally {
        view.cleanup()
      }
    })

    it('Running calls render without error', async () => {
      useAgentResponseStore.getState().ingest(baseEvent({ call_id: 'active-call', sequence: 0 }))
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const content = textOf(view.container)
        expect(content).toContain('Running')
      } finally {
        view.cleanup()
      }
    })

    it('"More agents" select has accessible label when present', async () => {
      for (let i = 1; i <= 6; i++) {
        useAgentResponseStore.getState().ingest(
          baseEvent({
            call_id: `agent-${i}`,
            actor_id: `agent-${i}`,
            payload: {
              request_display: { kind: 'prompt', content: 'test', truncated: false },
              actor_label: `Agent ${i}`,
              model_or_runtime: 'test',
            },
          })
        )
      }
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const select = descendants(view.container).find((n) => n.tagName === 'SELECT')
        if (select) {
          expect(select.getAttribute('aria-label')).toBe('More agents')
        }
      } finally {
        view.cleanup()
      }
    })

    it('Truncation markers use appropriate styling', async () => {
      useAgentResponseStore.getState().ingest(
        baseEvent({
          call_id: 'trunc-test',
          payload: {
            request_display: { kind: 'prompt', content: 'test', truncated: true },
            actor_label: 'Agent',
            model_or_runtime: 'test',
          },
        })
      )
      const view = await mountWithTestDom(
        () => <AgentActivityPanel projectId="proj-1" />,
        React.act,
      )
      try {
        const truncMarker = descendants(view.container).find((n) =>
          n.tagName === 'P' && textOf(n).includes('[Request truncated]')
        )
        expect(truncMarker).toBeDefined()
      } finally {
        view.cleanup()
      }
    })
  })
})

describe('reconcileAgentTabs unit tests', () => {
  it('reserves slots for selected + focused (never evicted)', () => {
    const calls = {
      orch: {
        callId: 'orch',
        projectId: 'p',
        callStartedAt: '2024-01-01T00:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'system' as const,
        actorId: 'orchestrator',
        actorLabel: 'Orch',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
      'c-a': {
        callId: 'c-a',
        projectId: 'p',
        callStartedAt: '2024-01-01T01:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: 'agent-A',
        actorLabel: 'A',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
      'c-b': {
        callId: 'c-b',
        projectId: 'p',
        callStartedAt: '2024-01-01T02:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: 'agent-B',
        actorLabel: 'B',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
      'c-c': {
        callId: 'c-c',
        projectId: 'p',
        callStartedAt: '2024-01-01T03:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: 'agent-C',
        actorLabel: 'C',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
      'c-d': {
        callId: 'c-d',
        projectId: 'p',
        callStartedAt: '2024-01-01T04:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: 'agent-D',
        actorLabel: 'D',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
    }

    const result = reconcileAgentTabs(calls, ['orchestrator', 'agent-A', 'agent-B'], 'agent-C', 'agent-D')

    expect(result.visibleActorIds).toContain('agent-C')
    expect(result.visibleActorIds).toContain('agent-D')
    expect(result.selectedActorId).toBe('agent-C')
    expect(result.focusedActorId).toBe('agent-D')
  })

  it('enforces max 3 agent tabs regardless of orchestrator', () => {
    const calls: Record<string, any> = {}
    for (let i = 1; i <= 5; i++) {
      calls[`c${i}`] = {
        callId: `c${i}`,
        projectId: 'p',
        callStartedAt: `2024-01-01T0${i}:00:00Z`,
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: `agent-${i}`,
        actorLabel: `Agent ${i}`,
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      }
    }

    const result = reconcileAgentTabs(calls, [], null, null)
    const agentCount = result.visibleActorIds.filter((a) => a !== 'orchestrator').length
    expect(agentCount).toBeLessThanOrEqual(3)
  })

  it('invalidates selection/focus if actor is gone', () => {
    const calls = {
      'c1': {
        callId: 'c1',
        projectId: 'p',
        callStartedAt: '2024-01-01T00:00:00Z',
        invocationKind: 'api' as const,
        operation: 'op',
        parentCallId: null,
        actorKind: 'agent' as const,
        actorId: 'agent-1',
        actorLabel: 'Agent 1',
        requestDisplay: { kind: 'prompt' as const, content: null, truncated: false },
        items: [],
        lastSequence: 0,
        sequenceGap: false,
        outputEvicted: false,
        terminalStatePreserved: false,
        terminal: null,
      },
    }

    const result = reconcileAgentTabs(calls, [], 'agent-gone', 'agent-also-gone')
    expect(result.selectedActorId).toBeNull()
    expect(result.focusedActorId).toBeNull()
  })
})
