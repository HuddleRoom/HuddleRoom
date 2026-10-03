import { create } from 'zustand'
import {
  AgentResponseEvent,
  AgentCallRecord,
  RequestDisplay,
  AgentCallItem,
  isAgentResponseEvent,
  JsonValue,
} from '@/lib/agentResponse'

export interface AgentResponseState {
  projectId: string | null
  projectName: string | null
  rawEvents: AgentResponseEvent[]
  calls: Record<string, AgentCallRecord>
  seenEventIds: Set<string>
  candidateRevision: number
  lifecycleAnnouncement: string
  setProject(projectId: string | null, projectName?: string): void
  ingest(event: AgentResponseEvent): void
  clear(): void
  clearLifecycleAnnouncement(): void
}

const RETENTION_RAW_EVENTS = 200
const RETENTION_DERIVED_CALLS = 200

function getActorLabel(event: AgentResponseEvent): string {
  if (event.event_type === 'agent_response.started') {
    const payload = event.payload as Record<string, unknown>
    const label = payload.actor_label
    if (typeof label === 'string' && label) return label
  }
  return 'Unknown agent'
}

function extractRequestDisplay(event: AgentResponseEvent): RequestDisplay {
  if (event.event_type === 'agent_response.started') {
    const payload = event.payload as Record<string, unknown>
    if (payload.request_display && typeof payload.request_display === 'object') {
      return payload.request_display as RequestDisplay
    }
  }
  return { kind: 'unavailable', content: null, truncated: false }
}

function eventToItem(event: AgentResponseEvent): AgentCallItem | null {
  if (event.event_type === 'agent_response.output') {
    const payload = event.payload as Record<string, unknown>
    return {
      type: 'output',
      stream: payload.stream as 'output' | 'reasoning' | 'stderr',
      text: payload.text as string,
      sequence: event.sequence,
    }
  }

  if (event.event_type === 'agent_response.tool_started') {
    const payload = event.payload as Record<string, unknown>
    return {
      type: 'tool_started',
      name: payload.name as string,
      arguments: payload.arguments as JsonValue,
      truncated: (payload.truncated as boolean) || false,
      sequence: event.sequence,
    }
  }

  if (event.event_type === 'agent_response.tool_finished') {
    const payload = event.payload as Record<string, unknown>
    return {
      type: 'tool_finished',
      name: payload.name as string,
      outcome: payload.outcome as string,
      result: payload.result as JsonValue,
      truncated: (payload.truncated as boolean) || false,
      sequence: event.sequence,
    }
  }

  return null
}

export const useAgentResponseStore = create<AgentResponseState>()((set, get) => ({
  projectId: null,
  projectName: null,
  rawEvents: [],
  calls: {},
  // ponytail: seenEventIds grows unbounded per project session by design—required so dedup
  // survives reconnect-replay after events age out of the 200-cap. Upgrade to FIFO bound if long sessions matter.
  seenEventIds: new Set(),
  candidateRevision: 0,
  lifecycleAnnouncement: '',

  setProject(projectId: string | null, projectName?: string) {
    const state = get()
    if (state.projectId === projectId) return

    set((s) => {
      const announcement =
        projectId && projectName
          ? `Switched to ${projectName}; prior project activity was cleared.`
          : ''

      return {
        projectId,
        projectName: projectName || null,
        rawEvents: [],
        calls: {},
        seenEventIds: new Set(),
        candidateRevision: 0,
        lifecycleAnnouncement: announcement,
      }
    })
  },

  clear() {
    set({
      projectId: null,
      projectName: null,
      rawEvents: [],
      calls: {},
      seenEventIds: new Set(),
      candidateRevision: 0,
      lifecycleAnnouncement: '',
    })
  },

  clearLifecycleAnnouncement() {
    set({ lifecycleAnnouncement: '' })
  },

  ingest(event: AgentResponseEvent) {
    if (!isAgentResponseEvent(event)) return

    set((s) => {
      // Skip if already seen
      if (s.seenEventIds.has(event.id)) return s

      // Add to seen IDs
      const newSeenIds = new Set(s.seenEventIds)
      newSeenIds.add(event.id)

      // Add to raw events
      let newRawEvents = [...s.rawEvents, event]
      let callsToUpdate = { ...s.calls }

      // Retention: keep ≤200 rawEvents
      // Remove oldest OUTPUT event first; if none, fold lifecycle fields and remove oldest raw
      if (newRawEvents.length > RETENTION_RAW_EVENTS) {
        const outputIdx = newRawEvents.findIndex(
          (e) => e.event_type === 'agent_response.output'
        )
        if (outputIdx !== -1) {
          newRawEvents.splice(outputIdx, 1)
        } else if (newRawEvents.length > 0) {
          // Remove oldest, folding lifecycle fields into derived record
          const oldestEvent = newRawEvents[0]
          if (
            oldestEvent.event_type === 'agent_response.started' ||
            oldestEvent.event_type === 'agent_response.terminal'
          ) {
            // Mark the call as outputEvicted by creating a new call object
            const callId = oldestEvent.call_id
            const existingCall = callsToUpdate[callId]
            if (existingCall) {
              callsToUpdate[callId] = { ...existingCall, outputEvicted: true }
            }
          }
          newRawEvents.shift()
        }
      }

      // Get or create call (immutable)
      const existingCall = callsToUpdate[event.call_id]
      let call: AgentCallRecord
      if (!existingCall) {
        call = {
          callId: event.call_id,
          projectId: event.project_id,
          callStartedAt: event.call_started_at,
          invocationKind: event.invocation_kind,
          operation: event.operation,
          parentCallId: event.parent_call_id,
          actorKind: event.actor_kind,
          actorId: event.actor_id,
          actorLabel: getActorLabel(event),
          requestDisplay: extractRequestDisplay(event),
          items: [],
          lastSequence: -1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          terminal: null,
        }
      } else {
        call = { ...existingCall }
      }

      // Update actor label if we have a better one (from started event)
      if (event.event_type === 'agent_response.started') {
        const label = getActorLabel(event)
        if (label && label !== 'Unknown agent') {
          call.actorLabel = label
        }
        // Preserve request_display from started event
        call.requestDisplay = extractRequestDisplay(event)
      }

      // Check for sequence gap: started events have sequence 0, so if we see
      // any non-zero sequence on a brand-new call (lastSequence === -1), it's a gap
      let hasGap = call.sequenceGap
      if (call.lastSequence === -1) {
        if (event.sequence > 0) {
          hasGap = true
        }
      } else if (event.sequence !== call.lastSequence + 1) {
        hasGap = true
      }
      call.sequenceGap = hasGap

      // Update last sequence
      call.lastSequence = Math.max(call.lastSequence, event.sequence)

      // Add item if applicable (new items array)
      const item = eventToItem(event)
      if (item) {
        call.items = [...call.items, item]
      }

      // Handle terminal event
      if (event.event_type === 'agent_response.terminal') {
        const payload = event.payload as Record<string, unknown>
        const terminalObj: { status: string; emittedAt: string; error?: string } = {
          status: payload.status as string,
          emittedAt: event.emitted_at,
        }
        if (typeof payload.error === 'string') {
          terminalObj.error = payload.error
        }
        call.terminal = terminalObj
        // Only mark as preserved if output was evicted (folding happened)
        if (call.outputEvicted) {
          call.terminalStatePreserved = true
        }
      }

      // Assign updated call back to calls
      callsToUpdate[event.call_id] = call

      // Increment candidateRevision only on started/terminal/project-clear
      let newRevision = s.candidateRevision
      if (
        event.event_type === 'agent_response.started' ||
        event.event_type === 'agent_response.terminal'
      ) {
        newRevision += 1
      }

      // Retention: keep ≤200 derived calls
      // Remove oldest TERMINAL call, else oldest active
      const callIds = Object.keys(callsToUpdate)
      if (callIds.length > RETENTION_DERIVED_CALLS) {
        // Find oldest terminal call
        let oldestTerminalId: string | null = null
        let oldestTerminalStartTime = Infinity

        // Find oldest active (nonterminal) call
        let oldestActiveId: string | null = null
        let oldestActiveStartTime = Infinity

        for (const callId of callIds) {
          const c = callsToUpdate[callId]
          const startTime = new Date(c.callStartedAt).getTime()

          if (c.terminal) {
            if (startTime < oldestTerminalStartTime) {
              oldestTerminalId = callId
              oldestTerminalStartTime = startTime
            }
          } else {
            if (startTime < oldestActiveStartTime) {
              oldestActiveId = callId
              oldestActiveStartTime = startTime
            }
          }
        }

        if (oldestTerminalId) {
          const { [oldestTerminalId]: _, ...remainingCalls } = callsToUpdate
          callsToUpdate = remainingCalls
        } else if (oldestActiveId) {
          const { [oldestActiveId]: _, ...remainingCalls } = callsToUpdate
          callsToUpdate = remainingCalls
        }
      }

      return {
        rawEvents: newRawEvents,
        calls: callsToUpdate,
        seenEventIds: newSeenIds,
        candidateRevision: newRevision,
      }
    })
  },
}))
