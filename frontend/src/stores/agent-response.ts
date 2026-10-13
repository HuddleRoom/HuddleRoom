import { create } from 'zustand'
import { parseUtc } from '@/lib/time'
import {
  AgentResponseEvent,
  AgentCallRecord,
  RequestDisplay,
  AgentCallItem,
  FinalResponseSnapshot,
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
const RETENTION_DERIVED_ITEMS = 128
const RETENTION_OUTPUT_BYTES = 64 * 1024

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

function extractModelOrRuntime(event: AgentResponseEvent): string {
  if (event.event_type === 'agent_response.started') {
    const value = (event.payload as Record<string, unknown>).model_or_runtime
    if (typeof value === 'string') return value
  }
  return 'Unknown runtime'
}

function boundedUtf8Suffix(text: string, maxBytes: number): string {
  let result = ''
  let bytes = 0
  const chars = Array.from(text)
  for (let index = chars.length - 1; index >= 0; index--) {
    const char = chars[index]
    const size = new TextEncoder().encode(char).length
    if (bytes + size > maxBytes) break
    result = char + result
    bytes += size
  }
  return result
}

function enforceCallBounds(call: AgentCallRecord): AgentCallRecord {
  let items = [...call.items]
  let evicted = call.outputEvicted
  const textBytes = () => items.reduce((sum, item) => sum + (item.type === 'output' ? new TextEncoder().encode(item.text).length : 0), 0)
  while (items.length > RETENTION_DERIVED_ITEMS || textBytes() > RETENTION_OUTPUT_BYTES) {
    const oldestOutput = items.findIndex((item) => item.type === 'output')
    if (oldestOutput >= 0) {
      const totalBytes = textBytes()
      if (items.length <= RETENTION_DERIVED_ITEMS && totalBytes > RETENTION_OUTPUT_BYTES) {
        const item = items[oldestOutput]
        if (item.type === 'output') {
          const remainingBytes = totalBytes - new TextEncoder().encode(item.text).length
          const keepBytes = Math.max(0, RETENTION_OUTPUT_BYTES - remainingBytes)
          const suffix = boundedUtf8Suffix(item.text, keepBytes)
          if (suffix) items[oldestOutput] = { ...item, text: suffix }
          else items.splice(oldestOutput, 1)
        }
      } else {
        items.splice(oldestOutput, 1)
      }
      evicted = true
    } else {
      items.shift()
      evicted = true
    }
  }
  return { ...call, items, outputEvicted: evicted }
}

function extractFinalResponse(payload: Record<string, unknown>): FinalResponseSnapshot | null {
  const value = payload.final_response
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const snapshot = value as Record<string, unknown>
  return {
    content: (snapshot.content ?? null) as JsonValue | null,
    tool_calls: (snapshot.tool_calls ?? null) as JsonValue | null,
    usage: (snapshot.usage ?? null) as JsonValue | null,
    content_truncated: snapshot.content_truncated === true,
    tool_calls_truncated: snapshot.tool_calls_truncated === true,
    usage_truncated: snapshot.usage_truncated === true,
    tool_count: typeof snapshot.tool_count === 'number' ? snapshot.tool_count : 0,
    text_present: snapshot.text_present === true,
  }
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
      if (s.projectId && event.project_id !== s.projectId) return s
      // Skip if already seen
      if (s.seenEventIds.has(event.id)) return s

      const current = s.calls[event.call_id]
      const stale = !!current && event.sequence <= current.lastSequence

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

      // Keep the raw bounded feed and remember stale IDs, but do not let a
      // replayed payload overwrite newer derived content. A delayed `started`
      // record can still enrich request/runtime metadata for an existing call.
      if (stale && event.event_type !== 'agent_response.started') {
        return { rawEvents: newRawEvents, calls: callsToUpdate, seenEventIds: newSeenIds }
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
          modelOrRuntime: extractModelOrRuntime(event),
          requestDisplay: extractRequestDisplay(event),
          items: [],
          lastSequence: -1,
          sequenceGap: false,
          outputEvicted: false,
          terminalStatePreserved: false,
          observationTruncated: false,
          finalResponse: null,
          terminal: null,
        }
      } else {
        call = { ...existingCall }
      }

      if (event.event_type === 'agent_response.output') {
        const payload = event.payload as Record<string, unknown>
        call.observationTruncated = call.observationTruncated || payload.observation_truncated === true
      }

      // Update actor label if we have a better one (from started event)
      if (event.event_type === 'agent_response.started') {
        const label = getActorLabel(event)
        if (label && label !== 'Unknown agent') {
          call.actorLabel = label
        }
        // Preserve request_display from started event
        call.requestDisplay = extractRequestDisplay(event)
        call.modelOrRuntime = extractModelOrRuntime(event)
      }

      // Check for sequence gap: started events have sequence 0, so if we see
      // any non-zero sequence on a brand-new call (lastSequence === -1), it's a gap
      let hasGap = call.sequenceGap
      if (stale && event.event_type === 'agent_response.started') {
        hasGap = call.sequenceGap
      } else if (call.lastSequence === -1) {
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
        call.terminal = {
          ...terminalObj,
          errorTruncated: payload.error_truncated === true,
          metadata: (payload.metadata ?? null) as JsonValue | null,
          metadataTruncated: payload.metadata_truncated === true,
        }
        call.finalResponse = extractFinalResponse(payload)
        call.observationTruncated = call.observationTruncated || payload.observation_truncated === true
        // Only mark as preserved if output was evicted (folding happened)
        if (call.outputEvicted) {
          call.terminalStatePreserved = true
        }
      }

      // Assign updated call back to calls
      callsToUpdate[event.call_id] = enforceCallBounds(call)

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
          const startTime = parseUtc(c.callStartedAt).getTime()

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
