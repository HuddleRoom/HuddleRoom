// Wire types matching backend contract exactly
export type InvocationKind = 'api' | 'cli_main' | 'cli_resume' | 'cli_meeting'

export interface RequestDisplay {
  kind: 'prompt' | 'continuation' | 'unavailable'
  content: unknown | null
  truncated: boolean
}

export interface FinalResponseSnapshot {
  content: JsonValue | null
  tool_calls: JsonValue | null
  usage: JsonValue | null
  content_truncated: boolean
  tool_calls_truncated: boolean
  usage_truncated: boolean
  tool_count: number
  text_present: boolean
}

export type JsonValue = string | number | boolean | null | Record<string, unknown> | unknown[]

export interface AgentResponseEvent {
  id: string
  project_id: string
  call_id: string
  sequence: number
  event_type:
    | 'agent_response.started'
    | 'agent_response.output'
    | 'agent_response.tool_started'
    | 'agent_response.tool_finished'
    | 'agent_response.terminal'
  emitted_at: string
  call_started_at: string
  invocation_kind: InvocationKind
  operation: string
  parent_call_id: string | null
  actor_kind: 'system' | 'agent'
  actor_id: string
  payload: Record<string, JsonValue>
}

// Derived call item types
export interface OutputItem {
  type: 'output'
  stream: 'output' | 'reasoning' | 'stderr'
  text: string
  sequence: number
}

export interface ToolStartedItem {
  type: 'tool_started'
  name: string
  arguments: JsonValue | null
  truncated: boolean
  sequence: number
}

export interface ToolFinishedItem {
  type: 'tool_finished'
  name: string
  outcome: string
  result: JsonValue | null
  truncated: boolean
  sequence: number
}

export type AgentCallItem = OutputItem | ToolStartedItem | ToolFinishedItem

export interface AgentCallRecord {
  callId: string
  projectId: string
  callStartedAt: string
  invocationKind: InvocationKind
  operation: string
  parentCallId: string | null
  actorKind: AgentResponseEvent['actor_kind']
  actorId: string
  actorLabel: string
  modelOrRuntime: string
  requestDisplay: RequestDisplay
  items: AgentCallItem[]
  lastSequence: number
  sequenceGap: boolean
  outputEvicted: boolean
  terminalStatePreserved: boolean
  observationTruncated: boolean
  finalResponse: FinalResponseSnapshot | null
  terminal: null | {
    status: string
    emittedAt: string
    error?: string
    errorTruncated?: boolean
    metadata?: JsonValue | null
    metadataTruncated?: boolean
  }
}

const OPERATION_LABELS: Record<string, string> = {
  goal_analysis: 'Goal Analysis',
  plan_generation: 'Plan Generation',
  task_execution: 'Task Execution',
  agent_call: 'Agent Call',
  orchestration: 'Orchestration',
}

export function friendlyOperation(operation: string): string {
  const known = Object.prototype.hasOwnProperty.call(OPERATION_LABELS, operation) ? OPERATION_LABELS[operation] : undefined
  if (known) return known
  return operation
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase()) || 'Unknown Operation'
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
  if (value && typeof value === 'object') {
    const object = value as Record<string, unknown>
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${stableJson(object[key])}`).join(',')}}`
  }
  try { return JSON.stringify(value) ?? String(value) } catch { return String(value) }
}

function principalText(value: unknown): string | null {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    const parts = value.map(principalText).filter((part): part is string => !!part)
    return parts.length ? parts.join(' ') : null
  }
  if (!value || typeof value !== 'object') return null
  const object = value as Record<string, unknown>
  for (const key of ['text', 'content', 'message', 'task', 'query']) {
    const result = principalText(object[key])
    if (result) return result
  }
  return null
}

export function requestPreview(display: RequestDisplay, maxLength = 240): string {
  if (display.kind === 'unavailable' || display.content == null) return ''
  const text = principalText(display.content) ?? stableJson(display.content)
  const limit = Math.max(0, maxLength)
  if (text.length <= limit) return text
  if (limit === 0) return ''
  return `${text.slice(0, Math.max(0, limit - 1))}…`
}

export function assembleStream(items: AgentCallItem[], stream: OutputItem['stream'] = 'output'): string {
  return items
    .filter((item): item is OutputItem => item.type === 'output' && item.stream === stream)
    .sort((a, b) => a.sequence - b.sequence)
    .map((item) => item.text)
    .join('')
}

export interface ResponsePresentation {
  value: JsonValue | undefined
  provisional: string
  mismatch: boolean
  finalLimited: boolean
  empty: boolean
  toolOnly: boolean
}

type StructuredTextParse = { parsed: true; value: JsonValue } | { parsed: false }

function parseStructuredText(text: string): StructuredTextParse {
  const trimmed = text.trim()
  if (!trimmed) return { parsed: false }
  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (
      parsed === null ||
      ['string', 'number', 'boolean'].includes(typeof parsed) ||
      Array.isArray(parsed) ||
      typeof parsed === 'object'
    ) {
      return { parsed: true, value: parsed as JsonValue }
    }
  } catch {
    // Incomplete or non-JSON text remains readable text.
  }
  return { parsed: false }
}

export function responsePresentation(call: AgentCallRecord): ResponsePresentation {
  const provisional = assembleStream(call.items)
  const terminal = call.terminal !== null
  const snapshot = terminal ? call.finalResponse : null
  const finalLimited = !!snapshot?.content_truncated
  const parsedProvisional = terminal ? parseStructuredText(provisional) : { parsed: false as const }
  const toolCount = snapshot?.tool_count ?? call.items.filter((item) => item.type !== 'output').length
  const hasToolCalls = toolCount > 0
  const toolOnly = terminal && hasToolCalls &&
    (snapshot
      ? !snapshot.text_present && (snapshot.content === null || snapshot.content === '')
      : !provisional)
  let value: JsonValue | undefined
  if (snapshot) {
    if (finalLimited && provisional) {
      value = parsedProvisional.parsed ? parsedProvisional.value : provisional
    } else if (snapshot.content !== null && snapshot.content !== '') {
      if (typeof snapshot.content === 'string') {
        const parsedSnapshot = parseStructuredText(snapshot.content)
        value = parsedSnapshot.parsed ? parsedSnapshot.value : snapshot.content
      } else {
        value = snapshot.content
      }
    }
  } else if (parsedProvisional.parsed) {
    value = parsedProvisional.value
  } else if (provisional) {
    value = provisional
  }
  const normalizedProvisional = parsedProvisional.parsed ? parsedProvisional.value : provisional
  const mismatch = terminal && !finalLimited && !!provisional &&
    (snapshot !== null
      ? value === undefined || stableJson(value) !== stableJson(normalizedProvisional)
      : false)
  const empty = terminal && !toolOnly && value === undefined
  return { value, provisional, mismatch, finalLimited, empty, toolOnly }
}

/**
 * Validate that a message is an AgentResponseEvent by checking common fields
 * and event-specific payload shape.
 */
export function isAgentResponseEvent(msg: unknown): msg is AgentResponseEvent {
  if (typeof msg !== 'object' || msg === null) return false

  const m = msg as Record<string, unknown>

  // Check common fields
  if (
    typeof m.id !== 'string' ||
    typeof m.project_id !== 'string' ||
    typeof m.call_id !== 'string' ||
    typeof m.sequence !== 'number' ||
    typeof m.event_type !== 'string' ||
    typeof m.emitted_at !== 'string' ||
    typeof m.call_started_at !== 'string' ||
    typeof m.invocation_kind !== 'string' ||
    typeof m.operation !== 'string' ||
    typeof m.actor_kind !== 'string' ||
    typeof m.actor_id !== 'string' ||
    typeof m.payload !== 'object' ||
    m.payload === null
  ) {
    return false
  }

  // Validate event_type and payload shape
  const payload = m.payload as Record<string, unknown>
  switch (m.event_type) {
    case 'agent_response.started': {
      const rd = payload.request_display
      if (typeof rd !== 'object' || rd === null) return false
      const rdObj = rd as Record<string, unknown>
      if (
        typeof rdObj.kind !== 'string' ||
        typeof rdObj.truncated !== 'boolean'
      ) {
        return false
      }
      if (
        typeof payload.actor_label !== 'string' &&
        payload.actor_label !== null
      ) {
        return false
      }
      if (typeof payload.model_or_runtime !== 'string') return false
      return true
    }

    case 'agent_response.output': {
      if (
        typeof payload.stream !== 'string' ||
        typeof payload.text !== 'string'
      ) {
        return false
      }
      return true
    }

    case 'agent_response.tool_started': {
      if (typeof payload.name !== 'string' || typeof payload.truncated !== 'boolean') {
        return false
      }
      return true
    }

    case 'agent_response.tool_finished': {
      if (
        typeof payload.name !== 'string' ||
        typeof payload.outcome !== 'string' ||
        typeof payload.truncated !== 'boolean'
      ) {
        return false
      }
      return true
    }

    case 'agent_response.terminal': {
      if (typeof payload.status !== 'string') return false
      if (typeof payload.error !== 'string' && payload.error !== null) return false
      if (typeof payload.error_truncated !== 'boolean') return false
      return true
    }

    default:
      return false
  }
}

/**
 * Sort calls by call_started_at ascending, tiebreak by call_id.
 */
export function orderedCalls(calls: Record<string, AgentCallRecord>) {
  return Object.values(calls).sort((a, b) =>
    a.callStartedAt.localeCompare(b.callStartedAt) || a.callId.localeCompare(b.callId)
  )
}

/**
 * Reconcile visible actor tabs: returns ordered actor IDs for the visible tabs
 * and stable actor selection/focus state.
 */
export interface ActorTabReconciliation {
  visibleActorIds: string[]
  selectedActorId: string | null
  focusedActorId: string | null
}

export function reconcileAgentTabs(
  calls: Record<string, AgentCallRecord>,
  previousVisibleActorIds: string[],
  selectedActorId: string | null,
  focusedActorId: string | null
): ActorTabReconciliation {
  // Collect all unique actors and track recency
  const allCalls = orderedCalls(calls)
  const actorIds = new Set<string>()
  const actorRecency = new Map<string, string>() // most recent callStartedAt per actor

  for (const call of allCalls) {
    actorIds.add(call.actorId)
    const existing = actorRecency.get(call.actorId)
    if (!existing || call.callStartedAt > existing) {
      actorRecency.set(call.actorId, call.callStartedAt)
    }
  }

  // Separate orchestrator from regular agents
  const orchestratorId = actorIds.has('orchestrator') ? 'orchestrator' : null
  const agentIds = Array.from(actorIds).filter((id) => id !== 'orchestrator')

  // Validate selection/focus still have calls
  let newSelectedActorId = selectedActorId && actorIds.has(selectedActorId) ? selectedActorId : null
  let newFocusedActorId = focusedActorId && actorIds.has(focusedActorId) ? focusedActorId : null

  // Build visible actor list: reserve slots for pinned (selected/focused) FIRST, then fill by recency
  // Cap: 1 orchestrator + 3 agents max
  const visibleActorIds: string[] = []

  // Always include orchestrator if it exists (reserved slot, never evicted)
  if (orchestratorId) {
    visibleActorIds.push(orchestratorId)
  }

  // Reserve slots for selected and focused (pinned, never evicted)
  if (newSelectedActorId && !visibleActorIds.includes(newSelectedActorId)) {
    visibleActorIds.push(newSelectedActorId)
  }

  if (newFocusedActorId && !visibleActorIds.includes(newFocusedActorId)) {
    visibleActorIds.push(newFocusedActorId)
  }

  // Fill remaining slots (max 3 agents total) with live-first (most recent)
  const agentsByRecency = [...agentIds].sort((a, b) => {
    const aTime = actorRecency.get(a) || ''
    const bTime = actorRecency.get(b) || ''
    return bTime.localeCompare(aTime) // descending (most recent first)
  })

  for (const agentId of agentsByRecency) {
    // Max agents: 3 if orchestrator exists, else 3 (total max 4)
    const maxAgents = orchestratorId ? 3 : 3
    const agentCount = visibleActorIds.filter((id) => id !== 'orchestrator').length
    if (agentCount >= maxAgents) break
    if (!visibleActorIds.includes(agentId)) {
      visibleActorIds.push(agentId)
    }
  }

  return {
    visibleActorIds,
    selectedActorId: newSelectedActorId,
    focusedActorId: newFocusedActorId,
  }
}
