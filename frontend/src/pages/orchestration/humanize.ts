import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
} from '@/lib/types'
import { ApiError } from '@/lib/api-client'
import { absolute } from '@/lib/time'

export const BASELINE_PROCESS_TYPES = [
  'goal_definition',
  'manager_selection',
  'agent_definition_review',
  'team_hierarchy',
  'effectiveness_review',
  'goal_closeout',
] as const satisfies readonly OrchestrationBaselineProcessType[]

const PROCESS_LABELS: Record<OrchestrationBaselineProcessType, string> = {
  goal_definition: 'Goal definition',
  manager_selection: 'Manager selection',
  agent_definition_review: 'Agent definition review',
  team_hierarchy: 'Team hierarchy',
  effectiveness_review: 'Effectiveness review',
  goal_closeout: 'Goal closeout',
}

// Static one-sentence description of what each baseline process does. Used
// by the Z4 outcome card as the fallback whenever a step hasn't produced
// (or is missing) the data its normal outcome card needs — Z4 must never
// render an empty shell.
const PROCESS_DESCRIPTIONS: Record<OrchestrationBaselineProcessType, string> = {
  goal_definition: 'The orchestrator reads the goal, asks any clarifying questions it needs, and classifies how much weight the work deserves.',
  manager_selection: 'The orchestrator decides who is accountable for approving decisions on this goal — an agent manager, a human, or no manager.',
  agent_definition_review: 'The orchestrator reviews each proposed agent definition for fit before the team is assembled.',
  team_hierarchy: 'The orchestrator assigns work functions to agents and records who reports to whom.',
  effectiveness_review: 'The orchestrator checks whether the goal is still on track and recommends whether to continue, revise, pause, or split it.',
  goal_closeout: 'The orchestrator confirms the evidence justifies closing the goal, then records why it was allowed to close or was cancelled.',
}

export function processDescription(type: OrchestrationBaselineProcessType): string {
  return PROCESS_DESCRIPTIONS[type]
}

const PROCESS_STATUS_PHRASES = {
  running: { label: 'In progress', tone: 'in_progress' },
  waiting_decision: { label: 'Needs you', tone: 'needs-you' },
  completed: { label: 'Done', tone: 'done' },
  skipped: { label: 'Skipped', tone: 'skipped' },
} as const

const GOAL_STATUS_LABELS: Record<string, string> = {
  active: 'Active',
  blocked: 'Blocked',
  paused: 'Paused',
  completed: 'Completed',
  cancelled: 'Cancelled',
}

const RUN_STATUS_LABELS: Record<string, string> = {
  running: 'Running',
  blocked: 'Blocked',
  paused: 'Paused',
  completed: 'Completed',
  cancelled: 'Cancelled',
}

const WARNING_SEVERITY_LABELS: Record<string, string> = {
  recommendation: 'Recommendation',
  warning: 'Warning',
  blocker: 'Blocker',
  hard_stop: 'Hard stop',
}

export function processLabel(type: string): string {
  return PROCESS_LABELS[type as OrchestrationBaselineProcessType] ?? type
}

// Shared step-display override: "Agent definition review" reads as a
// completed-state label, but while it's the selected/active step the
// operator is reading a live verb ("Reviewing agent definitions"). Used
// everywhere a step needs a human label (stepper, focus heading, queue tags,
// activity log).
export function stepDisplayLabel(type: OrchestrationBaselineProcessType): string {
  return type === 'agent_definition_review' ? 'Reviewing agent definitions' : processLabel(type)
}

export function statusPhrase(status: string) {
  return PROCESS_STATUS_PHRASES[status as keyof typeof PROCESS_STATUS_PHRASES]
    ?? { label: status, tone: 'idle' as const }
}

export function goalStatusLabel(status: string): string {
  return GOAL_STATUS_LABELS[status] ?? status
}

export function runStatusLabel(status: string): string {
  return RUN_STATUS_LABELS[status] ?? status
}

export function warningSeverityLabel(severity: string): string {
  return WARNING_SEVERITY_LABELS[severity] ?? severity
}

export function warningTypeLabel(type: string): string {
  const readable = type.replace(/_/g, ' ')
  return readable ? `${readable[0].toUpperCase()}${readable.slice(1)}` : readable
}

export function warningMessage(message: string): string {
  const skipped = /^Process '([^']+)' was skipped by human:[^:]+:\s*(.+)$/.exec(message)
  return skipped ? `${processLabel(skipped[1])} was skipped: ${skipped[2]}` : message
}

export function gateOverrideDecisionLabel(v: 'accept' | 'reject'): string {
  return v === 'accept' ? 'Accept' : 'Reject'
}

// Present-participle "what's happening right now" verb for the now-line (Z2)
// and the quiet-state current-activity suffix (Z3). stepDisplayLabel's single
// agent_definition_review override isn't reused here — every step gets its
// own activity phrase so the now-line reads as a sentence, not a status dump.
const STEP_ACTIVITY_VERB: Record<OrchestrationBaselineProcessType, string> = {
  goal_definition: 'Scoping the goal',
  manager_selection: 'Selecting a manager',
  agent_definition_review: 'Reviewing agent definitions',
  team_hierarchy: 'Structuring the team',
  effectiveness_review: 'Reviewing effectiveness',
  goal_closeout: 'Closing out the goal',
}

// One phrase for what a running step is doing — reused by the quiet-state
// "Orchestrator is working — {…}" suffix (spec: quiet-state detail).
export function stepActivityVerb(type: OrchestrationBaselineProcessType): string {
  return STEP_ACTIVITY_VERB[type] ?? stepDisplayLabel(type)
}

// A segment of the now-line — plain prose. (Kept as a segment list, not a
// single string, only because a couple of earlier design passes needed mixed
// styling; nothing currently needs more than one segment's worth of style, so
// this is just `{ text }`.)
export type NowLineSegment = { text: string }

// Pure composer for the narrative "now" line directly under the stepper
// (spec: narrative now line). Always returns at least one segment — the
// caller never has to guess whether there's something to show.
export function nowLine({
  goal,
  processes,
  checkpointItems,
  run,
}: {
  goal: Pick<OrchestrationGoal, 'status'>
  processes: readonly OrchestrationProcessRunRecord[]
  checkpointItems: readonly OrchestrationAuthorityDecisionRecord[]
  // Optional — callers/tests that don't need the baseline-gate line keep
  // working. Only affects the line while `run.phase === 'baseline'` and the
  // human hasn't authorized it yet (Task 12b).
  run?: Pick<OrchestrationRun, 'phase' | 'baseline_authorized'> | null
}): NowLineSegment[] {
  if (goal.status === 'cancelled') return [{ text: 'This goal was cancelled.' }]
  if (goal.status === 'completed') return [{ text: 'All steps complete — goal closed.' }]
  if (run && run.phase === 'baseline' && !run.baseline_authorized) {
    return processes.length === 0
      ? [{ text: 'Baseline is waiting — press Start baseline to begin.' }]
      : [{ text: 'Baseline is waiting — press Continue baseline to resume.' }]
  }
  if (processes.length === 0) return [{ text: 'Not started yet.' }]

  const byType = new Map(processes.map((process) => [process.process_type, process] as const))
  const current = processes.find((process) => process.status === 'waiting_decision')
    ?? processes.find((process) => process.status === 'running')

  if (!current) {
    const allTerminal = BASELINE_PROCESS_TYPES.every((type) => {
      const status = byType.get(type)?.status
      return status === 'completed' || status === 'skipped'
    })
    return allTerminal ? [{ text: 'All steps complete — sign-off pending.' }] : [{ text: 'Waiting to start the next step.' }]
  }

  const type = current.process_type as OrchestrationBaselineProcessType
  const index = BASELINE_PROCESS_TYPES.indexOf(type)
  const nextType = index >= 0 ? BASELINE_PROCESS_TYPES[index + 1] : undefined
  // "next" is only meaningful when that step genuinely hasn't started yet —
  // `current` is picked by priority (waiting-decision beats running), not by
  // BASELINE_PROCESS_TYPES order, so the sequence-adjacent step can already
  // be running/waiting/terminal on its own. Showing "next: X" while X is
  // already in progress elsewhere is a contradiction, not information.
  const next = nextType && !byType.has(nextType) ? nextType : undefined
  let text = current.status === 'waiting_decision'
    ? (() => {
      const count = checkpointItems.filter((item) => item.source_process_run_id === current.id).length
      return `${stepActivityVerb(type)} · ${count > 0 ? `${count} question${count === 1 ? '' : 's'} for you` : 'waiting on you'}`
    })()
    : stepActivityVerb(type)
  if (next) text += ` · next: ${stepDisplayLabel(next)}`
  return [{ text }]
}

// ============================================================================
// Error Classifier (Phase 1)
// ============================================================================

export type ErrorClass = 'auth' | 'rate_limit' | 'timeout' | 'not_found' | 'server' | 'unknown'

export interface ErrorRecordData {
  errorClass: ErrorClass
  what: string
  why: string
  doThis: string
  details: string
}

/**
 * Classifies an error (ApiError, TypeError, or string) and returns structured
 * humanized display data for ErrorRecord. Checks ApiError.status first, then
 * applies regex fallback to message text for raw string/Error inputs.
 * Details include HTTP status (if present) and error stack (if available).
 *
 * @param error - The error to classify (ApiError, TypeError, string, etc.)
 * @param options - Optional configuration
 * @param options.entity - Entity name for not_found phrasing (default: "item")
 */
export function errorRecord(error: unknown, options?: { entity?: string }): ErrorRecordData {
  const entity = options?.entity ?? 'item'
  const message = error instanceof Error ? error.message : String(error)
  const stack = error instanceof Error ? error.stack : undefined

  // Build details: HTTP status prefix (if available) + message + stack
  let details = ''
  if (error instanceof ApiError) {
    details = `HTTP ${error.status} — ${message}`
  } else {
    details = message
  }
  if (stack && message !== stack) {
    details += '\n' + stack
  }

  // Status-based classification (takes priority)
  if (error instanceof ApiError) {
    if (error.status === 401) {
      return {
        errorClass: 'auth',
        what: "The orchestrator's language-model key was rejected.",
        why: 'The provider returned an authentication error — the key is missing, invalid, or revoked.',
        doThis: 'Update the key in Settings.',
        details,
      }
    }
    if (error.status === 429) {
      return {
        errorClass: 'rate_limit',
        what: 'The language-model provider is rate-limiting requests.',
        why: 'Too many requests were sent in a short window.',
        doThis: 'Wait a moment, then retry.',
        details,
      }
    }
    if (error.status === 404) {
      const backPhrase = entity === 'item' ? 'the list' : `${entity}s`
      return {
        errorClass: 'not_found',
        what: `This ${entity} doesn't exist.`,
        why: 'It may have been deleted, or the link is stale.',
        doThis: `Go back to ${backPhrase}.`,
        details,
      }
    }
    if (error.status >= 500) {
      return {
        errorClass: 'server',
        what: 'The orchestrator could not answer.',
        why: 'The server returned an unexpected error.',
        doThis: 'Retry.',
        details,
      }
    }
    // Unknown ApiError (e.g., 409) — use message as what
    return {
      errorClass: 'unknown',
      what: message || 'Something went wrong.',
      why: 'No further detail is available.',
      doThis: 'Check Details, or try again.',
      details,
    }
  }

  // Regex-based classification for strings and Error objects (message-only path)
  const lowerMessage = message.toLowerCase()

  if (/unauthorized|invalid api key|authentication/i.test(message)) {
    return {
      errorClass: 'auth',
      what: "The orchestrator's language-model key was rejected.",
      why: 'The provider returned an authentication error — the key is missing, invalid, or revoked.',
      doThis: 'Update the key in Settings.',
      details,
    }
  }

  if (/rate.?limit|429/i.test(message)) {
    return {
      errorClass: 'rate_limit',
      what: 'The language-model provider is rate-limiting requests.',
      why: 'Too many requests were sent in a short window.',
      doThis: 'Wait a moment, then retry.',
      details,
    }
  }

  if (/timeout|timed out|Failed to fetch|ETIMEDOUT|ECONNRESET|network/i.test(message)) {
    return {
      errorClass: 'timeout',
      what: 'The request took too long and was dropped.',
      why: "The provider or network didn't respond in time.",
      doThis: 'Retry.',
      details,
    }
  }

  if (/\b404\b|not found/i.test(message)) {
    const backPhrase = entity === 'item' ? 'the list' : `${entity}s`
    return {
      errorClass: 'not_found',
      what: `This ${entity} doesn't exist.`,
      why: 'It may have been deleted, or the link is stale.',
      doThis: `Go back to ${backPhrase}.`,
      details,
    }
  }

  if (/\b5\d{2}\b|internal server error|bad gateway/i.test(message)) {
    return {
      errorClass: 'server',
      what: 'The orchestrator could not answer.',
      why: 'The server returned an unexpected error.',
      doThis: 'Retry.',
      details,
    }
  }

  // Unknown / unrecognized error
  return {
    errorClass: 'unknown',
    what: message || 'Something went wrong.',
    why: 'No further detail is available.',
    doThis: 'Check Details, or try again.',
    details,
  }
}
