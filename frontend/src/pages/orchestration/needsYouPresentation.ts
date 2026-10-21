import type {
  NeedsYouReference,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBlocker,
  OrchestrationGate,
  OrchestrationDecisionAnswerResult,
  OrchestrationAnswerSubmission,
  OrchestrationMeetingCommitment,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { warningMessage } from './humanize'

export type NeedsYouSourceKind = 'gate' | 'process' | 'action' | 'artifact'

export type NeedsYouSourceTarget = {
  [Kind in NeedsYouSourceKind]: { kind: Kind; id: string; goalId: string }
}[NeedsYouSourceKind]

export type QueueRow =
  | { kind: 'blocker'; key: string; blocker: OrchestrationBlocker }
  | { kind: 'lm-retry'; key: string; process: OrchestrationProcessRunRecord; message: string; warning?: OrchestrationWarningRecord }
  | { kind: 'answer'; key: string; decision: OrchestrationAuthorityDecisionRecord; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'answer-batch'; key: string; decisions: OrchestrationAuthorityDecisionRecord[]; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'goal-definition-questions'; key: string; decisions: OrchestrationAuthorityDecisionRecord[]; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'recovery-proposal'; key: string; decision: OrchestrationAuthorityDecisionRecord }
  | { kind: 'gate'; key: string; gate: OrchestrationGate; description: string | undefined }
  | { kind: 'meeting-commitment'; key: string; commitment: OrchestrationMeetingCommitment }
  | { kind: 'warning'; key: string; warning: OrchestrationWarningRecord; process: OrchestrationProcessRunRecord | undefined; canAcknowledge: boolean }

export interface QueueOccurrence {
  key: string
  occurredAt: string | null
}

export interface NeedsYouPresentation {
  title: string
  explanation: string
  nextStep: string
  actionOwner: 'owner' | 'maintainer' | 'unknown'
  details: string
  references: NeedsYouReference[]
}

export interface NeedsYouEditSnapshot {
  rowKey: string
  decisionIds: readonly string[]
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
}

function sameValue(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right)
}

function sameCanonicalValue(left: unknown, right: unknown): boolean {
  if (left === right) return true
  if (Array.isArray(left) || Array.isArray(right)) return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length && left.every((value, index) => sameCanonicalValue(value, right[index]))
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false
  const leftRecord = left as Record<string, unknown>
  const rightRecord = right as Record<string, unknown>
  const keys = Object.keys(leftRecord).sort()
  return sameValue(keys, Object.keys(rightRecord).sort())
    && keys.every((key) => sameCanonicalValue(leftRecord[key], rightRecord[key]))
}

export function validateNeedsYouEdit(
  snapshot: NeedsYouEditSnapshot,
  authoritative: readonly OrchestrationAuthorityDecisionRecord[] | undefined,
): 'current' | 'terminal' | 'stale' | 'unavailable' {
  if (authoritative === undefined) return 'unavailable'
  const currentById = new Map(authoritative.map((decision) => [decision.id, decision] as const))
  const pairs = snapshot.decisions.map((reviewed) => [reviewed, currentById.get(reviewed.id)] as const)
  if (pairs.some(([, current]) => current === undefined)) return 'unavailable'
  if (pairs.some(([, current]) => current!.status !== 'pending')) return 'terminal'
  for (const [reviewed, current] of pairs) {
    if (!current
      || current.goal_id !== reviewed.goal_id
      || current.run_id !== reviewed.run_id
      || current.decision_key !== reviewed.decision_key
      || (current.runtime_identity ?? null) !== (reviewed.runtime_identity ?? null)
      || (current.contract_version ?? null) !== (reviewed.contract_version ?? null)) return 'stale'
    const contextIsContract = reviewed.decision_key.startsWith('agent_definition_review:')
      || reviewed.decision_key.startsWith('team_hierarchy:')
    if (reviewed.contract_version == null && current.contract_version == null && (
      current.question !== reviewed.question
      || !sameValue(current.options, reviewed.options)
      || current.recommendation !== reviewed.recommendation
      || current.consequences !== reviewed.consequences
      || (contextIsContract && current.context !== reviewed.context)
      || !sameValue(current.recovery ?? null, reviewed.recovery ?? null)
    )) return 'stale'
  }
  return 'current'
}

const PYTHON_TRIM = /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g

export function validateRuntimeAnswerReceipt(
  reviewed: OrchestrationAuthorityDecisionRecord,
  submitted: { selectedOption: string; reason: string | null },
  result: OrchestrationDecisionAnswerResult,
): 'accepted' | 'stale' {
  const persisted = result.decision
  if (!persisted || typeof persisted !== 'object') return 'stale'
  const normalizedReason = submitted.reason === null ? null : submitted.reason.replace(PYTHON_TRIM, '')
  return reviewed.runtime_identity
    && persisted.id === reviewed.id
    && persisted.goal_id === reviewed.goal_id
    && persisted.run_id === reviewed.run_id
    && persisted.decision_key === reviewed.decision_key
    && (persisted.runtime_identity ?? null) === (reviewed.runtime_identity ?? null)
    && (persisted.contract_version ?? null) === (reviewed.contract_version ?? null)
    && persisted.status === 'answered'
    && persisted.selected_option === submitted.selectedOption
    && persisted.reason === normalizedReason
    ? 'accepted'
    : 'stale'
}

export function validateOrdinaryAnswerResult(
  reviewed: OrchestrationAuthorityDecisionRecord,
  submission: OrchestrationAnswerSubmission,
  result: OrchestrationDecisionAnswerResult,
): boolean {
  const persisted = result.decision
  const answer = submission.answers[submission.decision_ids.indexOf(reviewed.id)]
  if (!persisted || !answer || persisted.status !== 'answered' || !persisted.decided_at
    || persisted.id !== reviewed.id || persisted.goal_id !== reviewed.goal_id
    || persisted.run_id !== reviewed.run_id || persisted.source_process_run_id !== reviewed.source_process_run_id
    || persisted.decision_key !== reviewed.decision_key || persisted.authority !== 'human'
    || persisted.decided_by_user_id !== submission.actor_user_id || persisted.decided_by_agent_id != null
    || persisted.selected_option !== answer.selected_option
    || (persisted.contract_version ?? null) !== (reviewed.contract_version ?? null)
    || !sameValue(persisted.options, reviewed.options)) return false
  if (persisted.reason === answer.reason) return true
  if (!reviewed.decision_key.startsWith('team_hierarchy:agent:') || !persisted.reason) return false
  try {
    const metadata = JSON.parse(persisted.reason)
    return typeof metadata.created_agent_id === 'string'
      && (metadata.reason ?? null) === (answer.reason ?? null)
      && (answer.selected_option !== 'edit' || sameCanonicalValue(metadata.edited_agent, answer.edited_agent))
  } catch { return false }
}

const DEFAULT_TITLES: Record<QueueRow['kind'], string> = {
  answer: 'Owner guidance needed',
  'answer-batch': 'Agent definitions need review',
  'goal-definition-questions': 'Goal clarification needed',
  'recovery-proposal': 'Recovery proposal',
  gate: 'Verification failed',
  'lm-retry': 'Model request failed',
  'meeting-commitment': 'Meeting action needs attention',
  blocker: 'Work is blocked',
  warning: 'Warning needs attention',
}

const REFERENCE_UNAVAILABLE: Record<NeedsYouReference['kind'], string> = {
  task: 'Task unavailable', meeting: 'Meeting unavailable', agent: 'Agent unavailable',
  goal: 'Goal unavailable', gate: 'Gate unavailable', process: 'Process unavailable',
  action: 'Action unavailable', artifact: 'Plan candidate unavailable',
}

function validInstant(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const trimmed = value.trim()
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|[+-]\d{2}:\d{2})?$/.exec(trimmed)
  if (!match) return null
  const [, yearText, monthText, dayText, hourText, minuteText, secondText = '00', , zone] = match
  const [year, month, day, hour, minute, second] = [yearText, monthText, dayText, hourText, minuteText, secondText].map(Number)
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
  const daysInMonth = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  if (year < 1 || month < 1 || month > 12 || day < 1 || day > daysInMonth[month - 1]
    || hour > 23 || minute > 59 || second > 59) return null
  if (zone && zone !== 'Z') {
    const [offsetHour, offsetMinute] = zone.slice(1).split(':').map(Number)
    if (offsetHour > 23 || offsetMinute > 59) return null
  }
  const source = zone ? trimmed : `${trimmed}Z`
  const milliseconds = Date.parse(source)
  return Number.isFinite(milliseconds) ? new Date(milliseconds).toISOString() : null
}

function codepointCompare(a: string, b: string): number {
  const left = Array.from(a)
  const right = Array.from(b)
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const difference = left[index].codePointAt(0)! - right[index].codePointAt(0)!
    if (difference !== 0) return difference
  }
  return left.length - right.length
}

function decisionInstant(decision: OrchestrationAuthorityDecisionRecord): string | null {
  return validInstant(decision.asked_at) ?? validInstant(decision.created_at)
}

function groupHistory(
  row: Extract<QueueRow, { kind: 'answer-batch' | 'goal-definition-questions' }>,
  allDecisions: readonly OrchestrationAuthorityDecisionRecord[],
) {
  const processId = row.process?.id ?? row.decisions[0]?.source_process_run_id
  if (!processId) return row.decisions
  if (row.kind === 'answer-batch') {
    const matches = allDecisions.filter((decision) => decision.source_process_run_id === processId
      && decision.decision_key.startsWith('agent_definition_review:proposal:'))
    return matches.length > 0 ? matches : row.decisions
  }
  const round = /^(?:goal_definition:adaptive:)([^:]+):/.exec(row.decisions[0]?.decision_key ?? '')?.[1]
  if (!round) return row.decisions
  const prefix = `goal_definition:adaptive:${round}:`
  const matches = allDecisions.filter((decision) => decision.source_process_run_id === processId
    && decision.decision_key.startsWith(prefix))
  return matches.length > 0 ? matches : row.decisions
}

function blockerIdentity(blocker: OrchestrationBlocker): string {
  const typed = [
    typeof blocker.task_id === 'string' && blocker.task_id ? `task_id:${blocker.task_id}` : null,
    typeof blocker.gate_id === 'string' && blocker.gate_id ? `gate_id:${blocker.gate_id}` : null,
  ].filter((value): value is string => value !== null)
  if (typed.length > 0) return typed.join('|')
  const source = JSON.stringify({ kind: blocker.kind ?? null, reason: blocker.reason ?? null })
  let hash = 2166136261
  for (const character of Array.from(source)) {
    hash ^= character.codePointAt(0)!
    hash = Math.imul(hash, 16777619)
  }
  return `content:${(hash >>> 0).toString(16).padStart(8, '0')}`
}

export function queueOccurrence(
  row: QueueRow,
  allDecisions: readonly OrchestrationAuthorityDecisionRecord[],
): QueueOccurrence {
  if (row.kind === 'answer' || row.kind === 'recovery-proposal') {
    return { key: `${row.kind}:${row.decision.id}`, occurredAt: decisionInstant(row.decision) }
  }
  if (row.kind === 'warning') return { key: `warning:${row.warning.id}`, occurredAt: validInstant(row.warning.created_at) }
  if (row.kind === 'gate') return { key: `gate:${row.gate.id}`, occurredAt: validInstant(row.gate.failed_at) ?? validInstant(row.gate.created_at) }
  if (row.kind === 'meeting-commitment') {
    return { key: `meeting-commitment:${row.commitment.id}`, occurredAt: validInstant(row.commitment.created_at) }
  }
  if (row.kind === 'lm-retry') {
    if (row.warning) return { key: `lm-retry:${row.process.id}:warning:${row.warning.id}`, occurredAt: validInstant(row.warning.created_at) }
    const retry = row.process.outputs.lm_retry
    const identity = retry?.kind || retry?.model || 'checkpoint'
    return { key: `lm-retry:${row.process.id}:${identity}`, occurredAt: validInstant(row.process.created_at) }
  }
  if (row.kind === 'blocker') {
    return {
      key: `blocker:${row.blocker.kind ?? 'unknown'}:${blockerIdentity(row.blocker)}`,
      occurredAt: null,
    }
  }
  const history = groupHistory(row, allDecisions)
  const occurredAt = history.map(decisionInstant).filter((value): value is string => value !== null)
    .sort((a, b) => Date.parse(b) - Date.parse(a))[0] ?? null
  const processId = row.process?.id ?? row.decisions[0]?.source_process_run_id ?? 'unknown'
  if (row.kind === 'goal-definition-questions') {
    const round = /^goal_definition:adaptive:([^:]+):/.exec(row.decisions[0]?.decision_key ?? '')?.[1]
    return {
      key: round ? `goal-definition-questions:${processId}:round:${round}` : `goal-definition-questions:${processId}:${history.map((decision) => decision.id).sort(codepointCompare).join(',')}`,
      occurredAt,
    }
  }
  return {
    key: `answer-batch:${processId}:${history.map((decision) => decision.id).sort(codepointCompare).join(',')}`,
    occurredAt,
  }
}

export function compareQueueOccurrences(a: QueueOccurrence, b: QueueOccurrence): number {
  if (a.occurredAt !== null && b.occurredAt !== null) {
    const difference = Date.parse(b.occurredAt) - Date.parse(a.occurredAt)
    if (difference !== 0) return difference
    return codepointCompare(a.key, b.key)
  }
  if (a.occurredAt !== null) return -1
  if (b.occurredAt !== null) return 1
  return codepointCompare(a.key, b.key)
}

export function conciseNeedsYouTitle(candidate: string | null, fallback: string): string {
  const title = candidate?.trim() || fallback
  const codepoints = Array.from(title)
  if (codepoints.length <= 100) return title
  const fitting = codepoints.slice(0, 99).join('')
  if (/\s/u.test(codepoints[99] ?? '')) return `${fitting.trimEnd()}…`
  const boundary = fitting.search(/\s+\S*$/u)
  return `${(boundary > 0 ? fitting.slice(0, boundary) : fitting).trimEnd()}…`
}

function sourceReferences(row: QueueRow): readonly NeedsYouReference[] {
  if (row.kind === 'answer' || row.kind === 'recovery-proposal') return validReferences(row.decision.needs_you_references)
  if (row.kind === 'answer-batch' || row.kind === 'goal-definition-questions') return row.decisions.flatMap((decision) => validReferences(decision.needs_you_references))
  if (row.kind === 'warning' || row.kind === 'lm-retry') return validReferences(row.warning?.needs_you_references)
  return []
}

function validReferences(value: unknown): NeedsYouReference[] {
  if (!Array.isArray(value)) return []
  const kinds = new Set<NeedsYouReference['kind']>(['task', 'meeting', 'agent', 'goal', 'gate', 'process', 'action', 'artifact'])
  return value.filter((reference): reference is NeedsYouReference => {
    if (reference === null || typeof reference !== 'object') return false
    const candidate = reference as Record<string, unknown>
    return kinds.has(candidate.kind as NeedsYouReference['kind'])
      && typeof candidate.id === 'string' && candidate.id.length > 0
      && typeof candidate.project_id === 'string' && candidate.project_id.length > 0
      && (candidate.goal_id === null || typeof candidate.goal_id === 'string')
      && (candidate.label === null || typeof candidate.label === 'string')
  })
}

function uniqueReferences(row: QueueRow, references: readonly NeedsYouReference[]) {
  const unique = new Map<string, NeedsYouReference>()
  for (const reference of [...sourceReferences(row), ...validReferences(references)]) unique.set(`${reference.kind}:${reference.id}`, reference)
  return [...unique.values()]
}

function substituteIdentifiers(value: string, references: readonly NeedsYouReference[]): string {
  let result = value
  for (const reference of references) {
    const replacement = reference.label ?? REFERENCE_UNAVAILABLE[reference.kind]
    const escaped = reference.id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    result = result.replace(new RegExp(`(?<![A-Za-z0-9])${escaped}(?![A-Za-z0-9])`, 'g'), replacement)
  }
  return result
    .replace(/[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/gi, 'Reference unavailable')
    .replace(/\b(?:run|task|goal|gate|process|action|decision):[A-Za-z0-9._:-]{8,}\b/g, 'Reference unavailable')
}

export function safeNeedsYouText(value: string, references: readonly NeedsYouReference[]): string {
  return substituteIdentifiers(value, validReferences(references))
}

function technicalTitle(title: string | null, body: string): string | null {
  const trimmed = title?.trim() ?? ''
  if (!trimmed || trimmed === body.trim() || /^Runtime decision:/i.test(trimmed)) return null
  if (/^(?:run|task|goal|gate|process|action|decision):/i.test(trimmed)) return null
  return trimmed
}

function rowDetails(row: QueueRow): string {
  const decision = (source: OrchestrationAuthorityDecisionRecord) => ({
    id: source.id, goal_id: source.goal_id, run_id: source.run_id, decision_key: source.decision_key,
    runtime_identity: source.runtime_identity, contract_version: source.contract_version, title: source.title,
    status: source.status, authority: source.authority, authority_agent_id: source.authority_agent_id,
    source_process_run_id: source.source_process_run_id, question: source.question, context: source.context,
    options: source.options, recommendation: source.recommendation, consequences: source.consequences,
    related_gate_id: source.related_gate_id, related_action_id: source.related_action_id,
    asked_at: source.asked_at, created_at: source.created_at, recovery: source.recovery,
    needs_you_references: validReferences(source.needs_you_references),
  })
  const warning = (source: OrchestrationWarningRecord) => ({
    id: source.id, goal_id: source.goal_id, run_id: source.run_id, warning_type: source.warning_type,
    severity: source.severity, message: source.message, source_process_run_id: source.source_process_run_id,
    related_gate_id: source.related_gate_id, related_action_id: source.related_action_id,
    related_agent_id: source.related_agent_id, blocks_completion: source.blocks_completion,
    created_at: source.created_at, needs_you_references: validReferences(source.needs_you_references),
  })
  if (row.kind === 'answer' || row.kind === 'recovery-proposal') return JSON.stringify(decision(row.decision), null, 2)
  if (row.kind === 'answer-batch' || row.kind === 'goal-definition-questions') return JSON.stringify(row.decisions.map(decision), null, 2)
  if (row.kind === 'warning') return JSON.stringify(warning(row.warning), null, 2)
  if (row.kind === 'lm-retry') return JSON.stringify({
    process: { id: row.process.id, goal_id: row.process.goal_id, run_id: row.process.run_id, process_type: row.process.process_type, status: row.process.status, trigger_reason: row.process.trigger_reason, lm_retry: row.process.outputs.lm_retry, created_at: row.process.created_at },
    warning: row.warning ? warning(row.warning) : null,
  }, null, 2)
  if (row.kind === 'gate') return JSON.stringify({ id: row.gate.id, run_id: row.gate.run_id, success_criterion_key: row.gate.success_criterion_key, gate_type: row.gate.gate_type, status: row.gate.status, failure_reason: row.gate.failure_reason, created_at: row.gate.created_at, failed_at: row.gate.failed_at }, null, 2)
  if (row.kind === 'meeting-commitment') return JSON.stringify({ id: row.commitment.id, meeting_id: row.commitment.meeting_id, meeting_title: row.commitment.meeting_title, created_at: row.commitment.created_at, description: row.commitment.description, state: row.commitment.state, task_id: row.commitment.task_id, graph_run_id: row.commitment.graph_run_id, reason: row.commitment.reason }, null, 2)
  return JSON.stringify({ kind: row.blocker.kind, reason: row.blocker.reason, task_id: row.blocker.task_id, gate_id: row.blocker.gate_id }, null, 2)
}

function distinctCopy(...values: Array<string | null | undefined>): string {
  const seen = new Set<string>()
  return values.flatMap((value) => {
    const text = value?.trim()
    if (!text || seen.has(text)) return []
    seen.add(text)
    return [text]
  }).join('\n')
}

function plainContext(context: string | null): string | null {
  const value = context?.trim()
  if (!value) return null
  if (value.startsWith('{') || value.startsWith('[')) {
    try { JSON.parse(value); return null } catch { /* keep non-JSON prose */ }
  }
  return value
}

export function presentNeedsYouRow(row: QueueRow, references: readonly NeedsYouReference[]): NeedsYouPresentation {
  const verifiedReferences = uniqueReferences(row, references)
  let candidate: string | null = null
  let explanation = ''
  let nextStep = 'Review Details with a maintainer.'
  let actionOwner: NeedsYouPresentation['actionOwner'] = 'unknown'

  if (row.kind === 'answer') {
    candidate = technicalTitle(row.decision.title, row.decision.question)
    explanation = distinctCopy(row.decision.question, plainContext(row.decision.context), row.decision.consequences)
    nextStep = row.decision.recommendation === null ? 'Choose an answer.' : 'Accept the recommendation or choose another answer.'
    actionOwner = 'owner'
  } else if (row.kind === 'answer-batch') {
    explanation = `${row.decisions.length} agent definitions are ready for review.`
    nextStep = 'Review and submit each agent definition.'
    actionOwner = 'owner'
  } else if (row.kind === 'goal-definition-questions') {
    explanation = `${row.decisions.length} goal clarification question${row.decisions.length === 1 ? '' : 's'} need an answer.`
    nextStep = 'Answer each clarification question.'
    actionOwner = 'owner'
  } else if (row.kind === 'recovery-proposal') {
    explanation = row.decision.recovery?.problem ?? row.decision.question
    nextStep = row.decision.recovery?.status === 'unsupported' ? 'Provide the missing capability, then confirm it is available.' : 'Review the proposed recovery and approve or reject it.'
    actionOwner = 'owner'
  } else if (row.kind === 'gate') {
    candidate = technicalTitle(row.description ?? null, row.gate.failure_reason ?? '')
    explanation = row.gate.failure_reason ?? row.description ?? 'The required verification did not pass.'
    nextStep = 'Review the failed criterion and its source details.'
  } else if (row.kind === 'lm-retry') {
    explanation = 'The last model request did not complete.'
    nextStep = 'Retry the model request.'
    actionOwner = 'owner'
  } else if (row.kind === 'meeting-commitment') {
    candidate = technicalTitle(row.commitment.description, row.commitment.reason ?? '')
    explanation = row.commitment.reason ?? row.commitment.description
    nextStep = 'Review the meeting action and update its status.'
    actionOwner = 'owner'
  } else if (row.kind === 'blocker') {
    const hasTypedSource = typeof row.blocker.task_id === 'string' || typeof row.blocker.gate_id === 'string'
    explanation = row.blocker.kind === 'goal_definition_clarification_limit'
      ? 'Goal clarification reached its current limit.'
      : row.blocker.kind === 'evidence_gap' || hasTypedSource
        ? row.blocker.reason ?? 'Required evidence is unavailable.'
      : 'The cause could not be determined from the available structured data.'
    nextStep = row.blocker.kind === 'goal_definition_clarification_limit'
      ? 'Choose how to continue goal clarification.'
      : 'Review Details with a maintainer.'
    actionOwner = row.blocker.kind === 'goal_definition_clarification_limit' ? 'owner' : 'unknown'
  } else if (row.warning.warning_type === 'closeout_blocked') {
    candidate = 'Completion is blocked'
    explanation = distinctCopy(
      warningMessage(row.warning.message),
      'The listed causes need action. Resolving this notice alone does not fix them; it returns until they are resolved.',
    )
    nextStep = 'Address each listed cause. Closeout resumes automatically.'
    actionOwner = 'owner'
  } else {
    explanation = warningMessage(row.warning.message)
    nextStep = row.canAcknowledge ? 'Acknowledge this warning after reviewing it.' : 'Review and resolve this warning.'
    actionOwner = row.canAcknowledge ? 'owner' : 'unknown'
  }

  const safeExplanation = substituteIdentifiers(explanation, verifiedReferences)
  return {
    title: conciseNeedsYouTitle(candidate ? substituteIdentifiers(candidate, verifiedReferences) : null, DEFAULT_TITLES[row.kind]),
    explanation: safeExplanation,
    nextStep,
    actionOwner,
    details: rowDetails(row),
    references: verifiedReferences,
  }
}

export function needsYouReferencePath(
  reference: NeedsYouReference,
  activeProjectId: string,
): string | null {
  if (reference.label === null || reference.project_id !== activeProjectId) return null

  const id = encodeURIComponent(reference.id)
  if (reference.kind === 'task') return `/tasks?task=${id}`
  if (reference.kind === 'meeting') return `/meetings/${id}`
  if (reference.kind === 'agent') return `/agents/${id}`
  if (reference.kind === 'goal') return `/orchestration/${id}`
  if (reference.kind === 'artifact' && reference.goal_id) return `/orchestration/${encodeURIComponent(reference.goal_id)}#plan-review-${id}`
  return null
}
