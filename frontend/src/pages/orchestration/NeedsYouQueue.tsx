import { useEffect, useLayoutEffect, useId, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { AlertTriangle, Ban, CheckCircle2, HelpCircle, RotateCcw, ShieldAlert } from 'lucide-react'
import { Link } from 'react-router-dom'
import { toast } from 'sonner'
import { useUpdateActionItem } from '@/api/meetings'
import { Button, Textarea } from '@/components/common/uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'
import { parseUtc } from '@/lib/time'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationBlocker,
  OrchestrationGate,
  OrchestrationGoal,
  OrchestrationMeetingCommitment,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
  OrchestrationWarningRecord,
  NeedsYouReference,
  RecoveryProposalStatus,
} from '@/lib/types'
import { BASELINE_PROCESS_TYPES, stepActivityVerb, stepDisplayLabel, warningMessage } from './humanize'
import { RetryControl } from './ProcessFocus'
import { decisionOptionKey, isManagerSelectionDecision, normalizeDecisionOption, parseManagerComparison, type BaselineFocusAction } from './BaselineDashboard'
import { ZONE_TITLE_CLASS } from './zoneTitle'
import { useGoalAnnouncer } from './goalAnnouncer'
import {
  compareQueueOccurrences,
  needsYouReferencePath,
  presentNeedsYouRow,
  queueOccurrence,
  safeNeedsYouText,
  validateNeedsYouEdit,
  type NeedsYouEditSnapshot,
  type NeedsYouSourceTarget,
  type QueueRow,
} from './needsYouPresentation'

// Debounce window for the net-queue-change announcer (spec: motion item 5) —
// coalesces a burst of individual row resolutions/arrivals (e.g. a batch
// answer clearing five rows at once) into one screen-reader announcement
// instead of five.
const QUEUE_ANNOUNCE_DEBOUNCE_MS = 500

function isBaselineProcessType(type: string): type is OrchestrationBaselineProcessType {
  return (BASELINE_PROCESS_TYPES as readonly string[]).includes(type)
}

// Stale-inputs suggestion warnings carry `warning_type` =
// `${process_type}_stale_inputs` (orchestration_warning_service.py
// suggest_stale_inputs) — derive the process type back out, or null if this
// isn't one of the four gated baseline steps.
const STALE_INPUTS_SUFFIX = '_stale_inputs'
function staleInputsProcessType(warningType: string): OrchestrationBaselineProcessType | null {
  if (!warningType.endsWith(STALE_INPUTS_SUFFIX)) return null
  const processType = warningType.slice(0, -STALE_INPUTS_SUFFIX.length)
  return isBaselineProcessType(processType) ? processType : null
}

function agentDefinitionReviewBatchDecisions(
  processId: string,
  checkpointItems: readonly OrchestrationAuthorityDecisionRecord[],
  allDecisions: readonly OrchestrationAuthorityDecisionRecord[],
  processesById: Map<string, OrchestrationProcessRunRecord>,
) {
  if (processesById.get(processId)?.process_type !== 'agent_definition_review') return null
  const forProcess = checkpointItems.filter((item) => item.source_process_run_id === processId)
  if (forProcess.length === 0) return null
  if (!forProcess.every((item) => item.decision_key.startsWith('agent_definition_review:proposal:'))) return null
  const pendingIds = new Set(forProcess.map((item) => item.id))
  const ordered = allDecisions.filter((item) => pendingIds.has(item.id))
  return [...ordered, ...forProcess.filter((item) => !ordered.some((candidate) => candidate.id === item.id))]
}

// Goal-definition's adaptive clarification questions (backend key format
// "goal_definition:adaptive:{round}:{index}:{destination}", see
// orchestration_goal_definition.py DECISION_KEY_PREFIX) are always asked and
// answered as one round at a time — the process parks as soon as any of them
// is pending, so a process never has two rounds pending simultaneously.
// Collapsed to a single step-through queue card (spec D) instead of one
// "answer" row per question.
function goalDefinitionQuestionDecisions(
  processId: string,
  checkpointItems: readonly OrchestrationAuthorityDecisionRecord[],
  allDecisions: readonly OrchestrationAuthorityDecisionRecord[],
  processesById: Map<string, OrchestrationProcessRunRecord>,
) {
  if (processesById.get(processId)?.process_type !== 'goal_definition') return null
  const forProcess = checkpointItems.filter((item) => item.source_process_run_id === processId && item.decision_key.startsWith('goal_definition:adaptive:'))
  if (forProcess.length === 0) return null
  const round = /^goal_definition:adaptive:([^:]+):/.exec(forProcess[0].decision_key)?.[1]
  const pendingIds = new Set(forProcess.map((item) => item.id))
  const ordered = round
    ? allDecisions.filter((item) => item.source_process_run_id === processId
      && item.decision_key.startsWith(`goal_definition:adaptive:${round}:`)
      && pendingIds.has(item.id))
    : []
  return [...ordered, ...forProcess.filter((item) => !ordered.some((candidate) => candidate.id === item.id))]
}

// Newest recovery decision per source blocker, minus resolved ones.
// Rejected/failed/unsupported stay readable until a newer proposal supersedes them.
function visibleRecoveryDecisions(all: readonly OrchestrationAuthorityDecisionRecord[]) {
  const stamp = (decision: OrchestrationAuthorityDecisionRecord) => Date.parse(decision.created_at || decision.asked_at) || 0
  const newer = (a: OrchestrationAuthorityDecisionRecord, b: OrchestrationAuthorityDecisionRecord) =>
    stamp(a) !== stamp(b) ? stamp(a) > stamp(b) : a.id > b.id
  const newest = new Map<string, OrchestrationAuthorityDecisionRecord>()
  for (const decision of all) {
    if (!decision.recovery) continue
    const key = decision.recovery.source_blocker_id ?? decision.id
    const current = newest.get(key)
    if (!current || newer(decision, current)) newest.set(key, decision)
  }
  return new Map([...newest.values()].filter((d) => d.recovery!.status !== 'resolved').map((d) => [d.id, d] as const))
}

// Ranked, in the order the brief specifies: active blockers > LM retries >
// checkpoint decisions (server order preserved, with same-process
// agent-definition-review batches collapsed to one row) > failed gates > warnings.
export function buildQueueRows({
  run,
  processes,
  checkpointItems,
  allDecisions = [],
  recoveryDecisions = [],
  gates,
  warnings,
  criteriaByKey,
  meetingCommitments = [],
}: {
  run: OrchestrationRun | null
  processes: readonly OrchestrationProcessRunRecord[]
  checkpointItems: readonly OrchestrationAuthorityDecisionRecord[]
  allDecisions?: readonly OrchestrationAuthorityDecisionRecord[]
  recoveryDecisions?: readonly OrchestrationAuthorityDecisionRecord[]
  gates: readonly OrchestrationGate[]
  warnings: readonly OrchestrationWarningRecord[]
  meetingCommitments?: readonly OrchestrationMeetingCommitment[]
  criteriaByKey: Map<string, string | undefined>
}): QueueRow[] {
  const decisionHistory = allDecisions.length > 0 ? allDecisions : checkpointItems
  const processesById = new Map(processes.map((process) => [process.id, process] as const))
  const rows: QueueRow[] = []

  ;(run?.active_blockers ?? []).forEach((blocker) => {
    rows.push({ kind: 'blocker', key: '', blocker })
  })

  const lmRetryProcesses = processes.filter((process) => process.status === 'running' && process.outputs.lm_retry?.available === true)
  const lmRetryWarningIds = new Set(lmRetryProcesses.flatMap((process) => process.outputs.lm_retry?.warning_id ? [process.outputs.lm_retry.warning_id] : []))
  for (const process of lmRetryProcesses) {
    const linkedWarning = process.outputs.lm_retry?.warning_id
      ? warnings.find((warning) => warning.id === process.outputs.lm_retry?.warning_id)
      : undefined
    rows.push({
      kind: 'lm-retry',
      // updated_at folds in the failure instance (a process record is reused
      // across retries — process.id alone can't tell a resolved failure from a
      // fresh one), so justResolvedIds doesn't permanently hide a later failure
      // on the same process.
      key: '',
      process,
      warning: linkedWarning,
      message: linkedWarning ? warningMessage(linkedWarning.message) : 'The last language model request failed and can be retried.',
    })
  }

  const handledDecisionIds = new Set<string>()
  const visibleRecovery = visibleRecoveryDecisions([...recoveryDecisions, ...checkpointItems.filter((item) => !recoveryDecisions.some((d) => d.id === item.id))])
  for (const item of checkpointItems) {
    if (handledDecisionIds.has(item.id)) continue
    if (item.recovery) {
      handledDecisionIds.add(item.id)
      const visible = visibleRecovery.get(item.id)
      if (visible) rows.push({ kind: 'recovery-proposal', key: `recovery-proposal:${item.id}`, decision: visible })
      continue
    }
    const processId = item.source_process_run_id
    const questions = processId ? goalDefinitionQuestionDecisions(processId, checkpointItems, decisionHistory, processesById) : null
    if (questions) {
      for (const decision of questions) handledDecisionIds.add(decision.id)
      // Sorted decision ids scope the key to this exact set of pending
      // questions, so a later round (a disjoint id set once every question
      // in this round is answered) mounts a fresh card instead of reusing
      // stale local step-through state.
      rows.push({ kind: 'goal-definition-questions', key: '', decisions: questions, process: processesById.get(processId!) })
      continue
    }
    const batch = processId ? agentDefinitionReviewBatchDecisions(processId, checkpointItems, decisionHistory, processesById) : null
    if (batch) {
      for (const decision of batch) handledDecisionIds.add(decision.id)
      // Decision ids (not just processId) scope the key to this specific batch
      // instance, so a fresh set of proposals for the same process isn't hidden
      // by a stale justResolvedIds entry from an earlier, already-submitted batch.
      rows.push({ kind: 'answer-batch', key: '', decisions: batch, process: processesById.get(processId!) })
      continue
    }
    handledDecisionIds.add(item.id)
    rows.push({ kind: 'answer', key: `answer:${item.id}`, decision: item, process: item.source_process_run_id ? processesById.get(item.source_process_run_id) : undefined })
  }

  for (const decision of visibleRecovery.values()) {
    if (!handledDecisionIds.has(decision.id)) rows.push({ kind: 'recovery-proposal', key: `recovery-proposal:${decision.id}`, decision })
  }

  for (const gate of gates.filter((candidate) => candidate.status === 'failed')) {
    rows.push({ kind: 'gate', key: `gate:${gate.id}`, gate, description: criteriaByKey.get(gate.success_criterion_key) })
  }

  for (const commitment of meetingCommitments) {
    rows.push({ kind: 'meeting-commitment', key: `meeting-commitment:${commitment.id}`, commitment })
  }

  for (const warning of warnings.filter((candidate) => candidate.active && !lmRetryWarningIds.has(candidate.id))) {
    rows.push({
      kind: 'warning',
      key: `warning:${warning.id}`,
      warning,
      process: warning.source_process_run_id ? processesById.get(warning.source_process_run_id) : undefined,
      canAcknowledge: warning.severity === 'recommendation' || warning.severity === 'warning',
    })
  }

  for (const row of rows) row.key = queueOccurrence(row, decisionHistory).key
  return rows.sort((left, right) => compareQueueOccurrences(
    queueOccurrence(left, decisionHistory),
    queueOccurrence(right, decisionHistory),
  ))
}

function lowerFirst(text: string) {
  return text ? `${text[0].toLowerCase()}${text.slice(1)}` : text
}

// Quiet-state detail (ux finding): when nothing needs the operator but a step
// is actively running, name it instead of leaving a bare "working…".
//
// An empty `rows` array only means "on track" if the data that built it
// actually loaded — checkpoint/warnings queries default to `[]` while
// loading (same trap as the earned-toast arrival detector, review round 1
// finding 1), and a goal with no run at all can't be "on track" either. Both
// must be ruled out before the green claim, mirroring the deleted page-level
// banner's `run && !decisions.isLoading && !warnings.isLoading` gate (review
// round 2, finding I2).
function emptyStateCopy(
  goal: OrchestrationGoal,
  isWorking: boolean,
  runningStep: OrchestrationBaselineProcessType | null,
  run: OrchestrationRun | null,
  queriesErrored: boolean,
  queriesUnready: boolean,
) {
  if (goal.status === 'completed') return { tone: 'terminal' as const, text: 'This goal is complete.' }
  if (goal.status === 'cancelled') return { tone: 'terminal' as const, text: 'This goal was cancelled.' }
  if (queriesErrored) return { tone: 'error' as const, text: 'Some goal data failed to load — this queue may be incomplete.' }
  if (run === null) return { tone: 'loading' as const, text: 'No orchestration run is attached to this goal.' }
  if (queriesUnready) return { tone: 'loading' as const, text: 'Loading…' }
  if (isWorking) {
    return {
      tone: 'working' as const,
      text: runningStep ? `Orchestrator is working — ${lowerFirst(stepActivityVerb(runningStep))}…` : 'Orchestrator is working…',
    }
  }
  return { tone: 'on-track' as const, text: 'No blockers, failed gates, pending decisions, meeting obligations, or active warnings — this goal is on track.' }
}

export interface NeedsYouQueueProps {
  goal: OrchestrationGoal
  run: OrchestrationRun | null
  processes: readonly OrchestrationProcessRunRecord[]
  sourceProcesses?: readonly OrchestrationProcessRunRecord[]
  warnings: readonly OrchestrationWarningRecord[]
  gates: readonly OrchestrationGate[]
  meetingCommitments?: readonly OrchestrationMeetingCommitment[]
  checkpointItems: readonly OrchestrationAuthorityDecisionRecord[]
  allDecisions?: readonly OrchestrationAuthorityDecisionRecord[]
  authorityReady?: boolean
  invalidatedDecisions?: Readonly<Record<string, 'terminal' | 'stale' | 'unavailable'>>
  needsYouReferences?: readonly NeedsYouReference[]
  sourceActionIds?: readonly string[]
  // Every decision of the goal (any status); the source of non-pending
  // recovery proposals shown as FIX PROPOSAL cards.
  recoveryDecisions?: readonly OrchestrationAuthorityDecisionRecord[]
  onAnswerRecoveryProposal?: (
    decision: OrchestrationAuthorityDecisionRecord,
    option: RecoveryAnswerOption,
    reason: string,
    callbacks: { onSuccess: () => void; onError: (error: unknown) => void; onSettled: () => void },
  ) => void
  // True once the checkpoint query has resolved successfully at least once.
  // Arms the earned-toast arrival detector below — `checkpointItems` defaults
  // to `[]` before the first load, which is indistinguishable from "loaded,
  // genuinely empty" by length alone (review round 1, finding 1).
  checkpointLoaded: boolean
  // True if the checkpoint, warnings, or decisions query errored — gates the
  // on-track claim (review round 2, finding I2). Also gates the earned-toast
  // arrival detector's "loaded" precondition via checkpointLoaded above.
  queriesErrored: boolean
  // True while any of those three queries hasn't resolved yet (first load).
  queriesUnready: boolean
  deferredCount: number
  isWorking: boolean
  mutationBusy: boolean
  onSelectProcess: (type: OrchestrationBaselineProcessType) => void
  onRevealSource?: (target: NeedsYouSourceTarget) => void
  openAction: BaselineFocusAction | null
  onOpenAction: (action: BaselineFocusAction) => void
  onActionTrigger: (action: BaselineFocusAction, element: HTMLButtonElement | null) => void
  renderActionForm: (action: BaselineFocusAction, record: OrchestrationAuthorityDecisionRecord | OrchestrationWarningRecord, onSuccessFocus?: () => void, displayQuestion?: string, displayReferences?: readonly NeedsYouReference[], editState?: 'current' | 'terminal' | 'stale' | 'unavailable', onReleaseEdit?: () => void, formKey?: string) => ReactNode
  renderAgentDefinitionReviewBatch: (decisions: readonly OrchestrationAuthorityDecisionRecord[], onSuccessFocus: (() => void) | undefined, batchKey: string, displayReferences: readonly NeedsYouReference[], editState?: 'current' | 'terminal' | 'stale' | 'unavailable') => ReactNode
  renderTeamHierarchyProposal: (decision: OrchestrationAuthorityDecisionRecord, onSuccess: () => void, registerPrimary: (element: HTMLButtonElement | null) => void, displayReferences: readonly NeedsYouReference[], editState?: 'current' | 'terminal' | 'stale' | 'unavailable', formKey?: string, onReleaseEdit?: () => boolean, onBeginEdit?: () => void) => ReactNode
  // Accept-recommendation fast path (spec C): submits decision.recommendation
  // immediately with no form and no reason. Only offered when the decision
  // carries a non-null recommendation; "Choose differently…" still opens
  // renderActionForm's full radio+reason form.
  onAcceptDecision: (decision: OrchestrationAuthorityDecisionRecord, onSuccessFocus?: () => void) => void
  // Goal-definition step-through card (spec D): submits one adaptive
  // clarification question's free-text answer at a time. `callbacks.onSuccess`
  // advances this card's local question index (or, on the last question,
  // is provided by the card's caller to resolve the whole queue row).
  onAnswerGoalDefinitionQuestion: (
    decision: OrchestrationAuthorityDecisionRecord,
    answerText: string,
    callbacks: { onSuccess: () => void; onError: (error: unknown) => void; onSettled: () => void },
  ) => void
  retryPending: boolean
  retryError: string | null
  onRetryProcess: (processType: OrchestrationBaselineProcessType, trigger: HTMLButtonElement | null, onSuccessFocus?: () => void) => void
  // For stale-inputs suggestions, Approve reruns the derived step
  // then resolves the warning; Dismiss just resolves it. Both reuse the
  // existing rerun/resolve mutations (see BaselineDashboard).
  onApproveStaleInputs: (warning: OrchestrationWarningRecord, processType: OrchestrationBaselineProcessType, onSuccessFocus?: () => void) => void
  onAcknowledgeWarning: (warning: OrchestrationWarningRecord, onSuccessFocus?: () => void) => void
  onDismissStaleInputs: (warning: OrchestrationWarningRecord, onSuccessFocus?: () => void) => void
  // Renders the goal_definition_clarification_limit recovery action
  // (Proceed / Ask another round) as this blocker row's primary action.
  // Owned by the page (see BaselineDashboardProps); absent for other blocker
  // kinds, which stay reference-only (no recovery UI exists for them yet).
  renderBlockerRecovery?: (blocker: OrchestrationBlocker, register: (element: HTMLButtonElement | null) => void, onSuccessFocus?: () => void) => ReactNode
  onReleaseRetainedEdit?: (decisionIds: readonly string[], mode: 'dismiss' | 'review') => void
  suppressAnnouncement?: boolean
  announcementScope?: 'all' | 'human' | 'none'
}

type RetainedQueueEdit = {
  row: QueueRow
  snapshot: NeedsYouEditSnapshot
  activeDecisionId: string
  reviewEpoch: number
}

function decisionsForRow(row: QueueRow): readonly OrchestrationAuthorityDecisionRecord[] {
  if (row.kind === 'answer' || row.kind === 'recovery-proposal') return [row.decision]
  if (row.kind === 'answer-batch' || row.kind === 'goal-definition-questions') return row.decisions
  return []
}

function frozenEdit(row: QueueRow): RetainedQueueEdit | null {
  const decisions = decisionsForRow(row)
  if (decisions.length === 0) return null
  const frozenRow = structuredClone(row)
  const frozenDecisions = decisionsForRow(frozenRow)
  return {
    row: frozenRow,
    snapshot: { rowKey: row.key, decisionIds: frozenDecisions.map((decision) => decision.id), decisions: frozenDecisions },
    activeDecisionId: frozenDecisions[0].id,
    reviewEpoch: 0,
  }
}

function hasInlineDecisionForm(row: QueueRow) {
  return row.kind === 'answer-batch'
    || row.kind === 'goal-definition-questions'
    || row.kind === 'recovery-proposal'
    || row.kind === 'answer' && row.decision.decision_key.startsWith('team_hierarchy:agent:')
}

type FormAncestorNode = {
  tagName?: string
  parentElement?: FormAncestorNode | null
  parentNode?: FormAncestorNode | null
}

function targetIsInsideForm(target: EventTarget | null, boundary: EventTarget | null) {
  let node = target as FormAncestorNode | null
  const boundaryNode = boundary as FormAncestorNode | null
  while (node && node !== boundaryNode) {
    if (node.tagName === 'FORM') return true
    node = node.parentElement ?? node.parentNode ?? null
  }
  return false
}

const EYEBROW: Record<QueueRow['kind'], { text: string; color: string; icon: typeof Ban }> = {
  blocker: { text: 'BLOCKER', color: STATUS_COLORS.amber, icon: Ban },
  'lm-retry': { text: 'NEEDS RETRY', color: STATUS_COLORS.red, icon: RotateCcw },
  answer: { text: 'DECISION', color: STATUS_COLORS.blue, icon: HelpCircle },
  'answer-batch': { text: 'DECISIONS', color: STATUS_COLORS.blue, icon: HelpCircle },
  'goal-definition-questions': { text: 'QUESTIONS', color: STATUS_COLORS.blue, icon: HelpCircle },
  'recovery-proposal': { text: 'FIX PROPOSAL', color: STATUS_COLORS.amber, icon: ShieldAlert },
  gate: { text: 'GATE FAILED', color: STATUS_COLORS.red, icon: ShieldAlert },
  'meeting-commitment': { text: 'MEETING ITEM', color: STATUS_COLORS.amber, icon: HelpCircle },
  warning: { text: 'WARNING', color: STATUS_COLORS.amber, icon: AlertTriangle },
}

function NeedsYouDetails({ details }: { details: string }) {
  return <details className="mt-2">
    <summary className="inline-flex min-h-11 cursor-pointer items-center text-xs font-medium text-huddleroom-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-huddleroom-primary">Details</summary>
    <pre tabIndex={0} role="region" aria-label="Needs-you source details" className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-huddleroom-border p-3 font-mono text-xs">{details}</pre>
  </details>
}

function NamedReferences({
  references, activeProjectId, activeGoalId, gates, processes, sourceActionIds, onRevealSource,
}: {
  references: readonly NeedsYouReference[]
  activeProjectId: string
  activeGoalId: string
  gates: readonly OrchestrationGate[]
  processes: readonly OrchestrationProcessRunRecord[]
  sourceActionIds: readonly string[]
  onRevealSource?: (target: NeedsYouSourceTarget) => void
}) {
  const controls = references.flatMap((reference) => {
    if (reference.label === null || reference.project_id !== activeProjectId) return []
    const path = needsYouReferencePath(reference, activeProjectId)
    if (path) return [<Link key={`${reference.kind}:${reference.id}`} to={path} className="inline-flex min-h-11 min-w-0 max-w-full items-center break-words [overflow-wrap:anywhere] text-xs font-medium text-huddleroom-primary hover:underline">{reference.label}</Link>]
    if (!onRevealSource || reference.goal_id !== activeGoalId) return []
    const available = reference.kind === 'gate'
      ? gates.some((gate) => gate.id === reference.id)
      : reference.kind === 'process'
        ? processes.some((process) => process.id === reference.id)
        : reference.kind === 'action' && sourceActionIds.includes(reference.id)
    if (!available || (reference.kind !== 'gate' && reference.kind !== 'process' && reference.kind !== 'action')) return []
    const target: NeedsYouSourceTarget = { kind: reference.kind, id: reference.id, goalId: reference.goal_id }
    return [<button key={`${reference.kind}:${reference.id}`} type="button" className="inline-flex min-h-11 min-w-0 max-w-full items-center break-words [overflow-wrap:anywhere] text-left text-xs font-medium text-huddleroom-primary underline-offset-2 hover:underline" onClick={() => onRevealSource(target)}>{reference.label}</button>]
  })
  return controls.length > 0 ? <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1" aria-label="Related records">{controls}</div> : null
}

function referencesForRow(row: QueueRow, references: readonly NeedsYouReference[]) {
  const ids = new Set<string>()
  if (row.kind === 'blocker') {
    if (typeof row.blocker.task_id === 'string') ids.add(row.blocker.task_id)
    if (typeof row.blocker.gate_id === 'string') ids.add(row.blocker.gate_id)
  } else if (row.kind === 'gate') ids.add(row.gate.id)
  else if (row.kind === 'meeting-commitment') {
    ids.add(row.commitment.meeting_id)
    if (row.commitment.task_id) ids.add(row.commitment.task_id)
  } else if (row.kind === 'lm-retry') {
    ids.add(row.process.id)
    if (row.warning) ids.add(row.warning.id)
  } else if (row.kind === 'warning') {
    for (const id of [row.warning.source_process_run_id, row.warning.related_gate_id, row.warning.related_action_id, row.warning.related_agent_id]) {
      if (id) ids.add(id)
    }
  } else {
    const decisions = row.kind === 'answer' || row.kind === 'recovery-proposal' ? [row.decision] : row.decisions
    for (const decision of decisions) {
      for (const id of [decision.source_process_run_id, decision.related_gate_id, decision.related_action_id, decision.authority_agent_id]) {
        if (id) ids.add(id)
      }
      for (const reference of decision.needs_you_references ?? []) ids.add(reference.id)
    }
  }
  return references.filter((reference) => ids.has(reference.id))
}

// Short label for a decision's recommended option, used by the accept-
// recommendation fast path's "Accept: {label}" button (spec C).
function recommendationLabel(decision: OrchestrationAuthorityDecisionRecord) {
  const recommendation = decision.recommendation
  if (recommendation === null) return ''
  const option = decision.options.find((candidate) => decisionOptionKey(candidate) === recommendation)
  return (option && normalizeDecisionOption(option)?.label) || recommendation
}

// Adaptive clarification question context is stored as a single string:
// "{rationale} Destination: {destination}" (orchestration_goal_definition.py
// _continue_after_analysis). Split it back into its two parts for display.
function parseAdaptiveQuestionContext(context: string | null) {
  if (!context) return { rationale: null as string | null, destination: null as string | null }
  const match = /^(.*)\sDestination:\s(.+)$/s.exec(context)
  return match ? { rationale: match[1].trim() || null, destination: match[2].trim() || null } : { rationale: context, destination: null }
}

// Goal-definition step-through card (spec D): walks N pending free-text
// clarification questions of one round one at a time. `order` freezes the
// question sequence at mount — a new round produces a disjoint decision-id
// set, which changes this row's key and remounts a fresh card (see
// goalDefinitionQuestionDecisions), so this component never needs to
// reconcile a changing question set mid-round.
function GoalDefinitionQuestionCard({
  decisions,
  activeDecisionId,
  mutationBusy,
  reviewLocked = false,
  onAnswer,
  onAllAnswered,
  onActiveDecisionChange,
  displayReferences,
}: {
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  activeDecisionId?: string
  mutationBusy: boolean
  reviewLocked?: boolean
  onAnswer: NeedsYouQueueProps['onAnswerGoalDefinitionQuestion']
  onAllAnswered: () => void
  onActiveDecisionChange?: (decisionId: string) => void
  displayReferences: readonly NeedsYouReference[]
}) {
  const [order] = useState(() => decisions.map((decision) => decision.id))
  const [answeredCount, setAnsweredCount] = useState(() => {
    const activeIndex = activeDecisionId ? decisions.findIndex((decision) => decision.id === activeDecisionId) : 0
    return activeIndex >= 0 ? activeIndex : 0
  })
  const [values, setValues] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const byId = new Map(decisions.map((decision) => [decision.id, decision] as const))
  const { announceError } = useGoalAnnouncer()

  useEffect(() => { textareaRef.current?.focus() }, [answeredCount])
  useEffect(() => { if (error) announceError(error) }, [error, announceError])

  const total = order.length
  const currentId = order[answeredCount]
  const current = currentId ? byId.get(currentId) : undefined
  // Guards a stale/out-of-range index (e.g. the decisions prop shrank
  // unexpectedly) instead of crashing — Z4/queue cards must never throw.
  if (!current) return null
  const value = values[currentId] ?? ''
  const isLast = answeredCount === total - 1
  const { rationale, destination } = parseAdaptiveQuestionContext(current.context)
  const displayedQuestion = safeNeedsYouText(current.question, displayReferences)
  const displayedRationale = rationale ? safeNeedsYouText(rationale, displayReferences) : null
  const displayedDestination = destination ? safeNeedsYouText(destination, displayReferences) : null
  const disabled = mutationBusy || submitting
  const canSubmit = value.trim().length > 0

  function submit(event: FormEvent) {
    event.preventDefault()
    if (!canSubmit || disabled || reviewLocked) return
    setSubmitting(true)
    setError(null)
    onAnswer(current!, value.trim(), {
      onSuccess: () => {
        if (isLast) onAllAnswered()
        else {
          onActiveDecisionChange?.(order[answeredCount + 1])
          setAnsweredCount((count) => count + 1)
        }
      },
      onError: (submitError) => setError(submitError instanceof Error ? submitError.message : 'The request failed. Check your connection and try again.'),
      onSettled: () => setSubmitting(false),
    })
  }

  return <form onSubmit={submit} aria-busy={submitting} aria-disabled={reviewLocked || undefined}>
    <p className="text-xs font-medium text-huddleroom-text-muted">Question {answeredCount + 1} of {total}</p>
    <p className="mt-1 text-sm font-medium text-huddleroom-text-primary">{displayedQuestion}</p>
    {displayedRationale && <p className="mt-1 text-xs text-huddleroom-text-secondary">{displayedRationale}</p>}
    {displayedDestination && <p className="mt-1 text-xs text-huddleroom-text-muted">→ {displayedDestination}</p>}
    <Textarea ref={textareaRef} label="Answer" value={value} readOnly={reviewLocked} onChange={(event) => setValues((currentValues) => ({ ...currentValues, [currentId]: event.target.value }))} rows={3} disabled={disabled} />
    <div className="mt-1 flex gap-1" aria-hidden="true">
      {order.map((id, index) => (
        <span key={id} className={`h-1.5 w-1.5 rounded-full ${index === answeredCount ? 'bg-huddleroom-primary' : 'bg-huddleroom-border'}`} />
      ))}
    </div>
    {error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
    <div className="mt-3 flex justify-end">
      <Button type="submit" className="min-h-11" disabled={disabled || reviewLocked || !canSubmit}>{submitting ? 'Submitting…' : isLast ? 'Submit' : 'Next'}</Button>
    </div>
  </form>
}

const RECOVERY_STATUS_COPY: Partial<Record<RecoveryProposalStatus, string>> = {
  applying: 'Applying approved fix…',
  verifying: 'Verifying update…',
  retrying: 'Update verified. Retrying blocked work…',
  resolved: 'Blocked work is moving again.',
  rejected: 'Fix rejected. Nothing was changed.',
  failed: 'The fix did not complete.',
  unsupported: 'No executable fix is available with the current tools.',
}
const RECOVERY_LABEL_CLASS = 'text-xs font-semibold uppercase tracking-wide text-huddleroom-text-muted'
export type RecoveryAnswerOption = 'approve_fix' | 'reject' | 'capability_provided'

// A capability gap is "unsupported" (no fix to approve) but its decision stays pending until the owner
// marks the capability provided; only then is it actionable.
function isPendingGap(decision: OrchestrationAuthorityDecisionRecord) {
  return decision.recovery?.status === 'unsupported' && decision.status === 'pending'
}

// Simple FIX PROPOSAL card: Problem / Proposed updates / After approval, all
// server-supplied; one Approve fix or a Reject with required reason.
function RecoveryProposalCard({
  decision, mutationBusy, reviewLocked = false, accepted = false, onAnswer, registerPrimary, onAccepted, onBeginEdit, onCancel, onLeftQueue, displayReferences,
}: {
  decision: OrchestrationAuthorityDecisionRecord
  mutationBusy: boolean
  reviewLocked?: boolean
  accepted?: boolean
  onAnswer: NeedsYouQueueProps['onAnswerRecoveryProposal']
  registerPrimary: (element: HTMLButtonElement | null) => void
  onAccepted: () => void
  onBeginEdit: () => void
  onCancel: () => boolean
  onLeftQueue: (quiet: boolean) => void
  displayReferences: readonly NeedsYouReference[]
}) {
  const recovery = decision.recovery!
  const { status } = recovery
  const [busy, setBusy] = useState(false)
  const [answered, setAnswered] = useState(accepted)
  const [rejecting, setRejecting] = useState(false)
  const [reason, setReason] = useState('')
  const [error, setError] = useState<string | null>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const rejectRef = useRef<HTMLButtonElement>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)
  const inFlight = useRef(false)
  const answeredRef = useRef(accepted)
  const leftRef = useRef(onLeftQueue)
  leftRef.current = onLeftQueue
  const gapRef = useRef(false)
  const statusRef = useRef(status)
  statusRef.current = status
  const { announceError } = useGoalAnnouncer()
  const idBase = `recovery-${useId()}`

  const pending = status === 'pending_approval'
  const pendingGap = isPendingGap(decision)
  gapRef.current = pendingGap
  const disabled = mutationBusy || busy || (answered && (pending || pendingGap))

  useEffect(() => { if (rejecting) reasonRef.current?.focus() }, [rejecting])
  useEffect(() => {
    if (!accepted) return
    answeredRef.current = true
    setAnswered(true)
  }, [accepted])
  useEffect(() => { if (error) announceError(error) }, [error, announceError])
  // The card stays after approve (applying/verifying/retrying): keep focus on it.
  useEffect(() => { if (answered && !pending) headingRef.current?.focus() }, [answered, pending])
  // The row vanished after an answer. Unmount also happens on navigation or when a
  // newer proposal supersedes this card, so only celebrate when the last status
  // seen was one that precedes resolution; never toast otherwise.
  useEffect(() => () => {
    if (!answeredRef.current) return
    if (statusRef.current === 'retrying') leftRef.current(false)
    else if (gapRef.current) leftRef.current(true) // gap answered: move focus, no "moving again" toast
  }, [])

  function submit(option: RecoveryAnswerOption, text: string) {
    if (disabled || reviewLocked || inFlight.current || !onAnswer) { if (disabled || reviewLocked) headingRef.current?.focus(); return }
    inFlight.current = true
    setBusy(true)
    setError(null)
    onAnswer(decision, option, text, {
      onSuccess: () => { answeredRef.current = true; setAnswered(true); onAccepted() },
      onError: (submitError) => setError(submitError instanceof Error ? submitError.message : 'The request failed. Check your connection and try again.'),
      onSettled: () => { inFlight.current = false; setBusy(false) },
    })
  }

  const copy = pendingGap && answered ? 'Marked as provided. Re-diagnosing the blocker…' : RECOVERY_STATUS_COPY[status]
  const showAfter = pending || pendingGap || status === 'applying' || status === 'verifying' || status === 'retrying'
  const detail = status === 'rejected' ? decision.reason : status === 'failed' ? recovery.error : null
  const displayedDetail = detail ? safeNeedsYouText(detail, displayReferences) : null
  const displayedProblem = safeNeedsYouText(recovery.problem, displayReferences)
  const displayedUpdates = recovery.general_updates.map((update) => safeNeedsYouText(update, displayReferences))
  const displayedAfter = safeNeedsYouText(recovery.after_approval, displayReferences)
  const hasEvidence = (recovery.evidence?.length ?? 0) > 0
  const details = recovery.exact_details !== null || hasEvidence
    ? JSON.stringify({ ...(recovery.exact_details !== null && { exact_details: recovery.exact_details }), ...(hasEvidence && { evidence: recovery.evidence }) }, null, 2)
    : null

  return <div aria-busy={busy || (answered && (pending || pendingGap))}>
    <div className="flex items-center gap-3">
      <ShieldAlert aria-hidden="true" className="h-4 w-4 shrink-0" style={{ color: STATUS_COLORS.amber }} />
      <h3 ref={headingRef} tabIndex={-1} className="text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: STATUS_COLORS.amber }}>{pendingGap ? 'CAPABILITY NEEDED' : 'FIX PROPOSAL'}</h3>
    </div>
    <div className="mt-2 flex flex-col gap-3 text-sm text-huddleroom-text-primary">
      <section className="flex flex-col gap-1"><h4 className={RECOVERY_LABEL_CLASS}>Problem</h4><p id={`${idBase}-problem`}>{displayedProblem}</p></section>
      {!pendingGap && recovery.general_updates.length > 0 && <section className="flex flex-col gap-1">
        <h4 className={RECOVERY_LABEL_CLASS}>Proposed updates</h4>
        <ul className="flex list-disc flex-col gap-1 pl-5">{displayedUpdates.map((update, index) => <li key={index}>{update}</li>)}</ul>
      </section>}
      {showAfter && <section className="flex flex-col gap-1"><h4 className={RECOVERY_LABEL_CLASS}>{pendingGap ? "After you've provided it" : 'After approval'}</h4><p id={`${idBase}-after`}>{displayedAfter}</p></section>}
      {recovery.global_effect && <p role="note" className="flex items-start gap-2 text-sm text-huddleroom-text-primary">
        <ShieldAlert aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" style={{ color: STATUS_COLORS.amber }} />
        This affects the global agent roster, so it applies to every project and goal.
      </p>}
      {details && <details>
        <summary className="inline-flex min-h-11 cursor-pointer items-center text-sm text-huddleroom-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-huddleroom-primary">Show exact details</summary>
        <pre tabIndex={0} role="region" aria-label="Exact details" className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-huddleroom-border p-3 font-mono text-xs">{details}</pre>
      </details>}
      <div role="status" aria-live="polite" aria-atomic="true" className={copy ? 'text-sm text-huddleroom-text-secondary' : 'sr-only'}>
        {copy}{status === 'rejected' && displayedDetail ? ` ${displayedDetail}` : ''}
        {status === 'failed' && displayedDetail && <p className="mt-1 text-huddleroom-status-red">{displayedDetail}</p>}
      </div>
    </div>
    {pendingGap && <div className="mt-3 flex flex-wrap justify-end gap-3">
      <Button ref={registerPrimary} data-needs-you-primary type="button" variant="secondary" className="min-h-11 min-w-11" disabled={disabled || reviewLocked} aria-describedby={`${idBase}-problem ${idBase}-after`} onClick={() => submit('capability_provided', '')}>I've provided it</Button>
    </div>}
    {pending && <>
      <div className="mt-3 flex flex-wrap justify-end gap-3">
        <Button ref={registerPrimary} data-needs-you-primary type="button" className="min-h-11 min-w-11" disabled={disabled || reviewLocked} onClick={() => submit('approve_fix', '')}>Approve fix</Button>
        <Button ref={rejectRef} type="button" variant="secondary" className="min-h-11 min-w-11" disabled={disabled || reviewLocked} onClick={() => { onBeginEdit(); setRejecting(true) }}>Reject</Button>
      </div>
      {rejecting && <form className="mt-3 flex flex-col gap-3" onSubmit={(event) => { event.preventDefault(); if (reason.trim()) submit('reject', reason.trim()) }}>
        <Textarea ref={reasonRef} label="Reason" aria-required="true" value={reason} readOnly={reviewLocked} onChange={(event) => setReason(event.target.value)} rows={3} disabled={disabled} />
        <div className="flex flex-wrap justify-end gap-3">
          <Button type="submit" className="min-h-11 min-w-11" disabled={disabled || reviewLocked || !reason.trim()}>Confirm reject</Button>
          <Button type="button" variant="secondary" className="min-h-11 min-w-11" disabled={disabled} onClick={() => { setRejecting(false); setReason(''); if (!onCancel()) queueMicrotask(() => rejectRef.current?.focus()) }}>Cancel</Button>
        </div>
      </form>}
    </>}
    {error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
  </div>
}

function OwningStepTag({ process, onSelectProcess }: { process: OrchestrationProcessRunRecord | undefined; onSelectProcess: (type: OrchestrationBaselineProcessType) => void }) {
  if (!process || !isBaselineProcessType(process.process_type)) return null
  const type = process.process_type
  return <button type="button" className="inline-flex min-h-11 items-center text-xs font-medium text-huddleroom-primary underline-offset-2 hover:underline" onClick={() => onSelectProcess(type)}>
    {stepDisplayLabel(type)}
  </button>
}

const WAIVER_SCOPE = 'This removes the meeting obligation. Linked work will keep running; goal criteria and gates still apply.'
const COMMITMENT_STATES: Record<string, string> = {
  open: 'No linked work', assigned: 'Assigned — no linked work',
  task_in_flight: 'Linked task unfinished', graph_in_flight: 'Linked graph unfinished',
  task_failed: 'Linked task needs attention', graph_failed: 'Linked graph needs attention',
  done_unverified: 'Linked task done — awaiting verification',
}

function MeetingCommitmentRow({ commitment, projectId, mutationBusy, registerPrimary, onRemoved, title, explanation, nextStep, details, relatedRecords }: {
  commitment: OrchestrationMeetingCommitment
  projectId: string
  mutationBusy: boolean
  registerPrimary: (element: HTMLButtonElement | null) => void
  onRemoved: () => void
  title: string
  explanation: string
  nextStep: string
  details: string
  relatedRecords: ReactNode
}) {
  const update = useUpdateActionItem(projectId, commitment.meeting_id)
  const [expanded, setExpanded] = useState(false)
  const [reason, setReason] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const submitted = useRef(false)
  const rowRef = useRef<HTMLLIElement>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)
  const removalRef = useRef(onRemoved)
  removalRef.current = onRemoved
  const id = useId()
  const { active, announceError, announceQueue } = useGoalAnnouncer()
  const trimmed = reason.trim()
  const overLimit = trimmed.length > 500
  const pending = submitting || update.isPending
  useEffect(() => { if (expanded) reasonRef.current?.focus() }, [expanded])
  useEffect(() => { if (error && !pending) reasonRef.current?.focus() }, [error, pending])
  useLayoutEffect(() => {
    const row = rowRef.current
    return () => {
      if (row?.contains(document.activeElement)) removalRef.current()
    }
  }, [])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (submitted.current || pending || mutationBusy || !trimmed || overLimit) return
    submitted.current = true
    setSubmitting(true)
    setError(null)
    update.mutate({ actionItemId: commitment.id, data: { status: 'dismissed', reason: trimmed } }, {
      onSuccess: (item) => {
        const message = item.status === 'dismissed' ? 'Meeting obligation waived.'
          : item.status === 'completed' ? 'Meeting item already completed.'
          : `Meeting item updated — current status: ${item.status.replace(/_/g, ' ')}.`
        toast.success(message)
        announceQueue(message)
      },
      onError: (failure) => {
        const message = failure instanceof Error ? failure.message : 'Could not waive the meeting obligation. Try again.'
        setError(message)
        announceError(message)
      },
      onSettled: () => { submitted.current = false; setSubmitting(false) },
    })
  }

  return <li ref={rowRef} className="py-3 first:pt-0 last:pb-0 motion-safe:animate-[queue-row-in_180ms_ease-out]" aria-busy={pending}>
    <div className="flex flex-wrap items-start gap-3 sm:flex-nowrap">
      <HelpCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" style={{ color: STATUS_COLORS.amber }} />
      <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-0.5 sm:flex-nowrap">
        <p className="w-[104px] shrink-0 text-label font-semibold uppercase tracking-[0.1em]" style={{ color: STATUS_COLORS.amber }}>MEETING ITEM</p>
        <div className="min-w-0 flex-1">
          <h3 className="break-words text-sm font-medium text-huddleroom-text-primary">{title}</h3>
          {explanation !== title && <p className="mt-1 break-words text-sm text-huddleroom-text-secondary">{explanation}</p>}
          <p className="mt-1 text-xs text-huddleroom-text-muted">{nextStep}</p>
          <Link to={`/meetings/${commitment.meeting_id}`} className="inline-flex min-h-11 min-w-0 max-w-full items-center break-words [overflow-wrap:anywhere] text-xs font-medium text-huddleroom-primary hover:underline">{commitment.meeting_title}</Link>
          <p className="text-xs text-huddleroom-text-secondary">{COMMITMENT_STATES[commitment.state] ?? commitment.state.replace(/_/g, ' ')}</p>
          {relatedRecords}
          <NeedsYouDetails details={details} />
        </div>
      </div>
      <Button ref={(element) => { triggerRef.current = element; registerPrimary(element) }} type="button" variant="secondary" className="min-h-11" disabled={mutationBusy || pending} aria-expanded={expanded} aria-controls={`${id}-form`} onClick={() => setExpanded(true)}>Waive</Button>
    </div>
    {expanded && <form id={`${id}-form`} className="mt-2 space-y-3" onSubmit={submit} aria-busy={pending}>
      <p id={`${id}-scope`} className="text-sm text-huddleroom-text-secondary">{WAIVER_SCOPE}</p>
      <Textarea ref={reasonRef} label="Reason" required aria-required="true" rows={3} value={reason} disabled={pending || mutationBusy} onChange={(event) => setReason(event.target.value)} aria-invalid={overLimit || !!error} aria-describedby={`${id}-scope ${id}-validation${error ? ` ${id}-error` : ''}`} />
      <p id={`${id}-validation`} className="text-xs text-huddleroom-text-secondary">{overLimit ? 'Reason must be 500 characters or fewer.' : 'A reason of 1–500 characters is required.'}</p>
      {error && <p id={`${id}-error`} role={active ? undefined : 'alert'} className="text-xs text-huddleroom-status-red">{error}</p>}
      <div className="flex flex-wrap justify-end gap-3">
        <Button type="submit" className="min-h-11" disabled={mutationBusy || pending || !trimmed || overLimit}>{pending ? 'Waiving…' : 'Confirm waive'}</Button>
        <Button type="button" variant="secondary" className="min-h-11" disabled={mutationBusy || pending} onClick={() => { setExpanded(false); triggerRef.current?.focus() }}>Cancel</Button>
      </div>
    </form>}
  </li>
}

export function NeedsYouQueue({
  goal, run, processes, sourceProcesses, warnings, gates, meetingCommitments = [], checkpointItems, allDecisions = [], authorityReady = true, invalidatedDecisions = {}, needsYouReferences = [], sourceActionIds = [], recoveryDecisions = [], onAnswerRecoveryProposal, checkpointLoaded, queriesErrored, queriesUnready, deferredCount, isWorking, mutationBusy,
  onSelectProcess, onRevealSource, openAction, onOpenAction, onActionTrigger, renderActionForm, renderAgentDefinitionReviewBatch, renderTeamHierarchyProposal,
  onAcceptDecision, onAnswerGoalDefinitionQuestion,
  retryPending, retryError, onRetryProcess, onApproveStaleInputs, onDismissStaleInputs, onAcknowledgeWarning,
  renderBlockerRecovery, onReleaseRetainedEdit, suppressAnnouncement = false, announcementScope,
}: NeedsYouQueueProps) {
  const [justResolvedIds, setJustResolvedIds] = useState<Set<string>>(new Set())
  const [retainedEdit, setRetainedEdit] = useState<RetainedQueueEdit | null>(null)
  const [acceptedRecoveryKeys, setAcceptedRecoveryKeys] = useState<Set<string>>(new Set())
  const retainedEditRef = useRef<RetainedQueueEdit | null>(retainedEdit)
  retainedEditRef.current = retainedEdit
  const releasedFormEpochs = useRef<Record<string, number>>({})
  const cancelledEditKeys = useRef<Set<string>>(new Set())
  const rowRefs = useRef<Record<string, HTMLButtonElement | null>>({})
  const rowContainers = useRef<Record<string, HTMLLIElement | null>>({})
  const rowOrderRef = useRef<string[]>([])
  const headingRef = useRef<HTMLHeadingElement>(null)
  const { announceQueue } = useGoalAnnouncer()
  const scope = announcementScope ?? (suppressAnnouncement ? 'none' : 'all')

  const criteriaByKey = new Map(goal.success_criteria.filter((criterion) => criterion.key).map((criterion) => [criterion.key as string, criterion.description]))
  const referenceMetadata = Array.isArray(needsYouReferences) ? needsYouReferences : []
  const selectedRows = buildQueueRows({ run, processes, checkpointItems, allDecisions, recoveryDecisions, gates, warnings, criteriaByKey, meetingCommitments })
  const allRows = retainedEdit
    ? [...selectedRows.filter((row) => row.key !== retainedEdit.snapshot.rowKey), retainedEdit.row].sort((left, right) => compareQueueOccurrences(
      queueOccurrence(left, allDecisions), queueOccurrence(right, allDecisions),
    ))
    : selectedRows
  const rows = allRows.filter((row) => !justResolvedIds.has(row.key))
  const humanDecisions = allRows.flatMap((row) => {
    if (row.kind === 'answer' || row.kind === 'recovery-proposal') return [row.decision]
    if (row.kind === 'answer-batch' || row.kind === 'goal-definition-questions') return row.decisions
    return []
  }).filter((decision) => decision.authority === 'human' && decision.status === 'pending')
  const humanIds = [...new Set(humanDecisions.map((decision) => decision.id))].sort()
  const humanSignature = humanIds.join('|')
  const seenHumanIds = useRef<Set<string> | null>(null)
  const seenAnsweredIds = useRef<Set<string> | null>(null)
  const newestQuestionAt = useRef<number | null>(null)
  const newestAnswerAt = useRef<number | null>(null)
  const locallyAnsweredIds = useRef(new Set<string>())
  const answeredIds = allDecisions.filter((decision) => decision.authority === 'human' && decision.status === 'answered' && decision.decided_at)
    .map((decision) => decision.id).sort()
  const answeredSignature = answeredIds.join('|')
  useEffect(() => {
    if (scope !== 'human' || queriesUnready || queriesErrored || !authorityReady) return
    const questionTimes = new Map(humanDecisions.map((decision) => [decision.id, parseUtc(decision.asked_at).getTime()]))
    const answerTimes = new Map(allDecisions.map((decision) => [decision.id, decision.decided_at ? parseUtc(decision.decided_at).getTime() : NaN]))
    if (!seenHumanIds.current) {
      seenHumanIds.current = new Set(humanIds)
      seenAnsweredIds.current = new Set(answeredIds)
      newestQuestionAt.current = Math.max(...[...questionTimes.values()].filter(Number.isFinite), -Infinity)
      newestAnswerAt.current = Math.max(...[...answerTimes.values()].filter(Number.isFinite), -Infinity)
      return
    }
    const arriving = humanIds.filter((id) => !seenHumanIds.current!.has(id)
      && (!Number.isFinite(questionTimes.get(id)) || questionTimes.get(id)! >= (newestQuestionAt.current ?? -Infinity)))
    for (const id of humanIds) seenHumanIds.current!.add(id)
    newestQuestionAt.current = Math.max(newestQuestionAt.current ?? -Infinity, ...[...questionTimes.values()].filter(Number.isFinite))
    if (arriving.length) announceQueue(`${arriving.length} new question${arriving.length === 1 ? '' : 's'} need${arriving.length === 1 ? 's' : ''} you.`)
    const resolved = answeredIds.filter((id) => !seenAnsweredIds.current!.has(id) && !locallyAnsweredIds.current.has(id)
      && (!Number.isFinite(answerTimes.get(id)) || answerTimes.get(id)! >= (newestAnswerAt.current ?? -Infinity)))
    for (const id of answeredIds) seenAnsweredIds.current!.add(id)
    newestAnswerAt.current = Math.max(newestAnswerAt.current ?? -Infinity, ...[...answerTimes.values()].filter(Number.isFinite))
    if (resolved.length) announceQueue(`${resolved.length} question${resolved.length === 1 ? '' : 's'} answered.`)
  }, [scope, queriesUnready, queriesErrored, authorityReady, humanSignature, answeredSignature, announceQueue])
  rowOrderRef.current = rows.map((row) => row.key)
  // Terminal read-only recovery history needs no action, so it isn't announced as "needs you".
  const actionableCount = rows.filter((row) => editState(row) !== 'terminal'
    && !(row.kind === 'recovery-proposal' && !isPendingGap(row.decision) && ['rejected', 'failed', 'unsupported'].includes(row.decision.recovery!.status))).length
  const runningProcess = processes.find((process) => process.status === 'running')
  const runningStep = runningProcess && isBaselineProcessType(runningProcess.process_type) ? runningProcess.process_type : null

  const [announcement, setAnnouncement] = useState('')
  const checkpointCountRef = useRef<number | null>(null)

  useEffect(() => {
    if (!retainedEdit || retainedEdit.reviewEpoch === 0) return
    const row = rowContainers.current[retainedEdit.snapshot.rowKey]
    queueMicrotask(() => {
      const registeredPrimary = rowRefs.current[retainedEdit.snapshot.rowKey]
      const prefersRegisteredPrimary = retainedEdit.row.kind === 'recovery-proposal'
        || retainedEdit.row.kind === 'answer-batch'
        || retainedEdit.row.kind === 'answer' && retainedEdit.row.decision.decision_key.startsWith('team_hierarchy:agent:')
      if (prefersRegisteredPrimary && registeredPrimary && registeredPrimary.isConnected !== false) {
        registeredPrimary.focus()
        return
      }
      if (!row || typeof row.querySelector !== 'function') return
      const formTarget = row?.querySelector<HTMLElement>(
        'form textarea:not([readonly]):not([disabled]), form input:not([readonly]):not([disabled]), form select:not([disabled]), form button:not([disabled])',
      )
      const fallback = row.querySelector<HTMLElement>('[data-needs-you-primary]:not([disabled])')
      ;(formTarget ?? fallback)?.focus()
    })
  }, [retainedEdit?.reviewEpoch, retainedEdit?.snapshot.rowKey])

  // Earned toast (spec: motion item 5) — fires once when checkpoint questions
  // arrive while the page is open. checkpointItems already refreshes on any
  // orchestration.* WS event via the app-wide useWSQuerySync invalidation
  // bridge (see hooks/useWSQuerySync.ts), so this only has to notice the
  // resulting count going up — no separate WS subscription needed here.
  // Stays disarmed (ref left null, no toast) until checkpointLoaded flips
  // true — `checkpointItems` defaults to `[]` before the first successful
  // load, which is indistinguishable from "loaded, genuinely empty" by
  // length alone, so the ordinary 0→N transition on first data load must not
  // be mistaken for a live arrival (review round 1, finding 1).
  useEffect(() => {
    if (!checkpointLoaded || scope !== 'all') return
    const previous = checkpointCountRef.current
    if (previous !== null && checkpointItems.length > previous) toast('The orchestrator has questions about your goal')
    checkpointCountRef.current = checkpointItems.length
  }, [checkpointLoaded, checkpointItems.length, scope])

  // Debounced (~500ms) sr-only aria-live announcing net queue changes — kept
  // separate from the existing per-action aria-live regions (BaselineActionForm
  // errors, header lifecycle message), which stay authoritative/immediate.
  useEffect(() => {
    if (scope !== 'all' || queriesUnready || queriesErrored) return
    const timer = setTimeout(() => {
      setAnnouncement(actionableCount === 0 ? 'Queue clear.' : `${actionableCount} item${actionableCount === 1 ? '' : 's'} need${actionableCount === 1 ? 's' : ''} you.`)
    }, QUEUE_ANNOUNCE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [actionableCount, queriesErrored, queriesUnready, scope])

  // Flow announcement through the shared goal announcer's queue region.
  useEffect(() => {
    if (announcement) announceQueue(announcement)
  }, [announcement, announceQueue])

  function markResolved(key: string) {
    const resolved = selectedRows.find((row) => row.key === key)
    if (resolved?.kind === 'answer' || resolved?.kind === 'recovery-proposal') locallyAnsweredIds.current.add(resolved.decision.id)
    if (resolved?.kind === 'answer-batch' || resolved?.kind === 'goal-definition-questions') {
      for (const decision of resolved.decisions) locallyAnsweredIds.current.add(decision.id)
    }
    setJustResolvedIds((current) => {
      const next = new Set(current)
      next.add(key)
      return next
    })
  }

  function retainRow(row: QueueRow) {
    if (acceptedRecoveryKeys.has(row.key) || cancelledEditKeys.current.has(row.key)) return
    if (retainedEditRef.current?.snapshot.rowKey === row.key) return
    const next = frozenEdit(row)
    if (next) next.reviewEpoch = releasedFormEpochs.current[row.key] ?? 0
    retainedEditRef.current = next
    setRetainedEdit(next)
  }

  function beginEdit(row: QueueRow) {
    cancelledEditKeys.current.delete(row.key)
    retainRow(row)
  }

  function releaseRetained(rowKey: string) {
    const current = retainedEditRef.current
    if (!current || current.snapshot.rowKey !== rowKey) return
    const decisionIds = current.snapshot.decisionIds
    if (current.reviewEpoch > 0) {
      releasedFormEpochs.current[rowKey] = current.reviewEpoch
    }
    cancelledEditKeys.current.add(rowKey)
    retainedEditRef.current = null
    setRetainedEdit(null)
    onReleaseRetainedEdit?.(decisionIds, 'dismiss')
  }

  function reviewRetained(rowKey: string) {
    if (!retainedEdit || retainedEdit.snapshot.rowKey !== rowKey) return
    const authoritativeById = new Map(allDecisions.map((decision) => [decision.id, decision] as const))
    const replace = (decision: OrchestrationAuthorityDecisionRecord) => authoritativeById.get(decision.id) ?? decision
    let refreshed: QueueRow = retainedEdit.row
    if (refreshed.kind === 'answer' || refreshed.kind === 'recovery-proposal') refreshed = { ...refreshed, decision: replace(refreshed.decision) }
    else if (refreshed.kind === 'answer-batch' || refreshed.kind === 'goal-definition-questions') refreshed = { ...refreshed, decisions: refreshed.decisions.map(replace) }
    const next = frozenEdit(refreshed)
    if (!next) return
    setRetainedEdit({
      ...next,
      activeDecisionId: next.snapshot.decisionIds.includes(retainedEdit.activeDecisionId)
        ? retainedEdit.activeDecisionId
        : next.activeDecisionId,
      reviewEpoch: retainedEdit.reviewEpoch + 1,
    })
    onReleaseRetainedEdit?.(retainedEdit.snapshot.decisionIds, 'review')
  }

  function retainedValidationSnapshot(row: QueueRow) {
    if (!retainedEdit || retainedEdit.snapshot.rowKey !== row.key) return null
    return row.kind === 'goal-definition-questions'
      ? {
          rowKey: retainedEdit.snapshot.rowKey,
          decisionIds: [retainedEdit.activeDecisionId],
          decisions: retainedEdit.snapshot.decisions.filter((decision) => decision.id === retainedEdit.activeDecisionId),
        }
      : retainedEdit.snapshot
  }

  function editState(row: QueueRow) {
    const validationSnapshot = retainedValidationSnapshot(row)
    if (!validationSnapshot) return 'current' as const
    const authoritativeState = validateNeedsYouEdit(validationSnapshot, authorityReady ? allDecisions : undefined)
    if (authoritativeState === 'terminal') return authoritativeState
    for (const decisionId of validationSnapshot.decisionIds) {
      const invalidated = invalidatedDecisions[decisionId]
      if (invalidated) return invalidated
    }
    return authoritativeState
  }

  function setActiveDecision(rowKey: string, decisionId: string) {
    setRetainedEdit((current) => current?.snapshot.rowKey === rowKey ? { ...current, activeDecisionId: decisionId } : current)
  }

  // Moves focus to the next row's primary action after a row resolves (skipping
  // reference-only rows like blockers/gates, which register no ref); falls back
  // to the queue heading once the queue is empty.
  function focusNextRow(key: string, order = rowOrderRef.current ?? []) {
    const index = order.indexOf(key)
    queueMicrotask(() => {
      for (let i = index + 1; i < order.length; i += 1) {
        const candidate = rowRefs.current[order[i]]
        if (candidate && candidate.isConnected !== false) { candidate.focus(); return }
      }
      headingRef.current?.focus()
    })
  }

  // Retry has no BaselineFocusAction (it doesn't go through openAction/
  // renderActionForm), so it registers directly instead of via onActionTrigger.
  function registerRowRef(key: string, element: HTMLButtonElement | null) {
    rowRefs.current[key] = element
  }

  function registerRow(action: BaselineFocusAction, key: string, element: HTMLButtonElement | null) {
    onActionTrigger(action, element)
    registerRowRef(key, element)
  }

  function successToast(message: string) {
    toast.success(message)
  }

  if (rows.length === 0) {
    const empty = emptyStateCopy(goal, isWorking, runningStep, run, queriesErrored, queriesUnready)
    return <section aria-labelledby="needs-you-queue-heading" className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
      <h2 ref={headingRef} id="needs-you-queue-heading" tabIndex={-1} className={ZONE_TITLE_CLASS}>Needs you</h2>
      <div className="mt-2 flex items-center gap-2" role={empty.tone === 'error' ? 'alert' : undefined}>
        {empty.tone === 'working' && <span aria-hidden="true" className="h-2.5 w-2.5 shrink-0 rounded-full motion-safe:animate-pulse" style={{ backgroundColor: STATUS_COLORS.amber }} />}
        {empty.tone === 'on-track' && <span aria-hidden="true" className="shrink-0" style={{ color: STATUS_COLORS.green }}>✓</span>}
        {empty.tone === 'terminal' && <CheckCircle2 aria-hidden="true" className="h-4 w-4 shrink-0" style={{ color: STATUS_COLORS.green }} />}
        {empty.tone === 'error' && <AlertTriangle aria-hidden="true" className="h-4 w-4 shrink-0" style={{ color: STATUS_COLORS.red }} />}
        <p className={`text-sm ${empty.tone === 'loading' ? 'text-huddleroom-text-muted' : ''}`} style={{ color: empty.tone === 'working' ? STATUS_COLORS.amber : empty.tone === 'on-track' || empty.tone === 'terminal' ? STATUS_COLORS.green : empty.tone === 'error' ? STATUS_COLORS.red : undefined }}>{empty.text}</p>
      </div>
    </section>
  }

  return <section aria-labelledby="needs-you-queue-heading" className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
    <h2 ref={headingRef} id="needs-you-queue-heading" tabIndex={-1} className={ZONE_TITLE_CLASS}>Needs you</h2>
    <ul className="mt-2 divide-y divide-huddleroom-border">
      {rows.map((row) => {
        const presentation = presentNeedsYouRow(row, referencesForRow(row, referenceMetadata))
        const lifecycle = editState(row)
        const retainedDisplaced = retainedEdit?.snapshot.rowKey === row.key
          && !selectedRows.some((selected) => selected.key === row.key)
        const reviewEpoch = retainedEdit?.snapshot.rowKey === row.key
          ? retainedEdit.reviewEpoch
          : releasedFormEpochs.current[row.key] ?? 0
        const inlineContract = row.kind === 'recovery-proposal' || row.kind === 'answer' && row.decision.decision_key.startsWith('team_hierarchy:agent:')
          ? `:${row.decision.contract_version}:${row.decision.runtime_identity ?? ''}`
          : ''
        const formKey = `${row.key}:review:${reviewEpoch}${inlineContract}`
        const editBlocked = lifecycle !== 'current'
        const releaseEdit = () => releaseRetained(row.key)
        const activeDecisionId = retainedEdit?.snapshot.rowKey === row.key ? retainedEdit.activeDecisionId : undefined
        const authoritativeDecision = activeDecisionId
          ? allDecisions.find((candidate) => candidate.id === activeDecisionId)
          : decisionsForRow(row).map((decision) => allDecisions.find((candidate) => candidate.id === decision.id))
            .find((decision) => decision?.status !== 'pending')
        const retainedSnapshot = retainedValidationSnapshot(row)
        const refreshedReviewAvailable = lifecycle === 'stale' && authorityReady && retainedSnapshot !== null
          && retainedSnapshot.decisionIds.every((decisionId) => allDecisions.some((decision) => decision.id === decisionId && decision.status === 'pending'))
        const reviewStatus = lifecycle === 'current' ? null : <div role="status" aria-live="polite" aria-atomic="true" className="mt-3 rounded border border-huddleroom-border p-3 text-sm text-huddleroom-text-secondary">
          {lifecycle === 'terminal' ? <>
            <p className="font-medium text-huddleroom-text-primary">{authoritativeDecision?.status === 'answered' ? 'Answered elsewhere' : 'Decision closed elsewhere'}</p>
            <p>Authoritative status: {authoritativeDecision?.status ?? 'completed'}.</p>
            {authoritativeDecision?.selected_option !== null && authoritativeDecision?.selected_option !== undefined && <p>Selected answer: {safeNeedsYouText(authoritativeDecision.selected_option, presentation.references)}</p>}
            {authoritativeDecision?.reason && <p>Reason: {safeNeedsYouText(authoritativeDecision.reason, presentation.references)}</p>}
            <Button type="button" variant="secondary" className="mt-2 min-h-11" onClick={() => { releaseEdit(); focusNextRow(row.key) }}>Dismiss</Button>
          </> : <>
            <p>{lifecycle === 'stale'
              ? 'The decision changed. Review the refreshed decision before answering.'
              : 'The current decision could not be confirmed. Refresh and review it before answering.'}</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {refreshedReviewAvailable && <Button type="button" variant="secondary" className="min-h-11" onClick={() => reviewRetained(row.key)}>Review refreshed decision</Button>}
              <Button type="button" variant="secondary" className="min-h-11" onClick={() => { releaseEdit(); focusNextRow(row.key) }}>Dismiss</Button>
            </div>
          </>}
        </div>
        if (row.kind === 'meeting-commitment') {
          const priorOrder = rows.map((candidate) => candidate.key)
          return <MeetingCommitmentRow key={row.key} commitment={row.commitment} projectId={goal.project_id} mutationBusy={mutationBusy} registerPrimary={(element) => registerRowRef(row.key, element)} onRemoved={() => focusNextRow(row.key, priorOrder)} title={presentation.title} explanation={presentation.explanation} nextStep={presentation.nextStep} details={presentation.details} relatedRecords={<NamedReferences references={presentation.references} activeProjectId={goal.project_id} activeGoalId={goal.id} gates={gates} processes={processes} sourceActionIds={sourceActionIds} onRevealSource={onRevealSource} />} />
        }
        if (row.kind === 'recovery-proposal') {
          return <li key={row.key} ref={(element) => { rowContainers.current[row.key] = element }} className="py-3 first:pt-0 last:pb-0 motion-safe:animate-[queue-row-in_180ms_ease-out]" onClickCapture={() => retainRow(row)} onFocusCapture={(event) => {
            if (targetIsInsideForm(event.target, event.currentTarget)) retainRow(row)
          }}>
            <RecoveryProposalCard key={formKey}
              decision={row.decision} mutationBusy={mutationBusy} reviewLocked={editBlocked} accepted={acceptedRecoveryKeys.has(row.key)} onAnswer={onAnswerRecoveryProposal}
              displayReferences={presentation.references}
              registerPrimary={(element) => registerRowRef(row.key, element)}
              onAccepted={() => { setAcceptedRecoveryKeys((current) => new Set(current).add(row.key)); releaseEdit() }}
              onBeginEdit={() => beginEdit(row)}
              onCancel={() => {
                if (retainedDisplaced) {
                  releaseEdit()
                  focusNextRow(row.key)
                  return true
                }
                releaseEdit()
                return false
              }}
              onLeftQueue={(quiet) => {
                if (rowOrderRef.current.includes(row.key)) return
                setAcceptedRecoveryKeys((current) => { const next = new Set(current); next.delete(row.key); return next })
                delete releasedFormEpochs.current[row.key]
                cancelledEditKeys.current.delete(row.key)
                releaseEdit()
                markResolved(row.key)
                if (!quiet) successToast('Blocked work is moving again.')
                focusNextRow(row.key)
              }}
            />
            <NamedReferences references={presentation.references} activeProjectId={goal.project_id} activeGoalId={goal.id} gates={gates} processes={sourceProcesses ?? processes} sourceActionIds={sourceActionIds} onRevealSource={onRevealSource} />
            <NeedsYouDetails details={presentation.details} />
            {reviewStatus}
          </li>
        }
        const eyebrow = EYEBROW[row.kind]
        const Icon = eyebrow.icon
        // Compact card per the approved mockup: icon + eyebrow/message/tag on
        // the left, one primary-action trigger right-aligned beside it (not
        // stacked below). `below` holds content too big for that slot (the
        // opened accordion form, the batch form, or the multi-button blocker
        // recovery UI) and always renders full-width under the row.
        let message: ReactNode = <>
          <h3 className="break-words text-sm font-medium text-huddleroom-text-primary">{presentation.title}</h3>
          {presentation.explanation !== presentation.title && (() => {
            let explanation = presentation.explanation
            if (row.kind === 'answer' && openAction?.kind === 'answer' && openAction.id === row.decision.id) {
              const question = safeNeedsYouText(row.decision.question, presentation.references)
              if (explanation === question) explanation = ''
              else if (explanation.startsWith(`${question}\n`)) explanation = explanation.slice(question.length + 1)
            }
            return explanation ? <p className="mt-1 whitespace-pre-line break-words text-sm text-huddleroom-text-secondary">{explanation}</p> : null
          })()}
          <p className="mt-1 text-xs text-huddleroom-text-muted"><span className="font-medium">{presentation.actionOwner === 'owner' ? 'Owner' : presentation.actionOwner === 'maintainer' ? 'Maintainer' : 'Owner or maintainer'}:</span> {presentation.nextStep}</p>
        </>
        let tag: ReactNode = null
        let action: ReactNode = null
        let below: ReactNode = null

        if (row.kind === 'blocker') {
          const canRecover = row.blocker.kind === 'goal_definition_clarification_limit' && renderBlockerRecovery
          if (canRecover) below = <div className="mt-2">{renderBlockerRecovery(row.blocker, (element) => registerRowRef(row.key, element), () => focusNextRow(row.key))}</div>
        } else if (row.kind === 'lm-retry') {
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          below = <div className="mt-2 flex justify-end">
            <RetryControl
                buttonRef={(element) => registerRowRef(row.key, element)}
                onRetry={(trigger) => onRetryProcess(row.process.process_type as OrchestrationBaselineProcessType, trigger, () => { markResolved(row.key); successToast('Retry succeeded.'); focusNextRow(row.key) })}
                retryPending={retryPending} retryError={retryError} disabled={mutationBusy || retryPending}
            />
          </div>
        } else if (row.kind === 'answer' && row.decision.decision_key.startsWith('team_hierarchy:agent:')) {
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          below = renderTeamHierarchyProposal(row.decision, () => { releaseEdit(); markResolved(row.key); successToast('Proposal decision received.'); focusNextRow(row.key) }, (element) => registerRowRef(row.key, element), presentation.references, lifecycle, formKey, () => {
            if (retainedDisplaced) {
              releaseEdit()
              focusNextRow(row.key)
              return true
            }
            releaseEdit()
            return false
          }, () => beginEdit(row))
        } else if (row.kind === 'answer') {
          const focusAction: BaselineFocusAction = { kind: 'answer', id: row.decision.id }
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          // Accept-recommendation fast path (spec C): a non-null recommendation
          // gets a primary Accept button (submits immediately, no form, no
          // reason) plus a secondary "Choose differently…" that opens today's
          // full form. A null recommendation keeps the single Answer button.
          const managerComparison = isManagerSelectionDecision(row.decision.decision_key) ? parseManagerComparison(row.decision.context) : null
          if (managerComparison) {
            below = <dl className="mt-2 grid gap-1 rounded border border-huddleroom-border p-2 text-xs text-huddleroom-text-secondary"><div><dt className="inline font-medium text-huddleroom-text-muted">Deterministic: </dt><dd className="inline">{safeNeedsYouText(managerComparison.deterministic, presentation.references)}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">LLM: </dt><dd className="inline">{safeNeedsYouText(managerComparison.llm, presentation.references)}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">Rationale: </dt><dd className="inline">{safeNeedsYouText(managerComparison.rationale, presentation.references)}</dd></div></dl>
            action = <div className="flex shrink-0 items-center gap-2"><Button ref={(element) => registerRowRef(row.key, element)} type="button" className="min-h-11" disabled={mutationBusy || editBlocked} onClick={() => onAcceptDecision(row.decision, () => { releaseEdit(); markResolved(row.key); successToast('Answer received.'); focusNextRow(row.key) })}>Accept: {safeNeedsYouText(recommendationLabel(row.decision), presentation.references)}</Button><Button ref={(element) => onActionTrigger(focusAction, element)} type="button" variant="secondary" className="min-h-11" disabled={mutationBusy || editBlocked} onClick={() => { beginEdit(row); onOpenAction(focusAction) }}>Choose differently…</Button></div>
          } else if (row.decision.recommendation !== null) {
            action = <div className="flex shrink-0 items-center gap-2">
              <Button ref={(element) => registerRowRef(row.key, element)} type="button" className="min-h-11" disabled={mutationBusy || editBlocked} onClick={() => onAcceptDecision(row.decision, () => { releaseEdit(); markResolved(row.key); successToast('Answer received.'); focusNextRow(row.key) })}>
                Accept: {safeNeedsYouText(recommendationLabel(row.decision), presentation.references)}
              </Button>
              <Button ref={(element) => onActionTrigger(focusAction, element)} type="button" variant="secondary" className="min-h-11" disabled={mutationBusy || editBlocked} onClick={() => { beginEdit(row); onOpenAction(focusAction) }}>Choose differently…</Button>
            </div>
          } else {
            action = <Button ref={(element) => registerRow(focusAction, row.key, element)} type="button" className="min-h-11" disabled={mutationBusy || editBlocked} onClick={() => { beginEdit(row); onOpenAction(focusAction) }}>Answer</Button>
          }
          if (openAction?.kind === 'answer' && openAction.id === row.decision.id) {
            below = <div className="mt-1">{renderActionForm(focusAction, row.decision, () => { releaseEdit(); markResolved(row.key); successToast('Answer received.'); focusNextRow(row.key) }, safeNeedsYouText(row.decision.question, presentation.references), presentation.references, lifecycle, () => { releaseEdit(); focusNextRow(row.key) }, formKey)}</div>
          }
        } else if (row.kind === 'answer-batch') {
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          below = <div className="mt-2">{renderAgentDefinitionReviewBatch(row.decisions, () => { releaseEdit(); markResolved(row.key); successToast('Agent-definition review completed.'); focusNextRow(row.key) }, formKey, presentation.references, lifecycle)}</div>
        } else if (row.kind === 'goal-definition-questions') {
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          below = <div className="mt-2">
            <GoalDefinitionQuestionCard key={formKey}
              decisions={row.decisions} activeDecisionId={activeDecisionId} mutationBusy={mutationBusy} reviewLocked={editBlocked}
              onAnswer={(decision, answerText, callbacks) => onAnswerGoalDefinitionQuestion(decision, answerText, {
                ...callbacks, onSuccess: () => { locallyAnsweredIds.current.add(decision.id); callbacks.onSuccess() },
              })}
              displayReferences={presentation.references}
              onActiveDecisionChange={(decisionId) => setActiveDecision(row.key, decisionId)}
              onAllAnswered={() => { releaseEdit(); markResolved(row.key); successToast('Answers received.'); focusNextRow(row.key) }}
            />
          </div>
        } else if (row.kind === 'gate') {
          // Presentation copy already carries the failed criterion and reason.
        } else if (staleInputsProcessType(row.warning.warning_type)) {
          const processType = staleInputsProcessType(row.warning.warning_type)!
          tag = <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          action = <div className="flex shrink-0 items-center gap-2">
            <Button ref={(element) => registerRowRef(row.key, element)} type="button" className="min-h-11" disabled={mutationBusy} onClick={() => onApproveStaleInputs(row.warning, processType, () => { markResolved(row.key); focusNextRow(row.key) })}>Approve</Button>
            <Button type="button" variant="secondary" className="min-h-11" disabled={mutationBusy} onClick={() => onDismissStaleInputs(row.warning, () => { markResolved(row.key); focusNextRow(row.key) })}>Dismiss</Button>
          </div>
        } else {
          const focusAction: BaselineFocusAction = row.canAcknowledge ? { kind: 'acknowledge', id: row.warning.id } : { kind: 'resolve', id: row.warning.id }
          const label = row.canAcknowledge ? 'Acknowledge' : 'Resolve'
          const successFocus = () => { markResolved(row.key); successToast(row.canAcknowledge ? 'Risk acknowledged.' : 'Risk resolved.'); focusNextRow(row.key) }
          tag = row.warning.warning_type === 'start_text_conflict'
            ? <Link to={{ pathname: '/settings', hash: '#project-details' }} className="inline-flex min-h-11 items-center text-xs font-medium text-huddleroom-primary underline-offset-2 hover:underline">Edit project description</Link>
            : <OwningStepTag process={row.process} onSelectProcess={onSelectProcess} />
          action = <Button ref={(element) => registerRow(focusAction, row.key, element)} type="button" variant="secondary" className="min-h-11" disabled={mutationBusy} onClick={() => row.canAcknowledge ? onAcknowledgeWarning(row.warning, successFocus) : onOpenAction(focusAction)}>{label}</Button>
          if (!row.canAcknowledge && openAction?.kind === focusAction.kind && openAction.id === row.warning.id) {
            below = <div className="mt-1">{renderActionForm(focusAction, row.warning, successFocus)}</div>
          }
        }

        return <li key={row.key} ref={(element) => { rowContainers.current[row.key] = element }} className="py-3 first:pt-0 last:pb-0 motion-safe:animate-[queue-row-in_180ms_ease-out]" onClickCapture={() => {
          if (hasInlineDecisionForm(row)) retainRow(row)
        }} onFocusCapture={(event) => {
          if (targetIsInsideForm(event.target, event.currentTarget)) retainRow(row)
        }}>
          <div className="flex flex-wrap items-start gap-3 sm:flex-nowrap">
            <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" style={{ color: eyebrow.color }} />
            {/* Eyebrow reads as the row's key column (Record style: 11px
                uppercase, tracking-wide) — kept in its status color rather
                than muted, since the color is itself the signal (blocker vs
                decision vs warning). Message is the value beside it. */}
            <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-0.5 sm:flex-nowrap">
              <p className="w-[104px] shrink-0 text-label font-semibold uppercase tracking-[0.1em]" style={{ color: eyebrow.color }}>{eyebrow.text}</p>
              <div className="min-w-0 flex-1">
                <div className="text-sm text-huddleroom-text-primary">{message}</div>
                {tag && <div className="mt-0.5">{tag}</div>}
              </div>
            </div>
            {action && <div className="flex shrink-0 items-center">{action}</div>}
          </div>
          {below}
          {reviewStatus}
          <NamedReferences references={presentation.references} activeProjectId={goal.project_id} activeGoalId={goal.id} gates={gates} processes={sourceProcesses ?? processes} sourceActionIds={sourceActionIds} onRevealSource={onRevealSource} />
          <NeedsYouDetails details={presentation.details} />
        </li>
      })}
    </ul>
    {deferredCount > 0 && <p className="mt-3 border-t border-huddleroom-border pt-2 text-xs text-huddleroom-text-muted">{deferredCount} more queued</p>}
  </section>
}
