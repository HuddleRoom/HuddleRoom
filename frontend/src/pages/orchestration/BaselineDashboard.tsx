import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { ApiError } from '@/lib/api-client'
import { useCurrentUser } from '@/api/auth'
import {
  createOrchestrationAnswerSubmission,
  validateSavedAnswerReceipt,
  fetchOrchestrationDecision,
  useAcknowledgeOrchestrationWarning,
  type AgentDefinitionReviewBatchAnswerInput,
  type AgentDefinitionReviewBatchAnswerResult,
  useAnswerAgentDefinitionReviewBatch,
  useAnswerOrchestrationDecision,
  useAuthorizeOrchestrationBaseline,
  useBaselineDashboard,
  useOrchestrationHealth,
  useResolveOrchestrationWarning,
  useRunOrchestrationBaselineStep,
  useSkipOrchestrationProcess,
  useRerunOrchestrationBaselineStep,
  useRetryOrchestrationBaselineStep,
} from '@/api/orchestration'
import { Button, StatusBadge, Textarea } from '@/components/common/uiPrimitives'
import { Record } from '@/components/common/Record'
import { Tag } from '@/components/common/Tag'
import type { AgentCreatePayload } from '@/api/agents'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationBlocker,
  OrchestrationDecisionAnswerResult,
  OrchestrationAnswerReceipt,
  OrchestrationAnswerSubmission,
  OrchestrationAgentReviewRecord,
  OrchestrationDecisionOption,
  OrchestrationGate,
  OrchestrationGoal,
  OrchestrationMeetingCommitment,
  NeedsYouReference,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { BASELINE_PROCESS_TYPES, processLabel, statusPhrase } from './humanize'
import { frontierProcessType, isTerminalProcess, ProcessChain } from './ProcessChain'
import { ProcessFocus } from './ProcessFocus'
import { NeedsYouQueue, type RecoveryAnswerOption } from './NeedsYouQueue'
import { useGoalAnnouncer } from './goalAnnouncer'
import { hierarchyAgentResolver } from './hierarchyAgentNames'
import { HierarchyEvidence } from './HierarchyEvidence'
import {
  safeNeedsYouText,
  validateNeedsYouEdit,
  validateRuntimeAnswerReceipt,
  validateOrdinaryAnswerResult,
  type NeedsYouEditSnapshot,
  type NeedsYouSourceTarget,
} from './needsYouPresentation'

export type BaselineMutationOwner =
  `baseline:${'answer' | 'skip' | 'acknowledge' | 'resolve' | 'run' | 'rerun' | 'retry' | 'authorize'}:${string}`

export type BaselineFocusAction =
  | { kind: 'answer'; id: string }
  | { kind: 'skip'; id: string }
  | { kind: 'acknowledge'; id: string }
  | { kind: 'resolve'; id: string }

type ActionTarget =
  | { kind: 'answer'; action: Extract<BaselineFocusAction, { kind: 'answer' }>; decision: OrchestrationAuthorityDecisionRecord }
  | { kind: 'skip'; action: Extract<BaselineFocusAction, { kind: 'skip' }>; process: OrchestrationProcessRunRecord }
  | { kind: 'acknowledge' | 'resolve'; action: Extract<BaselineFocusAction, { kind: 'acknowledge' | 'resolve' }>; warning: OrchestrationWarningRecord }

export interface BaselineDashboardProps {
  projectId: string
  goalId: string
  goal: OrchestrationGoal
  run: OrchestrationRun | null
  // Fetched by the goal-detail query (not useBaselineDashboard) — threaded in
  // so the Needs-you queue can rank failed gates without a duplicate fetch.
  meetingCommitments?: readonly OrchestrationMeetingCommitment[]
  needsYouReferences?: readonly NeedsYouReference[]
  sourceActionIds?: readonly string[]
  gates: readonly OrchestrationGate[]
  mutationBusy: boolean
  acquireMutation: (owner: BaselineMutationOwner) => boolean
  releaseMutation: (owner: BaselineMutationOwner) => void
  getMutationGeneration?: () => number
  onMessage: (message: string) => void
  // Reports the currently-effective selected step whenever it changes, so the
  // page can keep the Details tabs' Activity "This step" scope synced to the
  // stepper selection without lifting the selection state itself. Optional —
  // existing callers/tests that don't need Activity sync keep working.
  onSelectedProcessTypeChange?: (type: OrchestrationBaselineProcessType) => void
  // Switches the Details tabs to Activity, scoped to this step (threaded from
  // ProcessFocus's "View activity for this step" control).
  onViewActivity?: () => void
  onRevealSource?: (target: NeedsYouSourceTarget) => void
  suppressQueueAnnouncement?: boolean
  queueAnnouncementScope?: 'all' | 'human' | 'none'
  // The goal_definition_clarification_limit recovery action (Proceed / Ask
  // another round) is owned by the page (shares its mutation-owner
  // coordination with lifecycle/gate-override controls) and rendered here as
  // a render prop, mirroring renderActionForm/renderAgentDefinitionReviewBatch.
  renderBlockerRecovery?: (blocker: OrchestrationBlocker, register: (element: HTMLButtonElement | null) => void, onSuccessFocus?: () => void) => ReactNode
}

export function currentProcesses(processes: readonly OrchestrationProcessRunRecord[] | null | undefined) {
  return (processes ?? []).filter((process) => process.superseded_by_id === null)
}

export function decisionOptionKey(option: OrchestrationDecisionOption | unknown) {
  if (typeof option === 'string') return option.trim() ? option : null
  return typeof option === 'object' && option !== null && !Array.isArray(option)
    && typeof (option as Record<string, unknown>).key === 'string'
    && (option as { key: string }).key.trim()
    ? (option as { key: string }).key
    : null
}

export function normalizeDecisionOption(option: OrchestrationDecisionOption | unknown) {
  const key = decisionOptionKey(option)
  if (typeof option === 'string' && key !== null) return { key, label: option }
  if (key === null || typeof option !== 'object' || option === null || Array.isArray(option)) return null
  const record = option as Record<string, unknown>
  return { key, label: typeof record.label === 'string' ? record.label : typeof record.description === 'string' ? record.description : key }
}

export function canSubmitReason(reason: string) {
  return reason.trim().length > 0
}

export function decisionRequiresReason(selectedOption: string | null, recommendation: string | null) {
  return recommendation !== null && selectedOption !== null && selectedOption !== recommendation
}

export function canSubmitDecision(selectedOption: string | null, recommendation: string | null, reason: string) {
  return selectedOption !== null && selectedOption.trim().length > 0
    && (!decisionRequiresReason(selectedOption, recommendation) || canSubmitReason(reason))
}

function actionError(error: unknown) {
  if (error instanceof ApiError && error.detail && typeof error.detail === 'object' && 'message' in error.detail && typeof error.detail.message === 'string') return error.detail.message
  return error instanceof Error ? error.message : 'The request failed. Check your connection and try again.'
}

function parseReviewContext(context: string | null) {
  if (!context) return null
  try {
    const parsed = JSON.parse(context)
    if (parsed && typeof parsed === 'object' && ('proposed_description' in parsed || 'proposed_persona' in parsed)) return parsed as Record<string, unknown>
  } catch { /* not JSON */ }
  return null
}

export function parseManagerComparison(context: string | null) {
  const match = /^Deterministic:\s*(.+)\nLLM:\s*(.+)\nRationale:\s*([\s\S]+)$/.exec(context ?? '')
  return match ? { deterministic: match[1], llm: match[2], rationale: match[3] } : null
}

export function isManagerSelectionDecision(decisionKey: string) {
  return decisionKey.startsWith('manager_selection:')
}

function actionOwner(action: BaselineFocusAction): BaselineMutationOwner {
  return `baseline:${action.kind}:${action.id}`
}

function canAnswerDecision(decision: OrchestrationAuthorityDecisionRecord) {
  return decision.status === 'pending' && decision.authority === 'human'
}

export function BaselineActionForm({
  target,
  mutationBusy,
  submitting,
  selectedOption,
  reason,
  error,
  firstOptionRef,
  reasonRef,
  submitRef,
  onAnswer,
  onSkip,
  onAcknowledge,
  onResolve,
  onSelectedOptionChange,
  onReasonChange,
  onCancel,
  displayQuestion,
  displayReferences = [],
  editState = 'current',
  onReleaseEdit,
}: {
  target: ActionTarget
  mutationBusy: boolean
  submitting: boolean
  selectedOption: string | null
  reason: string
  error: string | null
  firstOptionRef: React.RefObject<HTMLInputElement | null>
  reasonRef: React.RefObject<HTMLTextAreaElement | null>
  submitRef: React.RefObject<HTMLButtonElement | null>
  onAnswer: (decision: OrchestrationAuthorityDecisionRecord, selectedOption: string, reason: string) => void
  onSkip: (process: OrchestrationProcessRunRecord, reason: string) => void
  onAcknowledge: (warning: OrchestrationWarningRecord, reason: string) => void
  onResolve: (warning: OrchestrationWarningRecord, reason: string) => void
  onSelectedOptionChange: (option: string) => void
  onReasonChange: (reason: string) => void
  onCancel: () => void
  displayQuestion?: string
  displayReferences?: readonly NeedsYouReference[]
  editState?: 'current' | 'terminal' | 'stale' | 'unavailable'
  onReleaseEdit?: () => void
}) {
  const disabled = mutationBusy || submitting
  const reviewLocked = editState !== 'current'
  const options = target.kind === 'answer' ? target.decision.options.map((option) => ({ option, key: decisionOptionKey(option) })) : []
  const freeText = target.kind === 'answer' && options.length === 0
  const runtimeDecision = target.kind === 'answer' && Boolean(target.decision.runtime_identity)
  const textAnswer = runtimeDecision && options.length === 1 && options[0].key === 'acknowledge'
  // Match Python str.strip(): JavaScript trim removes FEFF and misses U+0085.
  const normalizedReason = runtimeDecision
    ? reason.replace(/^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+|[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+$/g, '')
    : reason
  const reasonTooLong = runtimeDecision && [...normalizedReason].length > 4000
  const reviewContext = target.kind === 'answer' ? parseReviewContext(target.decision.context) : null
  const managerComparison = target.kind === 'answer' && isManagerSelectionDecision(target.decision.decision_key) ? parseManagerComparison(target.decision.context) : null
  const answerRef = useRef<HTMLTextAreaElement>(null)
  const firstAnswerOption = options.findIndex((option) => option.key !== null)
  const requiresReason = textAnswer || target.kind === 'skip' || target.kind === 'answer' && decisionRequiresReason(selectedOption, target.decision.recommendation)
  const canSubmit = target.kind === 'answer'
    ? !reasonTooLong && (textAnswer ? normalizedReason.length > 0 : Boolean(selectedOption?.trim()) && (!requiresReason || (runtimeDecision ? normalizedReason.length > 0 : canSubmitReason(reason))))
    : target.kind === 'skip' ? canSubmitReason(reason) : true
  const label = target.kind === 'answer' ? textAnswer ? 'Answer' : 'Submit answer' : target.kind === 'skip' ? 'Skip process' : target.kind === 'acknowledge' ? 'Acknowledge risk' : 'Resolve risk'
  const { announceError } = useGoalAnnouncer()
  const display = (value: string) => safeNeedsYouText(value, displayReferences)

  useEffect(() => { if (error) announceError(error) }, [error, announceError])
  useEffect(() => {
    if (target.kind === 'answer' && freeText && answerRef.current) answerRef.current.focus()
    else if (target.kind === 'answer' && !textAnswer && firstOptionRef.current) firstOptionRef.current.focus()
    else reasonRef.current?.focus()
  }, [target.kind, target.action.id, freeText, textAnswer, firstOptionRef, reasonRef])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (reviewLocked) return
    if (target.kind === 'answer') {
      if (canSubmit) onAnswer(target.decision, textAnswer ? 'acknowledge' : selectedOption!, normalizedReason)
      return
    }
    if (target.kind === 'skip' && !canSubmitReason(reason)) return
    const warningReason = reason.trim() || 'Acknowledged by operator'
    if (target.kind === 'skip') onSkip(target.process, reason)
    else if (target.kind === 'acknowledge') onAcknowledge(target.warning, warningReason)
    else onResolve(target.warning, warningReason)
  }

  return (
    <form onSubmit={submit} aria-busy={submitting} aria-disabled={reviewLocked || undefined} className="mt-3 border-t border-huddleroom-border pt-3">
      <fieldset disabled={disabled}>
        {target.kind === 'answer' && <>
          <legend className="text-xs font-medium text-huddleroom-text-muted">Answer decision</legend>
          <p className="mt-2 text-sm font-medium text-huddleroom-text-primary">{displayQuestion ?? target.decision.question}</p>
          {target.decision.recommendation !== null && <p className="mt-2 text-xs text-huddleroom-text-secondary">Recommendation: {display(target.decision.recommendation)}</p>}
          {managerComparison && <dl className="mt-2 grid gap-1 rounded border border-huddleroom-border p-2 text-xs text-huddleroom-text-secondary"><div><dt className="inline font-medium text-huddleroom-text-muted">Deterministic: </dt><dd className="inline">{display(managerComparison.deterministic)}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">LLM: </dt><dd className="inline">{display(managerComparison.llm)}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">Rationale: </dt><dd className="inline">{display(managerComparison.rationale)}</dd></div></dl>}
          {(() => {
            if (!reviewContext) return null
            const reason = typeof reviewContext.reason === 'string' && reviewContext.reason.trim() ? reviewContext.reason : null
            const description = typeof reviewContext.original_description === 'string' ? reviewContext.original_description : null
            const proposedDescription = typeof reviewContext.proposed_description === 'string' ? reviewContext.proposed_description : null
            const persona = typeof reviewContext.original_persona === 'string' ? reviewContext.original_persona : null
            const proposedPersona = typeof reviewContext.proposed_persona === 'string' ? reviewContext.proposed_persona : null
            const problems = Array.isArray(reviewContext.problems) ? reviewContext.problems.filter((p): p is string => typeof p === 'string') : []
            const hasContent = reason || (description && proposedDescription) || (persona && proposedPersona) || problems.length > 0
            if (!hasContent) return null
            return <div className="mt-2 border-t border-huddleroom-border pt-2">
              {reason && <p className="text-sm text-huddleroom-text-secondary">{display(reason)}</p>}
              {description && proposedDescription && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Description</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Original: {display(description)}</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {display(proposedDescription)}</p>
              </div>}
              {persona && proposedPersona && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Persona</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Original: {display(persona)}</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {display(proposedPersona)}</p>
              </div>}
              {problems.length > 0 && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Problems</p>
                <ul className="mt-1 space-y-1">
                  {problems.map((problem, index) => <li key={index} className="text-xs text-huddleroom-text-secondary">• {display(problem)}</li>)}
                </ul>
              </div>}
            </div>
          })()}
          {textAnswer ? <Textarea ref={reasonRef} label="Answer" value={reason} readOnly={reviewLocked || submitting} onChange={(event) => onReasonChange(event.target.value)} rows={3} required /> : freeText ? <Textarea ref={answerRef} label="Answer" value={selectedOption ?? ''} readOnly={reviewLocked || submitting} onChange={(event) => onSelectedOptionChange(event.target.value)} rows={3} /> : (
            <div className="mt-2 space-y-2">
              {options.map(({ option, key }, index) => key === null ? <p key={index} className="text-sm text-huddleroom-text-secondary">Unavailable option</p> : (
                <label key={key} className="flex min-h-11 items-center gap-2 text-sm text-huddleroom-text-primary">
                  <input ref={index === firstAnswerOption ? firstOptionRef : undefined} type="radio" name={`decision-${target.decision.id}`} value={key} checked={selectedOption === key} disabled={reviewLocked} onChange={() => onSelectedOptionChange(key)} />
                  {display(normalizeDecisionOption(option)?.label ?? key)}
                </label>
              ))}
            </div>
          )}
        </>}
        {target.kind === 'answer' && !textAnswer && <Textarea ref={reasonRef} label={target.decision.recommendation === null ? 'Reason (optional)' : 'Reason (required when overriding the recommendation)'} value={reason} readOnly={reviewLocked || submitting} onChange={(event) => onReasonChange(event.target.value)} rows={3} required={requiresReason} />}
        {target.kind !== 'answer' && <Textarea ref={reasonRef} label={target.kind === 'skip' ? 'Reason (required)' : 'Reason (optional)'} value={reason} onChange={(event) => onReasonChange(event.target.value)} rows={3} required={requiresReason} />}
        {reasonTooLong && <p role="alert" className="mt-2 text-xs text-huddleroom-status-red">Answer or reason must be at most 4,000 Unicode characters.</p>}
        {error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
        <div className="mt-3 flex flex-wrap justify-end gap-2">
          <Button type="button" variant="secondary" className="min-h-11" disabled={mutationBusy || submitting} onClick={() => { onReleaseEdit?.(); onCancel() }}>Cancel</Button>
          <Button ref={submitRef} type="submit" className="min-h-11" disabled={disabled || reviewLocked || !canSubmit}>{submitting ? 'Submitting…' : label}</Button>
        </div>
      </fieldset>
    </form>
  )
}

type AgentDefinitionReviewAnswerState = {
  // Final decided value for this row — once non-null the row is done and its
  // controls are replaced with a status line. Null means still pending.
  selectedOption: string | null
  // In-progress radio pick for the reject/edit path, before the row's own
  // Confirm button turns it into `selectedOption` (SPR #87: approve is a
  // single click; reject/edit still need a reason/description first).
  choice: 'reject' | 'edit' | null
  reason: string
  editedDescription: string
  editedPersona: string
}

// The backend's batch-answer endpoint requires the full set of pending
// proposal ids in one request (409 "pending proposal set changed" otherwise)
// — there is no partial-submit endpoint. So a per-row Approve doesn't post by
// itself; it records that row's decision locally and, once every row in the
// batch has one, submits them all in a single request. The last row decided
// (or "Approve all remaining") is what actually triggers the POST.
function agentDefinitionApproveOption(decision: OrchestrationAuthorityDecisionRecord) {
  const approve = decision.options.find((option) => decisionOptionKey(option) === 'approve')
  return approve ? decisionOptionKey(approve)! : 'approve'
}

function AgentDefinitionReviewBatchForm({
  decisions,
  mutationBusy,
  submitting,
  error,
  onSubmit,
  displayReferences,
  reviewLocked = false,
}: {
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  mutationBusy: boolean
  submitting: boolean
  error: string | null
  onSubmit: (answers: AgentDefinitionReviewBatchAnswerInput[]) => void
  displayReferences: readonly NeedsYouReference[]
  reviewLocked?: boolean
}) {
  const [order] = useState(() => decisions.map((decision) => decision.id))
  const currentById = new Map(decisions.map((decision) => [decision.id, decision] as const))
  const orderedDecisions = [
    ...order.flatMap((id) => currentById.get(id) ? [currentById.get(id)!] : []),
    ...decisions.filter((decision) => !order.includes(decision.id)),
  ]
  const [answers, setAnswers] = useState<Record<string, AgentDefinitionReviewAnswerState>>(() => Object.fromEntries(
    decisions.map((decision) => {
      const context = parseReviewContext(decision.context)
      return [decision.id, {
        selectedOption: null,
        choice: null,
        reason: '',
        editedDescription: typeof context?.proposed_description === 'string' ? context.proposed_description : '',
        editedPersona: typeof context?.proposed_persona === 'string' ? context.proposed_persona : '',
      }]
    }),
  ))
  const rowRefs = useRef<Record<string, HTMLButtonElement | null>>({})
  const formRef = useRef<HTMLDivElement>(null)
  const disabled = mutationBusy || submitting
  const display = (value: string) => safeNeedsYouText(value, displayReferences)
  const { announceError } = useGoalAnnouncer()
  useEffect(() => { if (error) announceError(error) }, [error, announceError])
  const update = (decisionId: string, changes: Partial<AgentDefinitionReviewAnswerState>) => {
    if (reviewLocked) return
    setAnswers((current) => ({ ...current, [decisionId]: { ...current[decisionId], ...changes } }))
  }
  const canSubmit = (candidate: Record<string, AgentDefinitionReviewAnswerState>) => orderedDecisions.every((decision) => {
    const answer = candidate[decision.id]
    return answer?.selectedOption && (answer.selectedOption !== 'edit'
      || (answer.editedDescription.trim() && answer.editedPersona.trim()))
  })
  const submitIfComplete = (candidate: Record<string, AgentDefinitionReviewAnswerState>) => {
    if (reviewLocked || !canSubmit(candidate)) return
    onSubmit(orderedDecisions.map((decision) => {
      const { selectedOption, reason, editedDescription, editedPersona } = candidate[decision.id]
      return { decisionId: decision.id, selectedOption: selectedOption!, reason, editedDescription, editedPersona }
    }))
  }
  const focusNextControl = (nextState: Record<string, AgentDefinitionReviewAnswerState>, decisionIdToSkip?: string) => {
    const nextRow = orderedDecisions.find((d) => d.id !== decisionIdToSkip && nextState[d.id]?.selectedOption === null)
    if (nextRow) {
      queueMicrotask(() => rowRefs.current[nextRow.id]?.focus())
    } else {
      queueMicrotask(() => formRef.current?.focus())
    }
  }
  // Records one row's decision and, once every row in the batch has one,
  // submits — this is the only place a row actually finalizes.
  const decideAndMaybeSubmit = (decisionId: string, changes: Partial<AgentDefinitionReviewAnswerState>) => {
    if (reviewLocked) return
    const next = { ...answers, [decisionId]: { ...answers[decisionId], ...changes } }
    setAnswers(next)
    focusNextControl(next, decisionId)
    submitIfComplete(next)
  }
  const approveAllRemaining = () => {
    if (reviewLocked) return
    const next = { ...answers }
    for (const decision of orderedDecisions) {
      if (next[decision.id].selectedOption === null) next[decision.id] = { ...next[decision.id], selectedOption: agentDefinitionApproveOption(decision) }
    }
    setAnswers(next)
    focusNextControl(next)
    submitIfComplete(next)
  }
  const remaining = orderedDecisions.filter((decision) => answers[decision.id]?.selectedOption === null)
  const allComplete = remaining.length === 0

  return <div ref={formRef} aria-busy={submitting} aria-disabled={reviewLocked || undefined} className="mt-2 divide-y divide-huddleroom-border">
    <fieldset disabled={disabled} className="contents">
      {remaining.length > 1 && <div className="flex justify-end py-2">
        <Button type="button" variant="secondary" className="min-h-11" disabled={disabled || reviewLocked} onClick={approveAllRemaining}>
          Approve all remaining ({remaining.length})
        </Button>
      </div>}
      {orderedDecisions.map((decision) => {
        const answer = answers[decision.id]
        const context = parseReviewContext(decision.context)
        const hasDescription = context !== null && Object.prototype.hasOwnProperty.call(context, 'original_description')
        const hasPersona = context !== null && Object.prototype.hasOwnProperty.call(context, 'original_persona')
        const description = typeof context?.original_description === 'string' ? context.original_description : null
        const proposedDescription = typeof context?.proposed_description === 'string' ? context.proposed_description : null
        const persona = typeof context?.original_persona === 'string' ? context.original_persona : null
        const proposedPersona = typeof context?.proposed_persona === 'string' ? context.proposed_persona : null
        const decided = answer?.selectedOption !== null
        const decidedLabel = answer?.selectedOption === 'approve' ? 'Approved' : answer?.selectedOption === 'reject' ? 'Rejected' : answer?.selectedOption === 'edit' ? 'Edited' : null
        return <article key={decision.id} className="py-3 first:pt-0">
          <p className="text-sm font-medium text-huddleroom-text-primary">{display(decision.title || decision.question)}</p>
          {decision.title && <p className="mt-1 text-sm text-huddleroom-text-secondary">{display(decision.question)}</p>}
          {hasDescription && proposedDescription !== null && <div className="mt-2">
            <p className="text-xs font-medium text-huddleroom-text-muted">Description</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Current: {description ? display(description) : '(empty)'}</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedDescription ? display(proposedDescription) : '(empty)'}</p>
          </div>}
          {hasPersona && proposedPersona !== null && <div className="mt-2">
            <p className="text-xs font-medium text-huddleroom-text-muted">System prompt</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Current: {persona ? display(persona) : '(empty)'}</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedPersona ? display(proposedPersona) : '(empty)'}</p>
          </div>}
          {decided ? (
            <p className="mt-2 text-sm font-medium text-huddleroom-text-secondary">{decidedLabel}{submitting ? ' — submitting…' : ''}</p>
          ) : <>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button ref={(el) => { rowRefs.current[decision.id] = el }} data-needs-you-primary type="button" className="min-h-11" disabled={disabled || reviewLocked} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: agentDefinitionApproveOption(decision) })}>Approve</Button>
              <Button type="button" variant="secondary" className="min-h-11" aria-pressed={answer.choice === 'reject'} disabled={disabled || reviewLocked} onClick={() => update(decision.id, { choice: answer.choice === 'reject' ? null : 'reject' })}>Reject</Button>
              <Button type="button" variant="secondary" className="min-h-11" aria-pressed={answer.choice === 'edit'} disabled={disabled || reviewLocked} onClick={() => update(decision.id, { choice: answer.choice === 'edit' ? null : 'edit' })}>Edit</Button>
            </div>
            {answer.choice === 'reject' && <div className="mt-2">
              <Textarea label="Reason (required)" value={answer.reason} readOnly={reviewLocked || submitting} onChange={(event) => update(decision.id, { reason: event.target.value })} rows={2} required />
              <div className="mt-2 flex justify-end">
                <Button type="button" className="min-h-11" disabled={disabled || reviewLocked || !canSubmitReason(answer.reason)} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: 'reject' })}>Confirm reject</Button>
              </div>
            </div>}
            {answer.choice === 'edit' && <div className="mt-2">
              <Textarea label="Description" value={answer.editedDescription} readOnly={reviewLocked || submitting} onChange={(event) => update(decision.id, { editedDescription: event.target.value })} rows={3} required />
              <Textarea label="Persona" value={answer.editedPersona} readOnly={reviewLocked || submitting} onChange={(event) => update(decision.id, { editedPersona: event.target.value })} rows={3} required />
              <div className="mt-2 flex justify-end">
                <Button type="button" className="min-h-11" disabled={disabled || reviewLocked || !answer.editedDescription.trim() || !answer.editedPersona.trim()} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: 'edit' })}>Confirm edit</Button>
              </div>
            </div>}
          </>}
        </article>
      })}
      {error && <div className="mt-2">
        <p className="text-xs text-huddleroom-status-red">{error}</p>
        {allComplete && <Button type="button" className="mt-2 min-h-11" disabled={disabled || reviewLocked} onClick={() => submitIfComplete(answers)}>Retry submit</Button>}
      </div>}
    </fieldset>
  </div>
}

function preferredProcessType(processes: readonly OrchestrationProcessRunRecord[]) {
  const selected = processes.find((process) => process.status === 'waiting_decision')
    ?? processes.find((process) => process.status === 'running')
    ?? processes[0]
  return selected && BASELINE_PROCESS_TYPES.includes(selected.process_type as OrchestrationBaselineProcessType)
    ? selected.process_type as OrchestrationBaselineProcessType
    : 'goal_definition'
}

function proposalDefinition(decision: OrchestrationAuthorityDecisionRecord): AgentCreatePayload | null {
  let context: Record<string, unknown> | null = null
  try {
    const parsed = JSON.parse(decision.context ?? '')
    if (parsed && typeof parsed === 'object') context = parsed as Record<string, unknown>
  } catch { return null }
  const value = context?.proposal
  if (!value || typeof value !== 'object' || !('definition' in value)) return null
  const definition = (value as { definition?: unknown }).definition
  return definition && typeof definition === 'object' ? definition as AgentCreatePayload : null
}

function TeamHierarchyProposalForm({ decision, disabled, reviewLocked = false, submitting, error, onSubmit, onCancelEdit, onBeginEdit, primaryRef, displayReferences }: {
  decision: OrchestrationAuthorityDecisionRecord
  disabled: boolean
  reviewLocked?: boolean
  submitting: boolean
  error: string | null
  onSubmit: (option: string, editedAgent?: AgentCreatePayload) => void
  onCancelEdit?: () => boolean
  onBeginEdit?: () => void
  primaryRef?: (element: HTMLButtonElement | null) => void
  displayReferences: readonly NeedsYouReference[]
}) {
  const definition = proposalDefinition(decision)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(() => ({ ...definition, capabilities: definition?.capabilities ?? [], config: definition?.config ?? {} }))
  const [config, setConfig] = useState(() => JSON.stringify(definition?.config ?? {}, null, 2))
  const [validation, setValidation] = useState<string | null>(null)
  const editRef = useRef<HTMLButtonElement>(null)
  const { announceError } = useGoalAnnouncer()
  const display = (value: string) => safeNeedsYouText(value, displayReferences)
  useEffect(() => {
    if (!definition) { announceError('This proposal is missing its agent definition.'); return }
    const message = editing ? (validation || error) : error
    if (message) announceError(message)
  }, [definition, editing, validation, error, announceError])
  if (!definition) return <p className="mt-2 text-xs text-huddleroom-status-red">This proposal is missing its agent definition.</p>

  function submitEdit(event: FormEvent) {
    event.preventDefault()
    if (reviewLocked) return
    let parsed: unknown
    try { parsed = JSON.parse(config) } catch { setValidation('Config must be valid JSON.'); return }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') { setValidation('Config must be a JSON object.'); return }
    setValidation(null)
    onSubmit('edit', { ...draft, config: parsed as Record<string, unknown> } as AgentCreatePayload)
  }

  const field = (label: string, key: keyof AgentCreatePayload, required = false) => <label className="block text-xs text-huddleroom-text-secondary">{label}
    <input className="mt-1 min-h-11 w-full rounded border border-huddleroom-border bg-huddleroom-surface px-2 text-sm text-huddleroom-text-primary" required={required} readOnly={reviewLocked || submitting} value={String(draft[key] ?? '')} onChange={(event) => setDraft((current) => ({ ...current, [key]: event.target.value }))} />
  </label>
  const capabilities = Array.isArray(definition.capabilities)
    ? definition.capabilities.filter((capability): capability is string => typeof capability === 'string' && capability.trim().length > 0)
    : []
  return <div className="mt-2" aria-busy={submitting} aria-disabled={reviewLocked || undefined}>
    <Record keyWidth={84} rows={[
      { key: 'Role', value: definition.role ? <span className="font-semibold">{display(definition.role)}</span> : null },
      { key: 'Name', value: definition.name ? display(definition.name) : null },
      { key: 'Model', value: definition.model ? display(definition.model) : null },
      { key: 'Provider', value: definition.provider ? display(definition.provider) : null },
      { key: 'Adapter', value: definition.adapter_type ? display(definition.adapter_type) : null },
      { key: 'Description', value: definition.description ? <span className="block truncate" title={display(definition.description)}>{display(definition.description)}</span> : null },
    ]} />
    {capabilities.length > 0 && <div className="mt-2 flex flex-wrap gap-1">
      {capabilities.map((capability) => <Tag key={capability}>{display(capability)}</Tag>)}
    </div>}
    {definition.system_prompt && <details className="mt-2">
      <summary className="cursor-pointer text-xs font-medium text-huddleroom-text-muted hover:text-huddleroom-text-secondary">Show prompt</summary>
      <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs text-huddleroom-text-secondary">{display(definition.system_prompt)}</pre>
    </details>}
    {!editing ? <div className="mt-3 flex flex-wrap gap-2">
      <Button ref={primaryRef} data-needs-you-primary type="button" disabled={disabled || reviewLocked} onClick={() => onSubmit('approve')}>Approve</Button>
      <Button ref={editRef} type="button" variant="secondary" disabled={disabled || reviewLocked} onClick={() => { onBeginEdit?.(); setEditing(true) }}>Edit</Button>
      <Button type="button" variant="secondary" disabled={disabled || reviewLocked} onClick={() => onSubmit('reject')}>Reject</Button>
    </div> : <form className="mt-3 grid gap-2" onSubmit={submitEdit}>
      {field('Name', 'name', true)}{field('Role', 'role', true)}{field('Description', 'description')}{field('Provider', 'provider', true)}{field('Model', 'model', true)}
      <label className="block text-xs text-huddleroom-text-secondary">Adapter type<select className="mt-1 min-h-11 w-full rounded border border-huddleroom-border bg-huddleroom-surface px-2" disabled={reviewLocked} value={draft.adapter_type ?? 'api'} onChange={(event) => setDraft((current) => ({ ...current, adapter_type: event.target.value as AgentCreatePayload['adapter_type'] }))}><option value="api">api</option><option value="cli">cli</option><option value="routine">routine</option></select></label>
      {field('CLI runtime', 'cli_runtime')}
      <Textarea label="Capabilities (one per line)" value={(draft.capabilities ?? []).join('\n')} readOnly={reviewLocked || submitting} onChange={(event) => setDraft((current) => ({ ...current, capabilities: event.target.value.split('\n').filter(Boolean) }))} rows={3} />
      <Textarea label="System prompt" value={draft.system_prompt ?? ''} readOnly={reviewLocked || submitting} onChange={(event) => setDraft((current) => ({ ...current, system_prompt: event.target.value }))} rows={8} />
      <Textarea label="Config (JSON object)" value={config} readOnly={reviewLocked || submitting} onChange={(event) => setConfig(event.target.value)} rows={5} />
      {(validation || error) && <p className="text-xs text-huddleroom-status-red">{validation || error}</p>}
      <div className="flex justify-end gap-2"><Button type="button" variant="secondary" disabled={disabled} onClick={() => { setEditing(false); setDraft({ ...definition, capabilities: definition.capabilities ?? [], config: definition.config ?? {} }); setConfig(JSON.stringify(definition.config ?? {}, null, 2)); setValidation(null); if (!onCancelEdit?.()) queueMicrotask(() => editRef.current?.focus()) }}>Cancel</Button><Button type="submit" disabled={disabled || reviewLocked}>{submitting ? 'Submitting…' : 'Submit edit'}</Button></div>
    </form>}
    {!editing && error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
  </div>
}

function TeamHierarchyStatus({ decisions, process, reviews }: { decisions: readonly OrchestrationAuthorityDecisionRecord[]; process: OrchestrationProcessRunRecord | undefined; reviews: readonly OrchestrationAgentReviewRecord[] }) {
  const processId = process?.id ?? null
  const current = processId ? decisions.filter((item) => item.source_process_run_id === processId) : []
  const pending = current.filter((item) => item.status === 'pending' && item.decision_key.startsWith('team_hierarchy:agent:'))
  const approval = current.find((item) => item.status === 'pending' && item.decision_key === 'team_hierarchy:approval')
  const { announceQueue } = useGoalAnnouncer()
  useEffect(() => {
    if (pending.length && !approval) announceQueue(`Waiting for ${pending.length} proposed agent decision${pending.length === 1 ? '' : 's'} before hierarchy approval.`)
  }, [pending.length, approval, announceQueue])
  if (pending.length && !approval) return <p className="rounded border border-huddleroom-border p-3 text-sm text-huddleroom-text-secondary">Waiting for {pending.length} proposed agent decision{pending.length === 1 ? '' : 's'} before hierarchy approval.</p>
  const option = approval?.options.find((candidate) => typeof candidate === 'object' && candidate !== null && 'proposal' in candidate) as { proposal?: Record<string, unknown> } | undefined
  const proposal = option?.proposal
  if (!proposal) return null
  const approvalOption = (option as Record<string, unknown> | undefined)?.key === 'approve_with_documented_gaps'
    ? 'Approve with documented gaps'
    : null
  const outputs = process && typeof process.outputs === 'object' && process.outputs !== null && !Array.isArray(process.outputs) ? process.outputs as Record<string, unknown> : {}
  const revisionContext = typeof outputs.revision_context === 'object' && outputs.revision_context !== null && !Array.isArray(outputs.revision_context) ? outputs.revision_context as Record<string, unknown> : null
  const approvedUnused = Array.isArray(proposal.approved_but_unused_agent_ids) ? proposal.approved_but_unused_agent_ids : []
  const resolveAgent = hierarchyAgentResolver(reviews)
  return <section className="rounded border border-huddleroom-border p-3"><h3 className="text-sm font-medium text-huddleroom-text-primary">Proposed hierarchy</h3>
    <HierarchyEvidence proposal={proposal} reviews={reviews} revisionContext={revisionContext} />
    {approvalOption && <p className="mt-2 text-xs"><span className="font-medium text-huddleroom-text-muted">Approval:</span> {approvalOption}</p>}
    <p className="mt-2 text-xs"><span className="font-medium text-huddleroom-text-muted">Rationale:</span> {String(proposal.rationale ?? 'None')}</p><p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Self-review:</span> {String(proposal.self_review ?? 'None')}</p>
    <p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Documented gaps:</span> {Array.isArray(proposal.documented_gaps) && proposal.documented_gaps.length ? proposal.documented_gaps.join(', ') : 'None'}</p>
    <p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Approved but unused:</span> {approvedUnused.length ? approvedUnused.map((agent) => resolveAgent(agent) ?? 'Unknown agent').join(', ') : 'None'}</p>
  </section>
}

export function BaselineDashboard({ projectId, goalId, goal, run, gates, meetingCommitments = [], needsYouReferences = [], sourceActionIds = [], mutationBusy, acquireMutation, releaseMutation, onMessage, onSelectedProcessTypeChange, onViewActivity, onRevealSource, renderBlockerRecovery, suppressQueueAnnouncement, queueAnnouncementScope, getMutationGeneration }: BaselineDashboardProps) {
  const health = useOrchestrationHealth()
  const currentUser = useCurrentUser()
  const answer = useAnswerOrchestrationDecision(projectId)
  const answerAgentDefinitionBatch = useAnswerAgentDefinitionReviewBatch(projectId)
  const skip = useSkipOrchestrationProcess(projectId)
  const acknowledge = useAcknowledgeOrchestrationWarning(projectId)
  const resolve = useResolveOrchestrationWarning(projectId)
  const runStep = useRunOrchestrationBaselineStep(projectId)
  const rerun = useRerunOrchestrationBaselineStep(projectId)
  const retry = useRetryOrchestrationBaselineStep(projectId)
  const authorize = useAuthorizeOrchestrationBaseline(projectId)
  const dashboard = useBaselineDashboard(projectId, goalId, runStep.isPending || rerun.isPending || retry.isPending || authorize.isPending)
  const localGeneration = useRef(0)
  const generation = () => getMutationGeneration?.() ?? localGeneration.current
  function beginMutation(owner: BaselineMutationOwner) {
    if (!acquireMutation(owner)) return null
    localGeneration.current++
    const started = generation()
    return () => generation() === started
  }
  function guardedMicrotask(callback: () => void) {
    const started = generation()
    queueMicrotask(() => { if (generation() === started) callback() })
  }
  const [selectedProcessType, setSelectedProcessType] = useState<OrchestrationBaselineProcessType | null>(null)
  const [openAction, setOpenAction] = useState<BaselineFocusAction | null>(null)
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [reason, setReason] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [formErrorSource, setFormErrorSource] = useState<'action' | 'batch' | null>(null)
  const [invalidatedDecisions, setInvalidatedDecisions] = useState<Record<string, 'terminal' | 'stale' | 'unavailable'>>({})
  type FrozenAnswer = {
    submission: OrchestrationAnswerSubmission
    reviewed: OrchestrationAuthorityDecisionRecord[]
    owner: BaselineMutationOwner
    finish: () => void
    scope: string
  }
  const [savedAnswers, setSavedAnswers] = useState<OrchestrationAnswerReceipt[]>([])
  const [checkingAnswer, setCheckingAnswer] = useState<FrozenAnswer | null>(null)
  const [ordinarySubmitting, setOrdinarySubmitting] = useState(false)
  const checkingAnswerRef = useRef<FrozenAnswer | null>(null)
  const savedContexts = useRef(new Map<string, FrozenAnswer>())
  const savedAnswersRef = useRef(savedAnswers)
  savedAnswersRef.current = savedAnswers
  const answerScopeRef = useRef('')
  answerScopeRef.current = `${projectId}:${goalId}:${run?.id ?? ''}`
  const ordinaryBusy = ordinarySubmitting || checkingAnswer !== null

  const [runRerunError, setRunRerunError] = useState<string | null>(null)
  const [retryError, setRetryError] = useState<string | null>(null)
  const [authorizeError, setAuthorizeError] = useState<string | null>(null)
  const selectedGoal = useRef<string | null>(null)
  const currentGoalId = useRef(goalId)
  currentGoalId.current = goalId
  const actionTriggers = useRef<Record<string, HTMLButtonElement | null>>({})
  const focusHeadingRef = useRef<HTMLHeadingElement>(null)
  const firstOptionRef = useRef<HTMLInputElement>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)
  const submitRef = useRef<HTMLButtonElement>(null)
  const processes = currentProcesses(dashboard.processes.data)
  const decisions = useMemo(() => dashboard.decisions.data ?? [], [dashboard.decisions.data])
  const warnings = dashboard.warnings.data ?? []
  const checkpointItems = dashboard.checkpoint.data?.items ?? []
  const deferredCount = dashboard.checkpoint.data?.deferred_count ?? 0
  // Gates the Needs-you queue's on-track claim (review round 2, finding I2)
  // — an empty `rows` array only means "on track" once the data that built
  // it actually loaded, mirroring the deleted page-level banner's
  // `!decisions.isLoading && !warnings.isLoading` gate (now also covering
  // checkpoint, which the queue reads directly).
  const queueQueries = [dashboard.decisions, dashboard.warnings, dashboard.checkpoint]
  const queueQueriesErrored = queueQueries.some((query) => query.isError)
  const queueQueriesUnready = queueQueries.some((query) => query.isLoading || query.data === undefined)
  const authorityReady = !dashboard.decisions.isLoading && !dashboard.decisions.isError
    && dashboard.decisions.data !== undefined
  const savedIdentity = savedAnswers.map((receipt) => receipt.action_id ?? receipt.decision_ids.join(':')).join('|')
  useEffect(() => {
    let current = true
    for (const receipt of savedAnswersRef.current) {
      const key = receipt.action_id ?? receipt.decision_ids.join(':')
      const frozen = savedContexts.current.get(key)
      if (!frozen || frozen.scope !== answerScopeRef.current) continue
      // Follow the existing authority refresh, but reconcile only this immutable saved identity.
      void Promise.resolve().then(() => fetchOrchestrationDecision(projectId, goalId, frozen.reviewed[0].id)).then((result) => {
        if (!current || frozen.scope !== answerScopeRef.current || !result
            || !validateSavedAnswerReceipt(result.answer_receipt, frozen.submission)
            || !validateOrdinaryAnswerResult(frozen.reviewed[0], frozen.submission, result)) return
        const updated = result.answer_receipt!
        setSavedAnswers((existing) => existing.map((item) => {
          const itemKey = item.action_id ?? item.decision_ids.join(':')
          return itemKey === key && JSON.stringify(item) !== JSON.stringify(updated) ? updated : item
        }))
      }).catch(() => undefined) // A failed background read never revokes confirmed Saved.
    }
    return () => { current = false }
  }, [projectId, goalId, savedIdentity, dashboard.decisions.data, dashboard.decisions.dataUpdatedAt, dashboard.checkpoint.data, dashboard.checkpoint.dataUpdatedAt])


  useEffect(() => {
    if (!authorityReady) return
    setInvalidatedDecisions((current) => {
      const next = { ...current }
      let changed = false
      for (const [decisionId, state] of Object.entries(current)) {
        if (state === 'unavailable' && decisions.some((decision) => decision.id === decisionId)) {
          delete next[decisionId]
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [authorityReady, decisions])
  const effectiveProcessType = selectedProcessType ?? preferredProcessType(processes)
  const selectedProcess = processes.find((process) => process.process_type === effectiveProcessType)
  const debug = health.data?.debug_enabled === true
  const processState = dashboard.processes.isLoading ? 'loading' : dashboard.processes.isError || dashboard.processes.data === undefined ? 'error' : 'ready'
  const goalTerminal = goal.status === 'completed' || goal.status === 'cancelled'
  const frontierType = processState === 'ready' ? frontierProcessType(processes, goal, run) : null
  const canRunSelected = !goalTerminal && frontierType === effectiveProcessType
  // A process stuck on a retryable LM request (running, or waiting-decision
  // with a live retry checkpoint) can also be re-run, not just retried or
  // skipped — mirrors the pre-redesign stepper's `retryableStuck` rule
  // (commit d7e0ca4) and the backend rerun route's own `retryable_stuck`
  // eligibility (orchestration_debug_service.py rerun_last), which already
  // accepts a non-terminal process in that state (review round 4, finding U2).
  const retryStuck = (selectedProcess?.status === 'running' || selectedProcess?.status === 'waiting_decision')
    && selectedProcess?.outputs.lm_retry?.available === true
  const canRerunSelected = !goalTerminal && (isTerminalProcess(selectedProcess) || retryStuck)
  // ponytail: presence is persisted-status based; add live events only when the dashboard gains a push source.
  const isWorking = processes.some((process) => process.status === 'running')

  useEffect(() => {
    if (dashboard.processes.data === undefined || selectedGoal.current === goalId) return
    selectedGoal.current = goalId
    setSelectedProcessType(preferredProcessType(processes))
  }, [dashboard.processes.data, goalId, processes])

  // Keeps the page's Activity "This step" scope synced to the stepper
  // selection, without lifting the selection state itself (see
  // BaselineDashboardProps.onSelectedProcessTypeChange).
  useEffect(() => {
    onSelectedProcessTypeChange?.(effectiveProcessType)
  }, [effectiveProcessType, onSelectedProcessTypeChange])

  function openForm(action: BaselineFocusAction) {
    setSelectedOption(null)
    setReason('')
    setFormError(null)
    setFormErrorSource(null)
    setOpenAction(action)
  }

  function closeForm(action: BaselineFocusAction) {
    setOpenAction(null)
    guardedMicrotask(() => {
      const trigger = actionTriggers.current[`${action.kind}:${action.id}`]
      if (trigger && trigger.isConnected !== false) trigger.focus()
    })
  }

  function restoreSuccessFocus(action: BaselineFocusAction) {
    guardedMicrotask(() => {
      const trigger = actionTriggers.current[`${action.kind}:${action.id}`]
      // A skipped process leaves the frontier, so the refetch that follows
      // success unmounts its Skip trigger. The trigger is still connected in
      // this microtask (the refetch hasn't landed yet), so focusing it here
      // would drop focus to <body> once it's removed — send focus to the
      // heading instead, which always survives.
      if (action.kind !== 'skip' && trigger?.isConnected) trigger.focus()
      else focusHeadingRef.current?.focus()
    })
  }

  function setReviewState(decisionIds: readonly string[], state: 'terminal' | 'stale' | 'unavailable') {
    setInvalidatedDecisions((current) => ({
      ...current,
      ...Object.fromEntries(decisionIds.map((decisionId) => [decisionId, state])),
    }))
  }

  function clearReviewState(decisionIds: readonly string[]) {
    setInvalidatedDecisions((current) => {
      const next = { ...current }
      for (const decisionId of decisionIds) delete next[decisionId]
      return next
    })
  }

  async function preflightDecisions(reviewed: readonly OrchestrationAuthorityDecisionRecord[], isRelevant: () => boolean = () => true) {
    const snapshot: NeedsYouEditSnapshot = {
      rowKey: reviewed.map((decision) => decision.id).join(':'),
      decisionIds: reviewed.map((decision) => decision.id),
      decisions: reviewed,
    }
    let authoritative: readonly OrchestrationAuthorityDecisionRecord[] | undefined
    try {
      const result = await dashboard.decisions.refetch()
      authoritative = result.isSuccess === true && !result.isError ? result.data : undefined
    } catch {
      authoritative = undefined
    }
    const state = validateNeedsYouEdit(snapshot, authoritative)
    if (!isRelevant()) return false
    if (state === 'current') {
      clearReviewState(snapshot.decisionIds)
      return true
    }
    setReviewState(snapshot.decisionIds, state)
    const message = state === 'terminal'
      ? 'This decision was answered elsewhere. Review the authoritative outcome.'
      : state === 'stale'
        ? 'The decision changed. Review the refreshed decision before answering.'
        : 'The current decision could not be confirmed. Refresh and review it before answering.'
    setFormError(message)
    setFormErrorSource('action')
    onMessage(message)
    return false
  }

  function refreshAfterAcceptedSpecializedAnswer(result: OrchestrationDecisionAnswerResult) {
    void answer.acceptRuntimeResult(goalId, result)
    void Promise.all([
      dashboard.decisions.refetch(),
      dashboard.checkpoint.refetch(),
      dashboard.processes.refetch(),
      dashboard.warnings.refetch(),
    ]).catch(() => undefined)
  }

  function publishSaved(frozen: FrozenAnswer, result: OrchestrationDecisionAnswerResult) {
    if (frozen.scope !== answerScopeRef.current || !validateSavedAnswerReceipt(result.answer_receipt, frozen.submission)) return false
    const receipt = result.answer_receipt!
    savedContexts.current.set(receipt.action_id ?? receipt.decision_ids.join(':'), frozen)
    setSavedAnswers((current) => [...current.filter((item) => JSON.stringify(item.decision_ids) !== JSON.stringify(receipt.decision_ids)), receipt])
    checkingAnswerRef.current = null
    setCheckingAnswer(null)
    clearReviewState(receipt.decision_ids)
    onMessage(`Saved. ${receipt.application_state === 'pending' ? 'Application pending.'
      : receipt.application_state === 'failed' || receipt.application_state === 'blocked' ? `Application ${receipt.application_state}${receipt.reason ? `: ${receipt.reason}` : '.'}` : 'Application complete.'}`)
    frozen.finish()
    return true
  }

  async function reconcileAnswer(frozen: FrozenAnswer) {
    if (frozen.scope !== answerScopeRef.current) return false
    checkingAnswerRef.current = frozen
    setCheckingAnswer(frozen)
    onMessage('Checking saved answer…')
    try {
      const results = await Promise.all(frozen.reviewed.map((decision) => fetchOrchestrationDecision(projectId, goalId, decision.id)))
      if (frozen.scope !== answerScopeRef.current) return false
      if (results.every((result, index) => validateSavedAnswerReceipt(result.answer_receipt, frozen.submission)
          && validateOrdinaryAnswerResult(frozen.reviewed[index], frozen.submission, result))) {
        if (frozen.submission.kind === 'batch') answerAgentDefinitionBatch.acceptResult(goalId,
          { decisions: results.map((result) => result.decision), answer_receipt: results[0].answer_receipt }, frozen.submission)
        else answer.acceptResult(goalId, results[0], frozen.submission)
        return publishSaved(frozen, results[0])
      }
      if (results.some((result) => result.decision?.status !== 'pending')) {
        checkingAnswerRef.current = null
        setCheckingAnswer(null)
        setReviewState(frozen.submission.decision_ids, 'stale')
        const text = 'The saved outcome does not match this draft. Review the refreshed decision before answering.'
        setFormError(text)
        setFormErrorSource(frozen.submission.kind === 'batch' ? 'batch' : 'action')
        onMessage(text)
      }
    } catch { /* An unavailable readback leaves the frozen draft in Checking. */ }
    return false
  }

  async function checkFrozenAnswer() {
    const frozen = checkingAnswerRef.current
    if (!frozen || frozen.scope !== answerScopeRef.current) return
    const isCurrent = beginMutation(frozen.owner)
    if (!isCurrent) return
    setOrdinarySubmitting(true)
    try { await reconcileAnswer(frozen) } finally {
      if (isCurrent()) releaseMutation(frozen.owner)
      setOrdinarySubmitting(false)
    }
  }

  async function submitOrdinary(
    reviewedInput: readonly OrchestrationAuthorityDecisionRecord[],
    inputs: Parameters<typeof createOrchestrationAnswerSubmission>[2],
    kind: 'single' | 'batch',
    post: (callbacks: { onSuccess: (result: OrchestrationDecisionAnswerResult | AgentDefinitionReviewBatchAnswerResult) => void; onError: (error: unknown) => void; onSettled: () => void }) => void,
    finish: () => void,
  ) {
    if (!currentUser.data?.id || currentUser.isError || checkingAnswerRef.current) {
      onMessage('The current human actor could not be confirmed. Refresh before answering.')
      return
    }
    const reviewed = structuredClone([...reviewedInput])
    const submission = createOrchestrationAnswerSubmission(reviewed, currentUser.data.id, structuredClone(inputs), kind, run?.id ?? null)
    const owner: BaselineMutationOwner = kind === 'batch' ? 'baseline:answer:agent-definition-review-batch' : `baseline:answer:${reviewed[0].id}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    const frozen: FrozenAnswer = { submission, reviewed, owner, finish, scope: answerScopeRef.current }
    setOrdinarySubmitting(true)
    const targetsOpenForm = kind === 'batch' || openAction?.kind === 'answer' && submission.decision_ids.includes(openAction.id)
    if (targetsOpenForm) { setFormError(null); setFormErrorSource(null) }
    try {
      if (!await preflightDecisions(reviewed, isCurrent) || !isCurrent()) return
      let result: OrchestrationDecisionAnswerResult | AgentDefinitionReviewBatchAnswerResult
      try {
        result = await new Promise<OrchestrationDecisionAnswerResult | AgentDefinitionReviewBatchAnswerResult>((resolveResult, reject) => post({ onSuccess: resolveResult, onError: reject, onSettled: () => undefined }))
      } catch (error) {
        if (isCurrent() && kind === 'batch' && error instanceof ApiError && error.status === 409) {
          void Promise.all([dashboard.decisions.refetch(), dashboard.processes.refetch(), dashboard.warnings.refetch()]).catch(() => undefined)
          setReviewState(submission.decision_ids, 'stale')
          onMessage('The proposals changed and were reloaded — please review the updated agent-definition proposals.')
        } else if (isCurrent() && error instanceof ApiError && [400, 401, 403, 404, 409, 422].includes(error.status)) {
          if (error.status === 409) setReviewState(submission.decision_ids, 'stale')
          if (targetsOpenForm) { setFormError(actionError(error)); setFormErrorSource('action') }
          onMessage(actionError(error))
        } else if (isCurrent()) await reconcileAnswer(frozen)
        return
      }
      if (!isCurrent()) return
      if (!result || typeof result !== 'object') { await reconcileAnswer(frozen); return }
      if (kind === 'batch' && 'decisions' in result) {
        if (validateSavedAnswerReceipt(result.answer_receipt, submission)
            && result.decisions.length === reviewed.length
            && result.decisions.every((decision, index) => validateOrdinaryAnswerResult(reviewed[index], submission,
              { decision, answer_receipt: result.answer_receipt }))) {
          answerAgentDefinitionBatch.acceptResult(goalId, result, submission)
          publishSaved(frozen, { decision: result.decisions[0], answer_receipt: result.answer_receipt })
        } else await reconcileAnswer(frozen)
      } else if ('decision' in result && validateSavedAnswerReceipt(result.answer_receipt, submission)
          && validateOrdinaryAnswerResult(reviewed[0], submission, result)) {
        answer.acceptResult(goalId, result, submission)
        publishSaved(frozen, result)
      } else await reconcileAnswer(frozen)
    } finally {
      if (isCurrent()) releaseMutation(owner)
      setOrdinarySubmitting(false)
    }
  }

  function runMutation<T>(action: BaselineFocusAction, mutate: (callbacks: { onSuccess: (data: T) => void; onError: (error: unknown) => void; onSettled: () => void }) => void, success: string, onSuccessFocus?: () => void, acceptResult?: (data: T) => boolean) {
    const owner = actionOwner(action)
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    // Snapshot whether `action` is the form currently open — runMutation is
    // now also called for actions with no open form at all (the accept-
    // recommendation fast path). Closing/clearing openAction+formError must
    // stay scoped to the action that was actually acted on, so accepting one
    // row can't silently close or wipe an unrelated row's open form.
    const targetsOpenForm = openAction !== null && openAction.kind === action.kind && openAction.id === action.id
    if (targetsOpenForm) { setFormError(null); setFormErrorSource(null) }
    mutate({
      onSuccess: (data) => {
        if (!isCurrent()) return
        if (acceptResult && !acceptResult(data)) return
        if (targetsOpenForm) setOpenAction(null)
        onMessage(success)
        ;(onSuccessFocus ?? (() => restoreSuccessFocus(action)))()
      },
      // Actions with no open form (the accept-recommendation fast path) still
      // need error feedback — falling through to silence here left a failed
      // "Accept: X" click with no error, no toast, no announcement; the
      // button just re-enabled (review round 2, finding I1). onMessage flows
      // through the shared goal announcer's queue region.
      onError: (error) => {
        if (!isCurrent()) return
        if (targetsOpenForm) { setFormError(actionError(error)); setFormErrorSource('action'); submitRef.current?.focus() }
        else onMessage(actionError(error))
      },
      onSettled: () => { if (isCurrent()) { releaseMutation(owner) } },
    })
  }

  async function submitDecision(
    decision: OrchestrationAuthorityDecisionRecord,
    selectedOption: string,
    submitReason: string,
    success: string,
    onSuccessFocus?: () => void,
    editedAgent?: AgentCreatePayload,
  ) {
    if (!decision.runtime_identity && !decision.recovery) {
      const frozenAgent = editedAgent ? structuredClone(editedAgent) : undefined
      const payload = { goalId, decisionId: decision.id, selectedOption,
        reason: submitReason, contractVersion: decision.contract_version, editedAgent: frozenAgent }
      const targetsOpenForm = openAction?.kind === 'answer' && openAction.id === decision.id
      await submitOrdinary([decision], [payload], 'single',
        (callbacks) => answer.mutate(payload, callbacks),
        () => { if (targetsOpenForm) setOpenAction(null); (onSuccessFocus ?? (() => restoreSuccessFocus({ kind: 'answer', id: decision.id })))() })
      return
    }
    const preflightGeneration = generation()
    const preflightGoalId = goalId
    const reviewed = decision
    const submitted = { selectedOption, reason: reviewed.runtime_identity ? submitReason || null : submitReason.trim() || null }
    const recoverySubmission = reviewed.recovery && currentUser.data?.id
      ? createOrchestrationAnswerSubmission([structuredClone(reviewed)], currentUser.data.id,
        [{ decisionId: reviewed.id, selectedOption, reason: submitReason, contractVersion: reviewed.contract_version }], 'single', run?.id ?? null)
      : null
    if (reviewed.recovery && !recoverySubmission) { onMessage('The current human actor could not be confirmed. Refresh before answering.'); return }
    if (!await preflightDecisions([reviewed], () => generation() === preflightGeneration && currentGoalId.current === preflightGoalId)
      || generation() !== preflightGeneration || currentGoalId.current !== preflightGoalId) return
    const action: BaselineFocusAction = { kind: 'answer', id: reviewed.id }
    runMutation<OrchestrationDecisionAnswerResult>(action, (callbacks) => answer.mutate({
      goalId,
      decisionId: reviewed.id,
      contractVersion: reviewed.contract_version,
      runtimeIdentity: reviewed.runtime_identity,
      selectedOption,
      reason: submitReason,
      editedAgent,
    }, callbacks), success, onSuccessFocus, reviewed.runtime_identity ? (result) => {
      if (validateRuntimeAnswerReceipt(reviewed, submitted, result) === 'accepted') {
        refreshAfterAcceptedSpecializedAnswer(result)
        return true
      }
      setReviewState([reviewed.id], 'stale')
      void dashboard.decisions.refetch()
      const message = 'The decision changed while the answer was being recorded. Review the refreshed decision before answering.'
      setFormError(message)
      setFormErrorSource('action')
      onMessage(message)
      return false
    } : reviewed.recovery ? (result) => {
      if (recoverySubmission && validateOrdinaryAnswerResult(reviewed, recoverySubmission, result)) {
        refreshAfterAcceptedSpecializedAnswer(result)
        return true
      }
      setReviewState([reviewed.id], 'stale')
      onMessage('The decision changed while the answer was being recorded. Review the refreshed decision before answering.')
      return false
    } : undefined)
  }

  // onSuccessFocus lets a caller other than ProcessFocus (the Needs-you queue)
  // override where focus lands after a successful submit — ProcessFocus's Skip
  // keeps the default (same trigger, else the process heading); queue rows move
  // focus to the next queue row instead (see NeedsYouQueue).
  function renderActionForm(
    action: BaselineFocusAction,
    record: OrchestrationAuthorityDecisionRecord | OrchestrationWarningRecord | OrchestrationProcessRunRecord,
    onSuccessFocus?: () => void,
    displayQuestion?: string,
    displayReferences?: readonly NeedsYouReference[],
    editState: 'current' | 'terminal' | 'stale' | 'unavailable' = 'current',
    onReleaseEdit?: () => void,
    formKey?: string,
  ) {
    let target: ActionTarget | null = null
    if (action.kind === 'answer') target = { kind: 'answer', action, decision: record as OrchestrationAuthorityDecisionRecord }
    else if (action.kind === 'skip') target = { kind: 'skip', action, process: record as OrchestrationProcessRunRecord }
    else if (action.kind === 'acknowledge' || action.kind === 'resolve') target = { kind: action.kind, action, warning: record as OrchestrationWarningRecord }
    if (!target) return null
    return <BaselineActionForm key={formKey ?? (target.kind === 'answer' ? `${target.decision.id}:${target.decision.contract_version ?? target.decision.runtime_identity ?? 'legacy'}` : `${target.kind}:${target.action.id}`)}
      target={target} mutationBusy={mutationBusy || ordinaryBusy}
      submitting={action.kind === 'answer' ? answer.isPending || ordinaryBusy : action.kind === 'skip' ? skip.isPending : action.kind === 'acknowledge' ? acknowledge.isPending : resolve.isPending}
      selectedOption={selectedOption} reason={reason} error={formErrorSource === 'action' ? formError : null}
      firstOptionRef={firstOptionRef} reasonRef={reasonRef} submitRef={submitRef}
      onAnswer={(decision, option, submitReason) => { void submitDecision(decision, option, submitReason, 'Answer received.', onSuccessFocus) }}
      onSkip={(process, submitReason) => runMutation(action, (callbacks) => skip.mutate({ goalId, processType: process.process_type, reason: submitReason }, callbacks), 'Process skipped.', onSuccessFocus)}
      onAcknowledge={(warning, submitReason) => runMutation(action, (callbacks) => acknowledge.mutate({ goalId, warningId: warning.id, reason: submitReason }, callbacks), 'Risk acknowledged.', onSuccessFocus)}
      onResolve={(warning, submitReason) => runMutation(action, (callbacks) => resolve.mutate({ goalId, warningId: warning.id, reason: submitReason }, callbacks), 'Risk resolved.', onSuccessFocus)}
      onSelectedOptionChange={setSelectedOption} onReasonChange={setReason} onCancel={() => closeForm(action)}
      displayQuestion={displayQuestion}
      displayReferences={displayReferences}
      editState={editState}
      onReleaseEdit={onReleaseEdit}
    />
  }

  // Accept-recommendation fast path (spec C): submits decision.recommendation
  // immediately — no form, no reason. Reuses runMutation so it shares the
  // same mutation-owner locking, error surfacing, and focus handling as the
  // full answer form.
  function acceptRecommendation(decision: OrchestrationAuthorityDecisionRecord, onSuccessFocus?: () => void) {
    if (decision.recommendation === null) return
    void submitDecision(decision, decision.recommendation, '', 'Answer received.', onSuccessFocus)
  }

  // Goal-definition step-through card (spec D): submits one adaptive
  // clarification question's free-text answer. Doesn't route through
  // runMutation/openAction — the card lives entirely in the Needs-you queue
  // and manages its own local question index, so this only needs to own the
  // mutation-owner lock and error surfacing.
  async function answerGoalDefinitionQuestion(
    decision: OrchestrationAuthorityDecisionRecord,
    answerText: string,
    callbacks: { onSuccess: () => void; onError: (error: unknown) => void; onSettled: () => void },
  ) {
    if (!decision.runtime_identity) {
      const payload = { goalId, decisionId: decision.id, selectedOption: answerText, reason: '', contractVersion: decision.contract_version }
      await submitOrdinary([decision], [payload], 'single', (handlers) => answer.mutate(payload, handlers), callbacks.onSuccess)
      callbacks.onSettled()
      return
    }
    const preflightGeneration = generation()
    const preflightGoalId = goalId
    const reviewed = decision
    const submitted = { selectedOption: answerText, reason: reviewed.runtime_identity ? null : '' }
    const preflightCurrent = await preflightDecisions([reviewed], () => generation() === preflightGeneration && currentGoalId.current === preflightGoalId)
    if (generation() !== preflightGeneration || currentGoalId.current !== preflightGoalId) {
      callbacks.onSettled()
      return
    }
    if (!preflightCurrent) {
      callbacks.onError(new Error('The decision changed. Review the refreshed decision before answering.'))
      callbacks.onSettled()
      return
    }
    const owner: BaselineMutationOwner = `baseline:answer:${decision.id}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    setFormError(null)
    setFormErrorSource(null)
    answer.mutate({ goalId, decisionId: decision.id, contractVersion: decision.contract_version, runtimeIdentity: decision.runtime_identity, selectedOption: answerText, reason: '' }, {
      onSuccess: (result) => {
        if (!isCurrent()) return
        if (reviewed.runtime_identity && validateRuntimeAnswerReceipt(reviewed, submitted, result) !== 'accepted') {
          setReviewState([reviewed.id], 'stale')
          void dashboard.decisions.refetch()
          const staleError = new Error('The decision changed. Review the refreshed decision before answering.')
          callbacks.onError(staleError)
          onMessage(staleError.message)
          return
        }
      if (reviewed.runtime_identity) refreshAfterAcceptedSpecializedAnswer(result)
        onMessage('Answer received.')
        callbacks.onSuccess()
      },
      onError: (error) => {
        if (!isCurrent()) return
        setFormError(actionError(error))
        setFormErrorSource('action')
        callbacks.onError(error)
      },
      onSettled: () => {
        if (!isCurrent()) return
        releaseMutation(owner)
        callbacks.onSettled()
      },
    })
  }

  async function answerRecoveryProposal(
    decision: OrchestrationAuthorityDecisionRecord,
    option: RecoveryAnswerOption,
    reason: string,
    callbacks: { onSuccess: () => void; onError: (error: unknown) => void; onSettled: () => void },
  ) {
    const preflightGeneration = generation()
    const preflightGoalId = goalId
    const reviewed = decision
    const submitted = { selectedOption: option, reason: reviewed.runtime_identity ? reason || null : reason.trim() || null }
    const recoverySubmission = currentUser.data?.id
      ? createOrchestrationAnswerSubmission([structuredClone(reviewed)], currentUser.data.id,
        [{ decisionId: reviewed.id, selectedOption: option, reason, contractVersion: reviewed.contract_version }], 'single', run?.id ?? null)
      : null
    if (!reviewed.runtime_identity && !recoverySubmission) {
      callbacks.onError(new Error('The current human actor could not be confirmed. Refresh before answering.'))
      callbacks.onSettled()
      return
    }
    const preflightCurrent = await preflightDecisions([reviewed], () => generation() === preflightGeneration && currentGoalId.current === preflightGoalId)
    if (generation() !== preflightGeneration || currentGoalId.current !== preflightGoalId) {
      callbacks.onSettled()
      return
    }
    if (!preflightCurrent) {
      callbacks.onError(new Error('The decision changed. Review the refreshed decision before answering.'))
      callbacks.onSettled()
      return
    }
    const owner: BaselineMutationOwner = `baseline:answer:${decision.id}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) { callbacks.onSettled(); return }
    setFormError(null)
    setFormErrorSource(null)
    answer.mutate({ goalId, decisionId: decision.id, contractVersion: decision.contract_version, runtimeIdentity: decision.runtime_identity, selectedOption: option, reason }, {
      onSuccess: (result) => {
        if (!isCurrent()) return
        if (reviewed.runtime_identity && validateRuntimeAnswerReceipt(reviewed, submitted, result) !== 'accepted') {
          setReviewState([reviewed.id], 'stale')
          void dashboard.decisions.refetch()
          const staleError = new Error('The decision changed. Review the refreshed decision before answering.')
          callbacks.onError(staleError)
          onMessage(staleError.message)
          return
        }
        if (!reviewed.runtime_identity && (!recoverySubmission || !validateOrdinaryAnswerResult(reviewed, recoverySubmission, result))) {
          setReviewState([reviewed.id], 'stale')
          callbacks.onError(new Error('The decision changed. Review the refreshed decision before answering.'))
          return
        }
        refreshAfterAcceptedSpecializedAnswer(result)
        onMessage(option === 'approve_fix' ? 'Fix approved.' : option === 'reject' ? 'Fix rejected.' : 'Marked as provided. Re-diagnosing the blocker…')
        callbacks.onSuccess()
      },
      onError: (error) => {
        if (!isCurrent()) return
        callbacks.onError(error)
      },
      onSettled: () => {
        if (isCurrent()) releaseMutation(owner)
        callbacks.onSettled()
      },
    })
  }

  async function submitAgentDefinitionReviewBatch(reviewed: readonly OrchestrationAuthorityDecisionRecord[], answers: AgentDefinitionReviewBatchAnswerInput[], onSuccessFocus?: () => void) {
    const frozenAnswers = structuredClone(answers)
    await submitOrdinary(reviewed, frozenAnswers.map((item) => ({ ...item, reason: item.reason ?? '' })), 'batch',
      (callbacks) => answerAgentDefinitionBatch.mutate({ goalId, answers: frozenAnswers }, callbacks),
      () => { setOpenAction(null); guardedMicrotask(onSuccessFocus ?? (() => focusHeadingRef.current?.focus())) })
  }

  function answerTeamProposal(decision: OrchestrationAuthorityDecisionRecord, selectedOption: string, editedAgent: AgentCreatePayload | undefined, onSuccess?: () => void) {
    void submitDecision(decision, selectedOption, '', 'Proposal decision received.', onSuccess, editedAgent)
  }

  function mutateChain(processType: OrchestrationBaselineProcessType, action: 'run' | 'rerun', trigger: HTMLButtonElement | null) {
    const owner: BaselineMutationOwner = `baseline:${action}:${processType}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    setRunRerunError(null)
    onMessage(`${action === 'run' ? 'Running' : 'Re-running'} ${processLabel(processType)}.`)
    const mutation = action === 'run' ? runStep : rerun
    mutation.mutate({ goalId, processType }, {
      onSuccess: (result) => {
        if (!isCurrent()) return
        const status = result.process.status
        const reason = typeof result.process.reason === 'string' ? result.process.reason : 'The process did not start.'
        onMessage(status === 'running'
          ? `${action === 'run' ? 'Running' : 'Re-running'} ${processLabel(processType)}.`
          : status === 'idle' || status === 'not_started' || status === 'not-started' || !status
            ? `Nothing to run: ${reason}`
            : `${processLabel(processType)}: ${statusPhrase(String(status)).label}.`)
        setSelectedProcessType(processType)
      },
      onError: (mutationError) => {
        if (!isCurrent()) return
        const message = actionError(mutationError)
        setRunRerunError(message)
        onMessage(message)
      },
      onSettled: () => {
        if (!isCurrent()) return
        releaseMutation(owner)
        trigger?.focus()
      },
    })
  }

  // Gates the baseline chain per run.baseline_authorized — the
  // button itself lives in ProcessChain's Z2 header, this just owns the
  // mutation the same way mutateChain/retryProcess do.
  function authorizeBaseline(trigger: HTMLButtonElement | null) {
    const owner: BaselineMutationOwner = `baseline:authorize:${goalId}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    setAuthorizeError(null)
    onMessage('Authorizing baseline.')
    authorize.mutate({ goalId }, {
      onSuccess: (result) => {
        if (!isCurrent()) return
        onMessage(result.run?.baseline_authorized
        ? 'Baseline authorized.'
        : `Baseline: ${result.goal.status === 'paused' ? 'paused' : result.run?.status ?? 'not started'}.`)
      },
      onError: (mutationError) => {
        if (!isCurrent()) return
        const message = actionError(mutationError)
        setAuthorizeError(message)
        onMessage(message)
      },
      onSettled: () => {
        if (!isCurrent()) return
        releaseMutation(owner)
        trigger?.focus()
      },
    })
  }

  // Stale-inputs suggestion row wiring (Approve/Dismiss in the
  // Needs-you queue) — Approve reruns the step then resolves the warning so
  // it never resurfaces for the process run that triggered the rerun; Dismiss
  // just resolves it. Both reuse the existing rerun/resolve mutations, no new
  // endpoints.
  function approveStaleInputs(warningRecord: OrchestrationWarningRecord, processType: OrchestrationBaselineProcessType, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:rerun:${processType}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    onMessage(`Re-running ${processLabel(processType)}.`)
    rerun.mutate({ goalId, processType }, {
      onSuccess: () => {
        if (!isCurrent()) return
        resolve.mutate({ goalId, warningId: warningRecord.id, reason: 'approved rerun' }, {
          onSuccess: () => {
            if (!isCurrent()) return
            onMessage(`Re-running ${processLabel(processType)}.`)
            onSuccessFocus?.()
          },
          onError: (error) => { if (isCurrent()) { onMessage(actionError(error)) } },
          onSettled: () => { if (isCurrent()) { releaseMutation(owner) } },
        })
      },
      onError: (error) => {
        if (!isCurrent()) return
        onMessage(actionError(error))
        releaseMutation(owner)
      },
    })
  }

  function dismissStaleInputs(warningRecord: OrchestrationWarningRecord, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:resolve:${warningRecord.id}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    resolve.mutate({ goalId, warningId: warningRecord.id, reason: 'dismissed by human' }, {
      onSuccess: () => {
        if (!isCurrent()) return
        onMessage('Suggestion dismissed.'); onSuccessFocus?.() },
      onError: (error) => { if (isCurrent()) { onMessage(actionError(error)) } },
      onSettled: () => { if (isCurrent()) { releaseMutation(owner) } },
    })
  }

  function retryProcess(processType: OrchestrationBaselineProcessType, trigger: HTMLButtonElement | null, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:retry:${processType}`
    const isCurrent = beginMutation(owner)
    if (!isCurrent) return
    setRetryError(null)
    onMessage(`Retrying ${processLabel(processType)}.`)
    retry.mutate({ goalId, processType }, {
      onSuccess: (result) => {
        if (!isCurrent()) return
        if (result?.process?.['retry_failed'] === true) {
          const message = typeof result.process['error'] === 'string' ? result.process['error'] : 'Language model request failed.'
          setRetryError(message)
          onMessage(message)
          guardedMicrotask(() => trigger?.focus())
          return
        }
        onMessage(`Request succeeded. Continuing ${processLabel(processType)}.`)
        guardedMicrotask(onSuccessFocus ?? (() => focusHeadingRef.current?.focus()))
      },
      onError: (error) => {
        if (!isCurrent()) return
        if (error instanceof ApiError && error.status === 409) {
          void Promise.all([dashboard.processes.refetch(), dashboard.warnings.refetch()])
          onMessage('This retry is no longer available. Refreshing process state.')
          guardedMicrotask(() => focusHeadingRef.current?.focus())
        } else {
          const message = actionError(error)
          setRetryError(message)
          onMessage(message)
          guardedMicrotask(() => trigger?.focus())
        }
      },
      onSettled: () => { if (isCurrent()) { releaseMutation(owner) } },
    })
  }

  return <div data-testid="baseline-dashboard" className="flex min-w-0 flex-col gap-4">
    <TeamHierarchyStatus decisions={decisions} reviews={dashboard.agentReviews.data ?? []} process={processes.find((process) => process.process_type === 'team_hierarchy' && !process.superseded_by_id)} />
    <ProcessChain processes={processes} processState={processState} selectedProcessType={effectiveProcessType} onSelect={setSelectedProcessType} goal={goal} checkpointItems={checkpointItems} run={run} onAuthorize={authorizeBaseline} chainPending={mutationBusy || runStep.isPending || rerun.isPending || retry.isPending} authorizePending={authorize.isPending} authorizeError={authorizeError} />
    {processState === 'ready' && <>
      {savedAnswers.length > 0 && <section aria-label="Saved answers" className="mb-3 space-y-2">{savedAnswers.map((receipt) => <p key={receipt.action_id ?? receipt.decision_ids.join(':')} className="break-words rounded border border-huddleroom-border p-3 text-sm">
        <strong>Saved.</strong> {receipt.application_state === 'pending' ? 'Application pending.'
          : receipt.application_state === 'failed' || receipt.application_state === 'blocked' ? `Application ${receipt.application_state}${receipt.reason ? `: ${receipt.reason}` : '.'}` : 'Application complete.'}
      </p>)}</section>}
      {checkingAnswer && <div className="mb-3 rounded border border-huddleroom-border p-3 text-sm">
        <p>Checking saved answer… Your frozen draft is retained.</p>
        <Button type="button" variant="secondary" className="mt-2 min-h-11" disabled={ordinarySubmitting || mutationBusy} onClick={() => { void checkFrozenAnswer() }}>Check saved answer</Button>
      </div>}
      <NeedsYouQueue
        goal={goal} run={run} processes={processes} sourceProcesses={dashboard.processes.data ?? []} warnings={warnings} gates={gates} meetingCommitments={meetingCommitments}
        checkpointItems={checkpointItems} allDecisions={decisions} authorityReady={authorityReady} invalidatedDecisions={invalidatedDecisions} needsYouReferences={needsYouReferences} sourceActionIds={sourceActionIds}
        recoveryDecisions={decisions} onAnswerRecoveryProposal={answerRecoveryProposal} checkpointLoaded={dashboard.checkpoint.isSuccess}
        queriesErrored={queueQueriesErrored} queriesUnready={queueQueriesUnready}
        deferredCount={deferredCount} isWorking={isWorking}
        mutationBusy={mutationBusy || ordinaryBusy} onSelectProcess={setSelectedProcessType}
        onRevealSource={onRevealSource}
        openAction={openAction} onOpenAction={openForm} onActionTrigger={(action, element) => { actionTriggers.current[`${action.kind}:${action.id}`] = element }}
        renderActionForm={renderActionForm}
        renderAgentDefinitionReviewBatch={(batchDecisions, onSuccessFocus, batchKey, displayReferences, editState = 'current') => <AgentDefinitionReviewBatchForm key={batchKey} decisions={batchDecisions} mutationBusy={mutationBusy || ordinaryBusy} reviewLocked={editState !== 'current'} submitting={answerAgentDefinitionBatch.isPending || ordinaryBusy} error={formErrorSource === 'batch' ? formError : null} onSubmit={(answers) => { void submitAgentDefinitionReviewBatch(batchDecisions, answers, onSuccessFocus) }} displayReferences={displayReferences} />}
        renderTeamHierarchyProposal={(decision, onSuccess, registerPrimary, displayReferences, editState = 'current', formKey, onReleaseEdit, onBeginEdit) => {
          const isActive = openAction?.kind === 'answer' && openAction?.id === decision.id
          return <TeamHierarchyProposalForm key={formKey} decision={decision} disabled={mutationBusy || ordinaryBusy} reviewLocked={editState !== 'current'} submitting={answer.isPending || ordinaryBusy} error={isActive && formErrorSource === 'action' ? formError : null} onSubmit={(option, editedAgent) => answerTeamProposal(decision, option, editedAgent, onSuccess)} onCancelEdit={onReleaseEdit} onBeginEdit={onBeginEdit} primaryRef={registerPrimary} displayReferences={displayReferences} />
        }}
        onAcceptDecision={acceptRecommendation} onAnswerGoalDefinitionQuestion={answerGoalDefinitionQuestion}
        retryPending={retry.isPending} retryError={retryError} onRetryProcess={retryProcess}
        onApproveStaleInputs={approveStaleInputs} onDismissStaleInputs={dismissStaleInputs}
        onAcknowledgeWarning={(warningRecord, onSuccessFocus) => runMutation({ kind: 'acknowledge', id: warningRecord.id }, (callbacks) => resolve.mutate({ goalId, warningId: warningRecord.id, reason: 'Acknowledged by operator' }, callbacks), 'Risk acknowledged.', onSuccessFocus)}
        onReleaseRetainedEdit={(decisionIds, mode) => {
          clearReviewState(decisionIds)
          setSelectedOption(null)
          setReason('')
          setFormError(null)
          setFormErrorSource(null)
          if (mode === 'dismiss') setOpenAction(null)
        }}
        renderBlockerRecovery={renderBlockerRecovery} suppressAnnouncement={suppressQueueAnnouncement} announcementScope={queueAnnouncementScope}
      />
      <ProcessFocus selectedProcessType={effectiveProcessType} process={selectedProcess} goal={goal} decisions={decisions} decisionsUnready={dashboard.decisions.isLoading || dashboard.decisions.isError} warnings={warnings} reviews={dashboard.agentReviews.data ?? []} debug={debug} canSkip={!!selectedProcess && goal.status !== 'completed' && goal.status !== 'cancelled' && (selectedProcess.status === 'running' || selectedProcess.status === 'waiting_decision')} mutationBusy={mutationBusy || ordinaryBusy} canRun={canRunSelected} canRerun={canRerunSelected} runPending={runStep.isPending} rerunPending={rerun.isPending} runRerunError={runRerunError} onRun={(trigger) => mutateChain(effectiveProcessType, 'run', trigger)} onRerun={(trigger) => mutateChain(effectiveProcessType, 'rerun', trigger)} retryPending={retry.isPending} retryError={retryError} onRetry={(trigger) => retryProcess(effectiveProcessType, trigger)} openAction={openAction} onOpenAction={openForm} onActionTrigger={(action, element) => { actionTriggers.current[`${action.kind}:${action.id}`] = element }} headingRef={focusHeadingRef} renderActionForm={renderActionForm} onViewActivity={onViewActivity} />
    </>}
  </div>
}
