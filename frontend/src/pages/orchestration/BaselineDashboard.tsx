import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { ApiError } from '@/lib/api-client'
import {
  useAcknowledgeOrchestrationWarning,
  type AgentDefinitionReviewBatchAnswerInput,
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
  OrchestrationDecisionOption,
  OrchestrationGate,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { BASELINE_PROCESS_TYPES, processLabel } from './humanize'
import { frontierProcessType, isTerminalProcess, ProcessChain } from './ProcessChain'
import { ProcessFocus } from './ProcessFocus'
import { NeedsYouQueue } from './NeedsYouQueue'
import { useGoalAnnouncer } from './goalAnnouncer'

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
  gates: readonly OrchestrationGate[]
  mutationBusy: boolean
  acquireMutation: (owner: BaselineMutationOwner) => boolean
  releaseMutation: (owner: BaselineMutationOwner) => void
  onMessage: (message: string) => void
  // Reports the currently-effective selected step whenever it changes, so the
  // page can keep the Details tabs' Activity "This step" scope synced to the
  // stepper selection without lifting the selection state itself. Optional —
  // existing callers/tests that don't need Activity sync keep working.
  onSelectedProcessTypeChange?: (type: OrchestrationBaselineProcessType) => void
  // Switches the Details tabs to Activity, scoped to this step (threaded from
  // ProcessFocus's "View activity for this step" control).
  onViewActivity?: () => void
  suppressQueueAnnouncement?: boolean
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
}) {
  const disabled = mutationBusy || submitting
  const options = target.kind === 'answer' ? target.decision.options.map((option) => ({ option, key: decisionOptionKey(option) })) : []
  const freeText = target.kind === 'answer' && options.length === 0
  const reviewContext = target.kind === 'answer' ? parseReviewContext(target.decision.context) : null
  const managerComparison = target.kind === 'answer' && isManagerSelectionDecision(target.decision.decision_key) ? parseManagerComparison(target.decision.context) : null
  const answerRef = useRef<HTMLTextAreaElement>(null)
  const firstAnswerOption = options.findIndex((option) => option.key !== null)
  const requiresReason = target.kind !== 'answer' || decisionRequiresReason(selectedOption, target.decision.recommendation)
  const canSubmit = target.kind === 'answer'
    ? canSubmitDecision(selectedOption, target.decision.recommendation, reason)
    : canSubmitReason(reason)
  const label = target.kind === 'answer' ? 'Submit answer' : target.kind === 'skip' ? 'Skip process' : target.kind === 'acknowledge' ? 'Acknowledge risk' : 'Resolve risk'
  const { announceError } = useGoalAnnouncer()

  useEffect(() => { if (error) announceError(error) }, [error, announceError])
  useEffect(() => {
    if (target.kind === 'answer' && freeText && answerRef.current) answerRef.current.focus()
    else if (target.kind === 'answer' && firstOptionRef.current) firstOptionRef.current.focus()
    else reasonRef.current?.focus()
  }, [target.kind, target.action.id, freeText, firstOptionRef, reasonRef])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (target.kind === 'answer') {
      if (canSubmit) onAnswer(target.decision, selectedOption!, reason)
      return
    }
    if (!canSubmitReason(reason)) return
    if (target.kind === 'skip') onSkip(target.process, reason)
    else if (target.kind === 'acknowledge') onAcknowledge(target.warning, reason)
    else onResolve(target.warning, reason)
  }

  return (
    <form onSubmit={submit} aria-busy={submitting} className="mt-3 border-t border-huddleroom-border pt-3">
      <fieldset disabled={disabled}>
        {target.kind === 'answer' && <>
          <legend className="text-xs font-medium text-huddleroom-text-muted">Answer decision</legend>
          {target.decision.recommendation !== null && <p className="mt-2 text-xs text-huddleroom-text-secondary">Recommendation: {target.decision.recommendation}</p>}
          {managerComparison && <dl className="mt-2 grid gap-1 rounded border border-huddleroom-border p-2 text-xs text-huddleroom-text-secondary"><div><dt className="inline font-medium text-huddleroom-text-muted">Deterministic: </dt><dd className="inline">{managerComparison.deterministic}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">LLM: </dt><dd className="inline">{managerComparison.llm}</dd></div><div><dt className="inline font-medium text-huddleroom-text-muted">Rationale: </dt><dd className="inline">{managerComparison.rationale}</dd></div></dl>}
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
              {reason && <p className="text-sm text-huddleroom-text-secondary">{reason}</p>}
              {description && proposedDescription && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Description</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Original: {description}</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedDescription}</p>
              </div>}
              {persona && proposedPersona && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Persona</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Original: {persona}</p>
                <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedPersona}</p>
              </div>}
              {problems.length > 0 && <div className="mt-2">
                <p className="text-xs font-medium text-huddleroom-text-muted">Problems</p>
                <ul className="mt-1 space-y-1">
                  {problems.map((problem, index) => <li key={index} className="text-xs text-huddleroom-text-secondary">• {problem}</li>)}
                </ul>
              </div>}
            </div>
          })()}
          {freeText ? <Textarea ref={answerRef} label="Answer" value={selectedOption ?? ''} onChange={(event) => onSelectedOptionChange(event.target.value)} rows={3} /> : (
            <div className="mt-2 space-y-2">
              {options.map(({ option, key }, index) => key === null ? <p key={index} className="text-sm text-huddleroom-text-secondary">Unavailable option</p> : (
                <label key={key} className="flex min-h-11 items-center gap-2 text-sm text-huddleroom-text-primary">
                  <input ref={index === firstAnswerOption ? firstOptionRef : undefined} type="radio" name={`decision-${target.decision.id}`} value={key} checked={selectedOption === key} onChange={() => onSelectedOptionChange(key)} />
                  {normalizeDecisionOption(option)?.label ?? key}
                </label>
              ))}
            </div>
          )}
        </>}
        {target.kind === 'answer' && <Textarea ref={reasonRef} label={target.decision.recommendation === null ? 'Reason (optional)' : 'Reason (required when overriding the recommendation)'} value={reason} onChange={(event) => onReasonChange(event.target.value)} rows={3} required={requiresReason} />}
        {target.kind !== 'answer' && <Textarea ref={reasonRef} label="Reason (required)" value={reason} onChange={(event) => onReasonChange(event.target.value)} rows={3} required={requiresReason} />}
        {error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
        <div className="mt-3 flex flex-wrap justify-end gap-2">
          <Button type="button" variant="secondary" className="min-h-11" disabled={disabled} onClick={onCancel}>Cancel</Button>
          <Button ref={submitRef} type="submit" className="min-h-11" disabled={disabled || !canSubmit}>{submitting ? 'Submitting…' : label}</Button>
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
}: {
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  mutationBusy: boolean
  submitting: boolean
  error: string | null
  onSubmit: (answers: AgentDefinitionReviewBatchAnswerInput[]) => void
}) {
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
  const { announceError } = useGoalAnnouncer()
  useEffect(() => { if (error) announceError(error) }, [error, announceError])
  const update = (decisionId: string, changes: Partial<AgentDefinitionReviewAnswerState>) => {
    setAnswers((current) => ({ ...current, [decisionId]: { ...current[decisionId], ...changes } }))
  }
  const canSubmit = (candidate: Record<string, AgentDefinitionReviewAnswerState>) => decisions.every((decision) => {
    const answer = candidate[decision.id]
    return answer?.selectedOption && (answer.selectedOption !== 'edit'
      || (answer.editedDescription.trim() && answer.editedPersona.trim()))
  })
  const submitIfComplete = (candidate: Record<string, AgentDefinitionReviewAnswerState>) => {
    if (!canSubmit(candidate)) return
    onSubmit(decisions.map((decision) => {
      const { selectedOption, reason, editedDescription, editedPersona } = candidate[decision.id]
      return { decisionId: decision.id, selectedOption: selectedOption!, reason, editedDescription, editedPersona }
    }))
  }
  const focusNextControl = (nextState: Record<string, AgentDefinitionReviewAnswerState>, decisionIdToSkip?: string) => {
    const nextRow = decisions.find((d) => d.id !== decisionIdToSkip && nextState[d.id]?.selectedOption === null)
    if (nextRow) {
      queueMicrotask(() => rowRefs.current[nextRow.id]?.focus())
    } else {
      queueMicrotask(() => formRef.current?.focus())
    }
  }
  // Records one row's decision and, once every row in the batch has one,
  // submits — this is the only place a row actually finalizes.
  const decideAndMaybeSubmit = (decisionId: string, changes: Partial<AgentDefinitionReviewAnswerState>) => {
    const next = { ...answers, [decisionId]: { ...answers[decisionId], ...changes } }
    setAnswers(next)
    focusNextControl(next, decisionId)
    submitIfComplete(next)
  }
  const approveAllRemaining = () => {
    const next = { ...answers }
    for (const decision of decisions) {
      if (next[decision.id].selectedOption === null) next[decision.id] = { ...next[decision.id], selectedOption: agentDefinitionApproveOption(decision) }
    }
    setAnswers(next)
    focusNextControl(next)
    submitIfComplete(next)
  }
  const remaining = decisions.filter((decision) => answers[decision.id]?.selectedOption === null)
  const allComplete = remaining.length === 0

  return <div ref={formRef} aria-busy={submitting} className="mt-2 divide-y divide-huddleroom-border">
    <fieldset disabled={disabled} className="contents">
      {remaining.length > 1 && <div className="flex justify-end py-2">
        <Button type="button" variant="secondary" className="min-h-11" disabled={disabled} onClick={approveAllRemaining}>
          Approve all remaining ({remaining.length})
        </Button>
      </div>}
      {decisions.map((decision) => {
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
          <p className="text-sm font-medium text-huddleroom-text-primary">{decision.title || decision.question}</p>
          {decision.title && <p className="mt-1 text-sm text-huddleroom-text-secondary">{decision.question}</p>}
          {hasDescription && proposedDescription !== null && <div className="mt-2">
            <p className="text-xs font-medium text-huddleroom-text-muted">Description</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Current: {description || '(empty)'}</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedDescription || '(empty)'}</p>
          </div>}
          {hasPersona && proposedPersona !== null && <div className="mt-2">
            <p className="text-xs font-medium text-huddleroom-text-muted">System prompt</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Current: {persona || '(empty)'}</p>
            <p className="mt-1 whitespace-pre-wrap text-xs text-huddleroom-text-secondary">Proposed: {proposedPersona || '(empty)'}</p>
          </div>}
          {decided ? (
            <p className="mt-2 text-sm font-medium text-huddleroom-text-secondary">{decidedLabel}{submitting ? ' — submitting…' : ''}</p>
          ) : <>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button ref={(el) => { rowRefs.current[decision.id] = el }} type="button" className="min-h-11" disabled={disabled} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: agentDefinitionApproveOption(decision) })}>Approve</Button>
              <Button type="button" variant="secondary" className="min-h-11" aria-pressed={answer.choice === 'reject'} disabled={disabled} onClick={() => update(decision.id, { choice: answer.choice === 'reject' ? null : 'reject' })}>Reject</Button>
              <Button type="button" variant="secondary" className="min-h-11" aria-pressed={answer.choice === 'edit'} disabled={disabled} onClick={() => update(decision.id, { choice: answer.choice === 'edit' ? null : 'edit' })}>Edit</Button>
            </div>
            {answer.choice === 'reject' && <div className="mt-2">
              <Textarea label="Reason (required)" value={answer.reason} onChange={(event) => update(decision.id, { reason: event.target.value })} rows={2} required />
              <div className="mt-2 flex justify-end">
                <Button type="button" className="min-h-11" disabled={disabled || !canSubmitReason(answer.reason)} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: 'reject' })}>Confirm reject</Button>
              </div>
            </div>}
            {answer.choice === 'edit' && <div className="mt-2">
              <Textarea label="Description" value={answer.editedDescription} onChange={(event) => update(decision.id, { editedDescription: event.target.value })} rows={3} required />
              <Textarea label="Persona" value={answer.editedPersona} onChange={(event) => update(decision.id, { editedPersona: event.target.value })} rows={3} required />
              <div className="mt-2 flex justify-end">
                <Button type="button" className="min-h-11" disabled={disabled || !answer.editedDescription.trim() || !answer.editedPersona.trim()} onClick={() => decideAndMaybeSubmit(decision.id, { selectedOption: 'edit' })}>Confirm edit</Button>
              </div>
            </div>}
          </>}
        </article>
      })}
      {error && <div className="mt-2">
        <p className="text-xs text-huddleroom-status-red">{error}</p>
        {allComplete && <Button type="button" className="mt-2 min-h-11" disabled={disabled} onClick={() => submitIfComplete(answers)}>Retry submit</Button>}
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

function TeamHierarchyProposalForm({ decision, disabled, submitting, error, onSubmit, primaryRef }: {
  decision: OrchestrationAuthorityDecisionRecord
  disabled: boolean
  submitting: boolean
  error: string | null
  onSubmit: (option: string, editedAgent?: AgentCreatePayload) => void
  primaryRef?: (element: HTMLButtonElement | null) => void
}) {
  const definition = proposalDefinition(decision)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(() => ({ ...definition, capabilities: definition?.capabilities ?? [], config: definition?.config ?? {} }))
  const [config, setConfig] = useState(() => JSON.stringify(definition?.config ?? {}, null, 2))
  const [validation, setValidation] = useState<string | null>(null)
  const editRef = useRef<HTMLButtonElement>(null)
  const { announceError } = useGoalAnnouncer()
  useEffect(() => {
    if (!definition) { announceError('This proposal is missing its agent definition.'); return }
    const message = editing ? (validation || error) : error
    if (message) announceError(message)
  }, [definition, editing, validation, error, announceError])
  if (!definition) return <p className="mt-2 text-xs text-huddleroom-status-red">This proposal is missing its agent definition.</p>

  function submitEdit(event: FormEvent) {
    event.preventDefault()
    let parsed: unknown
    try { parsed = JSON.parse(config) } catch { setValidation('Config must be valid JSON.'); return }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') { setValidation('Config must be a JSON object.'); return }
    setValidation(null)
    onSubmit('edit', { ...draft, config: parsed as Record<string, unknown> } as AgentCreatePayload)
  }

  const field = (label: string, key: keyof AgentCreatePayload, required = false) => <label className="block text-xs text-huddleroom-text-secondary">{label}
    <input className="mt-1 min-h-11 w-full rounded border border-huddleroom-border bg-huddleroom-surface px-2 text-sm text-huddleroom-text-primary" required={required} value={String(draft[key] ?? '')} onChange={(event) => setDraft((current) => ({ ...current, [key]: event.target.value }))} />
  </label>
  const capabilities = Array.isArray(definition.capabilities)
    ? definition.capabilities.filter((capability): capability is string => typeof capability === 'string' && capability.trim().length > 0)
    : []
  return <div className="mt-2" aria-busy={submitting}>
    <Record keyWidth={84} rows={[
      { key: 'Role', value: definition.role ? <span className="font-semibold">{definition.role}</span> : null },
      { key: 'Name', value: definition.name || null },
      { key: 'Model', value: definition.model || null },
      { key: 'Provider', value: definition.provider || null },
      { key: 'Adapter', value: definition.adapter_type || null },
      { key: 'Description', value: definition.description ? <span className="block truncate" title={definition.description}>{definition.description}</span> : null },
    ]} />
    {capabilities.length > 0 && <div className="mt-2 flex flex-wrap gap-1">
      {capabilities.map((capability) => <Tag key={capability}>{capability}</Tag>)}
    </div>}
    {definition.system_prompt && <details className="mt-2">
      <summary className="cursor-pointer text-xs font-medium text-huddleroom-text-muted hover:text-huddleroom-text-secondary">Show prompt</summary>
      <pre className="mt-1 whitespace-pre-wrap break-words font-mono text-xs text-huddleroom-text-secondary">{definition.system_prompt}</pre>
    </details>}
    {!editing ? <div className="mt-3 flex flex-wrap gap-2">
      <Button ref={primaryRef} type="button" disabled={disabled} onClick={() => onSubmit('approve')}>Approve</Button>
      <Button ref={editRef} type="button" variant="secondary" disabled={disabled} onClick={() => setEditing(true)}>Edit</Button>
      <Button type="button" variant="secondary" disabled={disabled} onClick={() => onSubmit('reject')}>Reject</Button>
    </div> : <form className="mt-3 grid gap-2" onSubmit={submitEdit}>
      {field('Name', 'name', true)}{field('Role', 'role', true)}{field('Description', 'description')}{field('Provider', 'provider', true)}{field('Model', 'model', true)}
      <label className="block text-xs text-huddleroom-text-secondary">Adapter type<select className="mt-1 min-h-11 w-full rounded border border-huddleroom-border bg-huddleroom-surface px-2" value={draft.adapter_type ?? 'api'} onChange={(event) => setDraft((current) => ({ ...current, adapter_type: event.target.value as AgentCreatePayload['adapter_type'] }))}><option value="api">api</option><option value="cli">cli</option><option value="routine">routine</option></select></label>
      {field('CLI runtime', 'cli_runtime')}
      <Textarea label="Capabilities (one per line)" value={(draft.capabilities ?? []).join('\n')} onChange={(event) => setDraft((current) => ({ ...current, capabilities: event.target.value.split('\n').filter(Boolean) }))} rows={3} />
      <Textarea label="System prompt" value={draft.system_prompt ?? ''} onChange={(event) => setDraft((current) => ({ ...current, system_prompt: event.target.value }))} rows={8} />
      <Textarea label="Config (JSON object)" value={config} onChange={(event) => setConfig(event.target.value)} rows={5} />
      {(validation || error) && <p className="text-xs text-huddleroom-status-red">{validation || error}</p>}
      <div className="flex justify-end gap-2"><Button type="button" variant="secondary" disabled={disabled} onClick={() => { setEditing(false); queueMicrotask(() => editRef.current?.focus()) }}>Cancel</Button><Button type="submit" disabled={disabled}>{submitting ? 'Submitting…' : 'Submit edit'}</Button></div>
    </form>}
    {!editing && error && <p className="mt-2 text-xs text-huddleroom-status-red">{error}</p>}
  </div>
}

function TeamHierarchyStatus({ decisions, processId }: { decisions: readonly OrchestrationAuthorityDecisionRecord[]; processId: string | null }) {
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
  const list = (key: string) => Array.isArray(proposal[key]) ? proposal[key] as unknown[] : []
  return <section className="rounded border border-huddleroom-border p-3"><h3 className="text-sm font-medium text-huddleroom-text-primary">Proposed hierarchy</h3>
    <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Assignments</p><pre className="max-w-full overflow-hidden whitespace-pre-wrap break-words text-xs text-huddleroom-text-secondary">{JSON.stringify(list('assignments'), null, 2)}</pre>
    <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Reporting</p><pre className="max-w-full overflow-hidden whitespace-pre-wrap break-words text-xs text-huddleroom-text-secondary">{JSON.stringify(list('reporting_lines'), null, 2)}</pre>
    <p className="mt-2 text-xs"><span className="font-medium text-huddleroom-text-muted">Rationale:</span> {String(proposal.rationale ?? 'None')}</p><p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Self-review:</span> {String(proposal.self_review ?? 'None')}</p>
    <p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Documented gaps:</span> {list('documented_gaps').length ? list('documented_gaps').join(', ') : 'None'}</p><p className="mt-1 text-xs"><span className="font-medium text-huddleroom-text-muted">Approved but unused:</span> {list('approved_but_unused_agent_ids').length ? list('approved_but_unused_agent_ids').join(', ') : 'None'}</p>
  </section>
}

export function BaselineDashboard({ projectId, goalId, goal, run, gates, mutationBusy, acquireMutation, releaseMutation, onMessage, onSelectedProcessTypeChange, onViewActivity, renderBlockerRecovery, suppressQueueAnnouncement }: BaselineDashboardProps) {
  const dashboard = useBaselineDashboard(projectId, goalId)
  const health = useOrchestrationHealth()
  const answer = useAnswerOrchestrationDecision(projectId)
  const answerAgentDefinitionBatch = useAnswerAgentDefinitionReviewBatch(projectId)
  const skip = useSkipOrchestrationProcess(projectId)
  const acknowledge = useAcknowledgeOrchestrationWarning(projectId)
  const resolve = useResolveOrchestrationWarning(projectId)
  const runStep = useRunOrchestrationBaselineStep(projectId)
  const rerun = useRerunOrchestrationBaselineStep(projectId)
  const retry = useRetryOrchestrationBaselineStep(projectId)
  const authorize = useAuthorizeOrchestrationBaseline(projectId)
  const [selectedProcessType, setSelectedProcessType] = useState<OrchestrationBaselineProcessType | null>(null)
  const [openAction, setOpenAction] = useState<BaselineFocusAction | null>(null)
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [reason, setReason] = useState('')
  const [formError, setFormError] = useState<string | null>(null)
  const [formErrorSource, setFormErrorSource] = useState<'action' | 'batch' | null>(null)
  const [runRerunError, setRunRerunError] = useState<string | null>(null)
  const [retryError, setRetryError] = useState<string | null>(null)
  const [authorizeError, setAuthorizeError] = useState<string | null>(null)
  const selectedGoal = useRef<string | null>(null)
  const actionTriggers = useRef<Record<string, HTMLButtonElement | null>>({})
  const focusHeadingRef = useRef<HTMLHeadingElement>(null)
  const firstOptionRef = useRef<HTMLInputElement>(null)
  const reasonRef = useRef<HTMLTextAreaElement>(null)
  const submitRef = useRef<HTMLButtonElement>(null)
  const processes = currentProcesses(dashboard.processes.data)
  const decisions = dashboard.decisions.data ?? []
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
    queueMicrotask(() => actionTriggers.current[`${action.kind}:${action.id}`]?.focus())
  }

  function restoreSuccessFocus(action: BaselineFocusAction) {
    queueMicrotask(() => {
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

  function runMutation<T>(action: BaselineFocusAction, mutate: (callbacks: { onSuccess: (data: T) => void; onError: (error: unknown) => void; onSettled: () => void }) => void, success: string, onSuccessFocus?: () => void) {
    const owner = actionOwner(action)
    if (!acquireMutation(owner)) return
    // Snapshot whether `action` is the form currently open — runMutation is
    // now also called for actions with no open form at all (the accept-
    // recommendation fast path). Closing/clearing openAction+formError must
    // stay scoped to the action that was actually acted on, so accepting one
    // row can't silently close or wipe an unrelated row's open form.
    const targetsOpenForm = openAction !== null && openAction.kind === action.kind && openAction.id === action.id
    if (targetsOpenForm) { setFormError(null); setFormErrorSource(null) }
    mutate({
      onSuccess: () => {
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
        if (targetsOpenForm) { setFormError(actionError(error)); setFormErrorSource('action'); submitRef.current?.focus() }
        else onMessage(actionError(error))
      },
      onSettled: () => releaseMutation(owner),
    })
  }

  // onSuccessFocus lets a caller other than ProcessFocus (the Needs-you queue)
  // override where focus lands after a successful submit — ProcessFocus's Skip
  // keeps the default (same trigger, else the process heading); queue rows move
  // focus to the next queue row instead (see NeedsYouQueue).
  function renderActionForm(action: BaselineFocusAction, record: OrchestrationAuthorityDecisionRecord | OrchestrationWarningRecord | OrchestrationProcessRunRecord, onSuccessFocus?: () => void) {
    let target: ActionTarget | null = null
    if (action.kind === 'answer') target = { kind: 'answer', action, decision: record as OrchestrationAuthorityDecisionRecord }
    else if (action.kind === 'skip') target = { kind: 'skip', action, process: record as OrchestrationProcessRunRecord }
    else if (action.kind === 'acknowledge' || action.kind === 'resolve') target = { kind: action.kind, action, warning: record as OrchestrationWarningRecord }
    if (!target) return null
    return <BaselineActionForm
      target={target} mutationBusy={mutationBusy}
      submitting={action.kind === 'answer' ? answer.isPending : action.kind === 'skip' ? skip.isPending : action.kind === 'acknowledge' ? acknowledge.isPending : resolve.isPending}
      selectedOption={selectedOption} reason={reason} error={formErrorSource === 'action' ? formError : null}
      firstOptionRef={firstOptionRef} reasonRef={reasonRef} submitRef={submitRef}
      onAnswer={(decision, option, submitReason) => runMutation<OrchestrationDecisionAnswerResult>(action, (callbacks) => answer.mutate({
        goalId,
        decisionId: decision.id,
        selectedOption: option,
        reason: submitReason,
      }, callbacks), 'Answer received.', onSuccessFocus)}
      onSkip={(process, submitReason) => runMutation(action, (callbacks) => skip.mutate({ goalId, processType: process.process_type, reason: submitReason }, callbacks), 'Process skipped.', onSuccessFocus)}
      onAcknowledge={(warning, submitReason) => runMutation(action, (callbacks) => acknowledge.mutate({ goalId, warningId: warning.id, reason: submitReason }, callbacks), 'Risk acknowledged.', onSuccessFocus)}
      onResolve={(warning, submitReason) => runMutation(action, (callbacks) => resolve.mutate({ goalId, warningId: warning.id, reason: submitReason }, callbacks), 'Risk resolved.', onSuccessFocus)}
      onSelectedOptionChange={setSelectedOption} onReasonChange={setReason} onCancel={() => closeForm(action)}
    />
  }

  // Accept-recommendation fast path (spec C): submits decision.recommendation
  // immediately — no form, no reason. Reuses runMutation so it shares the
  // same mutation-owner locking, error surfacing, and focus handling as the
  // full answer form.
  function acceptRecommendation(decision: OrchestrationAuthorityDecisionRecord, onSuccessFocus?: () => void) {
    if (decision.recommendation === null) return
    const action: BaselineFocusAction = { kind: 'answer', id: decision.id }
    runMutation<OrchestrationDecisionAnswerResult>(action, (callbacks) => answer.mutate({
      goalId,
      decisionId: decision.id,
      selectedOption: decision.recommendation!,
      reason: '',
    }, callbacks), 'Answer received.', onSuccessFocus)
  }

  // Goal-definition step-through card (spec D): submits one adaptive
  // clarification question's free-text answer. Doesn't route through
  // runMutation/openAction — the card lives entirely in the Needs-you queue
  // and manages its own local question index, so this only needs to own the
  // mutation-owner lock and error surfacing.
  function answerGoalDefinitionQuestion(
    decision: OrchestrationAuthorityDecisionRecord,
    answerText: string,
    callbacks: { onSuccess: () => void; onError: (error: unknown) => void; onSettled: () => void },
  ) {
    const owner: BaselineMutationOwner = `baseline:answer:${decision.id}`
    if (!acquireMutation(owner)) return
    setFormError(null)
    setFormErrorSource(null)
    answer.mutate({ goalId, decisionId: decision.id, selectedOption: answerText, reason: '' }, {
      onSuccess: () => { onMessage('Answer received.'); callbacks.onSuccess() },
      onError: (error) => { setFormError(actionError(error)); setFormErrorSource('action'); callbacks.onError(error) },
      onSettled: () => { releaseMutation(owner); callbacks.onSettled() },
    })
  }

  function submitAgentDefinitionReviewBatch(answers: AgentDefinitionReviewBatchAnswerInput[], onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = 'baseline:answer:agent-definition-review-batch'
    if (!acquireMutation(owner)) return
    setFormError(null)
    setFormErrorSource(null)
    answerAgentDefinitionBatch.mutate({ goalId, answers }, {
      onSuccess: () => {
        setOpenAction(null)
        onMessage('Agent-definition review completed.')
        queueMicrotask(onSuccessFocus ?? (() => focusHeadingRef.current?.focus()))
      },
      onError: (error) => {
        if (error instanceof ApiError && error.status === 409) {
          void Promise.all([dashboard.decisions.refetch(), dashboard.processes.refetch(), dashboard.warnings.refetch()])
          onMessage('The proposals changed and were reloaded — please review the updated agent-definition proposals.')
          queueMicrotask(() => focusHeadingRef.current?.focus())
        } else {
          setFormError(actionError(error))
          setFormErrorSource('batch')
        }
      },
      onSettled: () => releaseMutation(owner),
    })
  }

  function answerTeamProposal(decision: OrchestrationAuthorityDecisionRecord, selectedOption: string, editedAgent: AgentCreatePayload | undefined, onSuccess?: () => void) {
    const action: BaselineFocusAction = { kind: 'answer', id: decision.id }
    runMutation<OrchestrationDecisionAnswerResult>(action, (callbacks) => answer.mutate({ goalId, decisionId: decision.id, selectedOption, reason: '', editedAgent }, callbacks), 'Proposal decision received.', onSuccess)
  }

  function mutateChain(processType: OrchestrationBaselineProcessType, action: 'run' | 'rerun', trigger: HTMLButtonElement | null) {
    const owner: BaselineMutationOwner = `baseline:${action}:${processType}`
    if (!acquireMutation(owner)) return
    setRunRerunError(null)
    const mutation = action === 'run' ? runStep : rerun
    mutation.mutate({ goalId, processType }, {
      onSuccess: () => { onMessage(`${action === 'run' ? 'Running' : 'Re-running'} ${processLabel(processType)}.`); setSelectedProcessType(processType) },
      onError: (mutationError) => {
        const message = actionError(mutationError)
        setRunRerunError(message)
        onMessage(message)
      },
      onSettled: () => {
        releaseMutation(owner)
        trigger?.focus()
      },
    })
  }

  // Task 12b: gates the baseline chain per run.baseline_authorized — the
  // button itself lives in ProcessChain's Z2 header, this just owns the
  // mutation the same way mutateChain/retryProcess do.
  function authorizeBaseline(trigger: HTMLButtonElement | null) {
    const owner: BaselineMutationOwner = `baseline:authorize:${goalId}`
    if (!acquireMutation(owner)) return
    setAuthorizeError(null)
    authorize.mutate({ goalId }, {
      onSuccess: () => onMessage('Baseline started.'),
      onError: (mutationError) => {
        const message = actionError(mutationError)
        setAuthorizeError(message)
        onMessage(message)
      },
      onSettled: () => {
        releaseMutation(owner)
        trigger?.focus()
      },
    })
  }

  // Task 12: stale-inputs suggestion row wiring (Approve/Dismiss in the
  // Needs-you queue) — Approve reruns the step then resolves the warning so
  // it never resurfaces for the process run that triggered the rerun; Dismiss
  // just resolves it. Both reuse the existing rerun/resolve mutations, no new
  // endpoints.
  function approveStaleInputs(warningRecord: OrchestrationWarningRecord, processType: OrchestrationBaselineProcessType, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:rerun:${processType}`
    if (!acquireMutation(owner)) return
    rerun.mutate({ goalId, processType }, {
      onSuccess: () => {
        resolve.mutate({ goalId, warningId: warningRecord.id, reason: 'approved rerun' }, {
          onSuccess: () => { onMessage(`Re-running ${processLabel(processType)}.`); onSuccessFocus?.() },
          onError: (error) => onMessage(actionError(error)),
          onSettled: () => releaseMutation(owner),
        })
      },
      onError: (error) => {
        onMessage(actionError(error))
        releaseMutation(owner)
      },
    })
  }

  function dismissStaleInputs(warningRecord: OrchestrationWarningRecord, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:resolve:${warningRecord.id}`
    if (!acquireMutation(owner)) return
    resolve.mutate({ goalId, warningId: warningRecord.id, reason: 'dismissed by human' }, {
      onSuccess: () => { onMessage('Suggestion dismissed.'); onSuccessFocus?.() },
      onError: (error) => onMessage(actionError(error)),
      onSettled: () => releaseMutation(owner),
    })
  }

  function retryProcess(processType: OrchestrationBaselineProcessType, trigger: HTMLButtonElement | null, onSuccessFocus?: () => void) {
    const owner: BaselineMutationOwner = `baseline:retry:${processType}`
    if (!acquireMutation(owner)) return
    setRetryError(null)
    retry.mutate({ goalId, processType }, {
      onSuccess: (result) => {
        if (result?.process?.['retry_failed'] === true) {
          const message = typeof result.process['error'] === 'string' ? result.process['error'] : 'Language model request failed.'
          setRetryError(message)
          onMessage(message)
          queueMicrotask(() => trigger?.focus())
          return
        }
        onMessage(`Request succeeded. Continuing ${processLabel(processType)}.`)
        queueMicrotask(onSuccessFocus ?? (() => focusHeadingRef.current?.focus()))
      },
      onError: (error) => {
        if (error instanceof ApiError && error.status === 409) {
          void Promise.all([dashboard.processes.refetch(), dashboard.warnings.refetch()])
          onMessage('This retry is no longer available. Refreshing process state.')
          queueMicrotask(() => focusHeadingRef.current?.focus())
        } else {
          const message = actionError(error)
          setRetryError(message)
          onMessage(message)
          queueMicrotask(() => trigger?.focus())
        }
      },
      onSettled: () => releaseMutation(owner),
    })
  }

  return <div data-testid="baseline-dashboard" className="flex min-w-0 flex-col gap-4">
    <TeamHierarchyStatus decisions={decisions} processId={processes.find((process) => process.process_type === 'team_hierarchy' && !process.superseded_by_id)?.id ?? null} />
    <ProcessChain processes={processes} processState={processState} selectedProcessType={effectiveProcessType} onSelect={setSelectedProcessType} goal={goal} checkpointItems={checkpointItems} run={run} onAuthorize={authorizeBaseline} authorizePending={authorize.isPending} authorizeError={authorizeError} />
    {processState === 'ready' && <>
      <NeedsYouQueue
        goal={goal} run={run} processes={processes} warnings={warnings} gates={gates}
        checkpointItems={checkpointItems} checkpointLoaded={dashboard.checkpoint.isSuccess}
        queriesErrored={queueQueriesErrored} queriesUnready={queueQueriesUnready}
        deferredCount={deferredCount} isWorking={isWorking}
        mutationBusy={mutationBusy} onSelectProcess={setSelectedProcessType}
        openAction={openAction} onOpenAction={openForm} onActionTrigger={(action, element) => { actionTriggers.current[`${action.kind}:${action.id}`] = element }}
        renderActionForm={renderActionForm}
        renderAgentDefinitionReviewBatch={(batchDecisions, onSuccessFocus) => <AgentDefinitionReviewBatchForm key={batchDecisions.map((decision) => decision.id).sort().join(',')} decisions={batchDecisions} mutationBusy={mutationBusy} submitting={answerAgentDefinitionBatch.isPending} error={formErrorSource === 'batch' ? formError : null} onSubmit={(answers) => submitAgentDefinitionReviewBatch(answers, onSuccessFocus)} />}
        renderTeamHierarchyProposal={(decision, onSuccess, registerPrimary) => {
          const isActive = openAction?.kind === 'answer' && openAction?.id === decision.id
          return <TeamHierarchyProposalForm decision={decision} disabled={mutationBusy} submitting={answer.isPending} error={isActive && formErrorSource === 'action' ? formError : null} onSubmit={(option, editedAgent) => answerTeamProposal(decision, option, editedAgent, onSuccess)} primaryRef={registerPrimary} />
        }}
        onAcceptDecision={acceptRecommendation} onAnswerGoalDefinitionQuestion={answerGoalDefinitionQuestion}
        retryPending={retry.isPending} retryError={retryError} onRetryProcess={retryProcess}
        onApproveStaleInputs={approveStaleInputs} onDismissStaleInputs={dismissStaleInputs}
        renderBlockerRecovery={renderBlockerRecovery} suppressAnnouncement={suppressQueueAnnouncement}
      />
      <ProcessFocus selectedProcessType={effectiveProcessType} process={selectedProcess} goal={goal} decisions={decisions} decisionsUnready={dashboard.decisions.isLoading || dashboard.decisions.isError} warnings={warnings} reviews={dashboard.agentReviews.data ?? []} debug={debug} canSkip={!!selectedProcess && goal.status !== 'completed' && goal.status !== 'cancelled' && (selectedProcess.status === 'running' || selectedProcess.status === 'waiting_decision')} mutationBusy={mutationBusy} canRun={canRunSelected} canRerun={canRerunSelected} runPending={runStep.isPending} rerunPending={rerun.isPending} runRerunError={runRerunError} onRun={(trigger) => mutateChain(effectiveProcessType, 'run', trigger)} onRerun={(trigger) => mutateChain(effectiveProcessType, 'rerun', trigger)} retryPending={retry.isPending} retryError={retryError} onRetry={(trigger) => retryProcess(effectiveProcessType, trigger)} openAction={openAction} onOpenAction={openForm} onActionTrigger={(action, element) => { actionTriggers.current[`${action.kind}:${action.id}`] = element }} headingRef={focusHeadingRef} renderActionForm={renderActionForm} onViewActivity={onViewActivity} />
    </>}
  </div>
}
