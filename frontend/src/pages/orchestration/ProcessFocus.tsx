import { Suspense, lazy, useEffect, useState, type ReactNode, type RefObject } from 'react'
import { Button, StatusBadge } from '@/components/common/uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'
import type {
  OrchestrationAgentReviewRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { errorRecord, processDescription, processLabel, statusPhrase, stepDisplayLabel } from './humanize'
const HierarchyGraph = lazy(() => import('./HierarchyGraph').then((m) => ({ default: m.HierarchyGraph })))
import { ZONE_TITLE_CLASS } from './zoneTitle'
import type { BaselineFocusAction } from './BaselineDashboard'
import { useGoalAnnouncer } from './goalAnnouncer'

// Shared retry button + pending/error UI. Reused by the queue in Phase C.
export function RetryControl({
  onRetry,
  retryPending,
  retryError,
  disabled,
  buttonRef,
}: {
  onRetry: (trigger: HTMLButtonElement | null) => void
  retryPending: boolean
  retryError: string | null
  disabled: boolean
  // Lets a caller (the Needs-you queue) register this button for its own
  // focus-management, on top of the disabled-sync ref this component already needs.
  buttonRef?: (element: HTMLButtonElement | null) => void
}) {
  const { announceError } = useGoalAnnouncer()
  useEffect(() => { if (retryError) announceError(errorRecord(retryError).what) }, [retryError, announceError])
  return (
    <div className="flex flex-col items-start gap-2">
      <p className="text-xs text-huddleroom-text-muted">Retry continues the failed request and keeps this run. Re-run starts a fresh run from the saved inputs.</p>
      <Button
        ref={(element) => { if (element) element.disabled = disabled; buttonRef?.(element) }}
        type="button"
        size="sm"
        className="min-h-11"
        disabled={disabled}
        onClick={(event) => { if (!disabled) onRetry(event.currentTarget) }}
      >
        {retryPending ? 'Retrying…' : 'Retry'}
      </Button>
      {retryError && (
        <div className="text-sm text-huddleroom-status-red">
          <p>{errorRecord(retryError).what}</p>
          <details className="mt-1 text-xs text-huddleroom-text-muted">
            <summary className="cursor-pointer">Details</summary>
            <code className="mt-1 block whitespace-pre-wrap break-all font-mono">{errorRecord(retryError).details}</code>
          </details>
        </div>
      )}
    </div>
  )
}

// Reused by NeedsYouQueue (source_process_run_id filtering) and ActivityLog
// (per-step timeline entries).
export function scoped<T extends { source_process_run_id: string | null }>(rows: readonly T[], process: OrchestrationProcessRunRecord | undefined) {
  return process ? rows.filter((row) => row.source_process_run_id === process.id) : []
}

function reviewCards(reviews: readonly OrchestrationAgentReviewRecord[], process: OrchestrationProcessRunRecord | undefined) {
  return scoped(reviews, process).map((review) => {
    const snapshot = review.definition_snapshot
    const name = typeof snapshot.name === 'string' ? snapshot.name : 'Agent'
    const role = typeof snapshot.role === 'string' ? snapshot.role : null
    const provider = typeof snapshot.provider === 'string' ? snapshot.provider : null
    const model = typeof snapshot.model === 'string' ? snapshot.model : null
    const capabilities = Array.isArray(snapshot.capabilities) ? snapshot.capabilities.filter((item): item is string => typeof item === 'string') : []
    const tools = isRecord(snapshot.config) ? stringItems(snapshot.config.tools) : []
    return {
      id: review.id, name, role, provider, model, capabilities, tools, assessment: review.fit_summary,
      proposed: stringItems(review.proposed_work_functions),
      approved: stringItems(review.approved_for_work_functions),
      risks: stringItems(review.risks),
      recommendations: stringItems(review.recommended_changes),
    }
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringItems(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.length > 0) : []
}

function displayValue(value: unknown, fallback = '—') {
  return typeof value === 'string' && value.trim() ? value : typeof value === 'number' || typeof value === 'boolean' ? String(value) : fallback
}

function looksLikeUuid(value: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
}

function processError(process: OrchestrationProcessRunRecord | undefined) {
  if (!process) return null
  const outputs = isRecord(process.outputs) ? process.outputs : null
  if (!outputs || typeof outputs.error !== 'string' || !outputs.error.trim()) return null
  const label = process.process_type === 'goal_definition' ? 'Goal analysis' : processLabel(process.process_type)
  return `${label} failed: ${outputs.error}`
}

function hierarchyItems(value: unknown, resolveAgent: (value: unknown) => string | null) {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!isRecord(item)) return [displayValue(item, '')].filter(Boolean)
    const parts = [item.work_function, item.role, resolveAgent(item.agent_id), item.reason].map((part) => displayValue(part, '')).filter(Boolean)
    return parts.length > 0 ? [parts.join(' · ')] : []
  })
}

// Shared by HierarchyCard (text, always rendered — also the a11y fallback)
// and HierarchyGraph (the reactflow diagram, guarded on this data being
// complete enough to draw).
function parseHierarchy(process: OrchestrationProcessRunRecord, reviews: readonly OrchestrationAgentReviewRecord[]) {
  const outputs = isRecord(process.outputs) ? process.outputs : {}
  const snapshot = process.status === 'waiting_decision' && isRecord(outputs.proposal) ? outputs.proposal : outputs
  const hierarchy = isRecord(snapshot.hierarchy) ? snapshot.hierarchy : snapshot
  const agentNames = new Map(reviews.flatMap((review) => {
    if (!review.agent_id) return []
    const name = typeof review.definition_snapshot.name === 'string' && review.definition_snapshot.name.trim()
      ? review.definition_snapshot.name
      : stringItems(review.approved_for_work_functions)[0]
    return name ? [[review.agent_id, name] as const] : []
  }))
  const resolveAgent = (value: unknown) => {
    if (typeof value !== 'string' || !value.trim()) return null
    return agentNames.get(value) ?? (looksLikeUuid(value) ? null : value)
  }
  const manager = isRecord(hierarchy.manager)
    ? displayValue(hierarchy.manager.name, resolveAgent(hierarchy.manager.id) ?? displayValue(hierarchy.manager.kind))
    : resolveAgent(snapshot.manager) ?? resolveAgent(hierarchy.manager) ?? 'Not recorded'
  const roles = isRecord(snapshot.role_to_agent) ? Object.entries(snapshot.role_to_agent).flatMap(([role, agent]) => {
    const name = resolveAgent(agent)
    return name ? [{ role, agent: name }] : []
  }) : []
  const missing = hierarchyItems(snapshot.missing_work_functions, resolveAgent)
  const weakFits = hierarchyItems(snapshot.weak_fits, resolveAgent)
  const verificationRecord = isRecord(snapshot.independent_verification) ? snapshot.independent_verification : null
  const verification = !verificationRecord
    ? 'Independent verification not recorded'
    : verificationRecord.required === false
      ? 'Independent verification not required'
      : verificationRecord.available === false || verificationRecord.possible === false
        ? 'Independent verification unavailable'
        : verificationRecord.available === true || verificationRecord.possible === true
          ? 'Independent verification available'
          : 'Independent verification not recorded'
  return { manager, roles, missing, weakFits, verification }
}

function HierarchyCard({ process, reviews }: { process: OrchestrationProcessRunRecord; reviews: readonly OrchestrationAgentReviewRecord[] }) {
  const { manager, roles, missing, weakFits, verification } = parseHierarchy(process, reviews)
  return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
    <h3 className="font-semibold text-huddleroom-text-primary">How the orchestrator structured the team</h3>
    <p className="mt-1 text-xs text-huddleroom-text-muted">Independent verification means a different agent than the one who did the work checks it before it's accepted.</p>
    <p className="mt-1">Manager: {manager}</p>
    {roles.map(({ role, agent }) => <p key={role} className="mt-1">{role}: {agent}</p>)}
    {missing.length > 0 && <p className="mt-1">Missing work functions: {missing.join(', ')}</p>}
    {weakFits.length > 0 && <p className="mt-1">Weak fits: {weakFits.join(', ')}</p>}
    <p className="mt-1">{verification}</p>
  </article>
}

function ManagerCard({ process, reviews }: { process: OrchestrationProcessRunRecord; reviews: readonly OrchestrationAgentReviewRecord[] }) {
  const outputs = isRecord(process.outputs) ? process.outputs : {}
  const agentNames = new Map(reviews.flatMap((review) => {
    if (!review.agent_id) return []
    const name = typeof review.definition_snapshot.name === 'string' && review.definition_snapshot.name.trim()
      ? review.definition_snapshot.name
      : stringItems(review.approved_for_work_functions)[0]
    return name ? [[review.agent_id, name] as const] : []
  }))
  const resolveAgent = (value: unknown) => {
    if (typeof value !== 'string' || !value.trim()) return null
    return agentNames.get(value) ?? (looksLikeUuid(value) ? null : value)
  }
  // Backend stores selected_manager as "agent:<uuid>", "human:<uuid>", "human", or null (spec 8).
  const rawManager = typeof outputs.selected_manager === 'string' ? outputs.selected_manager : ''
  const [managerKind, managerId] = rawManager.includes(':') ? [rawManager.slice(0, rawManager.indexOf(':')), rawManager.slice(rawManager.indexOf(':') + 1)] : [rawManager, '']
  const managerDisplay = managerKind === 'human'
    ? 'You (human)'
    : managerKind === 'agent' && managerId
      ? resolveAgent(managerId) ?? 'Unknown agent'
      : 'None selected'
  const authorityModel = typeof outputs.authority_model === 'string' ? outputs.authority_model.replace(/_/g, ' ').split(' ').map((w) => `${w[0]?.toUpperCase()}${w.slice(1)}`).join(' ') : displayValue(outputs.authority_model)
  const fitRationale = displayValue(outputs.manager_fit_rationale, '')
  // Candidate objects carry key/label/score/signals (no name/reason); label is "<name> (<role>)".
  const candidates = Array.isArray(outputs.candidates) ? outputs.candidates.flatMap((item) => {
    if (!isRecord(item)) return []
    const label = displayValue(item.label, '')
    return label ? [label] : []
  }) : []

  return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
    <h3 className="font-semibold text-huddleroom-text-primary">Who the orchestrator picked to manage this goal</h3>
    <p className="mt-1">Manager: {managerDisplay}</p>
    <p className="mt-1">Authority model: {authorityModel}</p>
    {fitRationale && <p className="mt-1">Fit rationale: {fitRationale}</p>}
    {candidates.length > 0 && <p className="mt-1">Candidates considered: {candidates.join(', ')}</p>}
  </article>
}

// Never renders an empty shell: every branch below (missing process, thin
// outputs) falls back to the static PROCESS_DESCRIPTIONS sentence.
function OutcomeFallback({ type }: { type: OrchestrationBaselineProcessType }) {
  return <p className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">{processDescription(type)}</p>
}

function GoalDefinitionCard({ process, goal }: { process: OrchestrationProcessRunRecord; goal: OrchestrationGoal }) {
  const outputs = isRecord(process.outputs) ? process.outputs : {}
  if (outputs.clarification_limit_reached === true) {
    const unresolved = stringItems(outputs.unresolved_questions)
    return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
      <h3 className="font-semibold text-huddleroom-text-primary">How the orchestrator scoped this goal</h3>
      <p className="mt-1">The orchestrator is proceeding on its best understanding — material ambiguity remained after clarification rounds, and all prior answers are preserved.</p>
      {unresolved.length > 0 && <>
        <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Unresolved questions</p>
        <ul className="mt-1 space-y-1">
          {unresolved.map((question, index) => <li key={index}>{question}</li>)}
        </ul>
      </>}
    </article>
  }
  const clarifications = Array.isArray(outputs.clarifications) ? outputs.clarifications.filter(isRecord) : []
  return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
    <h3 className="font-semibold text-huddleroom-text-primary">How the orchestrator scoped this goal</h3>
    <p className="mt-1">Weight: {displayValue(outputs.weight, goal.weight)}</p>
    {clarifications.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Clarifications</p>
      <ol className="mt-1 list-decimal space-y-1.5 pl-5">
        {clarifications.map((clarification, index) => <li key={index} className="min-w-0 break-words">
          <p>{displayValue(clarification.question, 'Question not recorded')}</p>
          <p className="text-xs text-huddleroom-text-muted">Answer: {displayValue(clarification.answer, '—')}{clarification.destination ? ` (→ ${displayValue(clarification.destination, '')})` : ''}</p>
        </li>)}
      </ol>
    </>}
    {goal.success_criteria.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Success criteria</p>
      <ol className="mt-1 list-decimal space-y-1 pl-5">
        {goal.success_criteria.map((criterion, index) => <li key={String(criterion.key ?? index)} className="min-w-0 break-words">{criterion.description ?? String(criterion.key ?? `criterion-${index + 1}`)}</li>)}
      </ol>
    </>}
  </article>
}

function EffectivenessReviewCard({ process }: { process: OrchestrationProcessRunRecord }) {
  const outputs = isRecord(process.outputs) ? process.outputs : {}
  const triggers = Array.isArray(outputs.triggers) ? outputs.triggers.filter(isRecord) : []
  const checks = Array.isArray(outputs.checks) ? outputs.checks.filter(isRecord) : []
  const recommended = displayValue(outputs.recommended_disposition, '')
  const decided = displayValue(outputs.selected_disposition, '')
  const decisionReason = displayValue(outputs.decision_reason, '')
  return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
    <h3 className="font-semibold text-huddleroom-text-primary">Why the orchestrator flagged this goal for review</h3>
    {triggers.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Triggers</p>
      <ul className="mt-1 space-y-1">
        {triggers.map((trigger, index) => <li key={index}>{displayValue(trigger.detail, displayValue(trigger.name, 'Trigger'))}</li>)}
      </ul>
    </>}
    {checks.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Checks</p>
      <ul className="mt-1 space-y-1">
        {checks.map((check, index) => <li key={index} className="flex items-start gap-1.5">
          <span aria-hidden="true" style={{ color: check.passed === true ? STATUS_COLORS.green : STATUS_COLORS.red }}>{check.passed === true ? '✓' : '✗'}</span>
          <span>{displayValue(check.detail, displayValue(check.name, 'Check'))}</span>
        </li>)}
      </ul>
    </>}
    {(recommended || decided) && <p className="mt-2">Recommended: {recommended || '—'} → Decided: {decided || 'not yet decided'}</p>}
    {decisionReason && <p className="mt-1">Reason: {decisionReason}</p>}
  </article>
}

// Real decision.context keys for the goal_closeout signoff decision
// (orchestration_service.py `_completion_manifest` / goal_closeout.py
// `_advance_full_closeout`): declared_success_criteria, criterion_evidence
// (a list of {criterion_key, evidence_ids}), accepted_risks (a list of
// {warning_id, type, severity, acknowledged_by} — the closeout preconditions'
// warning_disposition), overridden_gates.
function parseCloseoutContext(context: string | null) {
  if (!context) return null
  try {
    const parsed = JSON.parse(context)
    if (parsed && typeof parsed === 'object' && ('criterion_evidence' in parsed || 'accepted_risks' in parsed)) return parsed as Record<string, unknown>
  } catch { /* not JSON */ }
  return null
}

function GoalCloseoutCard({ process, decisions, decisionsUnready }: { process: OrchestrationProcessRunRecord; decisions: readonly OrchestrationAuthorityDecisionRecord[]; decisionsUnready: boolean }) {
  const outputs = isRecord(process.outputs) ? process.outputs : {}
  if (outputs.mode === 'cancellation') {
    return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
      <h3 className="font-semibold text-huddleroom-text-primary">Why this goal was cancelled</h3>
      <p className="mt-1">This goal was cancelled before completion.</p>
    </article>
  }
  const signoffId = typeof outputs.signoff_decision_id === 'string' ? outputs.signoff_decision_id : null
  const decision = signoffId ? decisions.find((candidate) => candidate.id === signoffId) : undefined
  // A signoff_decision_id with no matching decision is only "no sign-off
  // required" when the decisions query has actually settled successfully —
  // `decisionsUnready` covers both still-loading AND errored (a bare
  // isLoading boolean can't tell "loaded, genuinely no signoff" apart from
  // "failed to load"; readiness upstream only gates on processes, not
  // decisions). A signoff id with no match after a successful load is a data
  // inconsistency, not a loading state — falls through to today's text.
  if (signoffId && !decision && decisionsUnready) return <OutcomeFallback type="goal_closeout" />
  if (!signoffId || !decision) {
    return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
      <h3 className="font-semibold text-huddleroom-text-primary">Why this goal was allowed to close</h3>
      <p className="mt-1">Completed automatically — a trivial goal with accepted gates does not require a sign-off decision.</p>
    </article>
  }
  const approved = decision.selected_option === 'approve_completion'
  const authority = decision.authority === 'manager' ? 'the manager' : decision.authority === 'human' ? 'you' : decision.authority
  const context = parseCloseoutContext(decision.context)
  const criterionEvidence = context && Array.isArray(context.criterion_evidence) ? context.criterion_evidence.filter(isRecord) : []
  const acceptedRisks = context && Array.isArray(context.accepted_risks) ? context.accepted_risks.filter(isRecord) : []
  return <article className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
    <h3 className="font-semibold text-huddleroom-text-primary">{approved ? 'Why this goal was allowed to close' : 'Why this goal was kept open'}</h3>
    <p className="mt-1">{approved ? `Approved by ${authority}` : `Kept open by ${authority}`}{decision.reason ? `: ${decision.reason}` : ''}</p>
    {criterionEvidence.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Criteria evidence</p>
      <ul className="mt-1 space-y-1">
        {criterionEvidence.map((item, index) => <li key={index}>{displayValue(item.criterion_key, 'criterion')}: {stringItems(item.evidence_ids).length} evidence item(s)</li>)}
      </ul>
    </>}
    {acceptedRisks.length > 0 && <>
      <p className="mt-2 text-xs font-medium text-huddleroom-text-muted">Accepted risks</p>
      <ul className="mt-1 space-y-1">
        {acceptedRisks.map((risk, index) => <li key={index}>{displayValue(risk.type, 'Risk')} · {displayValue(risk.severity, '')}{risk.acknowledged_by ? ' · acknowledged' : ''}</li>)}
      </ul>
    </>}
  </article>
}

function OutcomeCard({ selectedProcessType, process, goal, decisions, decisionsUnready, reviews }: {
  selectedProcessType: OrchestrationBaselineProcessType
  process: OrchestrationProcessRunRecord | undefined
  goal: OrchestrationGoal
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  decisionsUnready: boolean
  reviews: readonly OrchestrationAgentReviewRecord[]
}) {
  if (!process) return <OutcomeFallback type={selectedProcessType} />
  if (selectedProcessType === 'goal_definition') return process.status === 'completed' ? <GoalDefinitionCard process={process} goal={goal} /> : <OutcomeFallback type={selectedProcessType} />
  if (selectedProcessType === 'manager_selection') return <ManagerCard process={process} reviews={reviews} />
  if (selectedProcessType === 'team_hierarchy') {
    const hierarchy = parseHierarchy(process, reviews)
    // Guard (spec: hierarchy graph): only draw the diagram once there's an
    // actual DAG to show (a resolved manager + at least one role assignment)
    // — otherwise fall back to the text card alone, same as before this
    // phase.
    const graphable = hierarchy.manager !== 'Not recorded' && hierarchy.roles.length > 0
    return <>
      {graphable && <Suspense fallback={<div aria-hidden="true" className="mt-4 overflow-hidden rounded border border-huddleroom-border bg-huddleroom-depth" style={{ height: 220 }} />}>
        <HierarchyGraph manager={hierarchy.manager} roles={hierarchy.roles} missing={hierarchy.missing} weakFits={hierarchy.weakFits} />
      </Suspense>}
      <HierarchyCard process={process} reviews={reviews} />
    </>
  }
  if (selectedProcessType === 'effectiveness_review') {
    const outputs = isRecord(process.outputs) ? process.outputs : {}
    return Array.isArray(outputs.triggers) || Array.isArray(outputs.checks) ? <EffectivenessReviewCard process={process} /> : <OutcomeFallback type={selectedProcessType} />
  }
  if (selectedProcessType === 'goal_closeout') return process.status === 'completed' ? <GoalCloseoutCard process={process} decisions={decisions} decisionsUnready={decisionsUnready} /> : <OutcomeFallback type={selectedProcessType} />
  // agent_definition_review's outcome is the roll-up line + per-agent review
  // cards rendered separately below (they need the scoped agent-review list,
  // not just the process outputs).
  return null
}

export interface ProcessFocusProps {
  selectedProcessType: OrchestrationBaselineProcessType
  process: OrchestrationProcessRunRecord | undefined
  goal: OrchestrationGoal
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  // Gates the goal-closeout outcome card's "no sign-off required" narrative —
  // that reading is only valid once the decisions query has actually
  // resolved *successfully* (readiness upstream only gates on processes).
  // True while loading OR errored — a bare isLoading flag can't tell "loaded,
  // no signoff" apart from "failed to load".
  decisionsUnready: boolean
  warnings: readonly OrchestrationWarningRecord[]
  reviews: readonly OrchestrationAgentReviewRecord[]
  debug: boolean
  canSkip: boolean
  mutationBusy: boolean
  canRun: boolean
  canRerun: boolean
  runPending: boolean
  rerunPending: boolean
  runRerunError: string | null
  onRun: (trigger: HTMLButtonElement | null) => void
  onRerun: (trigger: HTMLButtonElement | null) => void
  retryPending: boolean
  retryError: string | null
  onRetry: (trigger: HTMLButtonElement | null) => void
  openAction: BaselineFocusAction | null
  onOpenAction: (action: BaselineFocusAction) => void
  onActionTrigger: (action: BaselineFocusAction, element: HTMLButtonElement | null) => void
  headingRef: RefObject<HTMLHeadingElement | null>
  renderActionForm: (action: BaselineFocusAction, record: OrchestrationAuthorityDecisionRecord | OrchestrationWarningRecord | OrchestrationProcessRunRecord) => ReactNode
  // Switches the Details tabs (Z5) to Activity, scoped to this step. Optional
  // so existing callers/tests that don't wire it keep working.
  onViewActivity?: () => void
}

export function ProcessFocus({
  selectedProcessType,
  process,
  goal,
  decisions,
  decisionsUnready,
  warnings,
  reviews,
  debug,
  canSkip,
  mutationBusy,
  canRun,
  canRerun,
  runPending,
  rerunPending,
  runRerunError,
  onRun,
  onRerun,
  retryPending,
  retryError,
  onRetry,
  openAction,
  onOpenAction,
  onActionTrigger,
  headingRef,
  renderActionForm,
  onViewActivity,
}: ProcessFocusProps) {
  const [rawOpen, setRawOpen] = useState(false)
  const phrase = process ? statusPhrase(process.status) : { label: 'Not started', tone: 'idle' as const }
  const processDecisions = scoped(decisions, process)
  const processWarnings = scoped(warnings, process)
  const processReviews = scoped(reviews, process)
  const outputError = processError(process)
  const retry = process?.outputs.lm_retry
  const retryable = retry?.available === true
  // A process only needs the dedicated Retry control while it's actively
  // running the failed LM request — a step waiting on a decision recovers
  // through the decision, and a terminal step with stale retry metadata
  // just shows Re-run instead.
  const showRetryControl = process?.status === 'running' && retryable
  const showRun = canRun && !showRetryControl
  const { announceError } = useGoalAnnouncer()
  useEffect(() => { if (runRerunError) announceError(runRerunError) }, [runRerunError, announceError])
  useEffect(() => { if (outputError) announceError(outputError) }, [outputError, announceError])
  const controlsBusy = mutationBusy || runPending || rerunPending || retryPending
  const skipAction: BaselineFocusAction | null = process ? { kind: 'skip', id: process.id } : null

  return (
    <section aria-labelledby="baseline-process-focus-heading" className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
      {/* "Step detail" is a static zone label, not part of the heading's
          accessible name — #baseline-process-focus-heading still exposes just
          the step name (e.g. "Goal definition"), matching keyboard-nav tests. */}
      <p className={ZONE_TITLE_CLASS}>Step detail</p>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <h2 ref={headingRef} id="baseline-process-focus-heading" tabIndex={-1} className="text-sm font-semibold text-huddleroom-text-primary">{stepDisplayLabel(selectedProcessType)}</h2>
        <StatusBadge status={phrase.tone} label={phrase.label} />
        {showRun && <Button
          ref={(element) => { if (element) element.disabled = controlsBusy }}
          type="button" variant="secondary" size="sm" className="min-h-11" aria-label={`Run ${processLabel(selectedProcessType)}`}
          disabled={controlsBusy} onClick={(event) => { if (!controlsBusy) onRun(event.currentTarget) }}
        >{runPending ? 'Running…' : 'Run'}</Button>}
        {canRerun && <Button
          ref={(element) => { if (element) element.disabled = controlsBusy }}
          type="button" variant="secondary" size="sm" className="min-h-11" aria-label={`Re-run ${processLabel(selectedProcessType)}`}
          disabled={controlsBusy} onClick={(event) => { if (!controlsBusy) onRerun(event.currentTarget) }}
        >{rerunPending ? 'Re-running…' : 'Re-run'}</Button>}
        {canSkip && skipAction && <Button ref={(element) => onActionTrigger(skipAction, element)} type="button" variant="secondary" size="sm" className="min-h-11" disabled={mutationBusy} onClick={() => onOpenAction(skipAction)}>Skip</Button>}
      </div>
      {canSkip && <p className="mt-1 text-xs text-huddleroom-text-muted">Skip this process if it isn't needed for this goal — you can re-run it later.</p>}
      {skipAction && openAction?.kind === 'skip' && openAction.id === skipAction.id && process && renderActionForm(skipAction, process)}

      {runRerunError && <p className="mt-3 text-sm text-huddleroom-status-red">{runRerunError}</p>}
      {showRetryControl && <div className="mt-3">
        <RetryControl onRetry={onRetry} retryPending={retryPending} retryError={retryError} disabled={controlsBusy} />
        {retry?.hint && <p className="mt-2 text-xs text-huddleroom-text-muted">{retry.hint}</p>}
      </div>}

      {outputError && <p className="mt-3 text-sm text-huddleroom-status-red">{outputError}</p>}

      {onViewActivity && (
        <div className="mt-4 border-t border-huddleroom-border pt-3">
          <Button type="button" variant="ghost" size="sm" className="min-h-11" onClick={onViewActivity}>
            View activity for this step
          </Button>
        </div>
      )}

      {selectedProcessType === 'agent_definition_review' && process && processReviews.length > 0 && (() => {
        const outputs = isRecord(process.outputs) ? process.outputs : {}
        const warningCount = typeof outputs.warning_count === 'number' ? outputs.warning_count : processWarnings.length
        return <p className="mt-4 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
          <span className="block font-semibold text-huddleroom-text-primary">Who the orchestrator reviewed and approved</span>
          {processReviews.length} agent{processReviews.length === 1 ? '' : 's'} reviewed · {warningCount} warning{warningCount === 1 ? '' : 's'}
          {outputs.compressed === true && ' · Compressed (trivial goal)'}
        </p>
      })()}

      {reviewCards(reviews, process).map((review) => <article key={review.id} className="mt-4 border-t border-huddleroom-border pt-3">
        <h3 className="text-sm font-semibold text-huddleroom-text-primary">{review.name}{review.role ? ` · ${review.role}` : ''}</h3>
        {(review.provider || review.model) && <p className="mt-1 text-sm text-huddleroom-text-secondary">{review.provider ?? '—'} / {review.model ?? '—'}</p>}
        {review.capabilities.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Capabilities: {review.capabilities.join(', ')}</p>}
        {review.tools.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Tools: {review.tools.join(', ')}</p>}
        {review.proposed.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Proposed work functions: {review.proposed.join(', ')}</p>}
        {review.approved.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Approved work functions: {review.approved.join(', ')}</p>}
        {review.risks.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Risks: {review.risks.join(', ')}</p>}
        {review.recommendations.length > 0 && <p className="mt-1 text-sm text-huddleroom-text-secondary">Recommendations: {review.recommendations.join(', ')}</p>}
        {review.assessment && <p className="mt-1 text-sm text-huddleroom-text-secondary">Assessment: {review.assessment}</p>}
      </article>)}

      {selectedProcessType === 'agent_definition_review' && processReviews.length === 0 && <OutcomeFallback type="agent_definition_review" />}

      <OutcomeCard selectedProcessType={selectedProcessType} process={process} goal={goal} decisions={processDecisions} decisionsUnready={decisionsUnready} reviews={reviews} />

      {debug && process && <details open={rawOpen} className="mt-4 border-t border-huddleroom-border pt-3">
        <summary className="min-h-11 cursor-pointer py-3 text-sm font-medium text-huddleroom-text-primary" onClick={(event) => { event.preventDefault(); setRawOpen((open) => !open) }}>Raw data</summary>
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-huddleroom-depth p-3 font-mono text-xs text-huddleroom-text-primary">{JSON.stringify({ process, decisions: processDecisions, warnings: processWarnings, agentReviews: processReviews }, null, 2)}</pre>
      </details>}
    </section>
  )
}
