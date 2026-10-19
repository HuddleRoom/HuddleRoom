import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useOrchestrationGoal } from '@/api/orchestration'
import { Callout } from '@/components/common/Callout'
import { Record as RecordList } from '@/components/common/Record'
import { Tag } from '@/components/common/Tag'
import { Button, SkeletonRow, StatusBadge } from '@/components/common/uiPrimitives'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationChildGoal,
  OrchestrationEvidence,
  OrchestrationGoal,
  OrchestrationGoalDetail,
  OrchestrationPlanItemView,
  OrchestrationPlanReviewView,
  OrchestrationPlanView,
  OrchestrationRunSummary,
  OrchestrationTaskReport,
} from '@/lib/types'
import {
  acceptanceVerdictLabel,
  evidenceLabel,
  evidenceVerdict,
  goalStatusLabel,
  goalTypeLabel,
  itemStateLabel,
  planStatusLabel,
  planningModeLabel,
  runLabel,
  runStatusLabel,
  runTerminalReasonLabel,
} from './humanize'

const SUMMARY_CLASS = 'flex min-h-11 cursor-pointer items-center text-xs text-huddleroom-text-secondary'
const HEADING_CLASS = 'text-xs font-medium text-huddleroom-text-muted'

type Names = Readonly<Record<string, string>>

const shortId = (id: string) => id.slice(0, 8)

function MonoId({ id }: { id: string }) {
  return <span className="font-mono text-xs" title={id}>{shortId(id)}</span>
}

// Item-state -> StatusBadge colour key. Only a real `accepted` item is green.
const ITEM_STATE_BADGE: Record<string, string> = {
  queued: 'pending', started: 'running', evidence_pending: 'blocked',
  accepted: 'accepted', failed: 'failed', cancelled: 'cancelled',
}

const VERDICT_BADGE: Record<string, string> = {
  pending: 'pending', accepted: 'accepted', rejected: 'rejected', superseded: 'cancelled',
}

// ---------------------------------------------------------------------------
// Plan overview: runs + current plan record
// ---------------------------------------------------------------------------

function runDate(run: OrchestrationRunSummary): string {
  const date = new Date(run.completed_at ?? run.started_at)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleDateString()
}

function runTerminalText(run: OrchestrationRunSummary, runs: OrchestrationRunSummary[]): string | null {
  if (!run.terminal_reason) return null
  if (run.terminal_reason !== 'reclassified') return runTerminalReasonLabel(run.terminal_reason)
  const successor = runs.find((candidate) => candidate.supersedes_run_id === run.id)
  return successor?.goal_type_snapshot
    ? `Reclassified to ${goalTypeLabel(successor.goal_type_snapshot)}`
    : runTerminalReasonLabel(run.terminal_reason)
}

export interface PlanOverviewProps {
  runs: OrchestrationRunSummary[]
  plan: OrchestrationPlanView | null
  agentRoles?: Names
  // A revised plan awaiting approval sits on top of an accepted plan.
  pendingReplan?: boolean
  // Run plan-state status; drives the drafting copy while no plan record exists yet.
  planStatus?: string | null
  pendingDecisionId?: string | null
  onViewDecision?: (decisionId: string) => void
}

export function PlanOverview({ runs, plan, agentRoles = {}, pendingReplan = false, planStatus = plan?.status ?? null, pendingDecisionId = null, onViewDecision }: PlanOverviewProps) {
  const headingId = useId()
  const current = runs.find((run) => run.is_current)
  const earlier = runs.filter((run) => !run.is_current).sort((a, b) => b.started_at.localeCompare(a.started_at))
  const author = plan?.author_agent_id ? (agentRoles[plan.author_agent_id] ?? 'the plan author') : 'the plan author'
  const version = plan?.planning_version ?? null
  const revisionPending = plan?.kind !== 'roadmap'
    && (plan?.status === 'revision_requested' || plan?.status === 'revision_required')
  // ponytail: only the current plan version is projected; earlier ones are a count, not fetched.
  const earlierCount = version && version > 1 ? version - 1 : 0

  return <div className="flex flex-col gap-3">
    <section aria-labelledby={`${headingId}-runs`} className="flex flex-col gap-2">
      <h3 id={`${headingId}-runs`} className={HEADING_CLASS}>Runs</h3>
      {current && <RecordList keyWidth={120} rows={[{
        key: 'Run',
        value: <span className="flex flex-wrap items-center gap-2">
          <span>{runLabel(current.version_label)} · Current</span>
          <StatusBadge status={current.status} label={runStatusLabel(current.status)} />
        </span>,
      }]} />}
      {earlier.length > 0 && <details>
        <summary className={SUMMARY_CLASS}>▸ Earlier runs ({earlier.length})</summary>
        <ul className="flex flex-col gap-1 text-sm text-huddleroom-text-secondary">
          {earlier.map((run) => <li key={run.id} className="flex flex-wrap items-baseline gap-2">
            <span>{runLabel(run.version_label)} · Historical</span>
            {runTerminalText(run, runs) && <span className="text-xs">{runTerminalText(run, runs)}</span>}
            {runDate(run) && <span className="text-xs text-huddleroom-text-muted">{runDate(run)}</span>}
          </li>)}
        </ul>
      </details>}
    </section>

    <section aria-labelledby={`${headingId}-plan`} className="flex flex-col gap-2">
      <h3 id={`${headingId}-plan`} className={HEADING_CLASS}>Plan</h3>
      {pendingReplan && <Callout variant="info">
        <p>A revised plan is proposed and awaiting approval. Items below are the current accepted plan.</p>
        {pendingDecisionId && onViewDecision && <Button type="button" variant="ghost" className="min-h-11" onClick={() => onViewDecision(pendingDecisionId)}>View the pending approval</Button>}
      </Callout>}
      {plan && plan.artifact_id ? <>
        <RecordList keyWidth={120} rows={[
          { key: 'Kind', value: plan.kind ? planningModeLabel(plan.kind) : null },
          { key: 'Version', value: version ? `v${version} · Current` : null },
          { key: 'Author', value: plan.author_agent_id ? (agentRoles[plan.author_agent_id] ?? <MonoId id={plan.author_agent_id} />) : null },
          { key: 'Artifact', value: <MonoId id={plan.artifact_id} /> },
          { key: 'Status', value: plan.status ? <StatusBadge status={plan.status === 'requested' ? 'pending' : plan.status} label={revisionPending ? 'Accepted (current plan; revision pending)' : planStatusLabel(plan.status)} /> : null },
        ]} />
        {earlierCount > 0 && <p className="text-xs text-huddleroom-text-muted">
          {earlierCount === 1 ? '1 earlier version (v1)' : `${earlierCount} earlier versions (v1–v${earlierCount})`}, not shown here.
        </p>}
      </> : planStatus === 'requested' || planStatus === 'revision_requested'
        ? <p className="text-sm text-huddleroom-text-secondary">Plan is being drafted by {author}.</p>
        : <p className="text-sm text-huddleroom-text-secondary">No plan yet. Items appear here once the plan author&apos;s plan is accepted.</p>}
    </section>
  </div>
}

// ---------------------------------------------------------------------------
// Plan items
// ---------------------------------------------------------------------------

// Report rows are lists, never JSON. Tolerates a stray string / missing field.
function asLines(value: unknown): string[] {
  if (Array.isArray(value)) return value.filter((entry) => entry !== null && entry !== undefined).map(String)
  if (typeof value === 'string' && value) return [value]
  if (value && typeof value === 'object') return Object.entries(value).map(([key, entry]) => `${key}: ${String(entry)}`)
  return []
}

// Null when empty so the Record row is dropped instead of showing a bare label.
function lines(value: unknown): ReactNode {
  const entries = asLines(value)
  if (entries.length === 0) return null
  return <ul className="list-disc pl-4">{entries.map((line, index) => <li key={index} className="break-words">{line}</li>)}</ul>
}

function TitleList({ titles, empty }: { titles: string[]; empty: string }) {
  return titles.length === 0
    ? <p className="text-xs text-huddleroom-text-muted">{empty}</p>
    : <ul className="list-disc pl-4 text-sm">{titles.map((title, index) => <li key={index}>{title}</li>)}</ul>
}

export interface PlanItemListProps {
  plan: OrchestrationPlanView
  evidence: OrchestrationEvidence[]
  taskReports: OrchestrationTaskReport[]
  goal: Pick<OrchestrationGoal, 'objective' | 'success_criteria'>
  taskTitles?: Names
  agentRoles?: Names
  runVersion?: string | null
  contractFingerprint?: string | null
}

export function PlanItemList({
  plan, evidence, taskReports, goal, taskTitles = {}, agentRoles = {}, runVersion = null, contractFingerprint = null,
}: PlanItemListProps) {
  const headingId = useId()
  const itemsByKey = new Map(plan.items.map((item) => [item.key, item] as const))
  const titleOf = (key: string) => itemsByKey.get(key)?.title ?? key
  const revisionPending = plan.kind !== 'roadmap' && (plan.status === 'revision_requested' || plan.status === 'revision_required')
  const heading = revisionPending ? 'Current accepted plan (revision requested)' : 'Plan items'

  if (plan.items.length === 0) {
    if (plan.status === 'proposed') return <p className="text-sm text-huddleroom-text-secondary">No items in this proposal yet.</p>
    return null
  }

  function renderItem(item: OrchestrationPlanItemView) {
    const unmet = item.depends_on.filter((key) => itemsByKey.get(key)?.state !== 'accepted')
    const itemEvidence = item.gate_id ? evidence.filter((entry) => entry.gate_id === item.gate_id) : []
    const report = item.task_id ? taskReports.find((entry) => entry.task_id === item.task_id) : undefined
    const fullReport = report?.report ?? null
    const adoptedTitle = item.adopt_task_id ? (taskTitles[item.adopt_task_id] ?? null) : null
    const owner = [item.agent_id ? (agentRoles[item.agent_id] ?? shortId(item.agent_id)) : null, item.work_function].filter(Boolean).join(' · ')
    const deps = item.depends_on.map((key) => itemsByKey.get(key)).filter((dep): dep is OrchestrationPlanItemView => Boolean(dep))
    const contextEvidence = deps.flatMap((dep) => dep.gate_id
      ? evidence.filter((entry) => entry.gate_id === dep.gate_id && evidenceVerdict(entry) === 'accepted')
        .map((entry) => `${dep.title ?? dep.key} · ${evidenceLabel(entry.evidence_role ?? 'evidence', entry.review_stage ?? '')}`)
      : [])
    const contextReports = deps
      .filter((dep) => dep.task_id && taskReports.some((entry) => entry.task_id === dep.task_id && entry.report))
      .map((dep) => dep.title ?? dep.key)

    // Unreleased queued items wait on dependencies; released ones were unblocked (possibly by preliminary evidence).
    const runnabilityLine = item.state === 'queued' && !item.task_id && !item.child_goal_id && unmet.length > 0
      ? `Waiting on: ${unmet.map(titleOf).join(', ')}`
      : (item.state === 'queued' || item.state === 'started') && item.why_runnable?.trim()
        ? `Why runnable: ${item.why_runnable}`
        : null

    return <li key={item.key} className="flex flex-col gap-2 rounded-md border border-huddleroom-border p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-huddleroom-text-primary">{item.title ?? item.key}</span>
        <StatusBadge status={ITEM_STATE_BADGE[item.state] ?? 'neutral'} label={itemStateLabel(item.state)} />
        {item.priority !== 50 && <Tag>Priority {item.priority}</Tag>}
      </div>
      {runnabilityLine && <p className="text-sm text-huddleroom-text-secondary">{runnabilityLine}</p>}
      <RecordList keyWidth={120} rows={[
        { key: 'Depends on', value: item.depends_on.length > 0 ? item.depends_on.map(titleOf).join(', ') : 'None' },
        { key: 'Artifacts', value: item.artifact_ids.length > 0 ? <span className="flex flex-wrap gap-1">{item.artifact_ids.map((id) => <Tag key={id} className="font-mono"><span title={id}>{shortId(id)}</span></Tag>)}</span> : null },
        { key: 'Reuses task', value: item.adopt_task_id ? (adoptedTitle ?? <MonoId id={item.adopt_task_id} />) : null },
        { key: 'Owner', value: owner || null },
        { key: 'Report', value: report?.summary?.trim() || <span className="text-huddleroom-text-muted">No report yet.</span> },
      ]} />
      {itemEvidence.length > 0 && <ul aria-label="Evidence" className="flex flex-col gap-1 text-sm">
        {itemEvidence.map((entry) => {
          const verdict = evidenceVerdict(entry)
          return <li key={entry.id} className="flex flex-wrap items-center gap-2">
            <span>{evidenceLabel(entry.evidence_role ?? 'evidence', entry.review_stage ?? '')}</span>
            <StatusBadge status={VERDICT_BADGE[verdict] ?? 'neutral'} label={acceptanceVerdictLabel(verdict)} />
          </li>
        })}
      </ul>}
      {fullReport && <details>
        <summary className={SUMMARY_CLASS}>▸ Full report and provenance</summary>
        <div className="flex flex-col gap-2 pb-2">
          <RecordList keyWidth={120} rows={[
            { key: 'Status', value: typeof fullReport.status === 'string' ? acceptanceVerdictLabel(fullReport.status) : null },
            { key: 'Changes', value: lines(fullReport.changes) },
            { key: 'Evidence', value: lines(fullReport.evidence) },
            { key: 'Criterion progress', value: lines(fullReport.criterion_progress) },
            { key: 'Decisions', value: lines(fullReport.decisions) },
            { key: 'Risks', value: lines(fullReport.risks) },
            { key: 'Open questions', value: lines(fullReport.open_questions) },
            { key: 'Next step', value: typeof fullReport.next_step === 'string' ? fullReport.next_step : null },
            { key: 'Collaboration need', value: typeof fullReport.collaboration_need === 'string' ? fullReport.collaboration_need : null },
          ]} />
          <RecordList keyWidth={120} rows={[
            { key: 'Session', value: report?.session_id ? <MonoId id={report.session_id} /> : null },
            { key: 'Run', value: runVersion ? runLabel(runVersion) : null },
            { key: 'Contract at Start', value: contractFingerprint ? <MonoId id={contractFingerprint} /> : null },
          ]} />
        </div>
      </details>}
      <details>
        <summary className={SUMMARY_CLASS}>▸ Context given to the agent</summary>
        <div className="flex flex-col gap-2 pb-2">
          <section><h4 className={HEADING_CLASS}>Goal contract</h4>
            <TitleList empty="None" titles={[goal.objective, ...goal.success_criteria.map((criterion) => criterion.description ?? String(criterion.key ?? '')).filter(Boolean)]} />
          </section>
          <section><h4 className={HEADING_CLASS}>Accepted evidence</h4><TitleList empty="None" titles={contextEvidence} /></section>
          <section><h4 className={HEADING_CLASS}>Dependency reports</h4><TitleList empty="None" titles={contextReports} /></section>
          <section><h4 className={HEADING_CLASS}>Adoption candidates</h4>
            <TitleList empty="None" titles={item.adopt_task_id ? [adoptedTitle ?? shortId(item.adopt_task_id)] : []} />
          </section>
        </div>
      </details>
    </li>
  }

  return <section aria-labelledby={headingId} className="flex flex-col gap-2">
    <h3 id={headingId} className={HEADING_CLASS}>{heading}</h3>
    <p className="text-xs text-huddleroom-text-muted">Preliminary evidence can unblock dependent work but never satisfies a final criterion.</p>
    <ol aria-label="Plan items" className="flex flex-col gap-3">{plan.items.map(renderItem)}</ol>
  </section>
}

// ---------------------------------------------------------------------------
// Child goals (Roadmap)
// ---------------------------------------------------------------------------

function ChildRow({ projectId, child, nested }: { projectId: string; child: OrchestrationChildGoal; nested: boolean }) {
  const [open, setOpen] = useState(false)
  // Lazy: the child's own detail is fetched only after its "Next level" is opened.
  const next = useOrchestrationGoal(projectId, open ? child.id : undefined)
  const runText = child.run_version_label ? runLabel(child.run_version_label) : null
  return <li className="flex flex-col gap-1 rounded-md border border-huddleroom-border p-3">
    <div className="flex flex-wrap items-center gap-2">
      <Link to={`/orchestration/${child.id}`} className="text-sm font-medium text-huddleroom-primary">{child.objective}</Link>
      <Tag>{goalTypeLabel(child.goal_type)}</Tag>
      <StatusBadge status={child.status} label={goalStatusLabel(child.status)} />
      {runText && <span className="text-xs text-huddleroom-text-secondary">{runText}</span>}
      {child.roadmap_version !== null && <span className="text-xs text-huddleroom-text-secondary">Plan v{child.roadmap_version}</span>}
    </div>
    {!nested && child.goal_type === 'roadmap' && <details open={open}>
      <summary className={SUMMARY_CLASS} onClick={(event) => { event.preventDefault(); setOpen((current) => !current) }}>▸ Next level</summary>
      {open && (next.isLoading
        ? <div aria-busy="true" aria-label="Loading next level"><SkeletonRow /></div>
        : next.isError
          ? <p className="text-sm text-huddleroom-status-red">Failed to load the next level.</p>
          : <ChildGoalList projectId={projectId} goals={next.data?.child_goals ?? []} nested />)}
    </details>}
  </li>
}

export function ChildGoalList({ projectId, goals, nested = false }: { projectId: string; goals: OrchestrationChildGoal[]; nested?: boolean }) {
  if (goals.length === 0) return <p className="text-sm text-huddleroom-text-secondary">No child goals released yet.</p>
  return <ul aria-label={nested ? 'Next level child goals' : 'Child goals'} className="flex flex-col gap-2">
    {goals.map((child) => <ChildRow key={child.id} projectId={projectId} child={child} nested={nested} />)}
  </ul>
}

// ---------------------------------------------------------------------------
// Whole tab
// ---------------------------------------------------------------------------

export interface PlanTabProps {
  projectId: string
  detail: OrchestrationGoalDetail
  decisions?: readonly OrchestrationAuthorityDecisionRecord[]
  taskTitles?: Names
  agentRoles?: Names
  onViewDecision?: (decisionId: string) => void
  focusArtifactId?: string | null
}

function PlanReviewCandidate({ review, onViewDecision, focus }: { review: OrchestrationPlanReviewView; onViewDecision?: (decisionId: string) => void; focus: boolean }) {
  const candidateRef = useRef<HTMLElement>(null)
  useEffect(() => {
    if (!focus) return
    candidateRef.current?.scrollIntoView?.({ block: 'start' })
    candidateRef.current?.focus({ preventScroll: true })
  }, [focus, review.artifact_id])
  const version = review.identity.planning_version
  const authorityStatus = review.authority_status?.trim() ?? null
  const authorityOpen = ['pending', 'open', 'awaiting approval', 'awaiting review'].includes(authorityStatus?.toLowerCase() ?? '')
  const decisionAction = authorityOpen && review.authority_decision_id && onViewDecision
    ? <div className="flex flex-wrap gap-2">
      <Button type="button" className="min-h-11" disabled={review.state !== 'ready'} onClick={() => onViewDecision(review.authority_decision_id!)}>Approve plan v{version}</Button>
      <Button type="button" variant="secondary" className="min-h-11" onClick={() => onViewDecision(review.authority_decision_id!)}>Request revision</Button>
    </div> : null
  return <section id={`plan-review-${review.artifact_id}`} ref={candidateRef} tabIndex={-1} aria-label={`Plan candidate v${version}`} className="flex flex-col gap-3 rounded-md border border-huddleroom-border p-4">
    <div className="flex flex-wrap items-center gap-2">
      <h3 className={HEADING_CLASS}>Plan candidate v{version}</h3>
      <StatusBadge status={review.state === 'stale' ? 'blocked' : authorityOpen ? 'pending' : authorityStatus?.toLowerCase() === 'accepted' ? 'accepted' : 'neutral'} label={review.state === 'stale' ? 'Stale — revision required' : authorityOpen ? 'Awaiting review' : authorityStatus ?? 'Authority unavailable'} />
    </div>
    <p className="text-sm text-huddleroom-text-secondary">This Markdown is readable and editable, but only the canonical reviewed content can execute.</p>
    <Link to={review.review_url} className="inline-flex min-h-11 items-center text-sm text-huddleroom-primary hover:underline">Link to this candidate</Link>
    {review.reason && <Callout variant={review.state === 'stale' ? 'warning' : 'info'}>{review.reason}</Callout>}
    <RecordList keyWidth={150} rows={[
      { key: 'Artifact', value: <MonoId id={review.artifact_id} /> },
      { key: 'Review cycle', value: String(review.identity.review_cycle) },
      { key: 'Provenance', value: <MonoId id={review.identity.provenance_hash} /> },
      { key: 'Canonical content', value: <MonoId id={review.identity.canonical_content_hash} /> },
      { key: 'Rendered content', value: <MonoId id={review.identity.rendered_content_hash} /> },
      { key: 'Authority', value: review.authority_status ?? 'Awaiting authority decision' },
      { key: 'Export', value: review.export_path ?? 'No safe export is available' },
    ]} />
    <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded-md border border-huddleroom-border bg-huddleroom-depth p-3 text-sm">{review.markdown}</pre>
    {review.state === 'stale' && <Callout variant="warning">This candidate remains readable for revision, but it cannot be approved or executed.</Callout>}
    {decisionAction}
  </section>
}

export function PlanTab({ projectId, detail, decisions = [], taskTitles, agentRoles, onViewDecision, focusArtifactId = null }: PlanTabProps) {
  const headingId = useId()
  const { plan, run } = detail
  const runs = detail.runs ?? []
  if (!run && runs.length === 0) return <section className="border-t border-huddleroom-border pt-4">
    <h3 className={HEADING_CLASS}>Plan</h3>
    <p className="mt-2 text-sm text-huddleroom-text-secondary">No orchestration run is attached to this goal.</p>
  </section>

  const pendingReplan = plan?.status === 'accepted' && Boolean((run?.plan_state as { pending_replan?: unknown } | undefined)?.pending_replan)
  const pendingGateId = (run?.plan_state as { pending_replan?: { gate_id?: string } } | undefined)?.pending_replan?.gate_id
  const pendingDecision = pendingReplan && pendingGateId
    ? decisions.find((decision) => decision.status === 'pending' && decision.related_gate_id === pendingGateId)
    : undefined
  const isRoadmap = detail.goal.goal_type === 'roadmap' || plan?.kind === 'roadmap'
  const currentRun = runs.find((candidate) => candidate.is_current)
  const review = plan?.review ?? null

  return <section className="flex flex-col gap-4 border-t border-huddleroom-border pt-4">
    <PlanOverview
      runs={runs} plan={plan} agentRoles={agentRoles} pendingReplan={pendingReplan}
      planStatus={plan?.status ?? run?.plan_state?.status ?? null}
      pendingDecisionId={pendingDecision?.id ?? null} onViewDecision={onViewDecision}
    />
    {review && <PlanReviewCandidate review={review} onViewDecision={onViewDecision} focus={focusArtifactId === review.artifact_id} />}
    {plan && <PlanItemList
      plan={plan} evidence={detail.evidence} taskReports={detail.task_reports ?? []} goal={detail.goal}
      taskTitles={taskTitles} agentRoles={agentRoles}
      runVersion={currentRun?.version_label ?? null} contractFingerprint={run?.goal_contract_fingerprint ?? null}
    />}
    {isRoadmap && <section aria-labelledby={headingId} className="flex flex-col gap-2">
      <h3 id={headingId} className={HEADING_CLASS}>Child goals</h3>
      <ChildGoalList projectId={projectId} goals={detail.child_goals ?? []} />
    </section>}
  </section>
}
