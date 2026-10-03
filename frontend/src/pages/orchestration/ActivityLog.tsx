import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { StatusBadge, Button } from '@/components/common/uiPrimitives'
import { FilterChips } from '@/components/common/FilterChips'
import { LedgerRow } from '@/components/common/LedgerRow'
import { dateHeading } from '@/lib/time'
import type {
  OrchestrationAction,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationDecision,
  OrchestrationGoalDetail,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { errorRecord, stepDisplayLabel, warningMessage, warningTypeLabel } from './humanize'
import { scoped } from './ProcessFocus'

// Goal-wide orchestrator decision/action ledger (run-scoped, not attributable
// to any one baseline process step — distinct from the baseline
// OrchestrationAuthorityDecisionRecord/warning records below, which ARE
// process-scoped via source_process_run_id). Moved here from
// OrchestrationPage.tsx: this is exactly the "whole goal" half of the merge.
export type TimelineItem = {
  id: string
  kind: 'decision' | 'action'
  label: string
  status: string
  reason: string
  createdAt: string
  detail: unknown
  target?: string
  error?: string | null
  decisionRef?: { id: string; label: string; reason: string }
}

export function buildOrchestrationTimeline(detail: OrchestrationGoalDetail): TimelineItem[] {
  const decisionsById = new Map(detail.decisions.map((decision) => [decision.id, decision]))
  const decisionItems: TimelineItem[] = detail.decisions.map((decision: OrchestrationDecision) => ({
    id: decision.id,
    kind: 'decision',
    label: decision.decision_type,
    status: decision.validator_status,
    reason: decision.reason ?? decision.rejection_reason ?? 'No reason recorded.',
    createdAt: decision.created_at,
    detail: {
      parsed_decision: decision.parsed_decision,
      input_snapshot: decision.input_snapshot,
      llm_output: decision.llm_output,
    },
    error: decision.rejection_reason,
  }))
  const actionItems: TimelineItem[] = detail.actions.map((action: OrchestrationAction) => {
    const decision = action.decision_id ? decisionsById.get(action.decision_id) : undefined
    const requestReason = action.request.reason
    return {
      id: action.id,
      kind: 'action',
      label: action.action_type,
      status: action.status,
      reason: typeof requestReason === 'string'
        ? requestReason
        : decision?.reason ?? 'No reason recorded.',
      createdAt: action.created_at,
      detail: action.request,
      target: action.target_type && action.target_id
        ? `${action.target_type}:${action.target_id}`
        : undefined,
      error: action.error,
      decisionRef: action.decision_id
        ? {
            id: action.decision_id,
            label: decision?.decision_type ?? 'linked decision',
            reason: decision?.reason
              ?? decision?.rejection_reason
              ?? 'Linked decision is not in this ledger.',
          }
        : undefined,
    }
  })
  return [...decisionItems, ...actionItems].sort((left, right) =>
    right.createdAt.localeCompare(left.createdAt)
    || left.kind.localeCompare(right.kind)
    || left.id.localeCompare(right.id),
  )
}

// Unified shape both the per-step (process-scoped) and goal-wide (run-scoped
// ledger) sources are mapped into so the merged list can sort, filter, and
// render them uniformly.
export interface ActivityEntry {
  id: string
  anchorId?: string
  kind: string
  label?: string
  timestamp: string
  message: string
  processType: OrchestrationBaselineProcessType | null
  status?: string
  error?: string | null
  target?: string
  decisionRef?: { id: string; label: string; reason: string }
  // Only set for goal-ledger entries — per-step entries never had a raw
  // payload in the old inline Timeline block either.
  detail?: unknown
}

function isBaselineProcessType(type: string): type is OrchestrationBaselineProcessType {
  return type === 'goal_definition' || type === 'manager_selection' || type === 'agent_definition_review'
    || type === 'team_hierarchy' || type === 'effectiveness_review' || type === 'goal_closeout'
}

function answeredAuthorityDecisionEvent(
  decision: OrchestrationAuthorityDecisionRecord,
  processType: OrchestrationBaselineProcessType | null,
): ActivityEntry | null {
  if (decision.status !== 'answered') return null
  return {
    id: `${decision.id}:answered`, anchorId: `decision-${decision.id}`, kind: 'decision-answered',
    timestamp: decision.decided_at ?? decision.created_at,
    message: `You answered: ${decision.selected_option ?? 'Answer recorded'}${decision.reason ? ` — ${decision.reason}` : ''}`,
    processType,
  }
}

// Per-step lifecycle + decision + warning events for one process — moved
// here from ProcessFocus.tsx (its inline Timeline block was deleted; this is
// its replacement source of truth).
function perStepEvents(
  process: OrchestrationProcessRunRecord,
  decisions: readonly OrchestrationAuthorityDecisionRecord[],
  warnings: readonly OrchestrationWarningRecord[],
): ActivityEntry[] {
  const processType = isBaselineProcessType(process.process_type) ? process.process_type : null
  const label = processType ? stepDisplayLabel(processType) : process.process_type
  const events: ActivityEntry[] = [{
    id: `${process.id}:started`, kind: 'process-started', timestamp: process.started_at,
    message: `${label} started`, processType,
  }]
  if (process.status === 'completed' && process.completed_at) events.push({
    id: `${process.id}:completed`, kind: 'process-completed', timestamp: process.completed_at,
    message: `${label} completed`, processType,
  })
  if (process.status === 'skipped' && process.completed_at) events.push({
    id: `${process.id}:skipped`, kind: 'process-skipped', timestamp: process.completed_at,
    message: `${label} skipped`, processType,
  })
  for (const decision of scoped(decisions, process)) {
    events.push({
      id: `${decision.id}:asked`, kind: 'decision-asked', timestamp: decision.asked_at,
      message: decision.title || decision.question, processType,
    })
    const answered = answeredAuthorityDecisionEvent(decision, processType)
    if (answered) events.push(answered)
  }
  for (const warning of scoped(warnings, process)) {
    events.push({ id: `${warning.id}:flagged`, kind: 'warning-flagged', timestamp: warning.created_at, message: warningMessage(warning.message), processType })
    if (warning.acknowledged_at) events.push({
      id: `${warning.id}:acknowledged`, kind: 'warning-acknowledged', timestamp: warning.acknowledged_at,
      message: `${warningTypeLabel(warning.warning_type)} acknowledged`, processType,
    })
    if (warning.resolved_at) events.push({
      id: `${warning.id}:resolved`, kind: 'warning-resolved', timestamp: warning.resolved_at,
      message: warning.resolved_reason || `${warningTypeLabel(warning.warning_type)} resolved`, processType,
    })
  }
  return events
}

function goalLedgerEntries(detail: OrchestrationGoalDetail): ActivityEntry[] {
  return buildOrchestrationTimeline(detail).map((item) => ({
    id: `${item.kind}-${item.id}`,
    anchorId: item.kind === 'decision' ? `decision-${item.id}` : `action-${item.id}`,
    kind: item.kind,
    label: item.label,
    timestamp: item.createdAt,
    message: item.reason,
    processType: null,
    status: item.status,
    error: item.error,
    target: item.target,
    decisionRef: item.decisionRef,
    detail: item.detail,
  }))
}

// Mirrors the old goal-wide Timeline's debug gating exactly: the raw payload
// is always available behind a collapsed <details>, standard mode just
// strips the two verbose LLM-only fields.
function sanitizedDetail(value: unknown, debug: boolean) {
  if (debug) return value
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    const { llm_output, input_snapshot, ...rest } = value as Record<string, unknown>
    return rest
  }
  return value
}

export function buildActivityEntries({
  processes, decisions, warnings, detail,
}: {
  processes: readonly OrchestrationProcessRunRecord[]
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  warnings: readonly OrchestrationWarningRecord[]
  detail: OrchestrationGoalDetail
}): ActivityEntry[] {
  const perStep = processes.flatMap((process) => perStepEvents(process, decisions, warnings))
  const unscoped = decisions.flatMap((decision) => decision.source_process_run_id === null
    ? [answeredAuthorityDecisionEvent(decision, null)].filter((entry): entry is ActivityEntry => entry !== null)
    : [])
  return [...perStep, ...unscoped, ...goalLedgerEntries(detail)].sort((left, right) =>
    right.timestamp.localeCompare(left.timestamp) || right.id.localeCompare(left.id))
}

function humanizeKind(kind: string): string {
  const spaced = kind.replace(/-/g, ' ')
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

export type ActivityScope = 'step' | 'goal'

export interface ActivityLogProps {
  processes: readonly OrchestrationProcessRunRecord[]
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  warnings: readonly OrchestrationWarningRecord[]
  detail: OrchestrationGoalDetail
  selectedStepType: OrchestrationBaselineProcessType | null
  scope: ActivityScope
  onScopeChange: (scope: ActivityScope) => void
  debug: boolean
  focusDecisionId?: string | null
  focusDecisionPending?: boolean
  onDecisionFocused?: () => void
}

const LEDGER_PAGE_SIZE = 50

export function ActivityLog({ processes, decisions, warnings, detail, selectedStepType, scope, onScopeChange, debug, focusDecisionId = null, focusDecisionPending = false, onDecisionFocused }: ActivityLogProps) {
  const [activeKinds, setActiveKinds] = useState<Set<string>>(new Set())
  const [visibleCount, setVisibleCount] = useState(LEDGER_PAGE_SIZE)
  const scheduledFocusId = useRef<string | null>(null)
  const focusPendingRef = useRef(focusDecisionPending)
  focusPendingRef.current = focusDecisionPending
  const allEntries = buildActivityEntries({ processes, decisions, warnings, detail })
  const scopedEntries = scope === 'step' && selectedStepType
    ? allEntries.filter((entry) => entry.processType === selectedStepType)
    : allEntries
  const kinds = [...new Set(scopedEntries.map((entry) => entry.kind))]
  const entries = activeKinds.size === 0 ? scopedEntries : scopedEntries.filter((entry) => activeKinds.has(entry.kind))

  // A kind chip selected in one scope may not exist in another (e.g.
  // "This step" vs "Whole goal" or a different selected step) — an active
  // filter for a kind that's no longer present would otherwise silently
  // empty the list with no visible chip left to clear it.
  useLayoutEffect(() => {
    if (focusPendingRef.current) return
    setActiveKinds(new Set())
    setVisibleCount(LEDGER_PAGE_SIZE)
  }, [scope, selectedStepType])

  useLayoutEffect(() => {
    if (!focusDecisionId || !focusDecisionPending) return
    setActiveKinds(new Set())
    setVisibleCount(scopedEntries.length)
  }, [focusDecisionId, focusDecisionPending, scopedEntries.length])

  useLayoutEffect(() => {
    if (!focusDecisionPending || !focusDecisionId) {
      scheduledFocusId.current = null
      return
    }
    if (scheduledFocusId.current === focusDecisionId || activeKinds.size > 0 || visibleCount !== scopedEntries.length) return
    const target = document.getElementById?.(`decision-${focusDecisionId}`)
    if (!target) return
    scheduledFocusId.current = focusDecisionId
    const focus = () => {
      target.scrollIntoView?.({ block: 'center' })
      target.focus()
      onDecisionFocused?.()
    }
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(focus)
    else focus()
  }, [focusDecisionId, focusDecisionPending, onDecisionFocused, activeKinds.size, scopedEntries.length, visibleCount])

  function toggleKind(kind: string) {
    setActiveKinds((current) => {
      const next = new Set(current)
      if (next.has(kind)) next.delete(kind)
      else next.add(kind)
      return next
    })
  }

  return (
    <div className="flex flex-col gap-3">
      <div role="group" aria-label="Activity scope" className="inline-flex w-fit overflow-hidden rounded-md border border-huddleroom-border">
        {(['step', 'goal'] as const).map((value) => (
          <button
            key={value}
            type="button"
            aria-pressed={scope === value}
            disabled={value === 'step' && !selectedStepType}
            className={`min-h-11 px-3 text-xs font-medium disabled:cursor-not-allowed disabled:opacity-50 ${scope === value ? 'bg-huddleroom-primary text-white' : 'bg-huddleroom-surface text-huddleroom-text-secondary hover:bg-huddleroom-depth'}`}
            onClick={() => onScopeChange(value)}
          >
            {value === 'step' ? 'This step' : 'Whole goal'}
          </button>
        ))}
      </div>

      {kinds.length > 0 && (
        <FilterChips
          multi
          activeIds={activeKinds}
          onToggle={toggleKind}
          options={kinds.map((k) => ({ id: k, label: humanizeKind(k) }))}
          ariaLabel="Filter by event type"
        />
      )}

      {entries.length === 0 ? (
        <p className="text-sm text-huddleroom-text-secondary">No activity recorded{scope === 'step' ? ' for this step' : ''} yet.</p>
      ) : (
        <div className="flex flex-col gap-3">
          <ol className="divide-y divide-huddleroom-border">
            {entries.slice(0, visibleCount).map((entry, index) => {
            const heading = dateHeading(entry.timestamp)
            const boundary = index === 0 || heading !== dateHeading(entries[index - 1].timestamp)
            return (
              <li key={entry.id} id={entry.anchorId} tabIndex={entry.anchorId === `decision-${focusDecisionId}` ? -1 : undefined} className="py-3 first:pt-0 last:pb-0">
                {boundary && (
                  <div role="separator" aria-label={heading} className="mb-2 text-xs font-semibold text-huddleroom-text-muted">
                    {heading}
                  </div>
                )}
                <LedgerRow
                  iso={entry.timestamp}
                  action={<>{entry.label ?? entry.kind}{entry.status && <StatusBadge status={entry.status} />}</>}
                  detail={entry.message}
                >
                  {entry.processType && <span className="text-xs text-huddleroom-text-muted">{stepDisplayLabel(entry.processType)}</span>}
                  {entry.decisionRef ? (
                    <p className="mt-1 break-words text-xs text-huddleroom-text-secondary">
                      Caused by decision{' '}
                      <a href={`#decision-${entry.decisionRef.id}`} className="font-mono text-huddleroom-primary underline-offset-2 hover:underline">
                        {entry.decisionRef.id}
                      </a>
                      {' · '}{entry.decisionRef.label}: {entry.decisionRef.reason}
                    </p>
                  ) : entry.kind === 'action' ? (
                    <p className="mt-1 text-xs text-huddleroom-status-amber">No linked decision recorded.</p>
                  ) : null}
                  {entry.target && <p className="mt-1 break-all font-mono text-xs text-huddleroom-text-muted">{entry.target}</p>}
                  {entry.error && <p className="mt-1 text-xs text-huddleroom-status-red">{errorRecord(entry.error).what} — <a href="#needs-you-queue-heading" className="underline">see queue</a></p>}
                  {entry.detail !== undefined && (
                    <details className="mt-2">
                      <summary className="flex min-h-11 cursor-pointer items-center text-xs text-huddleroom-text-secondary">▸ {entry.kind} data</summary>
                      <pre className="mt-2 max-w-full overflow-auto whitespace-pre-wrap break-words rounded bg-huddleroom-depth p-3 font-mono text-xs text-huddleroom-text-primary">
                        {JSON.stringify(sanitizedDetail(entry.detail, debug), null, 2)}
                      </pre>
                    </details>
                  )}
                </LedgerRow>
              </li>
            )
            })}
          </ol>
          {entries.length > visibleCount && (
            <Button
              variant="secondary"
              size="md"
              onClick={() => setVisibleCount(entries.length)}
              className="self-start"
            >
              Show more
            </Button>
          )}
        </div>
      )}
    </div>
  )
}
