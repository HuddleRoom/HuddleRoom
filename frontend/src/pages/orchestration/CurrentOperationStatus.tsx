import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { StatusBadge, type StatusVariant } from '@/components/common/uiPrimitives'
import { absolute, parseUtc } from '@/lib/time'
import type { NeedsYouReference, OrchestrationStatusLine, OrchestrationSupervisionCondition } from '@/lib/types'
import { compareStatusOccurrences, currentOperationAnnouncementCandidates, type StatusOccurrence } from './currentOperationStatus'
import { useGoalAnnouncer } from './goalAnnouncer'
import { needsYouReferencePath } from './needsYouPresentation'

const BADGES: Record<OrchestrationSupervisionCondition, { status: StatusVariant; label: string }> = {
  unknown: { status: 'idle', label: 'Unknown' },
  working: { status: 'running', label: 'Working' },
  waiting: { status: 'ready', label: 'Waiting' },
  needs_you: { status: 'needs-you', label: 'Needs you' },
  needs_attention: { status: 'blocked', label: 'Needs attention' },
  paused: { status: 'paused', label: 'Paused' },
  stopped: { status: 'idle', label: 'Stopped' },
  cancelled: { status: 'cancelled', label: 'Cancelled' },
  completed: { status: 'completed', label: 'Completed' },
}

type RevealTarget = { kind: 'gate' | 'process' | 'action'; id: string; goalId: string }

export interface CurrentOperationStatusProps {
  statusLine: OrchestrationStatusLine | null
  condition: OrchestrationSupervisionCondition
  activeProjectId: string
  dataUpdatedAt: number
  isFetching: boolean
  isError: boolean
  onRevealSource(target: RevealTarget): void
  goalKey: string
}

function validTime(value: string | number | null): Date | null {
  if (value === null || value === '' || (typeof value === 'number' && value <= 0)) return null
  const date = parseUtc(value)
  return Number.isFinite(date.getTime()) ? date : null
}

function Time({ value }: { value: string | number | null }) {
  const date = validTime(value)
  return date ? <time dateTime={date.toISOString()} title={absolute(date.toISOString())}>{absolute(date.toISOString())}</time> : <>Unknown</>
}

function Reference({ reference, activeProjectId, onRevealSource }: {
  reference: NeedsYouReference | null
  activeProjectId: string
  onRevealSource: (target: RevealTarget) => void
}) {
  if (!reference) return null
  const label = reference.label?.trim()
  if (!label || reference.project_id !== activeProjectId) return <span>Reference unavailable</span>
  const path = needsYouReferencePath(reference, activeProjectId)
  if (path) return <Link className="underline break-words" to={path}>{label}</Link>
  if (reference.goal_id && ['gate', 'process', 'action'].includes(reference.kind)) {
    const kind = reference.kind as RevealTarget['kind']
    return <button type="button" className="underline break-words" onClick={() => onRevealSource({ kind, id: reference.id, goalId: reference.goal_id! })}>{label}</button>
  }
  return <span className="break-words">{label}</span>
}

export function CurrentOperationStatus({ statusLine, condition, activeProjectId, dataUpdatedAt, isFetching, isError,
  onRevealSource, goalKey }: CurrentOperationStatusProps) {
  const { announceQueue } = useGoalAnnouncer()
  const history = useRef<{ goalKey: string; initialized: boolean; seen: Set<string>; newest: StatusOccurrence | null }>({
    goalKey, initialized: false, seen: new Set(), newest: null,
  })
  useEffect(() => {
    const state = history.current
    if (state.goalKey !== goalKey) {
      state.goalKey = goalKey
      state.initialized = false
      state.seen.clear()
      state.newest = null
    }
    if (!statusLine) return
    const candidates = currentOperationAnnouncementCandidates(statusLine)
    const attention = statusLine.attention_summary
    const identity = (candidate: (typeof candidates)[number]) => candidate.source === 'event' ? `event:${candidate.key}` :
      `attention:${candidate.key}:${JSON.stringify({ kind: attention?.kind, summary: attention?.summary,
        subject: attention?.subject && [attention.subject.kind, attention.subject.id],
        unblock_actor: attention?.unblock_actor, unblock_actor_label: attention?.unblock_actor_label,
        next_check_at: attention?.next_check_at, additional_count: attention?.additional_count })}`
    if (!state.initialized) {
      state.initialized = true
      for (const candidate of candidates) state.seen.add(identity(candidate))
      state.newest = candidates.find((candidate) => candidate.occurred_at !== null) ?? null
      return
    }
    for (const candidate of [...candidates].reverse()) {
      const key = identity(candidate)
      if (state.seen.has(key)) continue
      state.seen.add(key)
      if (candidate.occurred_at && state.newest &&
        parseUtc(candidate.occurred_at).getTime() < parseUtc(state.newest.occurred_at!).getTime()) continue
      if (candidate.source === 'event' && state.newest && compareStatusOccurrences(candidate, state.newest) === 0) continue
      announceQueue(candidate.text)
    }
    const latestTimed = candidates.find((candidate) => candidate.occurred_at !== null)
    if (latestTimed && (!state.newest || compareStatusOccurrences(latestTimed, state.newest) < 0)) {
      state.newest = latestTimed
    }
  }, [statusLine, goalKey, announceQueue])

  if (!statusLine) return null
  const { current_state: current, attention_summary: attention, last_material_event: event } = statusLine
  const nextCheck = attention ? validTime(attention.next_check_at) : null
  const freshness = validTime(dataUpdatedAt)
  const badge = BADGES[condition]
  return <section aria-labelledby="current-operation-title" className="min-w-0 rounded-md border border-huddleroom-border bg-huddleroom-surface p-4 break-words">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h2 id="current-operation-title" className="min-w-0 text-sm font-bold tracking-[0.05em] text-huddleroom-text-primary">Current operation</h2>
      <StatusBadge status={badge.status} label={badge.label} />
    </div>
    <div className="mt-2 space-y-1 text-sm text-huddleroom-text-secondary">
      {(current.kind !== 'working' || current.activities.length === 0) && <p>{current.summary}</p>}
      {current.activities.map((activity) => <p key={activity.key} className="min-w-0">
        {(current.kind === 'working' || activity.summary !== current.summary) && <>{activity.summary}{' '}</>}
        {activity.actor && <>Agent: <Reference reference={activity.actor} activeProjectId={activeProjectId} onRevealSource={onRevealSource} />{' · '}</>}
        {activity.subject && <>Subject: <Reference reference={activity.subject} activeProjectId={activeProjectId} onRevealSource={onRevealSource} /></>}
        {activity.state === 'reported_active' && <> · Reported active at <Time value={activity.observed_at} /></>}
      </p>)}
      {attention && <p className="min-w-0">{attention.summary}{' '}
        {attention.subject && <>Subject: <Reference reference={attention.subject} activeProjectId={activeProjectId} onRevealSource={onRevealSource} />{' · '}</>}
        {' '}Who can unblock: {attention.unblock_actor_label || 'Unknown'}.{' '}
        {attention.kind === 'waiting' && <>Next check: {nextCheck ? <>{nextCheck.getTime() < Date.now() ? 'Scheduled check passed; current result unknown. ' : ''}<Time value={attention.next_check_at} /></> : 'Unknown'}</>}
        {attention.additional_count > 0 ? ` · ${attention.additional_count} more` : null}
      </p>}
      <p>Last update: {event ? <>{event.summary}{' '}{event.subject && <>Source: <Reference reference={event.subject} activeProjectId={activeProjectId} onRevealSource={onRevealSource} />{' · '}</>}<Time value={event.occurred_at} /></> : 'No recorded update yet'}</p>
    </div>
    <p className="mt-2 text-xs text-huddleroom-text-muted">Data as of {freshness ? <Time value={dataUpdatedAt} /> : 'Unknown'}{isError ? ' · not updating' : isFetching ? ' · refreshing…' : ''}</p>
  </section>
}
