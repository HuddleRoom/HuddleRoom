import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useUIStore } from '@/stores/ui'
import { useTaskCount } from '@/api/tasks'
import { useSessionCount } from '@/api/sessions'
import { useMeetingCount } from '@/api/meetings'
import { useGraphRunCount } from '@/api/graphs'
import { useRecentEvents } from '@/api/events'
import { useDashboardTriage } from '@/api/dashboard'
import { Button, PageHeader, QueryState, UI_COLORS, NoProjectSelected } from '@/components/common/uiPrimitives'
import { LedgerRow } from '@/components/common/LedgerRow'
import { StatTile } from '@/components/common/StatTile'
import { STATUS_COLORS } from '@/lib/statusColors'
import { humanizeEvent, collapseRuns } from '@/lib/humanizeEvents'
import { dateHeading } from '@/lib/time'
import type { HuddleRoomEvent } from '@/lib/types'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'

// `hasAttention` mirrors the Needs-attention panel above (blocked/failed
// tasks, failed graph runs/sessions, goals needing you, blocked goals). When
// true, this must never say "System steady" — that would contradict the
// panel — even if there's no active/ready work to summarize.
export function buildAttentionSummary({
  sessions,
  meetings,
  graphRuns,
  tasks,
  hasAnyError = false,
  hasAttention = false,
}: {
  sessions?: number
  meetings?: number
  graphRuns?: number
  tasks?: number
  hasAnyError?: boolean
  hasAttention?: boolean
}) {
  if (hasAnyError) {
    return { title: 'Status unavailable', detail: 'One or more status queries failed. Check connectivity.' }
  }

  const segments = [
    meetings ? `${meetings} active meeting${meetings === 1 ? '' : 's'}` : null,
    graphRuns ? `${graphRuns} graph run${graphRuns === 1 ? '' : 's'}` : null,
    tasks ? `${tasks} ready task${tasks === 1 ? '' : 's'}` : null,
    sessions ? `${sessions} running session${sessions === 1 ? '' : 's'}` : null,
  ].filter(Boolean) as string[]

  if (segments.length === 0) {
    if (hasAttention) {
      return { title: 'Needs attention', detail: 'Review the items above.' }
    }
    return { title: 'System steady', detail: 'No active work in this project right now.' }
  }
  return { title: 'Project activity', detail: segments.slice(0, 3).join(' · ') }
}

function buildPrimaryAction({
  tasks,
  meetings,
  sessions,
}: {
  tasks?: number
  meetings?: number
  sessions?: number
}) {
  if (tasks && tasks > 0) return { label: 'View ready tasks', path: '/tasks?status=ready' }
  if (meetings && meetings > 0) return { label: 'View meetings', path: '/meetings' }
  return {
    label: 'Review agents',
    path: '/agents',
  }
}

function getEventNavRoute(source: string): string | null {
  if (!source) return null
  const s = source.toLowerCase()
  if (s.includes('agent') || s.includes('session')) return '/agents'
  if (s.startsWith('graph.')) return '/graphs'
  if (s.includes('meeting')) return '/meetings'
  if (s.includes('task')) return '/tasks'
  return null
}

function getEventDotColor(source: string, eventType?: string): string {
  // Severity wins over source — check event_type first for failure/blocked keywords
  if (eventType) {
    const et = eventType.toLowerCase()
    if (et.includes('failed') || et.includes('error') || et.includes('timeout') || et.includes('abandon')) {
      return STATUS_COLORS.red
    }
    if (et.includes('blocked') || et.includes('stall')) {
      return STATUS_COLORS.amber
    }
  }

  // Fall back to source-based coloring
  if (!source) return UI_COLORS.sidebarGroupLabel
  const s = source.toLowerCase()
  if (s.includes('agent') || s.includes('session')) return UI_COLORS.textMuted
  if (s.startsWith('graph.') || s.includes('meeting')) return UI_COLORS.primary
  return UI_COLORS.sidebarGroupLabel
}

// One row in the activity feed. Wrapped in LedgerRow with time-of-day column,
// action (raw event code), and detail (humanized label).
// `count` is the collapseRuns run-length — shown as a ×N marker when > 1.
export function FeedRow({ event, count, isLast }: { event: HuddleRoomEvent; count: number; isLast: boolean }) {
  const navigate = useNavigate()
  const route = getEventNavRoute(event.source ?? '')

  return (
    <div
      role={route ? 'button' : undefined}
      tabIndex={route ? 0 : undefined}
      className="motion-safe:animate-[queue-row-in_150ms_cubic-bezier(0,0,0.2,1)]"
      style={{
        padding: '9px 16px',
        borderBottom: isLast ? 'none' : '1px solid rgba(226,232,240,0.8)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 12,
        cursor: route ? 'pointer' : 'default',
      }}
      onClick={() => {
        if (route) navigate(route)
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          if (route) navigate(route)
        }
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 6,
          height: 6,
          borderRadius: '50%',
          backgroundColor: getEventDotColor(event.source ?? '', event.event_type),
          flexShrink: 0,
          marginTop: 4,
        }}
      />
      <LedgerRow
        className="flex-1 min-w-0"
        iso={event.emitted_at}
        action={event.event_type}
        detail={<>{humanizeEvent(event.event_type)}{count > 1 && <span> ×{count}</span>}</>}
      >
        {event.source && (
          <span className="text-xs text-huddleroom-text-muted whitespace-nowrap flex-shrink-0">{event.source}</span>
        )}
      </LedgerRow>
    </div>
  )
}

export function DashboardPage() {
  useDocumentTitle('Dashboard')
  const navigate = useNavigate()
  const { activeProjectId, setCreateProjectOpen } = useUIStore()

  const sessionCountQuery = useSessionCount(activeProjectId, 'running')
  const meetingCountQuery = useMeetingCount(activeProjectId, 'active')
  const graphRunCountQuery = useGraphRunCount(activeProjectId, 'active')
  const taskCountQuery = useTaskCount(activeProjectId, 'ready')
  const eventsQuery = useRecentEvents(activeProjectId)
  const triage = useDashboardTriage(activeProjectId)

  const summary = buildAttentionSummary({
    sessions: sessionCountQuery.data?.count,
    meetings: meetingCountQuery.data?.count,
    graphRuns: graphRunCountQuery.data?.count,
    tasks: taskCountQuery.data?.count,
    hasAnyError: sessionCountQuery.isError || meetingCountQuery.isError || graphRunCountQuery.isError || taskCountQuery.isError || triage.isError,
    hasAttention: triage.hasAttention,
  })
  const primaryAction = buildPrimaryAction({
    tasks: taskCountQuery.data?.count,
    meetings: meetingCountQuery.data?.count,
    sessions: sessionCountQuery.data?.count,
  })

  if (!activeProjectId) {
    return (
      <NoProjectSelected
        pageTitle="Dashboard"
        title="Select a project to see what needs attention"
        detail="The dashboard shows running session counts, active meetings, graph runs, ready tasks, and a live event feed for one project at a time. Choose a workspace in the top-bar switcher to connect it."
        primaryAction={{ label: 'New project', onClick: () => setCreateProjectOpen(true) }}
        prerequisites={[
          'Your account can list at least one HuddleRoom project.',
          'The project switcher in the top bar shows the workspace you want to steer.',
        ]}
        alternatePaths={[
          'If the switcher is empty, confirm you are in the correct environment or ask an administrator to create a project.',
        ]}
      />
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <PageHeader title="Dashboard" className="mb-1" />

      {/* Triage-first "Needs attention" surface */}
      <section aria-live="polite" className="rounded-md px-4 py-3 bg-white" style={{ border: `1px solid ${UI_COLORS.border}` }}>
        {triage.isLoading ? (
          <div className="flex items-center gap-2">
            <div className="skeleton" style={{ flex: 1, height: 20, borderRadius: 4 }} />
          </div>
        ) : triage.hasAttention ? (
          <div className="flex flex-col gap-2">
            <span className="text-xs font-semibold" style={{ color: STATUS_COLORS.red }}>Needs attention</span>
            <div className="flex flex-wrap gap-2">
              {(triage.blockedTasks?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="blocked tasks"
                  value={triage.blockedTasks.count!}
                  color={STATUS_COLORS.amber}
                  onClick={() => navigate('/tasks?status=blocked')}
                />
              )}
              {(triage.failedTasks?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="failed tasks"
                  value={triage.failedTasks.count!}
                  color={STATUS_COLORS.red}
                  onClick={() => navigate('/tasks?status=failed')}
                />
              )}
              {(triage.failedGraphRuns?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="failed graph runs"
                  value={triage.failedGraphRuns.count!}
                  color={STATUS_COLORS.red}
                  onClick={() => navigate('/graphs?status=failed')}
                />
              )}
              {(triage.failedSessions?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="failed sessions"
                  value={triage.failedSessions.count!}
                  color={STATUS_COLORS.red}
                  onClick={() => navigate('/agents')}
                />
              )}
              {(triage.needsYouGoals?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="goals need you"
                  value={triage.needsYouGoals.count!}
                  color={STATUS_COLORS.red}
                  onClick={() =>
                    navigate(
                      triage.needsYouGoals.count === 1
                        ? `/orchestration/${triage.needsYouGoals.goals[0].id}#needs-you-queue-heading`
                        : '/orchestration'
                    )
                  }
                />
              )}
              {(triage.blockedGoals?.count ?? 0) > 0 && (
                <StatTile
                  layout="chip"
                  label="blocked goals"
                  value={triage.blockedGoals.count!}
                  color={STATUS_COLORS.amber}
                  onClick={() =>
                    navigate(
                      triage.blockedGoals.count === 1
                        ? `/orchestration/${triage.blockedGoals.goals[0].id}#needs-you-queue-heading`
                        : '/orchestration'
                    )
                  }
                />
              )}
            </div>
            {triage.isError && (
              <span className="text-xs font-medium" style={{ color: STATUS_COLORS.red }}>
                Some status checks failed to load.
              </span>
            )}
          </div>
        ) : triage.isError ? (
          <div className="text-xs font-medium" style={{ color: STATUS_COLORS.red }}>Status unavailable — one or more status queries failed.</div>
        ) : (
          <div className="text-xs font-medium" style={{ color: STATUS_COLORS.green }}>✓ System healthy — no blocked tasks or failed runs</div>
        )}
      </section>

      {/* Status bar: summary + stat chips + primary CTA */}
      <div
        role="status"
        aria-live="polite"
        className="bg-white rounded-md overflow-hidden"
        style={{ border: `1px solid ${UI_COLORS.border}` }}
      >
        <div className="flex items-center gap-3 flex-wrap px-4 py-3" style={{ borderBottom: `1px solid ${UI_COLORS.border}` }}>
          <div className="flex-1 min-w-0">
            {sessionCountQuery.isLoading || meetingCountQuery.isLoading || graphRunCountQuery.isLoading || taskCountQuery.isLoading || triage.isLoading ? (
              <div className="skeleton" style={{ width: '60%', height: 16, borderRadius: 4 }} />
            ) : (
              <>
                <h2 className="text-sm font-semibold" style={{ color: UI_COLORS.textPrimary }}>{summary.title}</h2>
                <p className="text-xs mt-0.5 leading-relaxed" style={{ color: UI_COLORS.textMuted }}>{summary.detail}</p>
              </>
            )}
          </div>
          <Button variant="primary" size="sm" onClick={() => navigate(primaryAction.path)}>
            {primaryAction.label}
          </Button>
        </div>
        <div className="flex flex-wrap gap-2 px-4 py-3">
          <StatTile label="Sessions" value={sessionCountQuery.data?.count} color={STATUS_COLORS.amber} to="/agents" />
          <StatTile label="Meetings" value={meetingCountQuery.data?.count} to="/meetings" />
          <StatTile label="Graphs" value={graphRunCountQuery.data?.count} to="/graphs" />
          <StatTile label="Ready tasks" value={taskCountQuery.data?.count} color={STATUS_COLORS.blue} to="/tasks" />
        </div>
      </div>

      {/* Activity feed — dominant surface */}
      <div className="bg-white rounded-md overflow-hidden flex flex-col" style={{ border: `1px solid ${UI_COLORS.border}` }}>
        <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: `1px solid ${UI_COLORS.border}` }}>
          <span className="text-sm font-semibold" style={{ color: UI_COLORS.textPrimary }}>Events</span>
          <span className="text-xs" style={{ color: UI_COLORS.textMuted }}>Newest first</span>
        </div>
        <div
          aria-label="Recent project events"
          style={{ minHeight: 200, maxHeight: 520, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}
        >
          <QueryState
            query={{
              isLoading: eventsQuery.isLoading,
              isError: eventsQuery.isError,
              data: eventsQuery.data,
              refetch: eventsQuery.refetch,
            }}
            skeleton="rows"
            skeletonCount={6}
            errorLabel="Failed to load recent events"
            emptyLabel="No recent activity"
            emptyDetail="Once agents, meetings, or graphs emit events, they will appear here with newest updates first."
          >
            {(events) => {
              const collapsed = collapseRuns(events.map((e) => ({ ...e, type: e.event_type })))
              let previousDate: string | null = null

              return collapsed.map(({ event, count }, idx) => {
                const currentDate = dateHeading(event.emitted_at)
                const showDateHeader = currentDate !== previousDate
                if (showDateHeader) previousDate = currentDate

                return (
                  <React.Fragment key={`group-${event.id}`}>
                    {showDateHeader && (
                      <div
                        role="separator"
                        aria-label={currentDate}
                        className="px-4 py-2 text-xs font-semibold text-huddleroom-text-muted uppercase tracking-wide"
                      >
                        {currentDate}
                      </div>
                    )}
                    <FeedRow event={event} count={count} isLast={idx === collapsed.length - 1} />
                  </React.Fragment>
                )
              })
            }}
          </QueryState>
        </div>
      </div>

    </div>
  )
}
