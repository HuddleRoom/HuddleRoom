const EVENT_LABELS: Record<string, string> = {
  'meeting.concluded': 'Meeting concluded',
  'meeting.scheduled': 'Meeting scheduled',
  'meeting.started': 'Meeting started',
  'meeting.turn_complete': 'Agent turn completed',
  'meeting.trace': 'Meeting trace',
  'meeting.decision': 'Decision recorded',
  'meeting.action_item': 'Action item created',
  'task.created': 'Task created',
  'task.updated': 'Task updated',
  'task.completed': 'Task completed',
  'task.failed': 'Task failed',
  'agent.created': 'Agent created',
  'agent.updated': 'Agent updated',
  'session.started': 'Session started',
  'session.ended': 'Session ended',
  'graph.run_started': 'Graph run started',
  'graph.run_advanced': 'Graph run advanced',
  'graph.run_completed': 'Graph run completed',
  'graph.run_failed': 'Graph run failed',
  'graph.run_escalated': 'Graph run escalated',
  'graph.run_external_resolution': 'Graph run resolved externally',
  // Below: real event types found via
  // `grep -rn "event_type\|\.emit(\|publish_event" ../rally --include="*.py" | grep -o '"[a-z_]*\.[a-z_]*"' | sort -u`
  'artifact.breaking_change': 'Artifact breaking change',
  'artifact.content_changed': 'Artifact content changed',
  'meeting.deadlocked': 'Meeting deadlocked',
  'meeting.decision_recorded': 'Decision recorded',
  'meeting.human_turn': 'Human turn requested',
  'meeting.request_pending': 'Request pending',
  'meeting.signal': 'Meeting signal',
  'meeting.signal_probe': 'Meeting signal probe',
  'meeting.timeout': 'Meeting timed out',
  'meeting.turn_failed': 'Agent turn failed',
  'orchestration.graph_start_requested': 'Graph start requested',
  'review.approved': 'Review approved',
  'session.cancelled': 'Session cancelled',
  'session.completed': 'Session completed',
  'session.failed': 'Session failed',
  'system.escalation_alert': 'Escalation alert',
  'task.status_changed': 'Task status changed',
  // The grep above requires a dot, so underscore-only MeetingEvent types (same
  // feed, served by rally/routers/events.py) need separate enumeration:
  // rally/services/meeting_outcome.py:582,592,643,718,732; meeting_runner.py:1588
  'meeting_final_pass': 'Meeting final pass',
  'final_pass_completed': 'Final pass completed',
  'partial_finalization': 'Partial finalization',
  'human_intervention_required': 'Human intervention required',
}

export function humanizeEvent(type: string): string {
  const known = EVENT_LABELS[type]
  if (known) return known
  const cleaned = type.replace(/[._]/g, ' ').trim()
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

export interface FeedEvent { type: string; [k: string]: unknown }
export interface CollapsedEvent<T extends { type: string }> { event: T; count: number }

export function collapseRuns<T extends { type: string }>(events: T[]): CollapsedEvent<T>[] {
  const out: CollapsedEvent<T>[] = []
  for (const e of events) {
    const last = out[out.length - 1]
    if (last && last.event.type === e.type) last.count += 1
    else out.push({ event: e, count: 1 })
  }
  return out
}
