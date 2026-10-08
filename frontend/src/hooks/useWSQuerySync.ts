import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { useWSStore } from '@/stores/ws'
import { useUIStore } from '@/stores/ui'
import type { Meeting } from '@/lib/types'

export function isOrchestrationEvent(eventType: string) {
  return eventType.startsWith('orchestration.')
}

export function orchestrationInvalidationKeys(projectId: string | null) {
  return [
    ['orchestration-goals', projectId],
    // Prefix-matches every orchestrationBaselineKey(projectId, goalId) sub-key
    // (processes, checkpoint, memory, etc.) via TanStack Query's default fuzzy
    // invalidateQueries matching. See useWSQuerySync.test.ts regression test.
    ['orchestration-goal', projectId],
    ['events', projectId],
  ] as const
}

export function invalidationKeysForEvent(eventType: string, projectId: string | null) {
  return isOrchestrationEvent(eventType)
    ? orchestrationInvalidationKeys(projectId)
    : []
}

export function useWSQuerySync() {
  const qc = useQueryClient()
  const events = useWSStore((s) => s.events)
  const projectId = useUIStore((s) => s.activeProjectId)
  const lastIdRef = useRef<string | null>(null)

  useEffect(() => {
    if (!events.length) return
    const lastId = lastIdRef.current
    const cutoff = lastId === null ? -1 : events.findIndex((e) => e.id === lastId)
    const newEvents = cutoff === -1 ? events : events.slice(0, cutoff)
    if (!newEvents.length) return
    lastIdRef.current = events[0].id

    let invalidateTasks = false
    let invalidateTaskCount = false
    let invalidateEvents = false
    let invalidateSessionCount = false
    let invalidateAgents = false
    let invalidateGraphs = false
    let invalidateGraphRuns = false
    let invalidateGraphRunCount = false
    let invalidateMeetings = false
    let invalidateMeetingCount = false
    let invalidateKnowledge = false
    let invalidateMemory = false
    let invalidateOrchestration = false
    const taskIds = new Set<string>()
    const meetingIds = new Set<string>()

    for (const event of newEvents) {
      const type = event.event_type
      const taskId = event.payload.task_id as string | undefined
      const meetingId = event.payload.meeting_id as string | undefined

      if (type === 'task.created' || type === 'task.status_changed' || type === 'task.assigned') {
        invalidateTasks = true
        invalidateTaskCount = true
        invalidateEvents = true
        if (taskId) taskIds.add(taskId)
      } else if (
        type === 'session.created' || type === 'session.cancelled' ||
        type === 'session.failed' || type === 'session.completed' || type === 'session.started'
      ) {
        invalidateSessionCount = true
        invalidateEvents = true
        if (taskId) {
          taskIds.add(taskId)
          invalidateTasks = true
          invalidateTaskCount = true
        }
        if (type === 'session.failed') {
          const errorMsg = event.payload.error as string | undefined
          toast.error(errorMsg ? `Session failed: ${errorMsg}` : 'Session failed')
        }
      } else if (type === 'agent.created') {
        invalidateAgents = true
        invalidateEvents = true
      } else if (type.startsWith('graph.')) {
        invalidateGraphs = true
        invalidateGraphRuns = true
        invalidateGraphRunCount = true
        invalidateEvents = true
      } else if (type === 'meeting.turn_complete' && meetingId && qc.getQueryData<Meeting>(['meeting', projectId, meetingId])?.resume_state?.resuming) {
        meetingIds.add(meetingId)
      } else if (type.startsWith('meeting.') && type !== 'meeting.turn_complete' && type !== 'meeting.human_turn') {
        invalidateMeetings = true
        invalidateMeetingCount = true
        invalidateEvents = true
        if (meetingId) meetingIds.add(meetingId)
      } else if (type.startsWith('knowledge.')) {
        invalidateKnowledge = true
        invalidateEvents = true
      } else if (type.startsWith('memory.')) {
        invalidateMemory = true
        invalidateEvents = true
      } else if (invalidationKeysForEvent(type, projectId).length > 0) {
        invalidateOrchestration = true
        invalidateEvents = true
      } else if (type.startsWith('artifact.')) {
        invalidateEvents = true
      }
    }

    if (invalidateTasks) qc.invalidateQueries({ queryKey: ['tasks', projectId] })
    if (invalidateTaskCount) qc.invalidateQueries({ queryKey: ['tasks', 'count', projectId] })
    if (invalidateSessionCount) qc.invalidateQueries({ queryKey: ['sessions', 'count', projectId] })
    if (invalidateAgents) qc.invalidateQueries({ queryKey: ['agents'] })
    if (invalidateGraphs) qc.invalidateQueries({ queryKey: ['graphs', projectId] })
    if (invalidateGraphRuns) qc.invalidateQueries({ queryKey: ['graph-runs', projectId] })
    if (invalidateGraphRunCount) qc.invalidateQueries({ queryKey: ['graph-runs', 'count', projectId] })
    if (invalidateMeetings) qc.invalidateQueries({ queryKey: ['meetings', projectId] })
    if (invalidateMeetingCount) qc.invalidateQueries({ queryKey: ['meetings', 'count', projectId] })
    if (invalidateKnowledge) qc.invalidateQueries({ queryKey: ['knowledge', projectId] })
    if (invalidateMemory) qc.invalidateQueries({ queryKey: ['memory', projectId] })
    if (invalidateOrchestration) {
      for (const queryKey of orchestrationInvalidationKeys(projectId)) {
        qc.invalidateQueries({ queryKey })
      }
    }
    if (invalidateEvents && !invalidateOrchestration) {
      qc.invalidateQueries({ queryKey: ['events', projectId] })
    }
    for (const taskId of taskIds) {
      qc.invalidateQueries({ queryKey: ['task', projectId, taskId] })
      qc.invalidateQueries({ queryKey: ['task', 'sessions', projectId, taskId] })
    }
    for (const meetingId of meetingIds) {
      qc.invalidateQueries({ queryKey: ['meeting', projectId, meetingId] })
      qc.invalidateQueries({ queryKey: ['meeting', 'turns', projectId, meetingId] })
      qc.invalidateQueries({ queryKey: ['meeting', 'agenda', projectId, meetingId] })
      qc.invalidateQueries({ queryKey: ['meeting', 'decisions', projectId, meetingId] })
      qc.invalidateQueries({ queryKey: ['meeting', 'action-items', projectId, meetingId] })
    }
  }, [events, projectId, qc])
}
