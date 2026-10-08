import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient } from '@tanstack/react-query'
import { orchestrationBaselineKey } from '@/api/orchestration'
import {
  invalidationKeysForEvent,
  isOrchestrationEvent,
  orchestrationInvalidationKeys,
  useWSQuerySync,
} from './useWSQuerySync'

const mocks = vi.hoisted(() => ({
  effects: [] as Array<() => void>,
  events: [] as Array<{
    id: string
    event_type: string
    payload: Record<string, unknown>
    emitted_at: string
  }>,
  invalidateQueries: vi.fn(),
  getQueryData: vi.fn(),
  projectId: 'project-1' as string | null,
}))

vi.mock('react', () => ({
  useEffect: (effect: () => void) => { mocks.effects.push(effect) },
  useRef: <T>(current: T) => ({ current }),
}))

vi.mock('@tanstack/react-query', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@tanstack/react-query')>()
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: mocks.invalidateQueries,
      getQueryData: mocks.getQueryData,
    }),
  }
})

vi.mock('sonner', () => ({ toast: { error: vi.fn() } }))

vi.mock('@/stores/ws', () => ({
  useWSStore: (selector: (state: { events: typeof mocks.events }) => unknown) =>
    selector({ events: mocks.events }),
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: { activeProjectId: string | null }) => unknown) =>
    selector({ activeProjectId: mocks.projectId }),
}))

beforeEach(() => {
  mocks.effects = []
  mocks.events = []
  mocks.invalidateQueries.mockReset()
  mocks.getQueryData.mockReset()
  mocks.projectId = 'project-1'
})

describe('isOrchestrationEvent', () => {
  it('matches orchestration events without swallowing adjacent domains', () => {
    expect(isOrchestrationEvent('orchestration.gate_overridden')).toBe(true)
    expect(isOrchestrationEvent('orchestration.goal_paused')).toBe(true)
    expect(isOrchestrationEvent('task.status_changed')).toBe(false)
    expect(isOrchestrationEvent('orchestration')).toBe(false)
  })

  it('dispatches list detail and recent-event invalidations', () => {
    const expected = [
      ['orchestration-goals', 'project-1'],
      ['orchestration-goal', 'project-1'],
      ['events', 'project-1'],
    ]
    expect(orchestrationInvalidationKeys('project-1')).toEqual(expected)
    expect(invalidationKeysForEvent('orchestration.gate_overridden', 'project-1')).toEqual(expected)
    expect(invalidationKeysForEvent('task.status_changed', 'project-1')).toEqual([])
  })

  it('invalidates each orchestration query prefix once for an event batch', () => {
    mocks.events = [
      {
        id: 'event-2',
        event_type: 'orchestration.goal_paused',
        payload: {},
        emitted_at: '2026-07-15T10:00:01Z',
      },
      {
        id: 'event-1',
        event_type: 'orchestration.gate_overridden',
        payload: {},
        emitted_at: '2026-07-15T10:00:00Z',
      },
    ]

    useWSQuerySync()
    expect(mocks.effects).toHaveLength(1)
    mocks.effects[0]()

    const queryKeys = mocks.invalidateQueries.mock.calls.map(
      ([options]) => (options as { queryKey: readonly unknown[] }).queryKey,
    )
    expect(queryKeys).toEqual([
      ['orchestration-goals', 'project-1'],
      ['orchestration-goal', 'project-1'],
      ['events', 'project-1'],
    ])
    expect(queryKeys.filter(([prefix]) => prefix === 'events')).toHaveLength(1)
  })

  it('prefix-invalidates orchestrationBaselineKey sub-keys on a real QueryClient', () => {
    const qc = new QueryClient()
    mocks.invalidateQueries.mockImplementation((options) => qc.invalidateQueries(options))

    const processesKey = [...orchestrationBaselineKey('project-1', 'goal-1'), 'processes']
    const checkpointKey = [...orchestrationBaselineKey('project-1', 'goal-1'), 'checkpoint']
    qc.setQueryData(processesKey, { seeded: true })
    qc.setQueryData(checkpointKey, { seeded: true })

    mocks.events = [
      {
        id: 'event-1',
        event_type: 'orchestration.goal_paused',
        payload: {},
        emitted_at: '2026-07-15T10:00:00Z',
      },
    ]

    useWSQuerySync()
    expect(mocks.effects).toHaveLength(1)
    mocks.effects[0]()

    expect(qc.getQueryState(processesKey)?.isInvalidated).toBe(true)
    expect(qc.getQueryState(checkpointKey)?.isInvalidated).toBe(true)
  })

  it('invalidates a cached resuming meeting on turn completion without changing ordinary meetings', () => {
    const qc = new QueryClient()
    mocks.invalidateQueries.mockImplementation((options) => qc.invalidateQueries(options))
    mocks.getQueryData.mockImplementation((queryKey) => qc.getQueryData(queryKey))
    const resumingKey = ['meeting', 'project-1', 'meeting-resuming']
    const ordinaryKey = ['meeting', 'project-1', 'meeting-ordinary']
    const resumingMeeting = { id: 'meeting-resuming', resume_state: { failed: true, resuming: true } }
    const ordinaryMeeting = { id: 'meeting-ordinary', resume_state: { failed: true } }
    qc.setQueryData(resumingKey, resumingMeeting)
    qc.setQueryData(ordinaryKey, ordinaryMeeting)
    mocks.events = [
      {
        id: 'event-ordinary', event_type: 'meeting.turn_complete',
        payload: { meeting_id: 'meeting-ordinary' }, emitted_at: '2026-07-15T10:00:01Z',
      },
      {
        id: 'event-resuming', event_type: 'meeting.turn_complete',
        payload: { meeting_id: 'meeting-resuming' }, emitted_at: '2026-07-15T10:00:00Z',
      },
    ]

    useWSQuerySync()
    mocks.effects[0]()

    expect(qc.getQueryState(resumingKey)?.isInvalidated).toBe(true)
    expect(qc.getQueryData(resumingKey)).toEqual(resumingMeeting)
    expect(qc.getQueryState(ordinaryKey)?.isInvalidated).toBe(false)
    expect(qc.getQueryData(ordinaryKey)).toEqual(ordinaryMeeting)
  })
})
