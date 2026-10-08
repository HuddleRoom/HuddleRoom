import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryObserver } from '@tanstack/react-query'
import { apiFetch } from '@/lib/api-client'
import {
  completeMeetingFinalReview,
  invalidateMeetingFinalReviewQueries,
  meetingKeys,
  useMeeting,
  useMeetingActionItems,
  useMeetingAgenda,
  useMeetingDecisions,
  useMeetingFinalReview,
  useMeetingSignals,
  useMeetingTurns,
  useResumeMeeting,
} from './meetings'
import type { Meeting, MeetingFinalReviewSubmit } from '@/lib/types'

const { useQueryMock, useMutationMock, useQueryClientMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn((options) => options),
  useMutationMock: vi.fn((options) => options),
  useQueryClientMock: vi.fn(),
}))

vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@tanstack/react-query')>()),
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock,
}))

vi.mock('@/lib/api-client', () => ({ apiFetch: vi.fn() }))

const apiFetchMock = vi.mocked(apiFetch)

describe('meeting final review API', () => {
  beforeEach(() => {
    apiFetchMock.mockReset()
    useQueryMock.mockClear()
    useMutationMock.mockClear()
    useQueryClientMock.mockReset()
  })

  it.each<[string, MeetingFinalReviewSubmit]>([
    ['items', {
      decisions_made: true,
      decisions_clear: true,
      action_items_needed: true,
      action_items: ['Implement Redis cache'],
    }],
    ['waiver', {
      decisions_made: true,
      decisions_clear: true,
      action_items_needed: false,
      action_items: [],
    }],
  ])('submits the %s final review payload and invalidates meeting queries', async (_kind, payload) => {
    apiFetchMock.mockResolvedValue(undefined)
    const invalidateQueries = vi.fn()

    await expect(completeMeetingFinalReview('meeting-1', payload)).resolves.toBeUndefined()
    await invalidateMeetingFinalReviewQueries({ invalidateQueries }, 'project-1', 'meeting-1')

    expect(apiFetchMock).toHaveBeenCalledWith('/api/v1/meetings/meeting-1/final-review', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    expect(invalidateQueries).toHaveBeenNthCalledWith(1, { queryKey: ['meeting', 'project-1', 'meeting-1'] })
    expect(invalidateQueries).toHaveBeenNthCalledWith(2, { queryKey: ['meeting', 'final-review', 'project-1', 'meeting-1'] })
    expect(invalidateQueries).toHaveBeenNthCalledWith(3, { queryKey: ['meeting', 'action-items', 'project-1', 'meeting-1'] })
  })

  it('includes the project in every meeting child key', () => {
    expect([
      meetingKeys.detail('project-1', 'meeting-1'),
      meetingKeys.turns('project-1', 'meeting-1'),
      meetingKeys.decisions('project-1', 'meeting-1'),
      meetingKeys.actionItems('project-1', 'meeting-1'),
      meetingKeys.finalReview('project-1', 'meeting-1'),
      meetingKeys.agenda('project-1', 'meeting-1'),
      meetingKeys.signals('project-1', 'meeting-1'),
    ]).toEqual([
      ['meeting', 'project-1', 'meeting-1'],
      ['meeting', 'turns', 'project-1', 'meeting-1'],
      ['meeting', 'decisions', 'project-1', 'meeting-1'],
      ['meeting', 'action-items', 'project-1', 'meeting-1'],
      ['meeting', 'final-review', 'project-1', 'meeting-1'],
      ['meeting', 'agenda', 'project-1', 'meeting-1'],
      ['meeting', 'signals', 'project-1', 'meeting-1'],
    ])
  })

  it('does not fetch or cache meeting children without a project', async () => {
    apiFetchMock.mockResolvedValue([])
    useMeeting(null, 'meeting-1')
    useMeetingTurns(null, 'meeting-1')
    useMeetingDecisions(null, 'meeting-1')
    useMeetingActionItems(null, 'meeting-1')
    useMeetingFinalReview(null, 'meeting-1', true)
    useMeetingAgenda(null, 'meeting-1')
    useMeetingSignals(null, 'meeting-1')

    for (const [options] of useQueryMock.mock.calls) {
      const queryClient = new QueryClient()
      const observer = new QueryObserver(queryClient, options)
      const unsubscribe = observer.subscribe(() => undefined)
      await Promise.resolve()
      unsubscribe()
      expect(queryClient.getQueryData(options.queryKey)).toBeUndefined()
    }
    expect(apiFetchMock).not.toHaveBeenCalled()
  })

  it('marks the cached failed turn as resuming after the resume request is accepted', () => {
    const setQueryData = vi.fn()
    const invalidateQueries = vi.fn()
    useQueryClientMock.mockReturnValue({ setQueryData, invalidateQueries })
    const meeting = {
      id: 'meeting-1', project_id: 'project-1', title: 'Retry', meeting_type: 'decision', status: 'active',
      participant_agent_ids: [], agenda_items: [], created_at: '', updated_at: '',
      resume_state: { failed: true, error: 'timed out', speaker_agent_id: 'agent-1' },
    } as Meeting
    const mutation = useResumeMeeting() as unknown as { onSuccess: (meeting: Meeting) => void }

    mutation.onSuccess(meeting)

    expect(setQueryData).toHaveBeenCalledWith(
      meetingKeys.detail('project-1', 'meeting-1'),
      expect.objectContaining({
        resume_state: { failed: true, error: 'timed out', speaker_agent_id: 'agent-1', resuming: true },
      }),
    )
  })
})
