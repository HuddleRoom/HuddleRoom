import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { descendants, findElement, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type { AgendaItem, Meeting, MeetingActionItem, MeetingDecision, MeetingFinalReview } from '@/lib/types'

const navigateMock = vi.fn()
const queryClientMock = { invalidateQueries: vi.fn(), setQueryData: vi.fn() }

let meetingIdMock: string
let meetingMock: Meeting
let agendaMock: AgendaItem[]
let decisionsMock: MeetingDecision[]
let actionItemsMock: MeetingActionItem[]
let finalReviewMock: MeetingFinalReview | null
let completeReviewPending: boolean
let meetingQueryErrorMock: unknown
let resumeMeetingPending: boolean
const submitHumanTurnMutateMock = vi.fn()
const endMeetingMutateMock = vi.fn()

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigateMock,
  useParams: () => ({ meetingId: meetingIdMock }),
  Link: ({ to, children, ...props }: { to: string; children: React.ReactNode }) => (
    <a href={to} {...props}>{children}</a>
  ),
}))

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => queryClientMock,
}))

vi.mock('@radix-ui/react-collapsible', () => ({
  Root: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Trigger: ({ children }: { children: React.ReactNode }) => <button>{children}</button>,
  Content: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Portal: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Overlay: () => null,
  Content: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Title: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Description: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Close: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

vi.mock('@radix-ui/react-dropdown-menu', () => ({
  Root: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Trigger: ({ children }: { children: React.ReactNode }) => <button>{children}</button>,
  Portal: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Content: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Item: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CheckboxItem: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ItemIndicator: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

vi.mock('sonner', () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

const { MockApiError } = vi.hoisted(() => ({
  MockApiError: class extends Error {
    status: number
    constructor(status: number, message: string) {
      super(message)
      this.status = status
    }
  },
}))

vi.mock('@/lib/api-client', () => ({
  getToken: () => null,
  ApiError: MockApiError,
}))

vi.mock('@/api/agents', () => ({
  useAllAgents: () => ({ items: [] }),
  useAllActiveAgents: () => ({ items: [] }),
}))

let wsStoreState = { turns: [], events: [], connected: false, connect: vi.fn(), disconnect: vi.fn() }

vi.mock('@/stores/meeting-ws', () => ({
  useMeetingWSStore: (selector: (state: unknown) => unknown) =>
    selector(wsStoreState),
}))

vi.mock('@/api/meetings', () => ({
  useMeetings: () => ({ data: { pages: [{ items: [] }] }, fetchNextPage: vi.fn(), hasNextPage: false, isFetchingNextPage: false }),
  useMeeting: () => (
    meetingQueryErrorMock
      ? { data: undefined, isLoading: false, isError: true, error: meetingQueryErrorMock, refetch: vi.fn() }
      : { data: meetingMock, isLoading: false, isError: false, error: null, refetch: vi.fn() }
  ),
  useMeetingTurns: () => ({ data: [] }),
  useMeetingDecisions: () => ({ data: decisionsMock, isLoading: false }),
  useMeetingActionItems: () => ({ data: actionItemsMock, isLoading: false }),
  useMeetingAgenda: () => ({ data: agendaMock }),
  useMeetingFinalReview: () => ({ data: finalReviewMock }),
  useCreateMeeting: () => ({ isPending: false, mutate: vi.fn() }),
  useCopyMeeting: () => ({ isPending: false, mutate: vi.fn() }),
  useSubmitHumanTurn: () => ({ isPending: false, mutate: submitHumanTurnMutateMock }),
  useEndMeeting: () => ({ isPending: false, mutate: endMeetingMutateMock }),
  useCancelMeeting: () => ({ isPending: false, mutate: vi.fn() }),
  useResumeMeeting: () => ({ isPending: resumeMeetingPending, mutate: vi.fn() }),
  useGrantTurn: () => ({ isPending: false, mutate: vi.fn() }),
  useAdvanceAgenda: () => ({ mutate: vi.fn() }),
  useVetoDecision: () => ({ isPending: false, mutate: vi.fn() }),
  useAddAgendaItem: () => ({ isPending: false, mutate: vi.fn() }),
  useUpdateActionItem: () => ({ mutate: vi.fn() }),
  useCompleteMeetingFinalReview: () => ({ isPending: completeReviewPending, mutate: vi.fn() }),
  meetingKeys: {
    detail: (projectId: string, meetingId: string) => ['meetings', projectId, meetingId, 'detail'],
    turns: (projectId: string, meetingId: string) => ['meetings', projectId, meetingId, 'turns'],
    agenda: (projectId: string, meetingId: string) => ['meetings', projectId, meetingId, 'agenda'],
    decisions: (projectId: string, meetingId: string) => ['meetings', projectId, meetingId, 'decisions'],
    actionItems: (projectId: string, meetingId: string) => ['meetings', projectId, meetingId, 'actionItems'],
  },
}))

describe('MeetingsPage', () => {
  beforeEach(() => {
    meetingIdMock = 'meeting-1'
    meetingMock = {
      id: 'meeting-1',
      project_id: 'project-1',
      title: 'Decision meeting',
      meeting_type: 'decision',
      status: 'concluded',
      participant_agent_ids: [],
      agenda_items: [],
      created_at: '2026-07-22T10:00:00Z',
      updated_at: '2026-07-22T10:00:00Z',
    }
    agendaMock = [
      {
        id: 'agenda-summary', meeting_id: 'meeting-1', title: 'Summary wins', order: 1,
        status: 'resolved', resolution_kind: 'kind-hidden', resolution_summary: 'summary-visible',
      },
      {
        id: 'agenda-kind', meeting_id: 'meeting-1', title: 'Kind wins', order: 2,
        status: 'resolved', resolution_kind: 'kind-visible', resolution_summary: null,
      },
      {
        id: 'agenda-status', meeting_id: 'meeting-1', title: 'Status fallback', order: 3,
        status: 'pending', resolution_kind: null, resolution_summary: null,
      },
    ]
    decisionsMock = [
      {
        id: 'decision-1',
        meeting_id: 'meeting-1',
        agenda_item_id: 'agenda-1',
        title: 'Cache strategy',
        chosen_option: 'Use Redis cache',
        content: 'Use Redis cache',
        rationale: 'Latency needs it.',
        decided_by: 'consensus',
        created_at: '2026-07-22T10:01:00Z',
      },
      {
        id: 'decision-2',
        meeting_id: 'meeting-1',
        agenda_item_id: 'agenda-2',
        title: 'Empty chosen option',
        chosen_option: '',
        rationale: 'Testing empty chosen',
        decided_by: 'consensus',
        created_at: '2026-07-22T10:02:00Z',
      },
    ]
    actionItemsMock = []
    finalReviewMock = null
    completeReviewPending = false
    meetingQueryErrorMock = null
    resumeMeetingPending = false
    wsStoreState = { turns: [], events: [], connected: false, connect: vi.fn(), disconnect: vi.fn() }
    navigateMock.mockReset()
    queryClientMock.invalidateQueries.mockReset()
    queryClientMock.setQueryData.mockReset()
    submitHumanTurnMutateMock.mockReset()
    endMeetingMutateMock.mockReset()
  })

  it('renders the full decided_by value for decisions', async () => {
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('by <span style="font-family:var(--huddleroom-font-mono)">consensus</span>')
    expect(markup).not.toContain('by <span style="font-family:var(--huddleroom-font-mono)">consensu</span>')
    expect(markup).toContain('summary-visible')
    expect(markup).not.toContain('kind-hidden')
    expect(markup).toContain('kind-visible')
    expect(markup).toMatch(/Status fallback<\/div><div[^>]*>pending<\/div>/)
  })

  it('renders an accessible pending final review with seeded action items', async () => {
    meetingMock.status = 'concluding'
    finalReviewMock = {
      reviewer_kind: 'organizer_user',
      reviewer_id: '12345678-1234-1234-1234-123456789abc',
      decisions_made: true,
      decisions_clear: true,
      suggested_action_items: ['Implement Redis cache', 'Document rollout'],
    }
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('Were decisions made?')
    expect(markup).toContain('Are decisions clearly stated?')
    expect(markup).toContain('Are action items needed?')
    expect(markup).toContain('aria-label="Final review action items"')
    expect(markup).toContain('Implement Redis cache\nDocument rollout')
    expect.soft(markup).toContain('<h2')
    expect.soft(markup).toContain('<fieldset')
    expect.soft(markup).toContain('<legend')
    expect.soft(markup).toContain('Review questions')
    expect.soft(markup).toContain('Requested reviewer <span style="font-family:var(--huddleroom-font-mono)">organizer_user / 12345678</span>')

    const view = await mountWithTestDom(() => <MeetingsPage />, React.act)

    try {
      const firstTextarea = findElement(view.container, 'textarea', ['aria-label', 'Final review action items'])
      const propsKey = Object.keys(firstTextarea ?? {}).find((key) => key.startsWith('__reactProps$'))
      const textareaProps = propsKey
        ? (firstTextarea as unknown as Record<string, { onChange: (event: { target: { value: string } }) => void }>)[propsKey]
        : undefined
      expect(firstTextarea).toBeDefined()
      expect(textareaProps).toBeDefined()
      await React.act(async () => textareaProps?.onChange({ target: { value: 'Edited meeting 1 item' } }))
      expect(firstTextarea?.value).toBe('Edited meeting 1 item')

      meetingIdMock = 'meeting-2'
      meetingMock = { ...meetingMock, id: meetingIdMock, status: 'concluding' }
      finalReviewMock = {
        reviewer_kind: 'orchestrator',
        reviewer_id: null,
        decisions_made: false,
        decisions_clear: false,
        suggested_action_items: ['Ship queue worker'],
      }
      await view.rerender()
      const nextTextarea = findElement(view.container, 'textarea', ['aria-label', 'Final review action items'])
      expect(nextTextarea?.value).toBe('Ship queue worker')
      expect(nextTextarea?.value).not.toBe('Edited meeting 1 item')

      meetingMock.status = 'concluded'
      await view.rerender()
      expect(findElement(view.container, 'textarea', ['aria-label', 'Final review action items'])).toBeUndefined()
    } finally {
      view.cleanup()
    }

    meetingMock.status = 'concluding'
    completeReviewPending = true
    const pendingMarkup = renderToStaticMarkup(<MeetingsPage />)
    const sectionStart = pendingMarkup.indexOf('aria-label="Meeting final review"')
    const sectionEnd = pendingMarkup.indexOf('</section>', sectionStart)
    const pendingSection = pendingMarkup.slice(sectionStart, sectionEnd)
    expect.soft(pendingSection).toContain('aria-busy="true"')
    expect.soft(pendingSection.match(/type="checkbox"[^>]*disabled/g)).toHaveLength(3)
    expect.soft(pendingSection).toMatch(/<textarea[^>]*disabled/)
    expect.soft(pendingSection).toMatch(/<button[^>]*disabled/)
    expect.soft(pendingSection).toContain('submitting...')
  })

  it('renders a model-error banner with a single Resume action', async () => {
    meetingMock.resume_state = { failed: true, error: 'The model timed out.' }
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('Model error')
    expect(markup).toContain('role="alert"')
    expect((markup.match(/Resume/g) ?? []).length).toBe(1)
  })

  it('shows a fresh model failure but replaces a claimed failure with a neutral resuming state', async () => {
    const { MeetingsPage } = await import('./MeetingsPage')
    meetingMock.resume_state = { failed: true, error: 'The model timed out.' }
    const failedMarkup = renderToStaticMarkup(<MeetingsPage />)

    meetingMock.resume_state = {
      failed: true, error: 'The model timed out.', resuming: true,
    } as NonNullable<Meeting['resume_state']> & { resuming: true }
    const resumingMarkup = renderToStaticMarkup(<MeetingsPage />)

    expect(failedMarkup).toContain('The model timed out.')
    expect((failedMarkup.match(/Resume/g) ?? []).length).toBe(1)
    expect(resumingMarkup).toContain('Resuming')
    expect(resumingMarkup).not.toContain('The model timed out.')
    expect((resumingMarkup.match(/Resume/g) ?? []).length).toBe(0)
  })

  it('renders ErrorRecord with a back-to-meetings action when the meeting fails to load', async () => {
    meetingQueryErrorMock = new ApiError(404, 'not found')
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Back to meetings')
  })

  it('renders Rejected/Dissent/Agreed/Veto rows for a fully-populated decision, expanded by default', async () => {
    decisionsMock = [
      {
        id: 'decision-rich',
        meeting_id: 'meeting-1',
        agenda_item_id: 'agenda-1',
        title: 'Rich decision',
        chosen_option: 'Use Postgres',
        content: 'Use Postgres',
        rationale: 'Fits the data model.',
        alternatives_rejected: ['Use MongoDB', 'Use MySQL'],
        participants_agreed: ['agent-a'],
        dissent: ['agent-b'],
        veto_reason: 'Budget risk',
        decided_by: 'consensus',
        created_at: '2026-07-22T10:01:00Z',
      },
    ]
    const { MeetingsPage } = await import('./MeetingsPage')

    // No click/expand needed — renderToStaticMarkup never interacts, yet the
    // decision's Record rows are already present, proving decisions render
    // expanded by default.
    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('Rejected')
    expect(markup).toContain('Use MongoDB, Use MySQL')
    expect(markup).toContain('Dissent')
    expect(markup).toContain('Agreed')
    expect(markup).toContain('Veto')
    expect(markup).toContain('Budget risk')
    expect(markup).toContain('Use Postgres')
  })

  it('drops the Chosen row when a decision has no chosen_option or content', async () => {
    decisionsMock = [
      {
        id: 'decision-empty',
        meeting_id: 'meeting-1',
        agenda_item_id: 'agenda-2',
        title: 'Empty decision',
        chosen_option: '',
        rationale: 'Nothing chosen yet.',
        decided_by: 'consensus',
        created_at: '2026-07-22T10:02:00Z',
      },
    ]
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).not.toContain('Chosen')
    expect(markup).toContain('Nothing chosen yet.')
  })

  it('renders a linked action item under its decision and omits it from the standalone action items list', async () => {
    decisionsMock = [
      {
        id: 'decision-1',
        meeting_id: 'meeting-1',
        agenda_item_id: 'agenda-1',
        title: 'Cache strategy',
        chosen_option: 'Use Redis cache',
        content: 'Use Redis cache',
        rationale: 'Latency needs it.',
        decided_by: 'consensus',
        created_at: '2026-07-22T10:01:00Z',
      },
    ]
    actionItemsMock = [
      {
        id: 'action-linked',
        meeting_id: 'meeting-1',
        description: 'Provision Redis cluster',
        status: 'in_progress',
        depends_on_decision_id: 'decision-1',
        created_at: '2026-07-22T10:03:00Z',
      },
      {
        id: 'action-unlinked',
        meeting_id: 'meeting-1',
        description: 'Write onboarding doc',
        status: 'in_progress',
        depends_on_decision_id: null,
        created_at: '2026-07-22T10:04:00Z',
      },
    ]
    const { MeetingsPage } = await import('./MeetingsPage')

    const markup = renderToStaticMarkup(<MeetingsPage />)

    expect(markup).toContain('→ Provision Redis cluster')
    expect(markup).toContain('Write onboarding doc')
    expect(markup).toContain('Action items (1)')
    // "Provision Redis cluster" must appear exactly once total — under the
    // decision, not duplicated in the standalone action-items section.
    expect(markup.split('Provision Redis cluster').length - 1).toBe(1)
  })

  it('submits the compose textarea on Cmd+Enter but not on plain Enter', async () => {
    meetingMock.status = 'active'
    const { MeetingsPage } = await import('./MeetingsPage')
    const view = await mountWithTestDom(() => <MeetingsPage />, React.act)

    try {
      const input = findElement(view.container, 'div', ['data-testid', 'meeting-human-input'])
      const textarea = input && findElement(input, 'textarea')
      expect(textarea).toBeDefined()

      const getProps = () => {
        const key = Object.keys(textarea ?? {}).find((k) => k.startsWith('__reactProps$'))
        return key
          ? (textarea as unknown as Record<string, {
              onChange: (e: { target: { value: string } }) => void
              onKeyDown: (e: { key: string; metaKey: boolean; ctrlKey: boolean; preventDefault: () => void }) => void
            }>)[key]
          : undefined
      }

      let props = getProps()
      await React.act(async () => props?.onChange({ target: { value: 'hello team' } }))
      expect(textarea?.value).toBe('hello team')

      props = getProps()
      await React.act(async () => props?.onKeyDown({ key: 'Enter', metaKey: false, ctrlKey: false, preventDefault: vi.fn() }))
      expect(submitHumanTurnMutateMock).not.toHaveBeenCalled()

      props = getProps()
      await React.act(async () => props?.onKeyDown({ key: 'Enter', metaKey: true, ctrlKey: false, preventDefault: vi.fn() }))
      expect(submitHumanTurnMutateMock).toHaveBeenCalledTimes(1)
      expect(submitHumanTurnMutateMock).toHaveBeenCalledWith('hello team', expect.anything())
    } finally {
      view.cleanup()
    }
  })

  it('gates End meeting behind the confirm dialog — the header button only opens it', async () => {
    meetingMock.status = 'active'
    const { MeetingsPage } = await import('./MeetingsPage')
    const view = await mountWithTestDom(() => <MeetingsPage />, React.act)

    try {
      const endButtons = () => descendants(view.container).filter((n) => n.tagName === 'BUTTON' && textOf(n) === 'End meeting')
      expect(endButtons()).toHaveLength(2)

      await React.act(async () => endButtons()[0].click())
      expect(endMeetingMutateMock).not.toHaveBeenCalled()

      await React.act(async () => endButtons()[1].click())
      expect(endMeetingMutateMock).toHaveBeenCalledTimes(1)
      expect(endMeetingMutateMock).toHaveBeenCalledWith('meeting-1', expect.anything())
    } finally {
      view.cleanup()
    }
  })

  it('renders exactly one Resume control, in the header, when resume_state.failed is true', async () => {
    meetingMock.resume_state = { failed: true, error: 'The model timed out.' }
    const { MeetingsPage } = await import('./MeetingsPage')
    const view = await mountWithTestDom(() => <MeetingsPage />, React.act)

    try {
      const resumeButtons = descendants(view.container).filter((n) => n.tagName === 'BUTTON' && textOf(n).includes('Resume'))
      expect(resumeButtons).toHaveLength(1)
    } finally {
      view.cleanup()
    }
  })

  it('skips poll interval when WS is connected, but polls when WS is disconnected', async () => {
    vi.useFakeTimers()
    try {
      const { MeetingsPage } = await import('./MeetingsPage')
      const view = await mountWithTestDom(() => <MeetingsPage />, React.act)

      try {
        // Initially disconnected — poll should start
        wsStoreState.connected = false
        await view.rerender()
        expect(queryClientMock.invalidateQueries).not.toHaveBeenCalled()

        // Advance 5.1 seconds (poll interval is 5s)
        await React.act(async () => vi.advanceTimersByTime(5100))
        expect(queryClientMock.invalidateQueries).toHaveBeenCalled()
        const callsWhenDisconnected = queryClientMock.invalidateQueries.mock.calls.length

        // Connect WS — poll should stop
        queryClientMock.invalidateQueries.mockReset()
        wsStoreState.connected = true
        await view.rerender()

        // Advance 5.1 seconds again — no new invalidations should occur
        await React.act(async () => vi.advanceTimersByTime(5100))
        expect(queryClientMock.invalidateQueries).not.toHaveBeenCalled()

        // Disconnect WS — poll should restart
        wsStoreState.connected = false
        await view.rerender()

        // Advance 5.1 seconds — poll should run again
        await React.act(async () => vi.advanceTimersByTime(5100))
        expect(queryClientMock.invalidateQueries).toHaveBeenCalled()
      } finally {
        view.cleanup()
      }
    } finally {
      vi.useRealTimers()
    }
  })
})
