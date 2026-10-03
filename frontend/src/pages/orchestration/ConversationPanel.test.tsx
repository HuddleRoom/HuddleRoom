import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act } from 'react'
import { changeControl, descendants, getButton, mountWithTestDom, TestEvent, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type {
  OrchestrationConversationHistory,
  OrchestrationConversationInvestigation,
  OrchestrationConversationInvestigationStatus,
  OrchestrationConversationTurn,
} from '@/lib/types'

const conversation = vi.hoisted(() => ({
  query: { data: undefined as OrchestrationConversationHistory | undefined, isLoading: false, isError: false, refetch: vi.fn() },
  mutation: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
  feedback: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
  submitSteering: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn(), reset: vi.fn() },
  withdrawSteering: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
  dismissProposal: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
}))
const radixDialog = vi.hoisted(() => ({ onOpenChange: null as ((open: boolean) => void) | null }))

// ponytail: the custom DOM harness runs under vitest's node environment, so
// @radix-ui/react-use-layout-effect's SSR-safety check resolves to a no-op at
// module import time and its real Portal never mounts here. Mock it, matching
// every other page test file that adopted the Dialog primitive.
vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ open, onOpenChange, children }: { open?: boolean; onOpenChange?: (open: boolean) => void; children: React.ReactNode }) => {
    radixDialog.onOpenChange = onOpenChange ?? null
    return open ? <>{children}</> : null
  },
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />,
  // Fire onOpenAutoFocus once per mount, mirroring Radix firing it each time the
  // dialog opens (Root above only mounts Content while `open` is true), so
  // Dialog's initialFocusRef behavior is exercised the same way it is in the app.
  Content: ({ children, onOpenAutoFocus, onCloseAutoFocus, ...props }: React.HTMLAttributes<HTMLDivElement> & { onOpenAutoFocus?: (event: Event) => void; onCloseAutoFocus?: (event: Event) => void }) => {
    React.useEffect(() => { onOpenAutoFocus?.(new Event('focus') as Event) }, []) // eslint-disable-line react-hooks/exhaustive-deps
    return <div {...props}>{children}</div>
  },
  Title: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => <h2 {...props}>{children}</h2>,
  Description: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props}>{children}</p>,
  Close: ({ children }: { children: React.ReactNode }) => <span onClick={() => radixDialog.onOpenChange?.(false)}>{children}</span>,
}))

vi.mock('@/api/orchestration', () => ({
  useOrchestrationConversation: () => conversation.query,
  useSubmitOrchestrationConversation: () => conversation.mutation,
  useRecordOrchestrationConversationFeedback: () => conversation.feedback,
  useSubmitOrchestrationSteering: () => conversation.submitSteering,
  useWithdrawOrchestrationSteering: () => conversation.withdrawSteering,
  useDismissOrchestrationSteeringProposal: () => conversation.dismissProposal,
}))

import { ConversationPanel } from './ConversationPanel'
import { GoalAnnouncerProvider } from './goalAnnouncer'

type FeedbackFixture = {
  feedback?: { feedback_id: string; rating: 'helpful' | 'not_helpful'; reason: 'unanswered' | 'incorrect' | 'missing_context' | 'stale_context' | 'unclear' | 'too_limited' | 'other' | null; created_at: string } | null
  feedback_eligible?: boolean
}

const turn = (overrides: Partial<OrchestrationConversationTurn> & FeedbackFixture = {}): OrchestrationConversationTurn => Object.assign({
  message_id: 'message-1', response_id: 'response-1', client_request_id: 'request-1', sequence: 1,
  actor_id: 'user-1', content: 'What is the current risk?', message_created_at: '2026-09-14T08:00:00Z',
  status: 'completed', run_id: null, answer: 'The validation gate is pending.', error: null,
  started_at: '2026-09-14T08:00:01Z', deadline_at: '2026-09-14T08:02:01Z', finished_at: '2026-09-14T08:00:02Z',
  created_at: '2026-09-14T08:00:00Z', updated_at: '2026-09-14T08:00:02Z', context_version: 'context-1',
  context_manifest: { run_id: null, excluded_categories: ['workspace'], truncated: false, sources: [] },
}, { feedback: null, feedback_eligible: false }, overrides)

const history = (items: OrchestrationConversationTurn[] = [turn()]): OrchestrationConversationHistory => ({
  items, total: items.length, omitted: 0, allowance: { enabled: true, limit: 1000, used: 100, remaining: 900 },
  steering: { enabled: true, eligibility: 'active', eligibility_reason: null, inbox_version: 0, direction_version: 0, requests: [], proposals: [] },
})

const investigation = (status: OrchestrationConversationInvestigationStatus = 'completed'): OrchestrationConversationInvestigation => {
  const attempted = !['pending', 'unavailable', 'limited'].includes(status)
  const active = status === 'pending' || status === 'running'
  return {
    investigation_id: 'investigation-1', status, objective: 'Compare recovery paths',
    attempt_count: attempted ? 1 : 0, repair_count: 0, retry_count: 0,
    sources: [{
      reference: 'rally/services/orchestration_conversation_service.py#L120-L140',
      operation: 'read', status: 'included', freshness_at: '2026-09-14T08:00:00Z', truncated: false,
    }],
    report: status === 'completed' ? {
      findings: 'Expired calls use provider result lookup.',
      uncertainty: 'Failover behavior was not observed.',
      sources: ['rally/services/orchestration_conversation_service.py#L120-L140'],
    } : null,
    error: status === 'failed' ? { code: 'invalid_investigation_report' } : null,
    started_at: attempted ? '2026-09-14T08:00:03Z' : null,
    deadline_at: attempted ? '2026-09-14T08:02:03Z' : null,
    finished_at: active ? null : '2026-09-14T08:00:04Z',
    created_at: '2026-09-14T08:00:02Z', updated_at: '2026-09-14T08:00:04Z',
  }
}

beforeEach(() => {
  conversation.query = { data: history(), isLoading: false, isError: false, refetch: vi.fn() }
  conversation.mutation = { isPending: false, isError: false, error: null, mutate: vi.fn() }
  conversation.feedback = { isPending: false, isError: false, error: null, mutate: vi.fn() }
  conversation.submitSteering = { isPending: false, isError: false, error: null, mutate: vi.fn(), reset: vi.fn() }
  conversation.withdrawSteering = { isPending: false, isError: false, error: null, mutate: vi.fn() }
  conversation.dismissProposal = { isPending: false, isError: false, error: null, mutate: vi.fn() }
})

async function mount() {
  return mountWithTestDom(() => <GoalAnnouncerProvider key="goal-1"><ConversationPanel projectId="project-1" goalId="goal-1" /></GoalAnnouncerProvider>, act)
}

function textarea(container: Parameters<typeof descendants>[0]) {
  return descendants(container).find((node) => node.tagName === 'TEXTAREA')!
}

function pressKey(control: object, key: string, shiftKey = false) {
  const event = Object.assign(new TestEvent('keydown'), { key, shiftKey })
  const propsKey = Object.keys(control).find((name) => name.startsWith('__reactProps$'))
  const onKeyDown = propsKey
    ? (control as Record<string, { onKeyDown?: (nextEvent: TestEvent & { key: string; shiftKey: boolean }) => void }>)[propsKey]?.onKeyDown
    : undefined
  onKeyDown?.(event)
  return event
}

function feedbackSection(container: Parameters<typeof descendants>[0], sequence: number) {
  return descendants(container).find((node) => node.getAttribute('aria-label') === `Feedback for message ${sequence}`)
}

function buttonIn(container: Parameters<typeof descendants>[0], label: string) {
  return descendants(container).find((node) => node.tagName === 'BUTTON' && textOf(node) === label)
}

const feedbackReasons = [
  ['Did not answer', 'unanswered'], ['Incorrect', 'incorrect'], ['Missing context', 'missing_context'],
  ['Stale context', 'stale_context'], ['Unclear', 'unclear'], ['Too limited', 'too_limited'], ['Other', 'other'],
] as const

describe('ConversationPanel', () => {
  it('keeps Send and Steer work before the inline draft and steering ledger', async () => {
    const steering = history().steering
    steering.requests = [{
      request_id: 'request-1', client_request_id: 'client-1', sequence: 1, directive: 'Prioritize validation.', target_type: 'goal', target_id: 'goal-1', scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Expected effect.', source_proposal_id: null, supersedes_request_id: null, status: 'pending', reason_code: 'submitted', submitted_at: '2026-09-16T10:00:00Z', considered_at: null, finished_at: null, updated_at: '2026-09-16T10:00:00Z', transitions: [{ status: 'pending', reason_code: 'submitted', actor: 'user-1', created_at: '2026-09-16T10:00:00Z' }], result_action_ids: [],
    }]
    conversation.query.data = { ...history(), steering }
    const view = await mount()
    try {
      const controls = descendants(view.container)
      const send = controls.findIndex((node) => node.tagName === 'BUTTON' && textOf(node) === 'Send message')
      const steer = controls.findIndex((node) => node.tagName === 'BUTTON' && textOf(node) === 'Steer work')
      const ledger = controls.findIndex((node) => node.getAttribute('aria-label') === 'Steering ledger')
      expect(send).toBeLessThan(steer)
      expect(steer).toBeLessThan(ledger)
      expect(textOf(view.container)).toContain('Transition history (1)')
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite' && textOf(node))).toHaveLength(1)
    } finally { view.cleanup() }
  })

  it('shows steering lifecycle state, linked actions, and pending-only withdrawal', async () => {
    const steering = history().steering
    steering.requests = [{
      request_id: 'request-1', client_request_id: 'client-1', sequence: 1, directive: 'Prioritize validation.', target_type: 'goal', target_id: 'goal-1',
      scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Applies to unstarted work.', source_proposal_id: null, supersedes_request_id: null,
      status: 'pending', reason_code: 'submitted', submitted_at: '2026-09-14T08:00:00Z', considered_at: null, finished_at: null, updated_at: '2026-09-14T08:00:00Z',
      transitions: [{ status: 'pending', reason_code: 'submitted', actor: 'user-1', created_at: '2026-09-14T08:00:00Z' }], result_action_ids: ['action-1'],
    }]
    conversation.query.data = { ...history(), steering }
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Submitted — awaiting control plane')
      expect(textOf(view.container)).toContain('submitted')
      expect(descendants(view.container).find((node) => node.tagName === 'A' && node.getAttribute('href') === '#action-action-1')).toBeDefined()
      await act(async () => getButton(view.container, 'Withdraw').click())
      expect(conversation.withdrawSteering.mutate).toHaveBeenCalledWith({ goalId: 'goal-1', requestId: 'request-1' }, expect.any(Object))
    } finally { view.cleanup() }
  })

  it('opens one editable replacement draft only from an applied direction', async () => {
    const steering = history().steering
    steering.requests = [{
      request_id: 'request-applied', client_request_id: 'client-1', sequence: 1, directive: 'Validate first.', target_type: 'goal', target_id: 'goal-1',
      scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Validate before work.', source_proposal_id: null, supersedes_request_id: null,
      status: 'applied', reason_code: 'applied', submitted_at: '2026-09-14T08:00:00Z', considered_at: '2026-09-14T08:00:01Z', finished_at: '2026-09-14T08:00:01Z', updated_at: '2026-09-14T08:00:01Z',
      transitions: [{ status: 'applied', reason_code: 'applied', actor: 'system', created_at: '2026-09-14T08:00:01Z' }], result_action_ids: [],
    }]
    conversation.query.data = { ...history(), steering }
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: vi.fn().mockReturnValue('replacement-uuid') }, configurable: true })
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Replace direction').click())
      const controls = descendants(view.container).filter((node) => node.tagName === 'TEXTAREA')
      expect((controls[1] as HTMLTextAreaElement).value).toBe('Validate first.')
      expect((controls[2] as HTMLTextAreaElement).value).toBe('Validate before work.')
      await act(async () => changeControl(controls[1] as never, 'Use an independent validator.'))
      await act(async () => getButton(view.container, 'Submit steering').click())
      expect(conversation.submitSteering.mutate).toHaveBeenCalledWith(expect.objectContaining({
        supersedesRequestId: 'request-applied', directive: 'Use an independent validator.',
        targetType: 'goal', targetId: 'goal-1', scope: 'run', lifetime: 'remaining_current_run',
      }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it('serializes steering controls and ignores a stale success after a newer failure', async () => {
    const steering = history().steering
    steering.requests = [{
      request_id: 'request-1', client_request_id: 'client-1', sequence: 1, directive: 'Prioritize validation.', target_type: 'goal', target_id: 'goal-1',
      scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Applies to unstarted work.', source_proposal_id: null, supersedes_request_id: null,
      status: 'pending', reason_code: 'submitted', submitted_at: '2026-09-14T08:00:00Z', considered_at: null, finished_at: null, updated_at: '2026-09-14T08:00:00Z',
      transitions: [{ status: 'pending', reason_code: 'submitted', actor: 'user-1', created_at: '2026-09-14T08:00:00Z' }], result_action_ids: [],
    }]
    conversation.query.data = { ...history(), steering }
    const callbacks: Array<{ onSuccess?: () => void }> = []
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: vi.fn().mockReturnValue('steering-uuid-1') }, configurable: true })
    conversation.submitSteering.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Steer work').click())
      const controls = descendants(view.container).filter((node) => node.tagName === 'TEXTAREA')
      await act(async () => changeControl(controls[1] as never, 'Prioritize validation.'))
      await act(async () => changeControl(controls[2] as never, 'Apply before unstarted work.'))
      await act(async () => getButton(view.container, 'Submit steering').click())
      expect(conversation.submitSteering.mutate).toHaveBeenCalledWith({
        goalId: 'goal-1', clientRequestId: 'steering-uuid-1', directive: 'Prioritize validation.', targetType: 'goal', targetId: 'goal-1',
        scope: 'run', lifetime: 'remaining_current_run', impactSummary: 'Apply before unstarted work.', sourceProposalId: null, supersedesRequestId: null,
      }, expect.any(Object))

      conversation.submitSteering = { ...conversation.submitSteering, isPending: true }
      await view.rerender()
      for (const label of ['Working…', 'Discard', 'Withdraw']) expect(getButton(view.container, label).getAttribute('disabled')).not.toBeNull()

      conversation.submitSteering = { ...conversation.submitSteering, isPending: false, isError: true, error: new ApiError(400, 'Steering request was not accepted.') }
      await view.rerender()
      expect(textOf(view.container)).toContain('Steering request was not accepted.')
      expect(view.activeElement?.getAttribute('tabindex')).toBe('-1')
      // ErrorRecord drops its own role="alert" inside GoalAnnouncerProvider and
      // announces through the shared assertive region instead.
      expect(descendants(view.activeElement as never).some((node) => node.getAttribute('role') === 'alert')).toBe(false)
      const assertiveAnnouncements = descendants(view.container).filter((node) => node.getAttribute('role') === 'alert' && node.getAttribute('aria-live') === 'assertive')
      expect(assertiveAnnouncements.some((node) => textOf(node) === 'Steering request was not accepted.')).toBe(true)
      await act(async () => callbacks[0]?.onSuccess?.())
      // The stale success must be ignored: the compose dialog (now titled "Steer
      // work"/"Review and apply steering") stays open with the error still shown.
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'dialog')).toBe(true)
      expect(textOf(view.container)).toContain('Steering request was not accepted.')
    } finally { view.cleanup() }
  })

  it('announces only the newest transition across requests with timestamp ties', async () => {
    const steering = history().steering
    steering.requests = [1, 2].map((sequence) => ({
      request_id: `request-${sequence}`, client_request_id: `client-${sequence}`, sequence, directive: `Directive ${sequence}`, target_type: 'goal' as const, target_id: 'goal-1',
      scope: 'run' as const, lifetime: 'remaining_current_run' as const, impact_summary: `Impact ${sequence}`, source_proposal_id: null, supersedes_request_id: null,
      status: sequence === 2 ? 'applied' as const : 'pending' as const, reason_code: sequence === 2 ? 'applied' : 'submitted', submitted_at: '2026-09-14T08:00:00Z', considered_at: null, finished_at: null, updated_at: '2026-09-14T08:00:00Z',
      transitions: [{ status: sequence === 2 ? 'applied' as const : 'pending' as const, reason_code: sequence === 2 ? 'applied' : 'submitted', actor: 'user-1', created_at: '2026-09-14T08:00:00Z' }], result_action_ids: [],
    }))
    conversation.query.data = { ...history(), steering }
    const view = await mount()
    try {
      const live = descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node))
      expect(live).toHaveLength(1)
      expect(textOf(live[0])).toContain('Applied as advisory direction · applied')
      expect(live[0].getAttribute('aria-atomic')).toBe('true')
      expect(descendants(view.container).filter((node) => node.tagName === 'DETAILS').every((node) => node.getAttribute('open') === null)).toBe(true)
      expect(descendants(view.container).filter((node) => node.tagName === 'LI' && /Directive [12]/.test(textOf(node))).map(textOf)).toEqual(expect.arrayContaining([
        expect.stringContaining('Directive 1'), expect.stringContaining('Directive 2'),
      ]))
    } finally { view.cleanup() }
  })
  it('nests one flat advisory report under its owning answer before timestamps and context', async () => {
    conversation.query.data = history([turn({ investigation: investigation() })])
    const view = await mount()
    try {
      const nested = descendants(view.container).find((node) =>
        node.tagName === 'SECTION' && node.getAttribute('aria-label') === 'Investigation for message 1',
      )
      expect(nested).toBeDefined()
      if (!nested) return

      expect(textOf(nested)).toContain('Advisory · unverified')
      expect(textOf(nested)).toContain('Scope: Compare recovery paths')
      expect(textOf(nested)).toContain('Findings: Expired calls use provider result lookup.')
      expect(textOf(nested)).toContain('Uncertainty: Failover behavior was not observed.')
      expect(textOf(nested)).toContain('This report is advisory and is not accepted evidence.')
      expect(textOf(nested)).toContain('read · included · freshness 2026-09-14T08:00:00Z · truncated false')
      expect(descendants(nested).filter((node) => node.tagName === 'SUMMARY').map(textOf)).toEqual(['Sources (1)'])
      expect(descendants(nested).find((node) => node.tagName === 'DETAILS')?.getAttribute('open')).toBeNull()
      expect(descendants(nested).filter((node) => node.tagName === 'A')).toHaveLength(0)
      expect(descendants(nested).filter((node) => ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'FORM'].includes(node.tagName))).toHaveLength(0)
      // Investigation status is no longer announced via a local live region nested in
      // the report — it flows through the shared page-level polite region instead.
      expect(descendants(nested).filter((node) => node.getAttribute('role') === 'status')).toHaveLength(0)
      const sharedAnnouncements = descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')
      expect(sharedAnnouncements.some((node) => textOf(node).includes('Investigation completed.'))).toBe(true)
      expect(nested.getAttribute('class')).toContain('min-w-0')
      expect(nested.getAttribute('class')).toContain('overflow-hidden')
      expect(nested.getAttribute('class')).not.toMatch(/(?:rounded|border|bg-rally|shadow)/)

      const row = nested.parentNode as typeof nested | null
      expect(row?.tagName).toBe('LI')
      const children = (row?.childNodes.filter((node) => node.nodeType === 1) ?? []) as typeof nested[]
      const answer = children.find((node) => textOf(node).includes('AnswerThe validation gate is pending.'))
      const firstTime = children.find((node) => node.tagName === 'TIME')
      const context = children.find((node) => descendants(node).some((child) => child.tagName === 'SUMMARY' && textOf(child) === 'Context for message 1'))
      expect(children.indexOf(nested)).toBeGreaterThan(children.indexOf(answer!))
      expect(children.indexOf(nested)).toBeLessThan(children.indexOf(firstTime!))
      expect(children.indexOf(nested)).toBeLessThan(children.indexOf(context!))
    } finally { view.cleanup() }
  })

  it('wraps long public investigation prose, source metadata, and safe error code at their actual elements', async () => {
    const reported = investigation()
    reported.objective = 'Compare recovery paths for reissued provider responses without widening authority. '.repeat(3)
    reported.report = {
      ...reported.report!,
      findings: 'Expired calls use provider result lookup before bounded recovery. '.repeat(3),
    }
    const failed = investigation('failed')
    conversation.query.data = history([
      turn({ investigation: reported }),
      turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, investigation: failed }),
    ])
    const view = await mount()
    try {
      const sections = descendants(view.container).filter((node) => node.tagName === 'SECTION' && /^Investigation for message /.test(node.getAttribute('aria-label') ?? ''))
      expect(sections).toHaveLength(2)
      if (sections.length !== 2) return

      const scope = descendants(sections[0]).find((node) => node.tagName === 'P' && textOf(node).startsWith('Scope: Compare recovery paths'))
      const findings = descendants(sections[0]).find((node) => node.tagName === 'P' && textOf(node).startsWith('Findings: Expired calls use provider result lookup'))
      const source = descendants(sections[0]).find((node) => node.tagName === 'LI' && textOf(node).includes('rally/services/orchestration_conversation_service.py#L120-L140'))
      const error = descendants(sections[1]).find((node) => node.tagName === 'P' && textOf(node) === 'invalid_investigation_report')
      expect(scope?.getAttribute('class')).toContain('break-words')
      expect(findings?.getAttribute('class')).toContain('break-words')
      expect(source?.getAttribute('class')).toContain('min-w-0')
      expect(source?.getAttribute('class')).toContain('break-all')
      expect(error?.getAttribute('class')).toContain('break-all')
    } finally { view.cleanup() }
  })

  it.each([
    ['pending', 'Investigation queued.'],
    ['running', 'Investigation in progress.'],
    ['completed', 'Investigation completed.'],
    ['limited', 'Investigation stopped at the conversation allowance limit. Send a new message when allowance is available.'],
    ['failed', 'Investigation failed safely. No finding was accepted.'],
    ['cancelled', 'Investigation cancelled. Send a new message to investigate again; late output is ignored.'],
    ['unavailable', 'Investigation is not enabled for this project. The conversation remains read-only.'],
    ['interrupted_unknown', 'Investigation outcome is unknown. Usage remains held and the orchestrator will not retry it automatically. Send a new message if allowance remains.'],
  ] as const)('renders safe %s investigation state without controls', async (status, copy) => {
    conversation.query.data = history([turn({ investigation: investigation(status) })])
    const view = await mount()
    try {
      const nested = descendants(view.container).find((node) =>
        node.tagName === 'SECTION' && node.getAttribute('aria-label') === 'Investigation for message 1',
      )
      expect(nested).toBeDefined()
      if (!nested) return

      expect(textOf(nested)).toContain(copy)
      expect(descendants(nested).filter((node) => node.tagName === 'A')).toHaveLength(0)
      expect(descendants(nested).filter((node) => ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'FORM'].includes(node.tagName))).toHaveLength(0)
      expect(textOf(nested)).not.toMatch(/workspace_root|provider_request_id|reserved_tokens|raw_error|excerpt|must-not-render/i)
      expect(textOf(nested).replace('Advisory · unverified', '').replace('This report is advisory and is not accepted evidence.', '')).not.toMatch(/\b(?:steer|steering|approve|approval|apply)\b/i)
      if (status === 'failed') expect(textOf(nested)).toContain('invalid_investigation_report')
      const finished = descendants(nested).find((node) => node.tagName === 'TIME')
      if (status === 'pending' || status === 'running') expect(finished).toBeUndefined()
      else {
        expect(finished?.getAttribute('datetime') ?? finished?.getAttribute('dateTime')).toBe('2026-09-14T08:00:04Z')
        expect(textOf(finished!)).toContain('Investigation finished Sep 14, 2026')
      }
    } finally { view.cleanup() }
  })

  it('keeps matching reports inspectable and omits subsections for absent or null investigations', async () => {
    const matching = investigation()
    matching.sources.push({
      reference: '[additional sources]', operation: 'search', status: 'omitted_by_limit', freshness_at: null, truncated: true,
    })
    matching.report = {
      findings: 'The validation gate is pending.', uncertainty: '',
      sources: ['rally/services/orchestration_conversation_service.py#L120-L140'],
    }
    const emptySources = investigation()
    emptySources.sources = []
    emptySources.report = null
    conversation.query.data = history([
      turn({ investigation: matching }),
      turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, investigation: emptySources }),
      turn({ message_id: 'message-3', response_id: 'response-3', sequence: 3, investigation: null }),
      turn({ message_id: 'message-4', response_id: 'response-4', sequence: 4 }),
    ])
    const view = await mount()
    try {
      const sections = descendants(view.container).filter((node) => node.tagName === 'SECTION' && /^Investigation for message /.test(node.getAttribute('aria-label') ?? ''))
      expect(sections.map((node) => node.getAttribute('aria-label'))).toEqual(['Investigation for message 1', 'Investigation for message 2'])
      expect(textOf(sections[0])).toContain('Findings: The validation gate is pending.')
      expect(textOf(sections[0])).toContain('Uncertainty: None reported.')
      expect(textOf(sections[0])).toContain('[additional sources] · search · omitted_by_limit · freshness — · truncated true')
      expect(descendants(sections[1]).filter((node) => node.tagName === 'SUMMARY').map(textOf)).toEqual(['Sources (0)'])
    } finally { view.cleanup() }
  })

  it('keeps repeated sanitized source metadata as separate rows without a React key warning', async () => {
    const repeated = investigation()
    repeated.sources = [
      { reference: '[unsafe path]', operation: 'read', status: 'unsafe', freshness_at: null, truncated: false },
      { reference: '[unsafe path]', operation: 'read', status: 'unsafe', freshness_at: null, truncated: true },
    ]
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    conversation.query.data = history([turn({ investigation: repeated })])
    const view = await mount()
    try {
      const nested = descendants(view.container).find((node) =>
        node.tagName === 'SECTION' && node.getAttribute('aria-label') === 'Investigation for message 1',
      )
      expect(nested).toBeDefined()
      if (!nested) return
      expect(descendants(nested).filter((node) => node.tagName === 'LI').map(textOf)).toEqual([
        '[unsafe path] · read · unsafe · freshness — · truncated false',
        '[unsafe path] · read · unsafe · freshness — · truncated true',
      ])
      expect(consoleError).not.toHaveBeenCalled()
    } finally {
      consoleError.mockRestore()
      view.cleanup()
    }
  })

  it('marks only the latest investigation status live across mixed history without moving composer focus', async () => {
    const first = investigation('completed')
    const second = investigation('running')
    second.investigation_id = 'investigation-2'
    conversation.query.data = history([
      turn({ investigation: first }),
      turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, investigation: second }),
      turn({ message_id: 'message-3', response_id: 'response-3', sequence: 3 }),
    ])
    const view = await mount()
    try {
      const sections = descendants(view.container).filter((node) => node.tagName === 'SECTION' && /^Investigation for message /.test(node.getAttribute('aria-label') ?? ''))
      expect(sections).toHaveLength(2)
      if (sections.length !== 2) return
      // Investigation sections carry no local live-region markup — status changes
      // flow through the single shared page-level polite region instead.
      for (const section of sections) {
        const investigationNodes = [section, ...descendants(section)]
        expect(investigationNodes.every((node) => node.getAttribute('role') !== 'status')).toBe(true)
        expect(investigationNodes.every((node) => !node.getAttribute('aria-live'))).toBe(true)
        expect(investigationNodes.every((node) => !node.getAttribute('aria-atomic'))).toBe(true)
      }
      const sharedPolite = descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && node.getAttribute('aria-live') === 'polite')
      expect(sharedPolite).toHaveLength(1)
      expect(textOf(sharedPolite[0])).toBe('Investigation in progress.')
      expect(sharedPolite[0].getAttribute('aria-atomic')).toBe('true')
      const historicalReport = descendants(sections[0]).find((node) => node.tagName === 'P' && textOf(node).startsWith('Findings:'))
      const historicalSources = descendants(sections[0]).find((node) => node.tagName === 'DETAILS')
      expect(historicalReport?.getAttribute('aria-live')).toBeNull()
      expect(historicalSources?.getAttribute('aria-live')).toBeNull()

      textarea(view.container).focus()
      second.status = 'completed'
      second.finished_at = '2026-09-14T08:00:04Z'
      await view.rerender()
      expect(view.activeElement).toBe(textarea(view.container))
    } finally { view.cleanup() }
  })

  it('renders advisory ledger rows, allowance, safe context, and run boundaries in server order', async () => {
    conversation.query.data = history([
      turn({ context_manifest: { run_id: null, excluded_categories: ['workspace'], truncated: true, sources: [{ source: 'goal', status: 'fresh', freshness_at: '2026-09-14T08:00:00Z', available: 2, included: 1, omitted: 1, truncated: true, references: ['goal-1'] }] } }),
      turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, run_id: 'run-2', status: 'failed', answer: null, error: { code: 'provider_failed' } }),
      turn({ message_id: 'message-3', response_id: 'response-3', sequence: 3, run_id: 'run-2', status: 'pending', answer: null, error: null, finished_at: null }),
      turn({ message_id: 'message-4', response_id: 'response-4', sequence: 4, run_id: null, status: 'interrupted_unknown', answer: null, error: null }),
    ])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Messages are advisory. They do not change goal state, plans, agents, or reservations.')
      expect(textOf(view.container)).toContain('Allowance: 900 remaining of 1000')
      expect(textOf(view.container)).toContain('Run context: No run attached')
      expect(textOf(view.container)).toContain('Run context: run-2')
      expect(textOf(view.container)).toContain('provider_failed')
      expect(textOf(view.container)).toContain('Awaiting response.')
      expect(textOf(view.container)).toContain('Response status is unknown after interruption.')
      expect(textOf(view.container)).toContain('freshness 2026-09-14T08:00:00Z')
      expect(textOf(view.container)).toContain('goal · fresh · freshness 2026-09-14T08:00:00Z · available 2 · included 1 · omitted 1 · truncated true')
      expect(textOf(view.container)).toContain('goal-1')
      expect(textOf(view.container)).not.toContain('response_id')
      const mandatoryAdvisory = 'Messages are advisory. They do not change goal state, plans, agents, or reservations.'
      expect(textOf(view.container)).toContain(mandatoryAdvisory)
      const controls = descendants(view.container).filter((node) =>
        ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(node.tagName) || node.getAttribute('role'),
      )
      expect(controls.map((node) => `${textOf(node)} ${node.getAttribute('aria-label') ?? ''} ${node.getAttribute('placeholder') ?? ''}`).join(' ')).not.toMatch(/\b(?:pause|resume|cancel|ask_human)\b/i)
      expect(descendants(view.container).filter((node) => node.tagName === 'DETAILS')).toHaveLength(4)
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'separator').map((node) => node.getAttribute('aria-label'))).toEqual([
        'Run context: No run attached', 'Run context: run-2', 'Run context: No run attached',
      ])
      expect(descendants(view.container).filter((node) => node.tagName === 'SUMMARY').map(textOf)).toEqual([
        'Context for message 1', 'Context for message 2', 'Context for message 3', 'Context for message 4',
      ])
      const times = descendants(view.container).filter((node) => node.tagName === 'TIME')
      const dateTime = (node: (typeof times)[number]) => node.getAttribute('datetime') ?? node.getAttribute('dateTime') ?? (node as unknown as { dateTime?: string }).dateTime
      expect(times.some((node) => dateTime(node) === '2026-09-14T08:00:00Z' && textOf(node).includes('Sep 14, 2026'))).toBe(true)
      expect(times.some((node) => dateTime(node) === '2026-09-14T08:00:02Z' && textOf(node).includes('Finished Sep 14, 2026'))).toBe(true)
      expect(descendants(view.container).filter((node) => node.tagName === 'A')).toHaveLength(0)
      expect(descendants(view.container).find((node) => node.tagName === 'OL' && node.getAttribute('aria-label') === 'Conversation history')?.getAttribute('class')).toContain('overflow-hidden')
      expect(descendants(view.container).find((node) => node.tagName === 'P' && textOf(node) === 'What is the current risk?')?.getAttribute('class')).toContain('break-words')
      expect(descendants(view.container).find((node) => node.tagName === 'P' && textOf(node) === 'The validation gate is pending.')?.getAttribute('class')).toContain('break-words')
      expect(descendants(view.container).find((node) => node.tagName === 'SPAN' && textOf(node) === 'goal-1')?.getAttribute('class')).toContain('break-all')
    } finally { view.cleanup() }
  })

  it('renders loading, retry, disabled, and exhausted states without hiding readable history', async () => {
    conversation.query = { data: undefined, isLoading: true, isError: false, refetch: vi.fn() }
    const view = await mount()
    try {
      expect(descendants(view.container).some((node) => node.getAttribute('aria-label') === 'Loading conversation')).toBe(true)
      expect((descendants(view.container).find((node) => node.tagName === 'TEXTAREA') as { getAttribute: (name: string) => string | null }).getAttribute('disabled')).not.toBeNull()
      await view.rerender(() => <GoalAnnouncerProvider key="goal-1"><ConversationPanel projectId="project-1" goalId="goal-1" /></GoalAnnouncerProvider>)
      conversation.query = { data: undefined, isLoading: false, isError: true, refetch: vi.fn() }
      await view.rerender(() => <GoalAnnouncerProvider key="goal-1"><ConversationPanel projectId="project-1" goalId="goal-1" /></GoalAnnouncerProvider>)
      expect(textOf(view.container)).toContain('Failed to load conversation. Retry to view messages.')
      expect(getButton(view.container, 'Retry').getAttribute('class')).toContain('min-h-11')
      await act(async () => getButton(view.container, 'Retry').click())
      expect(conversation.query.refetch).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('keeps history readable while disabled or exhausted', async () => {
    conversation.query.data = { ...history(), allowance: { enabled: false, limit: 0, used: 0, remaining: 0 } }
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Conversation is not enabled for this goal.')
      expect(textOf(view.container)).toContain('What is the current risk?')
      expect(textarea(view.container).getAttribute('disabled')).not.toBeNull()
      conversation.query.data = { ...history(), allowance: { enabled: true, limit: 10, used: 10, remaining: 0 } }
      await view.rerender()
      expect(textOf(view.container)).toContain('Allowance: 0 remaining of 10')
      expect(textOf(view.container)).toContain('Conversation allowance is exhausted. New messages are unavailable.')
      expect(textOf(view.container)).toContain('What is the current risk?')
    } finally { view.cleanup() }
  })

  it('hides the entire panel when allowance is disabled and there are no items', async () => {
    conversation.query.data = { ...history([]), allowance: { enabled: false, limit: 0, used: 0, remaining: 0 } }
    const view = await mount()
    try {
      const section = descendants(view.container).find((node) => node.tagName === 'SECTION' && node.getAttribute('aria-label') === 'Conversation')
      expect(section).toBeUndefined()
      expect(textOf(view.container)).not.toContain('CONVERSATION')
    } finally { view.cleanup() }
  })

  it('renders the exact empty state and wraps long ledger text without controls outside conversation', async () => {
    conversation.query.data = history([])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('No conversation messages yet. Send an advisory question to the orchestrator.')
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON').map(textOf).join(' ')).not.toMatch(/pause|resume|cancel|ask_human/i)
    } finally { view.cleanup() }
  })

  it('reports a completed response without an answer exactly', async () => {
    conversation.query.data = history([turn({ answer: null, finished_at: null })])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('No response recorded.')
    } finally { view.cleanup() }
  })

  it('uses the safe server error message for a failed submission', async () => {
    conversation.mutation = { isPending: false, isError: true, error: { detail: { message: 'Conversation service is unavailable.' } }, mutate: vi.fn() }
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Conversation service is unavailable.')
    } finally { view.cleanup() }
  })

  it('never exposes an arbitrary transport error message', async () => {
    conversation.mutation = { isPending: false, isError: true, error: new Error('internal socket address'), mutate: vi.fn() }
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Message was not sent. Check your connection and try again.')
      expect(textOf(view.container)).not.toContain('internal socket address')
    } finally { view.cleanup() }
  })

  it('sends nonblank content with a durable request id and clears it after an edit', async () => {
    const randomUUID = vi.fn().mockReturnValue('uuid-1')
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID }, configurable: true })
    const view = await mount()
    try {
      const textarea = descendants(view.container).find((node) => node.tagName === 'TEXTAREA') as HTMLTextAreaElement
      await act(async () => changeControl(textarea as never, 'Question'))
      await act(async () => getButton(view.container, 'Send message').click())
      expect(conversation.mutation.mutate).toHaveBeenCalledWith({ goalId: 'goal-1', clientRequestId: 'uuid-1', content: 'Question' }, expect.any(Object))
      await act(async () => getButton(view.container, 'Send message').click())
      expect(randomUUID).toHaveBeenCalledTimes(1)
      await act(async () => changeControl(textarea as never, 'Question changed'))
      await act(async () => getButton(view.container, 'Send message').click())
      expect(randomUUID).toHaveBeenCalledTimes(2)
    } finally { view.cleanup() }
  })

  it('retains failed IDs, dismisses error on edit, and resets draft/id after success', async () => {
    const randomUUID = vi.fn().mockReturnValueOnce('uuid-1').mockReturnValueOnce('uuid-2').mockReturnValueOnce('uuid-3')
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID }, configurable: true })
    const callbacks: Array<{ onSuccess?: () => void }> = []
    conversation.mutation.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => changeControl(textarea(view.container), 'Question'))
      await act(async () => getButton(view.container, 'Send message').click())
      conversation.mutation = { ...conversation.mutation, isError: true, error: { detail: { message: 'Safe failure.' } } }
      await view.rerender()
      expect(textOf(view.container)).toContain('Safe failure.')
      // The inline error paragraph no longer carries its own role="alert" — it is
      // still focused, and the message reaches the shared assertive region instead.
      expect(view.activeElement?.tagName).toBe('P')
      expect(textOf(view.activeElement!)).toBe('Safe failure.')
      const assertiveAnnouncements = descendants(view.container).filter((node) => node.getAttribute('role') === 'alert' && node.getAttribute('aria-live') === 'assertive')
      expect(assertiveAnnouncements.some((node) => textOf(node) === 'Safe failure.')).toBe(true)
      await act(async () => getButton(view.container, 'Send message').click())
      expect(conversation.mutation.mutate).toHaveBeenLastCalledWith({ goalId: 'goal-1', clientRequestId: 'uuid-1', content: 'Question' }, expect.any(Object))
      await act(async () => changeControl(textarea(view.container), 'Changed'))
      // The dismissable inline error paragraph is gone; the sr-only shared
      // assertive region may still hold the last-announced text (by design —
      // it's only overwritten by the next announcement), so check the visible,
      // focusable error paragraph specifically rather than the whole container.
      expect(descendants(view.container).some((node) => node.tagName === 'P' && node.getAttribute('tabindex') === '-1' && textOf(node) === 'Safe failure.')).toBe(false)
      await act(async () => getButton(view.container, 'Send message').click())
      expect(conversation.mutation.mutate).toHaveBeenLastCalledWith({ goalId: 'goal-1', clientRequestId: 'uuid-2', content: 'Changed' }, expect.any(Object))
      await act(async () => callbacks.at(-1)?.onSuccess?.())
      expect(textarea(view.container).value).toBe('')
      expect(view.activeElement).toBe(textarea(view.container))
      expect(textOf(view.container)).toContain('Message sent.')
      await act(async () => changeControl(textarea(view.container), 'Fresh request'))
      await act(async () => getButton(view.container, 'Send message').click())
      expect(conversation.mutation.mutate).toHaveBeenLastCalledWith({ goalId: 'goal-1', clientRequestId: 'uuid-3', content: 'Fresh request' }, expect.any(Object))
    } finally { view.cleanup() }
  })

  it('handles Enter, Shift+Enter, blank drafts, and local pending without blocking historical activity', async () => {
    const randomUUID = vi.fn().mockReturnValue('uuid-key')
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID }, configurable: true })
    conversation.query.data = history([turn({ status: 'running', answer: null, finished_at: null })])
    const view = await mount()
    try {
      await act(async () => changeControl(textarea(view.container), ''))
      await act(async () => pressKey(textarea(view.container), 'Enter'))
      expect(conversation.mutation.mutate).not.toHaveBeenCalled()
      await act(async () => changeControl(textarea(view.container), 'Question'))
      const shiftEnter = await act(async () => pressKey(textarea(view.container), 'Enter', true))
      expect(conversation.mutation.mutate).not.toHaveBeenCalled()
      expect(shiftEnter.defaultPrevented).toBe(false)
      await act(async () => changeControl(textarea(view.container), 'Question\nnext line'))
      expect(textarea(view.container).value).toBe('Question\nnext line')
      await act(async () => pressKey(textarea(view.container), 'Enter'))
      expect(conversation.mutation.mutate).toHaveBeenCalledTimes(1)
      conversation.mutation = { ...conversation.mutation, isPending: true }
      await view.rerender()
      expect(textarea(view.container).getAttribute('disabled')).not.toBeNull()
      expect(textOf(view.container)).toContain('Sending…')
    } finally { view.cleanup() }
  })

  it('keeps a proposal advisory until Submit, retries it with the same id, and discards it safely', async () => {
    const proposal = {
      proposal_id: 'proposal-1', response_id: 'response-1', status: 'proposed' as const,
      directive: 'Prioritize validation.', target_type: 'goal' as const, target_id: 'goal-1',
      scope: 'run' as const, lifetime: 'remaining_current_run' as const, impact_summary: 'Reduce release risk.',
      dismissed_at: null, promoted_request_id: null, created_at: '2026-09-16T10:00:00Z', updated_at: '2026-09-16T10:00:00Z',
    }
    conversation.query.data = history([turn({ proposed_steering: proposal })])
    Object.defineProperty(globalThis, 'crypto', { value: { randomUUID: vi.fn().mockReturnValue('proposal-request-id') }, configurable: true })
    const callbacks: Array<{ onError?: () => void }> = []
    conversation.submitSteering.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Review & apply').click())
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'dialog')).toBe(true)
      expect(textOf(view.container)).toContain('Review and apply steering')
      expect(descendants(view.container).filter((node) => node.tagName === 'TEXTAREA')[1].value).toBe('Prioritize validation.')
      expect(conversation.submitSteering.mutate).not.toHaveBeenCalled()
      expect(view.activeElement).toBe(descendants(view.container).filter((node) => node.tagName === 'TEXTAREA')[1])
      await act(async () => getButton(view.container, 'Submit steering').click())
      const payload = {
        goalId: 'goal-1', clientRequestId: 'proposal-request-id', directive: 'Prioritize validation.', targetType: 'goal', targetId: 'goal-1',
        scope: 'run', lifetime: 'remaining_current_run', impactSummary: 'Reduce release risk.', sourceProposalId: 'proposal-1', supersedesRequestId: null,
      }
      expect(conversation.submitSteering.mutate).toHaveBeenCalledWith(payload, expect.any(Object))
      conversation.submitSteering = { ...conversation.submitSteering, isError: true, error: new ApiError(503, 'Submit failed.') }
      await view.rerender()
      await act(async () => getButton(view.container, 'Retry').click())
      expect(conversation.submitSteering.mutate).toHaveBeenLastCalledWith(payload, expect.any(Object))
      expect(conversation.submitSteering.mutate).toHaveBeenCalledTimes(2)
      await act(async () => callbacks[1]?.onError?.())
      await act(async () => getButton(view.container, 'Discard').click())
      expect(conversation.submitSteering.reset).toHaveBeenCalledTimes(1)
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'dialog')).toBe(false)
    } finally { view.cleanup() }
  })

  it('renders feedback controls only for an explicitly eligible unrecorded answer', async () => {
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    const eligible = await mount()
    try {
      expect(textOf(eligible.container)).toContain('Was this answer useful?')
      expect(buttonIn(eligible.container, 'Helpful')).toBeDefined()
      expect(buttonIn(eligible.container, 'Needs work')).toBeDefined()
    } finally { eligible.cleanup() }

    for (const ineligible of [
      turn({ actor_id: 'another-actor', feedback: null, feedback_eligible: false }),
      turn({ status: 'running', answer: null, feedback: null, feedback_eligible: false }),
      turn({ status: 'failed', answer: null, feedback: null, feedback_eligible: false }),
      turn({ status: 'interrupted_unknown', answer: null, feedback: null, feedback_eligible: false }),
      turn({ answer: '', feedback: null, feedback_eligible: false }),
    ]) {
      conversation.query.data = history([ineligible])
      const view = await mount()
      try {
        expect(textOf(view.container)).not.toContain('Was this answer useful?')
        expect(feedbackSection(view.container, 1)).toBeUndefined()
      } finally { view.cleanup() }
    }
  })

  it.each([
    [{ rating: 'helpful', reason: null }, 'Feedback recorded: Helpful'],
    [{ rating: 'not_helpful', reason: 'unanswered' }, 'Feedback recorded: Needs work — Did not answer'],
    [{ rating: 'not_helpful', reason: 'incorrect' }, 'Feedback recorded: Needs work — Incorrect'],
    [{ rating: 'not_helpful', reason: 'missing_context' }, 'Feedback recorded: Needs work — Missing context'],
    [{ rating: 'not_helpful', reason: 'stale_context' }, 'Feedback recorded: Needs work — Stale context'],
    [{ rating: 'not_helpful', reason: 'unclear' }, 'Feedback recorded: Needs work — Unclear'],
    [{ rating: 'not_helpful', reason: 'too_limited' }, 'Feedback recorded: Needs work — Too limited'],
    [{ rating: 'not_helpful', reason: 'other' }, 'Feedback recorded: Needs work — Other'],
  ] as const)('renders persisted feedback quietly as %s', async (feedback, copy) => {
    conversation.query.data = history([turn({ feedback: { feedback_id: 'feedback-1', ...feedback, created_at: '2026-09-16T10:00:00Z' }, feedback_eligible: true })])
    const view = await mount()
    try {
      const section = feedbackSection(view.container, 1)
      expect(section).toBeDefined()
      if (!section) return
      expect(textOf(section)).toBe(copy)
      expect(buttonIn(section, 'Helpful')).toBeUndefined()
      expect(buttonIn(section, 'Needs work')).toBeUndefined()
      expect(descendants(section).filter((node) => node.getAttribute('role') || node.getAttribute('aria-live') || node.getAttribute('aria-atomic'))).toHaveLength(0)
      const persistedFeedback = descendants(section).find((node) => node.tagName === 'P' && textOf(node) === copy)
      expect(persistedFeedback).toBeDefined()
      expect(persistedFeedback?.getAttribute('role')).toBeNull()
      expect(persistedFeedback?.getAttribute('aria-live')).toBeNull()
      expect(persistedFeedback?.getAttribute('aria-atomic')).toBeNull()
      expect(section.getAttribute('class')).toContain('break-words')
    } finally { view.cleanup() }
  })

  it('records Helpful immediately with the required null reason', async () => {
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Helpful').click())
      expect(conversation.feedback.mutate).toHaveBeenCalledWith({
        goalId: 'goal-1', responseId: 'response-1', rating: 'helpful', reason: null,
      }, expect.any(Object))
    } finally { view.cleanup() }
  })

  it('uses a labelled native reason group and maps every reason to its fixed payload', async () => {
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    const view = await mount()
    try {
      const needsWork = getButton(view.container, 'Needs work')
      expect(needsWork.getAttribute('type')).toBe('button')
      await act(async () => needsWork.click())
      const section = feedbackSection(view.container, 1)
      expect(section).toBeDefined()
      if (!section) return
      const fieldset = descendants(section).find((node) => node.tagName === 'FIELDSET')
      expect(fieldset).toBeDefined()
      if (!fieldset) return
      expect(textOf(fieldset)).toContain('What needs work?')
      expect(fieldset.getAttribute('class')).toContain('min-w-0')
      expect(buttonIn(section, 'Record feedback')).toBeUndefined()
      expect(descendants(section).filter((node) => node.tagName === 'TEXTAREA')).toHaveLength(0)
      expect(descendants(section).filter((node) => node.tagName === 'INPUT' && node.getAttribute('type') === 'text')).toHaveLength(0)
      const radios = descendants(fieldset).filter((node) => node.tagName === 'INPUT' && node.getAttribute('type') === 'radio')
      expect(radios).toHaveLength(7)

      for (const [label, reason] of feedbackReasons) {
        const radio = radios.find((node) => node.getAttribute('value') === reason)
        const labelNode = descendants(fieldset).find((node) => node.tagName === 'LABEL' && textOf(node) === label)
        expect(radio).toBeDefined()
        expect(labelNode).toBeDefined()
        if (!radio || !labelNode) continue
        expect(labelNode.htmlFor).toBe(radio.id)
        expect(labelNode.getAttribute('class')).toContain('min-h-11')
        expect(labelNode.getAttribute('class')).toContain('min-w-0')
        expect(labelNode.getAttribute('class')).toContain('break-words')
      }
    } finally { view.cleanup() }
  })

  it.each(feedbackReasons)('records %s as its fixed %s payload', async (label, reason) => {
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Needs work').click())
      const section = feedbackSection(view.container, 1)!
      const radio = descendants(section).find((node) => node.tagName === 'INPUT' && node.getAttribute('value') === reason)!
      await act(async () => changeControl(radio, true))
      expect(radio.checked).toBe(true)
      const record = getButton(section, 'Record feedback')
      expect(record.getAttribute('type')).toBe('button')
      await act(async () => record.click())
      expect(conversation.feedback.mutate).toHaveBeenCalledWith({
        goalId: 'goal-1', responseId: 'response-1', rating: 'not_helpful', reason,
      }, expect.any(Object))
    } finally { view.cleanup() }
  })

  it('keeps the selected native radio visible and disables feedback controls while recording', async () => {
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Needs work').click())
      const section = feedbackSection(view.container, 1)
      if (!section) return
      const radio = descendants(section).find((node) => node.tagName === 'INPUT' && node.getAttribute('value') === 'missing_context')!
      await act(async () => changeControl(radio, true))
      conversation.feedback = { ...conversation.feedback, isPending: true }
      await view.rerender()
      expect(radio.checked).toBe(true)
      expect(radio.getAttribute('disabled')).not.toBeNull()
      expect(getButton(section, 'Recording…').getAttribute('disabled')).not.toBeNull()
      expect(descendants(section).filter((node) => ['BUTTON', 'INPUT'].includes(node.tagName)).every((node) => node.getAttribute('disabled') !== null)).toBe(true)
      expect(descendants(section).find((node) => node.tagName === 'DIV' && (node.getAttribute('class') ?? '').includes('flex-wrap'))?.getAttribute('class')).toContain('min-w-0')
    } finally { view.cleanup() }
  })

  it('retains a failed Needs-work selection under its own response, focuses one safe alert, and retries unchanged', async () => {
    const second = turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, feedback: null, feedback_eligible: true })
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true }), second])
    const view = await mount()
    try {
      const first = feedbackSection(view.container, 1)!
      await act(async () => getButton(first, 'Needs work').click())
      const firstRadio = descendants(first).find((node) => node.tagName === 'INPUT' && node.getAttribute('value') === 'missing_context')!
      await act(async () => changeControl(firstRadio, true))
      await act(async () => getButton(first, 'Record feedback').click())
      conversation.feedback = { ...conversation.feedback, isError: true, error: new ApiError(400, 'Feedback was not recorded. Check your connection and try again.') }
      await view.rerender()
      // ErrorRecord drops its own role="alert" inside GoalAnnouncerProvider —
      // the message is announced through the single shared assertive region instead.
      expect(descendants(first).filter((node) => node.getAttribute('role') === 'alert')).toHaveLength(0)
      const assertiveAnnouncements = descendants(view.container).filter((node) => node.getAttribute('role') === 'alert' && node.getAttribute('aria-live') === 'assertive')
      expect(assertiveAnnouncements.some((node) => textOf(node) === 'Feedback was not recorded. Check your connection and try again.')).toBe(true)
      const focusTarget = descendants(first).find((node) => node.getAttribute('tabindex') === '-1')!
      expect(view.activeElement).toBe(focusTarget)
      expect(firstRadio.checked).toBe(true)
      expect(descendants(feedbackSection(view.container, 2)!).filter((node) => node.getAttribute('role') === 'alert')).toHaveLength(0)
      await act(async () => getButton(first, 'Record feedback').click())
      expect(conversation.feedback.mutate).toHaveBeenLastCalledWith({
        goalId: 'goal-1', responseId: 'response-1', rating: 'not_helpful', reason: 'missing_context',
      }, expect.any(Object))
    } finally { view.cleanup() }
  })

  it('keeps only one active Needs-work draft and preserves keyboard-native button/radio semantics', async () => {
    const second = turn({ message_id: 'message-2', response_id: 'response-2', sequence: 2, feedback: null, feedback_eligible: true })
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true }), second])
    const view = await mount()
    try {
      const first = feedbackSection(view.container, 1)!
      const secondSection = feedbackSection(view.container, 2)!
      await act(async () => getButton(first, 'Needs work').click())
      await act(async () => changeControl(descendants(first).find((node) => node.tagName === 'INPUT' && node.getAttribute('value') === 'incorrect')!, true))
      await act(async () => getButton(secondSection, 'Needs work').click())
      expect(descendants(first).filter((node) => node.tagName === 'FIELDSET')).toHaveLength(0)
      const secondRadios = descendants(secondSection).filter((node) => node.tagName === 'INPUT' && node.getAttribute('type') === 'radio')
      expect(secondRadios).toHaveLength(7)
      expect(secondRadios.every((node) => node.getAttribute('role') === null)).toBe(true)
      expect(buttonIn(first, 'Helpful')?.getAttribute('type')).toBe('button')
      expect(buttonIn(secondSection, 'Needs work')?.getAttribute('type')).toBe('button')
      await act(async () => changeControl(secondRadios[0], true))
      expect(secondRadios[0].checked).toBe(true)
      expect(buttonIn(secondSection, 'Record feedback')).toBeDefined()
    } finally { view.cleanup() }
  })

  it('keeps one success announcement through the query refresh while persisted feedback stays quiet', async () => {
    const callbacks: Array<{ onSuccess?: () => void }> = []
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    conversation.feedback.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Helpful').click())
      await act(async () => callbacks[0]?.onSuccess?.())
      const announcements = descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.')
      expect(announcements).toHaveLength(1)
      expect(textOf(announcements[0])).toBe('Feedback recorded.')
      expect(announcements[0].getAttribute('aria-live')).toBe('polite')
      expect(announcements[0].getAttribute('aria-atomic')).toBe('true')

      conversation.query.data = history([turn({
        feedback: { feedback_id: 'feedback-1', rating: 'helpful', reason: null, created_at: '2026-09-16T10:00:00Z' },
        feedback_eligible: true,
      })])
      await view.rerender()
      const persistedFeedback = descendants(feedbackSection(view.container, 1)!).find((node) => node.tagName === 'P' && textOf(node) === 'Feedback recorded: Helpful')
      expect(persistedFeedback).toBeDefined()
      expect(descendants(persistedFeedback!).filter((node) => node.getAttribute('role') || node.getAttribute('aria-live') || node.getAttribute('aria-atomic'))).toHaveLength(0)
      expect(persistedFeedback?.getAttribute('role')).toBeNull()
      expect(persistedFeedback?.getAttribute('aria-live')).toBeNull()
      expect(persistedFeedback?.getAttribute('aria-atomic')).toBeNull()
      const announcementsAfterRefresh = descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.')
      expect(announcementsAfterRefresh).toHaveLength(1)
      expect(announcementsAfterRefresh[0].getAttribute('aria-live')).toBe('polite')
      expect(announcementsAfterRefresh[0].getAttribute('aria-atomic')).toBe('true')
    } finally { view.cleanup() }
  })

  it('keeps feedback confirmation mounted through query errors and recovery', async () => {
    const callbacks: Array<{ onSuccess?: () => void }> = []
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    conversation.feedback.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Helpful').click())
      await act(async () => callbacks[0]?.onSuccess?.())
      conversation.query = { data: undefined, isLoading: false, isError: true, refetch: vi.fn() }
      await view.rerender()
      const announcementDuringError = descendants(view.container).find((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.')
      expect(announcementDuringError).toBeDefined()

      conversation.query = { data: history([turn({ feedback: null, feedback_eligible: true })]), isLoading: false, isError: false, refetch: vi.fn() }
      await view.rerender()
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.')).toHaveLength(1)
    } finally { view.cleanup() }
  })

  it('ignores a late feedback success from a previous goal', async () => {
    const callbacks: Array<{ onSuccess?: () => void }> = []
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    conversation.feedback.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Helpful').click())
      await view.rerender(() => <GoalAnnouncerProvider key="goal-2"><ConversationPanel projectId="project-1" goalId="goal-2" /></GoalAnnouncerProvider>)
      await act(async () => callbacks[0]?.onSuccess?.())
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.')).toHaveLength(0)
    } finally { view.cleanup() }
  })

  it('does not carry a recorded feedback announcement into another goal', async () => {
    const callbacks: Array<{ onSuccess?: () => void }> = []
    conversation.query.data = history([turn({ feedback: null, feedback_eligible: true })])
    conversation.feedback.mutate = vi.fn((_input, options) => callbacks.push(options))
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Helpful').click())
      await act(async () => callbacks[0]?.onSuccess?.())
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.').length).toBe(1)
      await view.rerender(() => <GoalAnnouncerProvider key="goal-2"><ConversationPanel projectId="project-1" goalId="goal-2" /></GoalAnnouncerProvider>)
      expect(descendants(view.container).filter((node) => node.getAttribute('role') === 'status' && textOf(node) === 'Feedback recorded.').length).toBe(0)
    } finally { view.cleanup() }
  })
})
