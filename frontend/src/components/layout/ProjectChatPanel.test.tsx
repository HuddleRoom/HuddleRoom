import React from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { changeControl, descendants, getButton, mountWithTestDom, textOf, TestEvent } from '../../../tests/support/dom'
import type { ProjectChatAction, ProjectChatHistory, ProjectChatTurn, OrchestrationConversationTurn } from '@/lib/types'
import { ApiError } from '@/lib/api-client'

const state = vi.hoisted(() => ({
  query: { data: undefined as ProjectChatHistory | undefined, isLoading: false, isError: false, error: null as unknown, refetch: vi.fn() },
  mutation: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
  feedback: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
  contextGoal: 'Alpha',
}))
vi.mock('@/api/orchestration', () => ({
  useProjectChat: () => state.query,
  useSubmitProjectChat: () => state.mutation,
  useRecordOrchestrationConversationFeedback: () => state.feedback,
  useOrchestrationGoal: () => ({ data: { goal: { objective: state.contextGoal } } }),
}))
vi.mock('react-router-dom', () => ({ Link: ({ to, children }: { to: string; children: React.ReactNode }) => <a href={to}>{children}</a> }))
vi.mock('@/components/common/Dialog', () => ({
  Dialog: ({ open, title, role, children, footer, onOpenChange, onCloseAutoFocus }: {
    open: boolean; title: string; role?: string; children: React.ReactNode
    footer?: { primaryLabel: string; onPrimary: () => void; isPending?: boolean; primaryDisabled?: boolean }
    onOpenChange: (open: boolean) => void; onCloseAutoFocus?: (event: Event) => void
  }) => open ? <section role={role ?? 'dialog'} aria-label={title}><h2>{title}</h2>{children}
    <button onClick={() => { onOpenChange(false); onCloseAutoFocus?.(new Event('close')) }}>Close review</button>
    {footer && <button disabled={footer.isPending || footer.primaryDisabled} onClick={footer.onPrimary}>{footer.primaryLabel}</button>}
  </section> : null,
}))
import { ProjectChatPanel } from './ProjectChatPanel'

const action = (overrides: Partial<ProjectChatAction> = {}): ProjectChatAction => ({
  action_id: 'action-alpha', operation: 'pause', project_id: 'project-1', goal_id: 'goal-alpha', run_id: 'run-alpha',
  target_id: null, frozen_versions: { state: 'original' }, payload: {}, status: 'awaiting_confirmation', requires_confirmation: true,
  summary: 'Pauses Alpha.\n' + JSON.stringify({ contract: { objective: 'Alpha', goal_type: 'outcome', budget: { max_tokens: 100 } } }),
  ...overrides,
})
const turn = (overrides: Partial<ProjectChatTurn> = {}): ProjectChatTurn => ({
  id: 'turn-1', content: 'Pause this goal', answer: 'Review the action', status: 'completed', created_at: '2026-10-13T10:00:00Z',
  context_goal_id: 'goal-alpha', historical: false, source_references: [], citations: [], error: null,
  goal_turn: null, actions: [], receipts: [], ...overrides,
})
const history = (items: ProjectChatTurn[] = []): ProjectChatHistory => ({
  items, total: items.length, omitted: 0,
  allowance: { enabled: true, unlimited: false, limit: 1000, remaining: 900 },
  goal_allowances: { 'goal-alpha': { enabled: true, unlimited: false, limit: 1000, used: 100, remaining: 900 },
    'goal-beta': { enabled: true, unlimited: false, limit: 1000, used: 100, remaining: 900 } }, steering: {},
})
beforeEach(() => {
  state.query = { data: history(), isLoading: false, isError: false, error: null, refetch: vi.fn() }
  state.mutation = { isPending: false, isError: false, error: null, mutate: vi.fn() }
  state.feedback = { isPending: false, isError: false, error: null, mutate: vi.fn() }
  state.contextGoal = 'Alpha'
})
async function mount(goal: string | null = 'goal-alpha') {
  return mountWithTestDom(() => <ProjectChatPanel projectId="project-1" contextGoalId={goal} />, React.act)
}
const textarea = (container: Parameters<typeof descendants>[0]) => descendants(container).find((node) => node.tagName === 'TEXTAREA')!

describe('ProjectChatPanel', () => {
  it('renders one labeled composer and persistent history with context and historical links', async () => {
    state.query.data = history([turn({ historical: true, content: 'Old question', answer: 'Old answer',
      source_references: [{ kind: 'goal_conversation', id: 'response-original', goal_id: 'goal-alpha' }],
      citations: [{ type: 'goal', id: 'goal-alpha', label: 'Alpha source' }] })])
    const view = await mount()
    try {
      expect(descendants(view.container).filter((node) => node.tagName === 'TEXTAREA')).toHaveLength(1)
      expect(textOf(view.container)).toContain('Context: Alpha')
      expect(textOf(view.container)).not.toContain('Charged pool:')
      expect(textOf(view.container)).toContain('Human (historical)Old questionOrchestratorOld answer')
      expect(textOf(view.container)).toContain('Historical')
      expect(descendants(view.container).find((node) => node.tagName === 'A' && textOf(node) === 'Alpha source')?.getAttribute('href')).toBe('/orchestration/goal-alpha')
      expect(descendants(view.container).find((node) => node.getAttribute('aria-live') === 'polite')).toBeDefined()
    } finally { view.cleanup() }
  })

  it('submits text with a stable request ID after retry and frozen send context', async () => {
    const view = await mount()
    try {
      await React.act(async () => changeControl(textarea(view.container), 'Prioritize verification'))
      await React.act(async () => getButton(view.container, 'Send').click())
      const first = state.mutation.mutate.mock.calls[0][0]
      expect(first).toMatchObject({ content: 'Prioritize verification', contextGoalId: 'goal-alpha' })
      expect(first.clientRequestId).toMatch(/^[0-9a-f-]{36}$/)
      await React.act(async () => getButton(view.container, 'Send').click())
      expect(state.mutation.mutate.mock.calls[1][0].clientRequestId).toBe(first.clientRequestId)
      state.contextGoal = 'Beta'
      await view.rerender(() => <ProjectChatPanel projectId="project-1" contextGoalId="goal-beta" />)
      await React.act(async () => getButton(view.container, 'Send').click())
      expect(state.mutation.mutate.mock.calls[2][0].contextGoalId).toBe('goal-beta')
      expect(state.mutation.mutate.mock.calls[2][0].clientRequestId).not.toBe(first.clientRequestId)
    } finally { view.cleanup() }
  })

  it.each([false, true])('keeps history and no-model confirmations usable when pool unavailable (%s)', async (disabled) => {
    state.query.data = history([turn({ actions: [action()] })])
    state.query.data.goal_allowances['goal-alpha'] = { enabled: !disabled, unlimited: false, limit: disabled ? 0 : 1000, used: 1000, remaining: 0 }
    const view = await mount()
    try {
      expect(textarea(view.container).getAttribute('disabled')).toBeNull()
      await React.act(async () => getButton(view.container, 'Review pause').click())
      await React.act(async () => getButton(view.container, 'Confirm').click())
      expect(state.mutation.mutate).toHaveBeenCalledWith(expect.objectContaining({ content: 'yes',
        contextGoalId: 'goal-alpha', actionId: 'action-alpha' }), expect.any(Object))
      expect(textOf(view.container)).not.toContain('History and prepared confirmations remain available')
    } finally { view.cleanup() }
  })

  it('navigation does not alter an open review target and close restores its trigger focus', async () => {
    state.query.data = history([turn({ actions: [action()] })])
    const view = await mount()
    try {
      const trigger = getButton(view.container, 'Review pause')
      await React.act(async () => trigger.click())
      state.contextGoal = 'Beta'
      await view.rerender(() => <ProjectChatPanel projectId="project-1" contextGoalId="goal-beta" />)
      expect(textOf(view.container)).toContain('Context: Beta')
      const dialog = descendants(view.container).find((node) => node.getAttribute('role') === 'alertdialog')!
      expect(textOf(dialog)).toContain('Goal: Alpha (goal-alpha) · Run: run-alpha')
      expect(textOf(dialog)).not.toContain('Beta')
      await React.act(async () => getButton(view.container, 'Confirm').click())
      expect(state.mutation.mutate.mock.calls[0][0]).toMatchObject({ actionId: 'action-alpha', contextGoalId: 'goal-alpha' })
      await React.act(async () => getButton(view.container, 'Close review').click())
      expect(view.activeElement).toBe(trigger)
    } finally { view.cleanup() }
  })

  it('requires fresh explicit acceptance of the new action ID after stale review refresh', async () => {
    state.query.data = history([turn({ actions: [action()] })])
    const view = await mount()
    try {
      await React.act(async () => getButton(view.container, 'Review pause').click())
      await React.act(async () => getButton(view.container, 'Confirm').click())
      const callback = state.mutation.mutate.mock.calls[0][1].onSuccess
      const fresh = action({ action_id: 'action-refreshed', summary: 'New limits require review.\n{"contract":{"objective":"Alpha"}}' })
      state.query.data = history([turn({ actions: [fresh] })])
      await React.act(async () => callback(turn({ actions: [fresh], receipts: [{ action_id: fresh.action_id, operation: 'pause',
        project_id: 'project-1', goal_id: 'goal-alpha', run_id: 'run-alpha', status: 'pending', message: 'Review changed', error_code: 'chat_stale_review' }] })))
      await view.rerender()
      expect(textOf(view.container)).toContain('Review changed. Check the refreshed target and limits')
      expect(descendants(view.container).find((node) => node.getAttribute('role') === 'alertdialog')).toBeDefined()
      expect(state.mutation.mutate).toHaveBeenCalledTimes(1)
      await React.act(async () => getButton(view.container, 'Confirm').click())
      expect(state.mutation.mutate.mock.calls[1][0].actionId).toBe('action-refreshed')
    } finally { view.cleanup() }
  })

  it('does not bind bare yes across multiple pending reviews or require confirmation for recorded Resume', async () => {
    state.query.data = history([turn({ actions: [action(), action({ action_id: 'cancel-alpha', operation: 'cancel' })],
      receipts: [{ action_id: 'resume-alpha', operation: 'resume', project_id: 'project-1', goal_id: 'goal-alpha',
        run_id: 'run-alpha', status: 'completed', message: 'Resumed', error_code: null }] })])
    const view = await mount()
    try {
      await React.act(async () => changeControl(textarea(view.container), 'yes'))
      await React.act(async () => getButton(view.container, 'Send').click())
      expect(state.mutation.mutate.mock.calls[0][0].actionId).toBeUndefined()
      expect(textOf(view.container)).toContain('resume · completed: Resumed')
      expect(() => getButton(view.container, 'Review resume')).toThrow()
    } finally { view.cleanup() }
  })

  it.each(['running', 'completed', 'failed'])('renders investigation %s and feedback only when eligible', async (status) => {
    const goalTurn = { response_id: 'response-1', feedback_eligible: true, feedback: null,
      investigation: { status, objective: 'Inspect release', sources: [],
        report: status === 'completed' ? { findings: 'Gate verified', uncertainty: '' } : null,
        error: status === 'failed' ? { code: 'invalid_investigation_report' } : null } } as unknown as OrchestrationConversationTurn
    state.query.data = history([turn({ goal_turn: goalTurn })])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain(`Investigation: ${status}`)
      if (status === 'completed') expect(textOf(view.container)).toContain('Findings: Gate verified')
      if (status === 'failed') expect(textOf(view.container)).toContain('invalid_investigation_report')
      await React.act(async () => getButton(view.container, 'Helpful').click())
      expect(state.feedback.mutate).toHaveBeenCalledWith({ goalId: 'goal-alpha', responseId: 'response-1', rating: 'helpful', reason: null })
      state.query.data = history([turn({ goal_turn: { ...goalTurn, feedback_eligible: false } })])
      await view.rerender()
      expect(() => getButton(view.container, 'Helpful')).toThrow()
    } finally { view.cleanup() }
  })

  it('shows provider failure without optimistic success and pending announcements', async () => {
    state.query.data = history([turn({ status: 'interrupted_unknown', answer: null, error: { message: 'Outcome unknown' } })])
    state.mutation.isPending = true
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Outcome unknown')
      expect(textOf(view.container)).toContain('Sending…')
      expect(getButton(view.container, 'Send').getAttribute('disabled')).not.toBeNull()
      state.mutation = { isPending: false, isError: true, error: new ApiError(503, 'Service unavailable'), mutate: vi.fn() }
      await view.rerender()
      expect(textOf(view.container)).toContain('Service unavailable')
    } finally { view.cleanup() }
  })

  it.each([true, false])('uses the frozen Start envelope and preserves authorization availability (%s)', async (available) => {
    const envelope = { contract: { objective: 'Frozen Alpha', goal_type: 'outcome', constraints: { owned_files: ['src/alpha'] },
      budget: { max_tokens: 100 }, authority_model: 'no_manager' }, authorization: available ? {
      allowed: true, required_decision: null, reason_code: null, authority_source: 'root_start', run_id: 'run-alpha',
      contract_fingerprint: 'frozen', model_spend_remaining: { max_tokens: '88' }, external_write_targets: [],
      live_send_channels: [], workspace_mutation: 'direct', tool_constraints: ['no_external_writes', 'no_live_sends'],
    } : null }
    state.query.data = history([turn({ actions: [action({ operation: 'start_execution', summary: 'Authorize execution.\n' + JSON.stringify(envelope) })] })])
    const view = await mount()
    try {
      await React.act(async () => getButton(view.container, 'Review start execution').click())
      state.contextGoal = 'Beta'
      await view.rerender(() => <ProjectChatPanel projectId="project-1" contextGoalId="goal-beta" />)
      const dialog = descendants(view.container).find((node) => node.getAttribute('role') === 'alertdialog')!
      expect(textOf(dialog)).toContain('Frozen Alpha')
      expect(textOf(dialog)).toContain('run-alpha')
      expect(textOf(dialog)).not.toContain('Beta')
      expect(textOf(dialog)).toContain('max tokens: 100')
      expect(getButton(view.container, 'Start work').getAttribute('disabled') !== null).toBe(!available)
      if (available) {
        for (const label of ['Model spend left', 'External writes', 'Live sends', 'Workspace', 'Tool limits']) expect(textOf(dialog)).toContain(label)
        await React.act(async () => getButton(view.container, 'Start work').click())
        expect(state.mutation.mutate.mock.calls[0][0]).toMatchObject({ contextGoalId: 'goal-alpha', actionId: 'action-alpha', content: 'yes' })
      } else expect(textOf(dialog)).toContain('Authorization details are unavailable')
    } finally { view.cleanup() }
  })

  it('supports Ctrl/Cmd+Enter while normal Enter remains multiline', async () => {
    const view = await mount()
    try {
      const control = textarea(view.container)
      await React.act(async () => changeControl(control, 'Check the gate'))
      const propsKey = Object.keys(control).find((key) => key.startsWith('__reactProps$'))!
      const onKeyDown = (control as unknown as Record<string, { onKeyDown: (event: unknown) => void }>)[propsKey].onKeyDown
      await React.act(async () => onKeyDown(Object.assign(new TestEvent('keydown'), { key: 'Enter', ctrlKey: false, metaKey: false })))
      expect(state.mutation.mutate).not.toHaveBeenCalled()
      await React.act(async () => onKeyDown(Object.assign(new TestEvent('keydown'), { key: 'Enter', ctrlKey: true, metaKey: false })))
      expect(state.mutation.mutate).toHaveBeenCalledOnce()
      await React.act(async () => onKeyDown(Object.assign(new TestEvent('keydown'), { key: 'Enter', ctrlKey: false, metaKey: true })))
      expect(state.mutation.mutate).toHaveBeenCalledTimes(2)
    } finally { view.cleanup() }
  })
})


describe('SPR209 retired PROJECT allowance', () => {
  it.each([null, 'goal-alpha'])('submits with disabled exhausted legacy metadata in context %s', async (goal) => {
    state.query.data = history()
    state.query.data.allowance = { enabled: false, unlimited: false, limit: 0, remaining: 0 }
    state.query.data.goal_allowances['goal-alpha'] = { enabled: false, unlimited: false, limit: 0, used: 100, remaining: 0 }
    const view = await mount(goal)
    try {
      const control = textarea(view.container)
      expect(control.getAttribute('disabled')).toBeNull()
      expect(getButton(view.container, 'Send').getAttribute('disabled')).not.toBeNull()
      expect(textOf(view.container)).not.toMatch(/Charged pool|allowance is exhausted|remaining/i)
      await React.act(async () => changeControl(control, 'What should I do next?'))
      await React.act(async () => getButton(view.container, 'Send').click())
      expect(state.mutation.mutate).toHaveBeenCalledWith(expect.objectContaining({ content: 'What should I do next?', contextGoalId: goal }), expect.any(Object))
    } finally { view.cleanup() }
  })

  it('labels a historical limited investigation without active refill advice', async () => {
    const goalTurn = { investigation: { status: 'limited', objective: 'Inspect release', sources: [], report: null, error: null } } as unknown as OrchestrationConversationTurn
    state.query.data = history([turn({ historical: true, goal_turn: goalTurn })])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Historical investigation stopped at a retired conversation allowance limit. Send a new message to ask again.')
      expect(textOf(view.container)).not.toContain('if allowance remains')
    } finally { view.cleanup() }
  })
})


describe('SPR209 PROJECT readiness controls', () => {
  it.each(['loading', 'error', 'pending'])('blocks submission while %s', async (mode) => {
    if (mode === 'pending') state.mutation.isPending = true
    else state.query = { data: undefined, isLoading: mode === 'loading', isError: mode === 'error', error: null, refetch: vi.fn() }
    const view = await mount()
    try {
      expect(textarea(view.container).getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.container, 'Send').getAttribute('disabled')).not.toBeNull()
      if (mode === 'error') {
        await React.act(async () => getButton(view.container, 'Retry').click())
        expect(state.query.refetch).toHaveBeenCalledOnce()
      }
      expect(state.mutation.mutate).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })
})
