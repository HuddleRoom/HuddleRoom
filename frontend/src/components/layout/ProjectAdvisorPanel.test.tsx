import { describe, it, expect, beforeEach, vi } from 'vitest'
import React from 'react'
import { changeControl, descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type { ProjectAdvisorHistory, ProjectAdvisorTurn } from '@/lib/types'

const advisor = vi.hoisted(() => ({
  query: { data: undefined as ProjectAdvisorHistory | undefined, isLoading: false, isError: false, refetch: vi.fn() },
  mutation: { isPending: false, isError: false, error: null as unknown, mutate: vi.fn() },
}))

vi.mock('react-router-dom', () => ({
  Link: ({ children, to }: { children: React.ReactNode; to: string }) => <a href={to}>{children}</a>,
}))

vi.mock('@/api/orchestration', () => ({
  useProjectAdvisorConversation: () => advisor.query,
  useSubmitProjectAdvisorTurn: () => advisor.mutation,
}))

import { ProjectAdvisorPanel } from './ProjectAdvisorPanel'

const turn = (overrides: Partial<ProjectAdvisorTurn> = {}): ProjectAdvisorTurn => ({
  id: 'turn-1',
  question: 'What is the state of my project?',
  answer: 'Two goals are active; one is blocked on validation.',
  citations: [
    { type: 'goal', id: 'goal-1', label: 'Goal: Ship the release' },
    { type: 'decision', id: 'decision-1', goal_id: 'goal-1', label: 'Decision: Ship after review' },
    { type: 'meeting', id: 'meeting-1', label: 'Meeting: Release review' },
  ],
  off_topic: false,
  status: 'completed',
  created_at: '2026-09-18T08:00:00Z',
  ...overrides,
})

const history = (items: ProjectAdvisorTurn[] = [], allowance: Partial<ProjectAdvisorHistory['allowance']> = {}): ProjectAdvisorHistory => ({
  items,
  allowance: { enabled: true, unlimited: false, limit: 1000, remaining: 900, ...allowance },
})

beforeEach(() => {
  advisor.query = { data: history(), isLoading: false, isError: false, refetch: vi.fn() }
  advisor.mutation = { isPending: false, isError: false, error: null, mutate: vi.fn() }
})

async function mount() {
  return mountWithTestDom(() => <ProjectAdvisorPanel projectId="project-1" />, React.act)
}

function textarea(container: Parameters<typeof descendants>[0]) {
  return descendants(container).find((node) => node.tagName === 'TEXTAREA')!
}

describe('ProjectAdvisorPanel', () => {
  it('renders nothing when allowance is disabled', async () => {
    advisor.query.data = history([], { enabled: false, limit: 0, remaining: 0 })
    const view = await mount()
    try {
      expect(textOf(view.container)).toBe('')
    } finally { view.cleanup() }
  })

  it('renders the empty state when enabled with no turns', async () => {
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain("Ask the orchestrator about this project's goals, decisions, and recent activity.")
    } finally { view.cleanup() }
  })

  it("renders a turn's Question/Answer and citation chips with correct hrefs", async () => {
    advisor.query.data = history([turn()])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('What is the state of my project?')
      expect(textOf(view.container)).toContain('Two goals are active; one is blocked on validation.')
      const list = descendants(view.container).find((node) => node.tagName === 'OL' && node.getAttribute('aria-label') === 'Advisor conversation')
      expect(list).toBeDefined()
      const goalLink = descendants(view.container).find((node) => node.tagName === 'A' && node.getAttribute('href') === '/orchestration/goal-1')
      const decisionLink = descendants(view.container).find((node) => node.tagName === 'A' && node.getAttribute('href') === '/orchestration/goal-1#decision-decision-1')
      const meetingLink = descendants(view.container).find((node) => node.tagName === 'A' && node.getAttribute('href') === '/meetings/meeting-1')
      expect(goalLink).toBeDefined()
      expect(textOf(goalLink!)).toBe('Goal: Ship the release')
      expect(decisionLink).toBeDefined()
      expect(textOf(decisionLink!)).toBe('Decision: Ship after review')
      expect(meetingLink).toBeDefined()
      expect(textOf(meetingLink!)).toBe('Meeting: Release review')
    } finally { view.cleanup() }
  })

  it('keeps a historic decision citation without a goal id as a non-link chip', async () => {
    advisor.query.data = history([turn({ citations: [{ type: 'decision', id: 'decision-1', label: 'Decision: Historic' }] })])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Decision: Historic')
      expect(descendants(view.container).some((node) => node.tagName === 'A' && textOf(node) === 'Decision: Historic')).toBe(false)
    } finally { view.cleanup() }
  })

  it('renders only the deflection for an off-topic turn, with no citations', async () => {
    advisor.query.data = history([turn({
      off_topic: true,
      answer: null,
      citations: [{ type: 'goal', id: 'goal-1', label: 'Goal: Ship the release' }],
    })])
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain("That's outside what I can help with here — I can answer questions about this project's goals, decisions, and activity.")
      expect(descendants(view.container).filter((node) => node.tagName === 'A')).toHaveLength(0)
    } finally { view.cleanup() }
  })

  it('disables the composer and shows the exhausted message when remaining is 0', async () => {
    advisor.query.data = history([], { remaining: 0 })
    const view = await mount()
    try {
      expect(textarea(view.container).getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.container, 'Send question').getAttribute('disabled')).not.toBeNull()
      expect(textOf(view.container)).toContain('Advisor allowance is exhausted. New questions are unavailable.')
    } finally { view.cleanup() }
  })

  it('unlimited allowance: no counter line, composer enabled, never exhausted', async () => {
    advisor.query.data = history([], { unlimited: true, limit: -1, remaining: -1 })
    const view = await mount()
    try {
      expect(textOf(view.container)).not.toContain('Allowance:')
      expect(textOf(view.container)).not.toContain('Advisor allowance is exhausted')
      expect(textarea(view.container).getAttribute('disabled')).toBeNull()
      expect(getButton(view.container, 'Send question').getAttribute('disabled')).toBeNull()
    } finally { view.cleanup() }
  })

  it('renders the returned answer after a successful submit', async () => {
    const callbacks: Array<{ onSuccess?: () => void }> = []
    advisor.mutation.mutate = vi.fn((_content, options) => callbacks.push(options))
    const view = await mount()
    try {
      await React.act(async () => changeControl(textarea(view.container), 'What should I start next?'))
      await React.act(async () => getButton(view.container, 'Send question').click())
      expect(advisor.mutation.mutate).toHaveBeenCalledWith('What should I start next?', expect.any(Object))

      advisor.query.data = history([turn({ question: 'What should I start next?', answer: 'Start the validation goal.' })])
      await React.act(async () => callbacks.at(-1)?.onSuccess?.())
      await view.rerender()
      expect(textOf(view.container)).toContain('Start the validation goal.')
    } finally { view.cleanup() }
  })

  it('renders ErrorRecord on submit failure', async () => {
    advisor.mutation = { isPending: false, isError: true, error: new ApiError(400, 'Advisor service is unavailable.'), mutate: vi.fn() }
    const view = await mount()
    try {
      expect(textOf(view.container)).toContain('Advisor service is unavailable.')
      expect(descendants(view.container).some((node) => node.getAttribute('role') === 'alert')).toBe(true)
    } finally { view.cleanup() }
  })
})
