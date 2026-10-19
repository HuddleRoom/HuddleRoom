import React, { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { changeControl, descendants, getButton, getByLabel, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type { OrchestrationAuthorization, OrchestrationGoal } from '@/lib/types'

vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />,
  Content: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  Title: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => <h2 {...props}>{children}</h2>,
  Description: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props}>{children}</p>,
  Close: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

import { StartWorkDialog } from './StartWorkDialog'

const goal: OrchestrationGoal = {
  id: 'goal-1', project_id: 'project-1', objective: 'Ship it', success_criteria: [],
  orchestrator_context: {}, constraints: { owned_files: ['frontend/src'] }, budget: { max_tokens: 20000 },
  goal_type: 'outcome', supersedes_goal_id: null, status: 'active', weight: 'standard',
  weight_overridden_by: null, manager_agent_id: null, manager_user_id: null, authority_model: 'human_review',
  created_by_user_id: null, created_at: '2026-09-02T00:00:00Z', updated_at: '2026-09-02T00:00:00Z', needs_you_count: 0,
}

const allowed: OrchestrationAuthorization = {
  allowed: true, reason_code: null, required_decision: null, authority_source: 'root_start',
  run_id: 'run-1', contract_fingerprint: 'fp', model_spend_remaining: { tokens: '120000', usd: '4.20' },
  external_write_targets: [], live_send_channels: [], workspace_mutation: 'staged', tool_constraints: [],
}

function renderDialog(overrides: Partial<React.ComponentProps<typeof StartWorkDialog>> = {}) {
  return <StartWorkDialog open goal={goal} authorization={allowed} missingRecurringPolicy={false} isPending={false} error={null} onOpenChange={vi.fn()} onConfirm={vi.fn()} {...overrides} />
}

describe('StartWorkDialog', () => {
  it('makes the selected planning or implementation scope explicit before Start', async () => {
    const onConfirm = vi.fn()
    const view = await mountWithTestDom(() => renderDialog({ onConfirm, initialScope: 'planning_only' }), act)
    try {
      expect(textOf(view.document.body)).toContain('Plan only — review and verify the candidate. Implementation needs a later Start.')
      expect(getByLabel(view.document.body, 'Plan only').checked).toBe(true)
      const scopes = descendants(view.document.body).filter((node) => node.tagName === 'INPUT' && node.name === 'start-scope')
      await act(async () => changeControl(scopes[1], true))
      await act(async () => getButton(view.document.body, 'Start work').click())
      expect(onConfirm).toHaveBeenCalledWith('execution')
    } finally {
      view.cleanup()
    }
  })

  it('shows the goal start summary and confirms once', async () => {
    const onConfirm = vi.fn()
    const view = await mountWithTestDom(() => renderDialog({ onConfirm }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('Goal type')
      expect(text).toContain('Outcome')
      expect(text).toContain('Authority')
      expect(text).toContain('Human Review')
      expect(text).toContain('Scope')
      expect(text).toContain('frontend/src')
      expect(text).toContain('Budget')
      expect(text).toContain('max tokens: 20000')
      await act(async () => getButton(view.document.body, 'Start work').click())
      expect(onConfirm).toHaveBeenCalledTimes(1)
    } finally {
      view.cleanup()
    }
  })

  it('renders nested scope and budget payloads without hiding or coercing them', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      goal: {
        ...goal,
        constraints: { owned_files: ['frontend/src'], delivery: { environment: 'staging', checks: ['lint', 'test'] } },
        budget: { tokens: { max: 20000, warning_at: 16000 } },
      },
    }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('{"owned_files":["frontend/src"],"delivery":{"environment":"staging","checks":["lint","test"]}}')
      expect(text).toContain('tokens: {"max":20000,"warning_at":16000}')
      expect(text).not.toContain('[object Object]')
      expect(text).not.toContain('No explicit scope')
    } finally {
      view.cleanup()
    }
  })

  it('uses an explicit fallback for empty scope and budget', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      goal: { ...goal, constraints: {}, budget: {} },
    }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('No explicit scope')
      expect(text).toContain('No explicit budget')
      expect(text).not.toContain('Scope{}')
    } finally {
      view.cleanup()
    }
  })

  it('uses an assertive alert for a typed conflict', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      error: new ApiError(409, 'HTTP 409', { conflict: 'already_started' }),
    }), act)
    try {
      const alert = descendants(view.document.body).find((element) => element.getAttribute('role') === 'alert')
      expect(alert?.getAttribute('aria-live')).toBe('assertive')
      expect(textOf(view.document.body)).toContain('This goal has already been started.')
    } finally {
      view.cleanup()
    }
  })

  it('keeps its controls disabled while starting', async () => {
    const view = await mountWithTestDom(() => renderDialog({ isPending: true }), act)
    try {
      expect(getButton(view.document.body, 'Starting…').getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.document.body, 'Cancel').getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('lists the effective authorization envelope', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      goal: { ...goal, goal_type: 'roadmap' },
      authorization: {
        ...allowed,
        external_write_targets: ['github:main'],
        live_send_channels: ['slack'],
        tool_constraints: ['no_live_sends', 'no_external_writes'],
      },
    }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('Roadmap')
      expect(text).toContain('What Start allows')
      expect(text).toContain('tokens 120,000 · usd 4.20')
      expect(text).toContain('Allowed: github:main')
      expect(text).toContain('Allowed: slack')
      expect(text).toContain('Staged (reviewed before it lands)')
      expect(text).toContain('No live sends, No external writes')
      expect(text).not.toContain('no_live_sends')
      expect(getButton(view.document.body, 'Start work').getAttribute('disabled')).toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('uses explicit fallbacks for an empty envelope', async () => {
    const view = await mountWithTestDom(() => renderDialog({ authorization: { ...allowed, model_spend_remaining: {} } }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('No measured budget')
      expect(text).toContain('Not allowed')
      expect(text).toContain('None')
    } finally {
      view.cleanup()
    }
  })

  it('blocks Start for a Continuous goal without a recurring policy', async () => {
    const view = await mountWithTestDom(() => renderDialog({ missingRecurringPolicy: true }), act)
    try {
      expect(textOf(view.document.body)).toContain('This Continuous goal has no accepted recurring policy yet.')
      expect(getButton(view.document.body, 'Start work').getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('blocks Start with the reason and needed decision when authorization is denied', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      authorization: { ...allowed, allowed: false, reason_code: 'budget_exhausted', required_decision: 'increase_budget' },
    }), act)
    try {
      expect(textOf(view.document.body)).toContain("Start isn't available: the budget is used up. Needed: A budget increase.")
      expect(getButton(view.document.body, 'Start work').getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('keeps Start enabled when the needed decision is Start itself (first Start or re-Start)', async () => {
    const view = await mountWithTestDom(() => renderDialog({
      authorization: { ...allowed, allowed: false, reason_code: 'contract_changed_since_start', required_decision: 'start' },
    }), act)
    try {
      expect(textOf(view.document.body)).not.toContain("Start isn't available")
      expect(getButton(view.document.body, 'Start work').getAttribute('disabled')).toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('words a re-Start with the reason and a Start again action', async () => {
    const view = await mountWithTestDom(() => renderDialog({ restartReason: 'the goal changed after you started it' }), act)
    try {
      const text = textOf(view.document.body)
      expect(text).toContain('Start work again?')
      expect(text).toContain('The goal changed after you started it. Starting again re-authorizes the orchestrator with the limits below.')
      expect(getButton(view.document.body, 'Start again').getAttribute('disabled')).toBeNull()
      expect(text).not.toContain('Start work on this goal?')
    } finally {
      view.cleanup()
    }
  })

  it('disables Start when authorization details are unavailable', async () => {
    const view = await mountWithTestDom(() => renderDialog({ authorization: null }), act)
    try {
      expect(textOf(view.document.body)).toContain('Authorization details are unavailable. Refresh and try again.')
      expect(getButton(view.document.body, 'Start work').getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })
})
