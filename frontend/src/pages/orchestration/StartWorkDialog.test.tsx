import React, { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { ApiError } from '@/lib/api-client'
import type { OrchestrationGoal } from '@/lib/types'

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

function renderDialog(overrides: Partial<React.ComponentProps<typeof StartWorkDialog>> = {}) {
  return <StartWorkDialog open goal={goal} isPending={false} error={null} onOpenChange={vi.fn()} onConfirm={vi.fn()} {...overrides} />
}

describe('StartWorkDialog', () => {
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
      expect(getButton(view.document.body, 'Working…').getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.document.body, 'Cancel').getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })
})
