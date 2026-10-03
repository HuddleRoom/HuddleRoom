import { describe, expect, it, vi } from 'vitest'
import React, { act } from 'react'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type {
  OrchestrationAction,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationDecision,
  OrchestrationGoalDetail,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { ActivityLog, buildActivityEntries, type ActivityLogProps } from './ActivityLog'

const ISO = '2026-08-05T08:00:00Z'

function goalDetail(overrides: Partial<OrchestrationGoalDetail> = {}): OrchestrationGoalDetail {
  return {
    id: 'goal-1',
    objective: 'Test goal',
    success_criteria: [],
    decisions: [],
    actions: [],
    orchestrator_context: {},
    constraints: {},
    budget: {},
    status: 'active',
    weight: 'standard',
    weight_overridden_by: null,
    manager_agent_id: null,
    manager_user_id: null,
    authority_model: null,
    created_by_user_id: null,
    created_at: ISO,
    updated_at: ISO,
    ...overrides,
  }
}

function decision(index: number, overrides: Partial<OrchestrationDecision> = {}): OrchestrationDecision {
  return {
    id: `decision-${index}`,
    run_id: 'run-1',
    decision_type: `decision-type-${index}`,
    reason: `Reason for decision ${index}`,
    rejection_reason: null,
    validator_status: 'approved',
    parsed_decision: null,
    input_snapshot: {},
    llm_output: '',
    created_at: new Date(new Date(ISO).getTime() + index * 1000).toISOString(),
    ...overrides,
  }
}

function action(index: number, overrides: Partial<OrchestrationAction> = {}): OrchestrationAction {
  return {
    id: `action-${index}`,
    run_id: 'run-1',
    decision_id: null,
    action_type: `action-type-${index}`,
    status: 'completed',
    request: { reason: `Action reason ${index}` },
    target_type: null,
    target_id: null,
    error: null,
    created_at: new Date(new Date(ISO).getTime() + index * 1000).toISOString(),
  }
}

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1',
    goal_id: 'goal-1',
    run_id: 'run-1',
    process_type: 'team_hierarchy',
    process_version: 1,
    status: 'completed',
    trigger_reason: 'test',
    input_snapshot: {},
    outputs: {},
    skipped_by: null,
    override_reason: null,
    superseded_by_id: null,
    started_at: ISO,
    completed_at: ISO,
    created_at: ISO,
    updated_at: ISO,
    ...overrides,
  }
}

function authorityDecision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'authority-decision-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'team_hierarchy:approval',
    title: 'Approve the hierarchy', status: 'answered', authority: 'human', authority_agent_id: null,
    source_process_run_id: 'process-1', question: 'Approve this hierarchy?', context: null, options: ['approve'],
    recommendation: null, consequences: null, selected_option: 'approve', reason: null,
    decided_by_user_id: 'user-1', decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: ISO, decided_at: ISO, created_at: ISO, updated_at: ISO,
    ...overrides,
  }
}

function makeProps(overrides: Partial<ActivityLogProps> = {}): ActivityLogProps {
  return {
    processes: [],
    decisions: [],
    warnings: [],
    detail: goalDetail(),
    selectedStepType: null,
    scope: 'goal',
    onScopeChange: () => {},
    debug: false,
    ...overrides,
  }
}

async function mount(props: Partial<ActivityLogProps> = {}) {
  const render = () => <ActivityLog {...makeProps(props)} />
  const view = await mountWithTestDom(render, act)
  return view
}

describe('ActivityLog', () => {
  it('renders exactly 50 entries initially when there are 200+ entries', async () => {
    const entries = Array.from({ length: 210 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      const listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems).toHaveLength(50)
    } finally {
      view.cleanup()
    }
  })

  it('shows "Show more" button when entries exceed LEDGER_PAGE_SIZE', async () => {
    const entries = Array.from({ length: 210 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      const button = getButton(view.container, 'Show more')
      expect(button).toBeDefined()
    } finally {
      view.cleanup()
    }
  })

  it('reveals all entries when "Show more" button is clicked', async () => {
    const entries = Array.from({ length: 210 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      // Initially 50 entries
      let listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems).toHaveLength(50)

      // Click "Show more"
      const button = getButton(view.container, 'Show more')
      await act(async () => {
        button.click()
      })

      // All entries should now be visible
      listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems).toHaveLength(210)
    } finally {
      view.cleanup()
    }
  })

  it('does not show "Show more" button when entries count is <= LEDGER_PAGE_SIZE', async () => {
    const entries = Array.from({ length: 40 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      const buttons = descendants(view.container).filter(
        (node) => node.tagName === 'BUTTON' && textOf(node).includes('Show more'),
      )
      expect(buttons).toHaveLength(0)
    } finally {
      view.cleanup()
    }
  })

  it('resets visible count when scope changes', async () => {
    const entries = Array.from({ length: 210 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    let scope = 'goal' as const
    const view = await mount({
      detail,
      scope,
      onScopeChange: (newScope) => {
        scope = newScope
      },
    })

    try {
      // Click "Show more" to reveal all
      const button = getButton(view.container, 'Show more')
      await act(async () => {
        button.click()
      })

      let listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems).toHaveLength(210)

      // Simulate scope change by rerendering
      await view.rerender(() => (
        <ActivityLog
          {...makeProps({
            detail,
            scope: 'step' as const,
            selectedStepType: 'team_hierarchy',
          })}
        />
      ))

      // Should be back to 50
      listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems.length).toBeLessThanOrEqual(50)
    } finally {
      view.cleanup()
    }
  })

  it('hides "Show more" button after clicking it', async () => {
    const entries = Array.from({ length: 210 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      const button = getButton(view.container, 'Show more')
      await act(async () => {
        button.click()
      })

      // Button should no longer exist
      const buttons = descendants(view.container).filter(
        (node) => node.tagName === 'BUTTON' && textOf(node).includes('Show more'),
      )
      expect(buttons).toHaveLength(0)
    } finally {
      view.cleanup()
    }
  })

  it('renders exactly 50 entries when there are exactly 50 entries', async () => {
    const entries = Array.from({ length: 50 }, (_, i) => decision(i))
    const detail = goalDetail({ decisions: entries })
    const view = await mount({ detail })

    try {
      const listItems = descendants(view.container).filter((node) => node.tagName === 'LI')
      expect(listItems).toHaveLength(50)

      // No "Show more" button
      const buttons = descendants(view.container).filter(
        (node) => node.tagName === 'BUTTON' && textOf(node).includes('Show more'),
      )
      expect(buttons).toHaveLength(0)
    } finally {
      view.cleanup()
    }
  })
})

describe('ActivityLog steering anchors', () => {
  it('keeps timeline ordering and gives decisions and actions stable anchors', () => {
    const detail = {
      decisions: [{ id: 'decision-1', decision_type: 'review', validator_status: 'accepted', reason: 'reason', rejection_reason: null, created_at: '2026-09-16T10:00:00Z', parsed_decision: {}, input_snapshot: {}, llm_output: null }],
      actions: [{ id: 'action-1', decision_id: 'decision-1', action_type: 'delegate', status: 'completed', request: {}, target_type: null, target_id: null, error: null, created_at: '2026-09-16T11:00:00Z' }],
    } as never
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })

    expect(entries.map(({ id, anchorId }) => [id, anchorId])).toEqual([
      ['action-action-1', 'action-action-1'], ['decision-decision-1', 'decision-decision-1'],
    ])
  })

  it('anchors and focuses an answered authority decision without a goal-ledger decision', async () => {
    const authority = authorityDecision()
    const entries = buildActivityEntries({
      processes: [process()], decisions: [authority], warnings: [], detail: goalDetail(),
    })
    expect(entries.find((entry) => entry.id === `${authority.id}:answered`)?.anchorId)
      .toBe(`decision-${authority.id}`)

    const view = await mount({ processes: [process()], decisions: [authority] })
    try {
      const focus = vi.fn()
      const scrollIntoView = vi.fn()
      ;(view.document as unknown as { getElementById: (id: string) => HTMLElement | null }).getElementById = (id) => {
        const target = descendants(view.container).find((node) => node.getAttribute('id') === id)
        if (target) Object.assign(target, { focus, scrollIntoView })
        return target as unknown as HTMLElement ?? null
      }
      await view.rerender(() => <ActivityLog {...makeProps({
        processes: [process()], decisions: [authority], focusDecisionId: authority.id, focusDecisionPending: true,
      })} />)

      const target = descendants(view.container).find((node) => node.getAttribute('id') === `decision-${authority.id}`)
      expect(target).toBeDefined()
      expect(target?.getAttribute('tabindex')).toBe('-1')
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)
    } finally {
      view.cleanup()
    }
  })

  it('anchors an unscoped answered authority decision that lacks decided_at', () => {
    const authority = authorityDecision({ decided_at: null, source_process_run_id: null })
    const entries = buildActivityEntries({
      processes: [process()], decisions: [authority], warnings: [], detail: goalDetail(),
    })

    expect(entries.find((entry) => entry.id === `${authority.id}:answered`)).toMatchObject({
      anchorId: `decision-${authority.id}`,
      timestamp: authority.created_at,
      processType: null,
    })
  })
})
