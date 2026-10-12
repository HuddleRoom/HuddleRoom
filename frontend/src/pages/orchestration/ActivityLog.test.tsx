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
    ...overrides,
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
  it('labels goal type, plan and adoption actions and gives each its own filter chip', async () => {
    const detail = goalDetail({
      actions: [
        action(1, { action_type: 'correct_goal_type', request: { requested_goal_type: 'roadmap', previous_goal_type: 'outcome', rationale: 'Too big' } }),
        action(2, { action_type: 'accept_plan' }),
        action(3, { action_type: 'adopt_task' }),
        action(4),
      ],
    })
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
    const byKind = (kind: string) => entries.find((entry) => entry.kind === kind)
    expect(byKind('type-corrected')).toMatchObject({ label: 'Goal type corrected', message: 'Outcome → Roadmap: Too big' })
    expect(byKind('plan-accepted')?.label).toBe('Plan accepted')
    expect(byKind('work-adopted')?.label).toBe('Existing work adopted')
    expect(entries.find((entry) => entry.label === 'action-type-4')?.kind).toBe('action')

    const view = await mount({ detail })
    try {
      const text = textOf(view.container)
      for (const label of ['Goal type corrected', 'Plan accepted', 'Existing work adopted']) expect(text).toContain(label)
    } finally {
      view.cleanup()
    }
  })
  function hierarchyDetail(): OrchestrationGoalDetail {
    const at = (seconds: number) => new Date(new Date(ISO).getTime() + seconds * 1000).toISOString()
    return goalDetail({
      goal: { id: 'goal-1', goal_type: 'outcome', goal_type_source: 'auto', goal_type_rationale: 'Single deliverable', created_at: ISO },
      runs: [],
      plan: { artifact_id: 'art-1', summary: 'Build the thing', items: [{ key: 'p1', title: 'Write API', gate_id: 'gate-1', adopted: false }] },
      actions: [
        action(1, { action_type: 'correct_goal_type', request: { requested_goal_type: 'roadmap', previous_goal_type: 'outcome' } }),
        action(2, { action_type: 'accept_plan' }),
        action(3, { action_type: 'adopt_task' }),
        action(4, { action_type: 'expand_plan_item', request: { plan_item_id: 'p1' } }),
        action(5, { action_type: 'report_consumed', target_type: 'artifact', target_id: 'art-1' }),
      ],
      supervision: { workers: [{ session_id: 's1', task_id: 't1', task_title: 'Write API', session_status: 'running', started_at: at(10) }] },
      evidence: [{ id: 'ev1', gate_id: 'gate-1', source_type: 'task_report', evidence_role: 'delivery', review_stage: 'final', verdict: 'accepted', created_at: at(11) }],
      gates: [{ id: 'gate-1', success_criterion_key: 'api', status: 'accepted', accepted_at: at(12) }],
    } as unknown as Partial<OrchestrationGoalDetail>)
  }

  it('humanizes evidence status, criteria, queued needs and corrected type source', () => {
    const base = hierarchyDetail()
    const detail = {
      ...base,
      goal: { ...base.goal, goal_type: 'roadmap', success_criteria: [{ key: 'api', description: 'The API is documented' }] },
      runs: [{ goal_type_snapshot: 'outcome' }],
      plan: { ...base.plan, items: [{ key: 'p1', title: 'Write API', gate_id: 'gate-1', adopted: false, task_id: 't-1' }] },
      authorization_wait: { task_ids: ['t-1'], reason_code: 'start_required', required_decision: 'start' },
      evidence: [{ id: 'ev1', gate_id: 'gate-1', source_type: 'task_report', evidence_role: 'readiness', review_stage: 'preliminary', verdict: 'candidate', acceptance_verdict: null, created_at: base.evidence[0].created_at }],
    } as unknown as OrchestrationGoalDetail
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
    const find = (kind: string) => entries.find((e) => e.kind === kind)
    expect(find('type-resolved')?.message).toBe('Outcome')
    expect(find('evidence-recorded')).toMatchObject({ status: 'pending', statusLabel: 'Pending' })
    expect(find('evidence-recorded')?.message).toContain('for The API is documented')
    expect(find('work-queued')?.message).toBe('Queued: Write API · Needs: Start again')
  })

  it('shows when queued work is waiting for a free slot', () => {
    const base = hierarchyDetail()
    const detail = {
      ...base,
      plan: { ...base.plan, items: [{ key: 'p1', title: 'Write API', task_id: 't-1', adopted: false }] },
      capacity_wait: { root_goal_id: 'goal-1', available_slots: 0, queued_task_ids: ['t-1'] },
    } as OrchestrationGoalDetail
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
    expect(entries.find((entry) => entry.kind === 'work-queued')?.message)
      .toBe('Queued: Write API · Waiting for a free slot')
  })

  it('shows one queue event for completed release and expansion of the same run item', () => {
    const detail = hierarchyDetail()
    detail.actions = [
      action(1, { action_type: 'release_item', request: { plan_item_id: 'p1' } }),
      action(2, { action_type: 'expand_plan_item', request: { plan_item_id: 'p1' } }),
      action(3, { action_type: 'release_item', request: { plan_item_id: 'p2' } }),
      action(4, { action_type: 'release_item', run_id: 'run-2', request: { plan_item_id: 'p1' } }),
      action(5, { action_type: 'release_item', status: 'failed', request: { plan_item_id: 'p1' } }),
    ]
    const queued = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
      .filter((entry) => entry.kind === 'work-queued')
    expect(queued).toHaveLength(4)
    expect(queued.map((entry) => entry.id)).not.toContain('action-action-1')
    expect(queued.some((entry) => entry.status === 'failed')).toBe(true)
  })

  it('derives distinct events for every hierarchy kind with copy and one chip each', async () => {
    const detail = hierarchyDetail()
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
    const labels: Record<string, string> = {
      'type-resolved': 'Goal type chosen', 'type-corrected': 'Goal type corrected', 'plan-ingested': 'Plan received',
      'plan-accepted': 'Plan accepted', 'work-adopted': 'Existing work adopted', 'work-queued': 'Work queued',
      'work-started': 'Work started', 'evidence-recorded': 'Evidence recorded', 'work-accepted': 'Work accepted',
    }
    for (const [kind, label] of Object.entries(labels)) expect(entries.find((e) => e.kind === kind)?.label).toBe(label)
    const message = (kind: string) => entries.find((e) => e.kind === kind)?.message
    expect(message('work-queued')).toBe('Queued: Write API')
    expect(message('plan-ingested')).toBe('Build the thing')
    expect(message('type-resolved')).toContain('Outcome · ')
    expect(message('type-resolved')).toContain('Single deliverable')
    expect(message('evidence-recorded')).toContain('for api')
    expect(message('work-accepted')).toBe('Write API accepted')

    const view = await mount({ detail })
    try {
      const chips = descendants(view.container).filter((n) => n.tagName.toLowerCase() === 'button')
      for (const label of Object.values(labels)) {
        expect(chips.filter((chip) => textOf(chip) === label)).toHaveLength(1)
      }
    } finally {
      view.cleanup()
    }
  })

  it('filtering by Work queued hides derived Started rows', async () => {
    const view = await mount({ detail: hierarchyDetail() })
    try {
      const chip = getButton(view.container, 'Work queued')
      await act(async () => { chip.click() })
      const text = textOf(view.container)
      expect(text).toContain('Queued: Write API')
      expect(text).not.toContain('Started: Write API')
    } finally {
      view.cleanup()
    }
  })

  it('keeps report_consumed on a task target as a generic action', () => {
    const detail = goalDetail({ actions: [action(1, { action_type: 'report_consumed', target_type: 'task', target_id: 't1' })] })
    const entries = buildActivityEntries({ processes: [], decisions: [], warnings: [], detail })
    expect(entries.find((e) => e.label === 'report_consumed')?.kind).toBe('action')
  })
})
