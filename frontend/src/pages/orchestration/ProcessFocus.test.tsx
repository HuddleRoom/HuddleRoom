import { describe, expect, it } from 'vitest'
import React, { act, createRef } from 'react'
import { descendants, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type {
  OrchestrationAgentReviewRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { ProcessFocus, type ProcessFocusProps } from './ProcessFocus'

const ISO = { started: '2026-08-05T08:00:00Z' }

function goal(overrides: Partial<OrchestrationGoal> = {}): OrchestrationGoal {
  return {
    id: 'goal-1', project_id: 'project-1', objective: 'Test goal',
    success_criteria: [], orchestrator_context: {}, constraints: {}, budget: {},
    status: 'active', weight: 'standard', weight_overridden_by: null,
    manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null,
    created_at: ISO.started, updated_at: ISO.started, needs_you_count: 0,
    ...overrides,
  }
}

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'team_hierarchy',
    process_version: 1, status: 'running', trigger_reason: 'test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function makeProps(overrides: Partial<ProcessFocusProps> = {}): ProcessFocusProps {
  return {
    selectedProcessType: 'team_hierarchy',
    process: undefined,
    goal: goal(),
    decisions: [],
    decisionsUnready: false,
    warnings: [],
    reviews: [],
    debug: false,
    canSkip: false,
    mutationBusy: false,
    canRun: false,
    canRerun: false,
    runPending: false,
    rerunPending: false,
    runRerunError: null,
    onRun: () => {},
    onRerun: () => {},
    retryPending: false,
    retryError: null,
    onRetry: () => {},
    openAction: null,
    onOpenAction: () => {},
    onActionTrigger: () => {},
    headingRef: createRef(),
    renderActionForm: () => null,
    ...overrides,
  }
}

async function mount(props: Partial<ProcessFocusProps> = {}) {
  const render = () => <ProcessFocus {...makeProps(props)} />
  const view = await mountWithTestDom(render, act)
  return view
}

describe('ProcessFocus', () => {
  it('renders lm_retry hint text when present and retry control is shown', async () => {
    const testProcess = process({
      status: 'running',
      outputs: {
        lm_retry: {
          available: true,
          kind: 'team_hierarchy',
          warning_id: null,
          model: 'claude-sonnet-5-5',
          hint: 'UPGRADE HINT TEXT',
        },
      },
    })
    const view = await mount({
      process: testProcess,
    })
    try {
      expect(textOf(view.container)).toContain('UPGRADE HINT TEXT')
    } finally {
      view.cleanup()
    }
  })

  it('does not render hint when hint is null', async () => {
    const testProcess = process({
      status: 'running',
      outputs: {
        lm_retry: {
          available: true,
          kind: 'team_hierarchy',
          warning_id: null,
          model: 'claude-sonnet-5-5',
          hint: null,
        },
      },
    })
    const view = await mount({
      process: testProcess,
    })
    try {
      expect(textOf(view.container)).not.toContain('UPGRADE HINT TEXT')
    } finally {
      view.cleanup()
    }
  })

  it('does not render hint when retry control is not shown (available is false)', async () => {
    const testProcess = process({
      status: 'running',
      outputs: {
        lm_retry: {
          available: false,
          kind: 'team_hierarchy',
          warning_id: null,
          model: null,
          hint: 'UPGRADE HINT TEXT',
        },
      },
    })
    const view = await mount({
      process: testProcess,
    })
    try {
      expect(textOf(view.container)).not.toContain('UPGRADE HINT TEXT')
    } finally {
      view.cleanup()
    }
  })

  it('distinguishes recorded empty hierarchy coverage from legacy evidence that was not recorded', async () => {
    const recordedEmpty = process({
      status: 'completed',
      outputs: { feedback_results: [], responsibility_coverage: [] },
    })
    const emptyView = await mount({ process: recordedEmpty })
    try {
      expect(textOf(emptyView.container)).toContain('Feedback requested')
      expect(textOf(emptyView.container)).toContain('No requested feedback recorded.')
      expect(textOf(emptyView.container)).toContain('Responsibility coverage')
      expect(textOf(emptyView.container)).toContain('No uncovered responsibilities.')
    } finally {
      emptyView.cleanup()
    }

    const legacyView = await mount({ process: process({ status: 'completed', outputs: {} }) })
    try {
      expect(textOf(legacyView.container)).toContain('Feedback requested: Not recorded')
      expect(textOf(legacyView.container)).toContain('Responsibility coverage: Not recorded')
    } finally {
      legacyView.cleanup()
    }
  })

  it('renders retained hierarchy source feedback and outcomes from the saved proposal', async () => {
    const retained = process({
      status: 'completed',
      outputs: {
        revision_context: { feedback_text: 'Preserve the accepted product responsibility.' },
        proposal: {
          assignments: [], reporting_lines: [],
          feedback_results: [{ item_id: 'feedback:product', source_excerpt: 'Retain Product evidence.', disposition: 'applied', reason: 'The accepted definition covers sequencing.' }],
          responsibility_coverage: [
            { responsibility_id: 'product', responsibility: 'Product sequencing', status: 'covered', agent_ref: 'product-id', reason: 'Accepted Product evidence.' },
            { responsibility_id: 'ux', responsibility: 'UX consent', status: 'gap', agent_ref: null, reason: 'No eligible UX capability.' },
          ],
        },
      },
    })
    const view = await mount({
      process: retained,
      reviews: [{ id: 'review-product', goal_id: 'goal-1', agent_id: 'product-id', definition_snapshot: { name: 'Product Owner' }, disposition: 'accepted', reasons: [], created_at: ISO.started, updated_at: ISO.started }],
    })
    try {
      const rendered = textOf(view.container)
      for (const text of [
        'Feedback requested', 'Preserve the accepted product responsibility.', 'Outcomes',
        'Retain Product evidence.', 'applied', 'The accepted definition covers sequencing.',
        'Responsibility coverage', 'Product sequencing', 'Product Owner', 'Domain owner — lacks execution role',
        'Blocking gaps', 'UX consent', 'No eligible UX capability.',
      ]) expect(rendered).toContain(text)
      expect(descendants(view.container).filter((node) => node.tagName === 'H4').map(textOf)).toEqual(expect.arrayContaining([
        'Feedback requested', 'Outcomes', 'Responsibility coverage', 'Blocking gaps',
      ]))
      await view.rerender()
      expect(textOf(view.container)).toContain('Preserve the accepted product responsibility.')
      expect(textOf(view.container)).toContain('No eligible UX capability.')
    } finally {
      view.cleanup()
    }
  })
})
