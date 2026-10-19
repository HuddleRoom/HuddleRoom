import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, createRef } from 'react'
import { descendants, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type {
  Agent,
  OrchestrationAgentReviewRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { ProcessFocus, type ProcessFocusProps } from './ProcessFocus'

const agentsQuery = vi.hoisted(() => ({ data: { items: [] as Agent[], next_cursor: null }, isSuccess: true, isError: false, isPending: false }))
vi.mock('@/api/agents', () => ({ useAgents: () => agentsQuery }))
beforeEach(() => Object.assign(agentsQuery, { data: { items: [], next_cursor: null }, isSuccess: true, isError: false, isPending: false }))

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

function review(index = 1): OrchestrationAgentReviewRecord {
  return {
    id: `review-${index}`, goal_id: 'goal-1', run_id: 'run-1', agent_id: `agent-${index}`,
    source_process_run_id: 'process-1', review_context: null, proposed_work_functions: ['review'],
    definition_snapshot: { name: `Reviewed agent ${index}`, provider: 'historic-provider', model: `historic-model-${index}` },
    fit_summary: 'Saved assessment', strengths: [], risks: [], recommended_changes: [],
    approved_for_work_functions: ['review'], created_at: ISO.started, updated_at: ISO.started,
  }
}

function agent(index = 1): Agent {
  return {
    id: `agent-${index}`, name: `Current agent ${index}`, role: 'reviewer', provider: 'current-provider',
    model: `current-model-${index}`, adapter_type: 'api', cli_runtime: null, system_prompt: null,
    description: null, capabilities: ['review'], is_active: true, config: {},
    created_at: ISO.started, updated_at: ISO.started,
  }
}

describe('retained G4 identity and roster', () => {
  it.each(['Run', 'Re-run', 'Retry', 'Skip'])('retained old %s callback is inert after a successor arrives', async (label) => {
    const callback = vi.fn()
    let props = makeProps({
      process: process({ outputs: label === 'Retry' ? { lm_retry: { available: true, kind: 'team_hierarchy', warning_id: null, model: null, hint: null } } : {} }),
      canRun: true, canRerun: true, canSkip: true,
      onRun: callback, onRerun: callback, onRetry: callback, onOpenAction: callback,
    })
    const view = await mountWithTestDom(() => <ProcessFocus {...props} />, act)
    try {
      const button = descendants(view.container).find((node) => node.tagName === 'BUTTON' && textOf(node) === label)!
      expect(button).toBeDefined()
      const key = Object.keys(button).find((value) => value.startsWith('__reactProps$'))!
      const retained = (button as unknown as Record<string, { onClick: (event: unknown) => void }>)[key].onClick
      props = { ...props, process: { ...props.process!, superseded_by_id: 'replacement' } }
      await view.rerender()
      await act(async () => retained({ currentTarget: button }))
      expect(callback).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('test_g4_superseded_history_is_inert even with enabled parent controls and an open form', async () => {
    const callbacks = { onRun: vi.fn(), onRerun: vi.fn(), onRetry: vi.fn(), onOpenAction: vi.fn() }
    const renderActionForm = vi.fn(() => <button>Confirm skip</button>)
    const props = makeProps({
      ...callbacks, canRun: true, canRerun: true, canSkip: true,
      process: process({ status: 'running', superseded_by_id: 'replacement-process', outputs: { lm_retry: { available: true, kind: 'team_hierarchy', warning_id: null, model: null, hint: null } } }),
      reviews: [review()], openAction: { kind: 'skip', id: 'process-1' }, renderActionForm,
    })
    const view = await mountWithTestDom(() => <ProcessFocus {...props} />, act)
    try {
      expect(textOf(view.container)).toContain('Superseded')
      expect(textOf(view.container)).toContain('replacement-process')
      expect(textOf(view.container)).toContain('Saved assessment')
      expect(descendants(view.container).filter((node) => node.tagName === 'BUTTON')).toHaveLength(0)
      expect(renderActionForm).not.toHaveBeenCalled()
      for (const callback of Object.values(callbacks)) expect(callback).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('test_g4_current_and_reviewed_roster_are_separate for five agents and runtime updates', async () => {
    agentsQuery.data.items = Array.from({ length: 5 }, (_, i) => agent(i + 1))
    const reviews = Array.from({ length: 5 }, (_, i) => review(i + 1))
    const props = makeProps({ selectedProcessType: 'agent_definition_review', process: process({ process_type: 'agent_definition_review', status: 'completed' }), reviews })
    const view = await mountWithTestDom(() => <ProcessFocus {...props} />, act)
    try {
      for (let i = 1; i <= 5; i++) {
        expect(textOf(view.container)).toContain(`historic-model-${i}`)
        expect(textOf(view.container)).toContain(`current-model-${i}`)
      }
      expect(textOf(view.container)).toContain('Reviewed configuration')
      expect(textOf(view.container)).toContain('Current settings')
      agentsQuery.data.items = agentsQuery.data.items.map((row) => ({ ...row, model: 'updated-runtime' }))
      await view.rerender()
      expect(textOf(view.container)).toContain('updated-runtime')
      expect(textOf(view.container)).toContain('historic-model-1')
      expect(reviews[0].definition_snapshot.model).toBe('historic-model-1')
    } finally { view.cleanup() }
  })

  it.each(['failed', 'loading', 'missing'] as const)('current %s never falls back to cached or reviewed settings', async (state) => {
    Object.assign(agentsQuery, { data: { items: state === 'missing' ? [] : [agent()], next_cursor: null }, isSuccess: state === 'missing', isError: state === 'failed', isPending: state === 'loading' })
    const view = await mount({ selectedProcessType: 'agent_definition_review', process: process(), reviews: [review()] })
    try {
      expect(textOf(view.container)).toContain(state === 'loading' ? 'Current settings loading' : 'Current settings unavailable')
      expect(textOf(view.container)).toContain('historic-model-1')
      expect(textOf(view.container)).not.toContain('current-model-1')
    } finally { view.cleanup() }
  })

  it('unknown current native model remains unknown after a fresh mount', async () => {
    agentsQuery.data.items = [{ ...agent(), model: '', adapter_type: 'cli', cli_runtime: 'codex' }]
    const view = await mount({ process: process(), reviews: [review()] })
    try {
      expect(textOf(view.container)).toContain('Unknown / provider default')
      expect(textOf(view.container)).toContain('historic-model-1')
    } finally { view.cleanup() }
  })
})

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
