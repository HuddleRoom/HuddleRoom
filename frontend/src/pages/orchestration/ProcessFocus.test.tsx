import { describe, expect, it } from 'vitest'
import React, { act, createRef } from 'react'
import { mountWithTestDom, textOf } from '../../../tests/support/dom'
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
          model: 'claude-3-sonnet',
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
          model: 'claude-3-sonnet',
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
})
