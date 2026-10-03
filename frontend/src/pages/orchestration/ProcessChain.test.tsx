import { describe, expect, it, vi } from 'vitest'
import React, { act } from 'react'
import { getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type { OrchestrationGoal, OrchestrationProcessRunRecord, OrchestrationRun } from '@/lib/types'
import { ProcessChain } from './ProcessChain'

const ISO = '2026-08-05T08:00:00Z'

function goal(overrides: Partial<OrchestrationGoal> = {}): OrchestrationGoal {
  return {
    id: 'goal-1', project_id: 'project-1', objective: 'Ship the baseline gate.',
    success_criteria: [], orchestrator_context: {}, constraints: {}, budget: {},
    status: 'active', weight: 'standard', weight_overridden_by: null,
    manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null,
    created_at: ISO, updated_at: ISO,
    ...overrides,
  }
}

function run(overrides: Partial<OrchestrationRun> = {}): OrchestrationRun {
  return {
    id: 'run-1', goal_id: 'goal-1', status: 'running', phase: 'baseline', condition: 'nominal',
    baseline_authorized: false, event_cursor: null, plan_state: {}, active_blockers: [],
    budget_state: {}, retry_state: {}, started_at: ISO, completed_at: null, created_at: ISO, updated_at: ISO,
    ...overrides,
  }
}

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition',
    process_version: 1, status: 'completed', trigger_reason: 'test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: ISO, completed_at: ISO, created_at: ISO, updated_at: ISO,
    ...overrides,
  }
}

async function mountChain(props: Partial<React.ComponentProps<typeof ProcessChain>> = {}) {
  const defaults: React.ComponentProps<typeof ProcessChain> = {
    processes: [], processState: 'ready', selectedProcessType: 'goal_definition', onSelect: () => undefined,
  }
  const view = await mountWithTestDom(() => <ProcessChain {...defaults} {...props} />, act)
  return view
}

describe('ProcessChain baseline authorization button', () => {
  it('shows "Start baseline" when unauthorized and no baseline process rows exist yet', async () => {
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: false }), onAuthorize: vi.fn(), processes: [] })
    try {
      expect(textOf(getButton(view.container, 'Start baseline'))).toBe('Start baseline')
    } finally {
      view.cleanup()
    }
  })

  it('shows "Continue baseline" when unauthorized and a baseline process row already exists', async () => {
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: false }), onAuthorize: vi.fn(), processes: [process()] })
    try {
      expect(textOf(getButton(view.container, 'Continue baseline'))).toBe('Continue baseline')
      expect(() => getButton(view.container, 'Start baseline')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('hides the button once the run is authorized', async () => {
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: true }), onAuthorize: vi.fn(), processes: [] })
    try {
      expect(() => getButton(view.container, 'Start baseline')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('hides the button once the run has moved past the baseline phase', async () => {
    const view = await mountChain({ goal: goal(), run: run({ phase: 'ready', baseline_authorized: false }), onAuthorize: vi.fn(), processes: [] })
    try {
      expect(() => getButton(view.container, 'Start baseline')).toThrow()
      expect(() => getButton(view.container, 'Continue baseline')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('hides the button when no run is attached', async () => {
    const view = await mountChain({ goal: goal(), run: null, onAuthorize: vi.fn(), processes: [] })
    try {
      expect(() => getButton(view.container, 'Start baseline')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('calls onAuthorize with the clicked button element on click', async () => {
    const onAuthorize = vi.fn()
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: false }), onAuthorize, processes: [] })
    try {
      await act(async () => getButton(view.container, 'Start baseline').click())
      expect(onAuthorize).toHaveBeenCalledTimes(1)
      expect(onAuthorize.mock.calls[0][0]).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('shows a pending label and disables the button while the mutation is in flight', async () => {
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: false }), onAuthorize: vi.fn(), authorizePending: true, processes: [] })
    try {
      const button = getButton(view.container, 'Starting…')
      expect(button.getAttribute('disabled')).not.toBeNull()
    } finally {
      view.cleanup()
    }
  })

  it('surfaces an authorize error under the header', async () => {
    const view = await mountChain({ goal: goal(), run: run({ baseline_authorized: false }), onAuthorize: vi.fn(), authorizeError: 'The baseline could not be started.', processes: [] })
    try {
      expect(textOf(view.container)).toContain('The baseline could not be started.')
    } finally {
      view.cleanup()
    }
  })
})
