import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, createRef } from 'react'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationGoalDetail,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'

const memorySectionMock = vi.fn((
  _projectId: string | null,
  _goalId: string | undefined,
  sectionKey: string | undefined,
) => (
  sectionKey === 'runbook'
    ? { data: { body: 'Confirm the legal approval before closing out.' }, isLoading: false, isError: false }
    : { data: undefined, isLoading: false, isError: false }
))
const debugStepMock = { isPending: false, mutate: vi.fn() }
const debugRerunMock = { isPending: false, mutate: vi.fn() }
const conversationMock = {
  data: { items: [], total: 0, omitted: 0, allowance: { enabled: true, limit: 100, used: 0, remaining: 100 }, steering: { enabled: false, eligibility: 'unstarted', eligibility_reason: null, inbox_version: 0, direction_version: 0, requests: [], proposals: [] } },
  isLoading: false, isError: false, refetch: vi.fn(),
}

vi.mock('@/api/orchestration', () => ({
  useOrchestrationMemorySection: (...args: [string | null, string | undefined, string | undefined]) => memorySectionMock(...args),
  useStepOrchestrationBaseline: () => debugStepMock,
  useRerunLastOrchestrationBaseline: () => debugRerunMock,
  useOrchestrationConversation: () => conversationMock,
  useSubmitOrchestrationConversation: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useRecordOrchestrationConversationFeedback: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useSubmitOrchestrationSteering: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useWithdrawOrchestrationSteering: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
  useDismissOrchestrationSteeringProposal: () => ({ isPending: false, isError: false, error: null, mutate: vi.fn() }),
}))

import { DetailsTabs, type DetailsTabsHandle } from './DetailsTabs'

const ISO = { started: '2026-08-05T08:00:00Z', completed: '2026-08-05T09:00:00Z' }

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition',
    process_version: 1, status: 'running', trigger_reason: 'test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: ISO.started, completed_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function decision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'some_decision',
    title: 'Approve the plan', status: 'pending', authority: 'human', authority_agent_id: null,
    source_process_run_id: null, question: 'Approve the plan?', context: null, options: ['approve'],
    recommendation: null, consequences: null, selected_option: null, reason: null,
    decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: ISO.started, decided_at: null, created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function warning(overrides: Partial<OrchestrationWarningRecord> = {}): OrchestrationWarningRecord {
  return {
    id: 'warning-1', goal_id: 'goal-1', run_id: 'run-1', warning_type: 'coverage_gap', severity: 'warning',
    message: 'Coverage evidence is incomplete.', source_process_run_id: null, related_gate_id: null,
    related_action_id: null, related_agent_id: null, source_agent_review_id: null,
    related_authority_decision_id: null, acknowledged_by: null, acknowledged_at: null, active: true,
    resolved_by: null, resolved_reason: null, resolved_at: null, blocks_completion: false,
    created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function goalDetail(overrides: Partial<OrchestrationGoalDetail> = {}): OrchestrationGoalDetail {
  return {
    goal: {
      id: 'goal-1', project_id: 'project-1', objective: 'Ship the details tabs.',
      success_criteria: [], orchestrator_context: {}, constraints: {}, budget: {},
      goal_type: 'outcome', supersedes_goal_id: null,
      status: 'active', weight: 'standard', weight_overridden_by: null,
      manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null,
      created_at: ISO.started, updated_at: ISO.started, needs_you_count: 0,
    },
    run: null,
    decisions_count: 0,
    decisions: [{
      id: 'ledger-decision-1', run_id: 'run-1', decision_type: 'request_verification',
      input_snapshot: {}, llm_output: {}, parsed_decision: {}, validator_status: 'accepted',
      rejection_reason: null, reason: 'Request independent validation',
      created_at: '2026-08-05T07:00:00Z', updated_at: '2026-08-05T07:00:00Z',
    }],
    actions_count: 0,
    actions: [],
    gates_count: 0,
    gates: [],
    evidence_count: 0,
    evidence: [],
    agent_suggestions_count: 0,
    agent_suggestions: [],
    timeline: [],
    ...overrides,
  }
}

const memoryFixture = {
  data: {
    toc: [{ section_key: 'runbook', title: 'Runbook', summary: 'Manual procedure.', section_type: 'markdown', always_load: false, toc_order: 2, updated_at: ISO.started }],
    always_loaded: [{
      id: 'memory-intro', project_id: 'project-1', goal_id: 'goal-1', run_id: null, section_key: 'introduction',
      title: 'Introduction', section_type: 'markdown', body: 'Operator preface.', summary: 'What the operator needs to know.',
      always_load: true, toc_order: 1, created_by: 'orchestrator', created_from_event_id: null, updated_from_event_id: null,
      created_at: ISO.started, updated_at: ISO.started,
    }],
    preface: {
      objective: 'Ship the details tabs.', goal_status: 'active' as const, goal_weight: 'standard' as const, run_status: 'running' as const,
      current_process: { process_type: 'goal_definition', status: 'running' as const }, manager: 'agent-manager', hierarchy: 'Proposed',
      constraints: 'No unverified manager assignment.', active_warnings: [{ severity: 'blocker' as const, warning_type: 'legal_approval', message: 'Blocker: legal approval', acknowledged: false }],
      recent_decisions: [], open_blockers: [], skipped_processes: [], introduction: 'Operator preface.',
      always_loaded: [{ section_key: 'introduction', summary: 'What the operator needs to know.' }], toc: [{ section_key: 'runbook', title: 'Runbook' }],
    },
  },
  isLoading: false,
  isError: false,
}

function Harness(props: Partial<React.ComponentProps<typeof DetailsTabs>> & { handleRef?: React.Ref<DetailsTabsHandle> } = {}) {
  const { handleRef, ...rest } = props
  return (
    <DetailsTabs
      ref={handleRef}
      projectId="project-1"
      goalId="goal-1"
      detail={goalDetail()}
      processes={[]}
      decisions={[]}
      warnings={[]}
      selectedStepType={null}
      memory={memoryFixture}
      debug={false}
      goalPlanContent={<p>Goal-plan content.</p>}
      gatesContent={<p>Gates content.</p>}
      delegationsContent={<p>Delegations content.</p>}
      suggestionsContent={<p>Suggestions content.</p>}
      {...rest}
    />
  )
}

async function mount(props: Partial<React.ComponentProps<typeof Harness>> = {}) {
  const render = (nextProps = props) => <Harness {...nextProps} />
  const view = await mountWithTestDom(() => render(), act)
  return { ...view, rerender: (nextProps: Partial<React.ComponentProps<typeof Harness>> = props) => view.rerender(() => render(nextProps)) }
}

function tabPanel(container: Parameters<typeof descendants>[0], key: string) {
  const panel = descendants(container).find((node) => node.getAttribute('id') === `details-panel-${key}`)
  if (!panel) throw new Error(`Tab panel not found: ${key}`)
  return panel
}

beforeEach(() => {
  memorySectionMock.mockClear()
  debugStepMock.mutate.mockReset()
  debugRerunMock.mutate.mockReset()
})

describe('DetailsTabs', () => {
  it('places Conversation before the Activity log', async () => {
    const view = await mount()
    try {
      const activity = tabPanel(view.container, 'ledger')
      const content = textOf(activity)
      const conversationIndex = content.indexOf('Conversation')
      const logIndex = content.indexOf('Whole goal')
      expect(conversationIndex).toBeGreaterThanOrEqual(0)
      expect(logIndex).toBeGreaterThanOrEqual(0)
      expect(conversationIndex).toBeLessThan(logIndex)
    } finally { view.cleanup() }
  })

  it('defaults to the Activity tab and switches tabs on click, hiding inactive panels', async () => {
    const view = await mount()
    try {
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()
      expect(tabPanel(view.container, 'gates').getAttribute('hidden')).not.toBeNull()

      await act(async () => getButton(view.container, 'Gates').click())

      expect(tabPanel(view.container, 'gates').getAttribute('hidden')).toBeNull()
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).not.toBeNull()
      expect(textOf(tabPanel(view.container, 'gates'))).toContain('Gates content.')
    } finally {
      view.cleanup()
    }
  })

  it('renders both the goal-plan and suggestions content on the Plan tab, and delegations content on the Delegations tab', async () => {
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Plan').click())
      const plan = tabPanel(view.container, 'plan')
      expect(textOf(plan)).toContain('Goal-plan content.')
      expect(textOf(plan)).toContain('Suggestions content.')

      await act(async () => getButton(view.container, 'Delegations').click())
      expect(textOf(tabPanel(view.container, 'delegations'))).toContain('Delegations content.')
    } finally {
      view.cleanup()
    }
  })

  it('merges per-step and goal-wide entries in the Activity tab (Whole goal scope)', async () => {
    const goalDefinition = process({ id: 'process-goal-def', process_type: 'goal_definition', status: 'completed', completed_at: ISO.completed })
    const view = await mount({
      processes: [goalDefinition],
      decisions: [decision({ source_process_run_id: goalDefinition.id, title: 'Per-step decision' })],
      warnings: [warning({ source_process_run_id: goalDefinition.id, message: 'Per-step warning' })],
      detail: goalDetail(),
    })
    try {
      const activity = tabPanel(view.container, 'ledger')
      // Per-step (process-scoped) entry:
      expect(textOf(activity)).toContain('Per-step decision')
      // Goal-wide (run-scoped orchestrator ledger) entry:
      expect(textOf(activity)).toContain('request_verification')
      expect(textOf(activity)).toContain('Request independent validation')
    } finally {
      view.cleanup()
    }
  })

  it('scopes Activity to "This step" and stays synced to the stepper-selected step', async () => {
    const goalDefinition = process({ id: 'process-goal-def', process_type: 'goal_definition', status: 'completed', completed_at: ISO.completed })
    const managerSelection = process({ id: 'process-manager', process_type: 'manager_selection', status: 'running', started_at: ISO.completed })
    const view = await mount({
      processes: [goalDefinition, managerSelection],
      selectedStepType: 'goal_definition',
    })
    try {
      await act(async () => getButton(view.container, 'This step').click())
      const activity = tabPanel(view.container, 'ledger')
      expect(textOf(activity)).toContain('Goal definition completed')
      expect(textOf(activity)).not.toContain('Manager selection started')
      // Goal-wide ledger entries aren't attributable to any step, so "This step" excludes them too.
      expect(textOf(activity)).not.toContain('request_verification')

      // Stepper selection changes while scope stays "This step" — Activity follows.
      await view.rerender({ processes: [goalDefinition, managerSelection], selectedStepType: 'manager_selection' })
      expect(textOf(tabPanel(view.container, 'ledger'))).toContain('Manager selection started')
      expect(textOf(tabPanel(view.container, 'ledger'))).not.toContain('Goal definition completed')
    } finally {
      view.cleanup()
    }
  })

  it('exposes an imperative handle that switches to Activity scoped to This step', async () => {
    const ref = createRef<DetailsTabsHandle>()
    const goalDefinition = process({ id: 'process-goal-def', process_type: 'goal_definition', status: 'completed', completed_at: ISO.completed })
    const view = await mount({ handleRef: ref, processes: [goalDefinition], selectedStepType: 'goal_definition' })
    try {
      await act(async () => getButton(view.container, 'Gates').click())
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).not.toBeNull()

      await act(async () => ref.current?.showStepActivity())

      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()
      expect(getButton(view.container, 'This step').getAttribute('aria-pressed')).toBe('true')
    } finally {
      view.cleanup()
    }
  })

  it('opens and focuses an older cited decision in the whole-goal ledger', async () => {
    const ref = createRef<DetailsTabsHandle>()
    const decisions = Array.from({ length: 51 }, (_, index) => ({
      id: `decision-${index}`, run_id: 'run-1', decision_type: 'review', input_snapshot: {}, llm_output: {}, parsed_decision: {},
      validator_status: 'accepted' as const, rejection_reason: null, reason: 'Review recorded',
      created_at: `2026-08-05T${String(index % 24).padStart(2, '0')}:00:00Z`, updated_at: ISO.started,
    }))
    const view = await mount({ handleRef: ref, detail: goalDetail({ decisions }) })
    try {
      const focus = vi.fn()
      const scrollIntoView = vi.fn()
      ;(view.document as unknown as { getElementById: (id: string) => HTMLElement | null }).getElementById = (id) => {
        const target = descendants(view.container).find((node) => node.getAttribute('id') === id)
        if (target) Object.assign(target, { focus, scrollIntoView })
        return target as unknown as HTMLElement ?? null
      }
      await act(async () => getButton(view.container, 'Plan').click())
      await act(async () => ref.current?.showDecisionActivity('decision-0'))

      const target = descendants(view.container).find((node) => node.getAttribute('id') === 'decision-decision-0')
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()
      expect(getButton(view.container, 'Whole goal').getAttribute('aria-pressed')).toBe('true')
      expect(target).toBeDefined()
      expect(target?.getAttribute('tabindex')).toBe('-1')
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)

      await view.rerender({ handleRef: ref, detail: goalDetail({ decisions }) })
      expect(focus).toHaveBeenCalledTimes(1)
      expect(scrollIntoView).toHaveBeenCalledTimes(1)
    } finally { view.cleanup() }
  })

  it('handles the same decision again after navigating through a non-decision hash', async () => {
    const decisionNavigation = 'decision-link:/orchestration/goal-1#decision-ledger-decision-1'
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Gates').click())
      await view.rerender({ decisionFocusId: 'ledger-decision-1', decisionNavigation })
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()

      await act(async () => getButton(view.container, 'Gates').click())
      await view.rerender({ decisionFocusId: null, decisionNavigation: null })
      expect(tabPanel(view.container, 'gates').getAttribute('hidden')).toBeNull()

      await view.rerender({ decisionFocusId: 'ledger-decision-1', decisionNavigation })
      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()
    } finally { view.cleanup() }
  })

  it('filters Activity entries by kind chips', async () => {
    const goalDefinition = process({ id: 'process-goal-def', process_type: 'goal_definition', status: 'completed', completed_at: ISO.completed })
    const view = await mount({
      processes: [goalDefinition],
      decisions: [decision({ source_process_run_id: goalDefinition.id, title: 'Per-step decision' })],
    })
    try {
      const activity = tabPanel(view.container, 'ledger')
      expect(textOf(activity)).toContain('Goal definition completed')
      expect(textOf(activity)).toContain('Per-step decision')

      await act(async () => getButton(view.container, 'Decision asked').click())

      expect(textOf(activity)).toContain('Per-step decision')
      expect(textOf(activity)).not.toContain('Goal definition completed')
    } finally {
      view.cleanup()
    }
  })

  it('renders Memory preface as label/value rows, always_loaded expanded, and fetches a toc section on open', async () => {
    const view = await mount()
    try {
      await act(async () => getButton(view.container, 'Memory').click())
      const memory = tabPanel(view.container, 'memory')
      expect(textOf(memory)).toContain('Ship the details tabs.')
      expect(textOf(memory)).toContain('Introduction')
      expect(textOf(memory)).toContain('Operator preface.')
      expect(textOf(memory)).toContain('Runbook')
      expect(textOf(memory)).not.toContain('Confirm the legal approval before closing out.')

      const runbookSummary = descendants(view.container).find((node) => node.tagName === 'SUMMARY' && textOf(node) === 'Runbook')
      if (!runbookSummary) throw new Error('Runbook summary not found')
      await act(async () => runbookSummary.click())

      expect(memorySectionMock).toHaveBeenCalledWith('project-1', 'goal-1', 'runbook')
      expect(textOf(tabPanel(view.container, 'memory'))).toContain('Confirm the legal approval before closing out.')
    } finally {
      view.cleanup()
    }
  })

  it('gates the Debug tab behind debug_enabled', async () => {
    const withoutDebug = await mount({ debug: false })
    try {
      expect(() => getButton(withoutDebug.container, 'Debug')).toThrow()
    } finally {
      withoutDebug.cleanup()
    }

    const withDebug = await mount({ debug: true })
    try {
      await act(async () => getButton(withDebug.container, 'Debug').click())
      expect(textOf(tabPanel(withDebug.container, 'debug'))).toContain('Baseline debug')
    } finally {
      withDebug.cleanup()
    }
  })

  it('renders debugContent inside the Debug tabpanel when debug is true', async () => {
    const view = await mount({ debug: true, debugContent: <p>Danger zone content.</p> })
    try {
      await act(async () => getButton(view.container, 'Debug').click())
      const debug = tabPanel(view.container, 'debug')
      expect(textOf(debug)).toContain('Danger zone content.')
      // debugContent renders ahead of the baseline debug panel.
      expect(textOf(debug).indexOf('Danger zone content.')).toBeLessThan(textOf(debug).indexOf('Baseline debug'))
    } finally {
      view.cleanup()
    }
  })

  it('does not render debugContent or a Debug tab at all when debug is false', async () => {
    const view = await mount({ debug: false, debugContent: <p>Danger zone content.</p> })
    try {
      expect(textOf(view.container)).not.toContain('Danger zone content.')
      expect(() => getButton(view.container, 'Debug')).toThrow()
    } finally {
      view.cleanup()
    }
  })

  it('exposes an imperative handle that switches to the Debug tab and focuses it', async () => {
    const ref = createRef<DetailsTabsHandle>()
    const view = await mount({ handleRef: ref, debug: true })
    try {
      expect(tabPanel(view.container, 'debug').getAttribute('hidden')).not.toBeNull()

      await act(async () => ref.current?.showDebugTab())

      expect(tabPanel(view.container, 'debug').getAttribute('hidden')).toBeNull()
      const debugTab = descendants(view.container).find((node) => node.getAttribute('id') === 'details-tab-debug')
      expect(debugTab?.getAttribute('aria-selected')).toBe('true')
    } finally {
      view.cleanup()
    }
  })

  it('showDebugTab() is a no-op when the Debug tab is not rendered (!debug)', async () => {
    const ref = createRef<DetailsTabsHandle>()
    const view = await mount({ handleRef: ref, debug: false })
    try {
      await act(async () => ref.current?.showDebugTab())

      expect(tabPanel(view.container, 'ledger').getAttribute('hidden')).toBeNull()
      expect(() => getButton(view.container, 'Debug')).toThrow()
    } finally {
      view.cleanup()
    }
  })
})
