import { beforeEach, describe, expect, it, vi } from 'vitest'
import React, { act, createRef } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationChildGoal,
  OrchestrationEvidence,
  OrchestrationGoalDetail,
  OrchestrationPlanItemView,
  OrchestrationRunSummary,
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

const goalDetailMock = vi.fn((_projectId: string | null, _goalId: string | undefined): {
  data?: OrchestrationGoalDetail; isLoading: boolean; isError: boolean
} => ({ data: undefined, isLoading: false, isError: false }))

vi.mock('@/api/orchestration', () => ({
  useOrchestrationGoal: (projectId: string | null, goalId: string | undefined) => goalDetailMock(projectId, goalId),
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

import { ChildGoalList, DetailsTabs, PlanItemList, PlanOverview, PlanTab, type DetailsTabsHandle } from './DetailsTabs'

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
  goalDetailMock.mockClear()
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

// ---------------------------------------------------------------------------
// Plan tab
// ---------------------------------------------------------------------------

function run(overrides: Partial<OrchestrationRunSummary> = {}): OrchestrationRunSummary {
  return {
    id: 'run-2', status: 'running', phase: 'planning', cycle_key: null, supersedes_run_id: 'run-1',
    terminal_reason: null, goal_type_snapshot: 'roadmap', planning_mode_snapshot: 'roadmap',
    goal_contract_fingerprint: 'fp-2', started_at: '2026-08-06T08:00:00Z', completed_at: null,
    version_label: 'v2', is_current: true,
    ...overrides,
  }
}

function item(overrides: Partial<OrchestrationPlanItemView> = {}): OrchestrationPlanItemView {
  return {
    key: 'build', unit_type: 'task', title: 'Build the thing', work_function: 'implementation', agent_id: 'agent-1',
    depends_on: [], artifact_ids: [], why_runnable: 'No dependencies', priority: 50, adopt_task_id: null,
    adopted: false, task_id: 'task-build', child_goal_id: null, gate_id: 'gate-build', state: 'started',
    ...overrides,
  }
}

function evidenceRow(overrides: Partial<OrchestrationEvidence> = {}): OrchestrationEvidence {
  return {
    id: 'evidence-1', run_id: 'run-2', gate_id: 'gate-build', source_type: 'task', source_id: null,
    observed_event_id: null, producer_agent_id: null, verdict: 'candidate', evidence_metadata: {},
    evidence_role: 'readiness', review_stage: 'preliminary', acceptance_verdict: 'pending',
    created_at: ISO.started, updated_at: ISO.started,
    ...overrides,
  }
}

function child(overrides: Partial<OrchestrationChildGoal> = {}): OrchestrationChildGoal {
  return {
    id: 'child-1', objective: 'Deliver phase one', status: 'active', goal_type: 'roadmap', goal_type_source: 'parent_auto',
    planning_mode: 'roadmap', roadmap_item_key: 'phase-1', roadmap_version: 3, current_run_id: 'child-run-1',
    current_run_status: 'running', run_version_label: 'v1',
    ...overrides,
  }
}

const planBase = {
  status: 'accepted', artifact_id: 'artifact-abcdef123456', kind: 'delivery' as const, author_agent_id: 'agent-planner',
  planning_version: 3, planning_task_id: 'task-plan', summary: null, scope: null,
}

async function mountPlan(detail: OrchestrationGoalDetail, props: Partial<React.ComponentProps<typeof PlanTab>> = {}) {
  return mountWithTestDom(() => <MemoryRouter><PlanTab projectId="project-1" detail={detail} {...props} /></MemoryRouter>, act)
}

function detailsBySummary(container: Parameters<typeof descendants>[0], summary: string) {
  const found = descendants(container).find((node) => node.tagName === 'DETAILS'
    && node.childNodes.some((c) => 'tagName' in c && (c as typeof node).tagName === 'SUMMARY' && textOf(c).startsWith(`▸ ${summary}`)))
  if (!found) throw new Error(`details not found: ${summary}`)
  return found
}

function summaryOf(details: ReturnType<typeof detailsBySummary>) {
  return details.childNodes.find((c) => 'tagName' in c && (c as typeof details).tagName === 'SUMMARY') as typeof details
}

describe('Plan tab', () => {
  it('labels current and historical runs and plan versions', async () => {
    const runs = [
      run({ id: 'run-1', version_label: 'v1', is_current: false, supersedes_run_id: null, terminal_reason: 'reclassified', status: 'cancelled', goal_type_snapshot: 'outcome' }),
      run(),
    ]
    const view = await mountPlan(goalDetail({ runs, plan: { ...planBase, items: [] }, run: null }))
    try {
      const text = textOf(view.container)
      expect(text).toContain('Run 2 · Current')
      expect(text).toContain('Earlier runs (1)')
      expect(text).toContain('Run 1 · Historical')
      expect(text).toContain('Reclassified to Roadmap')
      expect(text).toContain('v3 · Current')
      expect(text).toContain('2 earlier versions (v1–v2), not shown here.')
      expect(text).toContain('Delivery plan')
      expect(descendants(view.container).some((n) => n.getAttribute('title') === 'artifact-abcdef123456')).toBe(true)
      expect(detailsBySummary(view.container, 'Earlier runs').getAttribute('open')).toBeNull()
    } finally { view.cleanup() }
  })

  it('shows empty, drafting and empty-proposal states', async () => {
    const empty = await mountPlan(goalDetail({ runs: [run()], plan: null, run: null }))
    try { expect(textOf(empty.container)).toContain("No plan yet. Items appear here once the plan author's plan is accepted.") } finally { empty.cleanup() }
    const drafting = await mountPlan(goalDetail({ runs: [run()], plan: { ...planBase, status: 'requested', artifact_id: null, items: [] }, run: null }), { agentRoles: { 'agent-planner': 'Planner' } })
    try { expect(textOf(drafting.container)).toContain('Plan is being drafted by Planner.') } finally { drafting.cleanup() }
    const proposed = await mountPlan(goalDetail({ runs: [run()], plan: { ...planBase, status: 'proposed', items: [] }, run: null }))
    try { expect(textOf(proposed.container)).toContain('No items in this proposal yet.') } finally { proposed.cleanup() }
  })

  it('shows the pending-replan callout and the revision heading', async () => {
    const pendingRun = { plan_state: { pending_replan: { gate_id: 'gate-replan' } } } as unknown as OrchestrationGoalDetail['run']
    const view = await mountPlan(goalDetail({ runs: [run()], run: pendingRun, plan: { ...planBase, items: [item()] } }))
    try { expect(textOf(view.container)).toContain('A revised plan is proposed and awaiting approval. Items below are the current accepted plan.') } finally { view.cleanup() }
    const revision = await mountPlan(goalDetail({ runs: [run()], run: null, plan: { ...planBase, status: 'revision_requested', items: [item()] } }))
    try { expect(textOf(revision.container)).toContain('Current accepted plan (revision requested)') } finally { revision.cleanup() }
  })

  it('renders item cards with dependencies, artifacts, adoption, owner and priority', async () => {
    const plan = { ...planBase, items: [
      item({ key: 'design', title: 'Design it', state: 'started', task_id: null, gate_id: null }),
      item({ key: 'build', depends_on: ['design'], state: 'queued', task_id: null, artifact_ids: ['artifact-zzzzzzzz9999'], adopt_task_id: 'task-old', priority: 80, why_runnable: 'Design finished' }),
    ] }
    const view = await mountPlan(goalDetail({ runs: [run()], run: null, plan }), { taskTitles: { 'task-old': 'Earlier spike' }, agentRoles: { 'agent-1': 'Developer' } })
    try {
      const text = textOf(view.container)
      expect(text).toContain('Preliminary evidence can unblock dependent work but never satisfies a final criterion.')
      expect(text).toContain('Waiting on: Design it')
      expect(text).toContain('Depends onDesign it')
      expect(text).toContain('Reuses taskEarlier spike')
      expect(text).toContain('OwnerDeveloper · implementation')
      expect(text).toContain('Priority 80')
      expect(text).not.toContain('Priority 50')
      expect(text).toContain('Queued')
      expect(descendants(view.container).some((n) => n.getAttribute('title') === 'artifact-zzzzzzzz9999')).toBe(true)
      expect(descendants(view.container).some((n) => n.getAttribute('aria-label') === 'Plan items')).toBe(true)
    } finally { view.cleanup() }
  })

  it('shows why-runnable once dependencies are met', async () => {
    const plan = { ...planBase, items: [
      item({ key: 'design', title: 'Design it', state: 'accepted' }),
      item({ key: 'build', depends_on: ['design'], state: 'queued', why_runnable: 'Design finished' }),
    ] }
    const view = await mountPlan(goalDetail({ runs: [run()], run: null, plan }))
    try { expect(textOf(view.container)).toContain('Why runnable: Design finished') } finally { view.cleanup() }
  })

  it('does not claim released items wait on dependencies and hides empty runnability', async () => {
    const plan = { ...planBase, items: [
      item({ key: 'design', title: 'Design it', state: 'started', task_id: 'task-design', why_runnable: '' }),
      item({ key: 'build', title: 'Build it', depends_on: ['design'], state: 'queued', task_id: 'task-build', why_runnable: 'Preliminary design evidence' }),
      item({ key: 'ship', title: 'Ship it', depends_on: ['design'], state: 'accepted', why_runnable: 'Was runnable' }),
    ] }
    const view = await mountPlan(goalDetail({ runs: [run()], run: null, plan }))
    try {
      const card = (title: string) => descendants(view.container).find((n) => n.tagName === 'LI' && textOf(n).startsWith(title))!
      expect(textOf(card('Build it'))).toContain('Why runnable: Preliminary design evidence')
      expect(textOf(card('Build it'))).not.toContain('Waiting on')
      expect(textOf(card('Ship it'))).not.toContain('Why runnable')
      expect(textOf(card('Ship it'))).not.toContain('Waiting on')
      expect(textOf(card('Design it'))).not.toContain('Why runnable')
    } finally { view.cleanup() }
  })

  it('prefixes disclosures, drops empty report rows and shows the report row', async () => {
    const plan = { ...planBase, items: [item()] }
    const view = await mountPlan(goalDetail({
      runs: [run()], run: null, plan,
      task_reports: [{ task_id: 'task-build', session_id: 's1', summary: null, report: { status: 'done', changes: [], risks: ['Slow'] } }],
    }))
    try {
      const summaries = descendants(view.container).filter((n) => n.tagName === 'SUMMARY').map((n) => textOf(n))
      expect(summaries.length).toBeGreaterThan(1)
      expect(summaries.every((t) => t.startsWith('▸ '))).toBe(true)
      const full = textOf(detailsBySummary(view.container, 'Full report and provenance'))
      expect(full).toContain('StatusDone')
      expect(full).toContain('Slow')
      expect(full).not.toContain('Changes')
      expect(full).not.toContain('Decisions')
      expect(textOf(view.container)).toContain('ReportNo report yet.')
    } finally { view.cleanup() }
  })

  it('Report row shows placeholder for an empty summary', async () => {
    const plan = { ...planBase, items: [item()] }
    const view = await mountPlan(goalDetail({
      runs: [run()], run: null, plan,
      task_reports: [{ task_id: 'task-build', session_id: 's1', summary: '  ', report: { status: 'done', changes: [], risks: [] } }],
    }))
    try {
      expect(textOf(view.container)).toContain('ReportNo report yet.')
    } finally { view.cleanup() }
  })

  it('shows one earlier version singularly and drafting copy from the run plan state', async () => {
    const one = await mountPlan(goalDetail({ runs: [run()], run: null, plan: { ...planBase, planning_version: 2, items: [] } }))
    try { expect(textOf(one.container)).toContain('1 earlier version (v1), not shown here.') } finally { one.cleanup() }
    const drafting = await mountPlan(goalDetail({
      runs: [run()], plan: null, run: { plan_state: { status: 'requested' } } as unknown as OrchestrationGoalDetail['run'],
    }))
    try { expect(textOf(drafting.container)).toContain('Plan is being drafted by the plan author.') } finally { drafting.cleanup() }
  })

  it('never shows a pending-evidence item as Accepted and keeps stage and verdict separate', async () => {
    const plan = { ...planBase, items: [item({ state: 'evidence_pending' })] }
    const view = await mountPlan(goalDetail({ runs: [run()], run: null, plan, evidence: [evidenceRow()] }))
    try {
      const card = descendants(view.container).find((n) => n.tagName === 'LI' && textOf(n).startsWith('Build the thing'))!
      const evidence = descendants(card).find((n) => n.getAttribute('aria-label') === 'Evidence')!
      expect(textOf(evidence)).toContain('Readiness · Preliminary')
      expect(textOf(evidence)).toContain('Pending')
      expect(textOf(evidence)).not.toContain('Accepted')
      expect(textOf(descendants(card)[0]!)).toContain('Waiting for evidence')
      expect(textOf(descendants(card)[0]!)).not.toContain('Accepted')
    } finally { view.cleanup() }
  })

  it('shows the concise summary while the full report stays in a collapsed details as lists', async () => {
    const plan = { ...planBase, items: [item()] }
    const report = {
      status: 'Implemented', changes: ['Added parser'], evidence: ['Test run green'], criterion_progress: { c1: 'half done' },
      decisions: ['Use streams'], risks: ['Slow on big files'], open_questions: ['Which encoding?'], next_step: 'Review', collaboration_need: 'Needs QA',
    }
    const view = await mountPlan(goalDetail({
      runs: [run()], plan, run: { goal_contract_fingerprint: 'fp-at-start', plan_state: {} } as unknown as OrchestrationGoalDetail['run'], contract_fingerprint: 'current-fp-999999',
      task_reports: [{ task_id: 'task-build', session_id: 'session-abcdef12', summary: 'Implemented · next: Review', report }],
    }))
    try {
      const full = detailsBySummary(view.container, 'Full report and provenance')
      expect(full.getAttribute('open')).toBeNull()
      expect(textOf(full)).not.toContain('Implemented · next: Review')
      const card = descendants(view.container).find((n) => n.tagName === 'LI' && textOf(n).startsWith('Build the thing'))!
      expect(textOf(card)).toContain('Implemented · next: Review')
      const fullText = textOf(full)
      for (const expected of ['Added parser', 'Test run green', 'c1: half done', 'Use streams', 'Slow on big files', 'Which encoding?', 'Review', 'Needs QA', 'Run 2', 'Contract at Start']) {
        expect(fullText).toContain(expected)
      }
      expect(fullText).not.toContain('{')
      expect(descendants(full).some((n) => n.getAttribute('title') === 'fp-at-start')).toBe(true)
      expect(descendants(full).some((n) => n.getAttribute('title') === 'current-fp-999999')).toBe(false)
      expect(descendants(full).some((n) => n.tagName === 'UL')).toBe(true)
    } finally { view.cleanup() }
  })

  it('says when there is no report and scopes the agent context to titles', async () => {
    const plan = { ...planBase, items: [
      item({ key: 'design', title: 'Design it', state: 'accepted', task_id: 'task-design', gate_id: 'gate-design' }),
      item({ key: 'build', depends_on: ['design'], adopt_task_id: 'task-old' }),
    ] }
    const goal = { ...goalDetail().goal, objective: 'Ship the details tabs.', success_criteria: [{ key: 'crit-1', description: 'Parser handles big files' }] }
    const view = await mountPlan(goalDetail({
      goal, runs: [run()], run: null, plan,
      evidence: [evidenceRow({ id: 'e2', gate_id: 'gate-design', evidence_role: 'delivery', review_stage: 'final', acceptance_verdict: 'accepted' })],
      task_reports: [{ task_id: 'task-design', session_id: 'session-x', summary: 'Designed', report: { status: 'Designed' } }],
    }), { taskTitles: { 'task-old': 'Earlier spike' } })
    try {
      const cards = descendants(view.container).filter((n) => n.tagName === 'LI' && textOf(n).startsWith('Build the thing'))
      expect(textOf(cards[0]!)).toContain('No report yet.')
      const context = detailsBySummary(cards[0]!, 'Context given to the agent')
      const text = textOf(context)
      expect(text).toContain('Ship the details tabs.')
      expect(text).toContain('Parser handles big files')
      expect(text).toContain('Design it · Delivery · Final')
      expect(text).toContain('Earlier spike')
      expect(text).not.toContain('task-design')
      expect(text).not.toContain('session-x')
      expect(context.getAttribute('open')).toBeNull()
    } finally { view.cleanup() }
  })

  it('renders direct children only and fetches the next level lazily, one level deep', async () => {
    const grandchild = child({ id: 'grand-1', objective: 'Deliver sub-phase', goal_type: 'roadmap', roadmap_version: 1 })
    goalDetailMock.mockImplementation((_p, goalId) => goalId === 'child-1'
      ? { data: goalDetail({ child_goals: [grandchild] }), isLoading: false, isError: false }
      : { data: undefined, isLoading: false, isError: false })
    const view = await mountPlan(goalDetail({
      goal: { ...goalDetail().goal, goal_type: 'roadmap' }, runs: [run()], run: null, plan: { ...planBase, kind: 'roadmap', items: [] },
      child_goals: [child(), child({ id: 'child-2', objective: 'Ship docs', goal_type: 'outcome', current_run_id: null })],
    }))
    try {
      let text = textOf(view.container)
      expect(text).toContain('Child goals')
      expect(text).toContain('Deliver phase one')
      expect(text).toContain('Run 1')
      expect(text).toContain('Plan v3')
      expect(text).not.toContain('Historical')
      expect(text).not.toContain('Deliver sub-phase')
      expect(goalDetailMock.mock.calls.every(([, goalId]) => goalId === undefined)).toBe(true)
      expect(descendants(view.container).filter((n) => n.tagName === 'SUMMARY' && textOf(n) === '▸ Next level')).toHaveLength(1)

      await act(async () => summaryOf(detailsBySummary(view.container, 'Next level')).click())
      expect(goalDetailMock).toHaveBeenCalledWith('project-1', 'child-1')
      text = textOf(view.container)
      expect(text).toContain('Deliver sub-phase')
      // grandchildren do not offer another level
      expect(descendants(view.container).filter((n) => n.tagName === 'SUMMARY' && textOf(n) === '▸ Next level')).toHaveLength(1)
      const links = descendants(view.container).filter((n) => n.tagName === 'A').map((n) => n.getAttribute('href'))
      expect(links).toContain('/orchestration/child-1')
    } finally { view.cleanup() }
  })

  it('shows an empty child list and exports the building blocks', async () => {
    expect(typeof PlanOverview).toBe('function')
    expect(typeof PlanItemList).toBe('function')
    const view = await mountWithTestDom(() => <MemoryRouter><ChildGoalList projectId="project-1" goals={[]} /></MemoryRouter>, act)
    try { expect(textOf(view.container)).toContain('No child goals released yet.') } finally { view.cleanup() }
  })

  it('mounts the plan inside the Plan tab', async () => {
    const view = await mountWithTestDom(() => <MemoryRouter><Harness detail={goalDetail({ runs: [run()], plan: { ...planBase, items: [item()] }, run: null })} /></MemoryRouter>, act)
    try {
      expect(textOf(tabPanel(view.container, 'plan'))).toContain('Build the thing')
      expect(textOf(tabPanel(view.container, 'plan'))).toContain('Goal-plan content.')
    } finally { view.cleanup() }
  })
})
