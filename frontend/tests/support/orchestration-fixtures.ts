// Shared goal/run fixture + live-route mock harness for the orchestration
// goal-detail page. Extracted from orchestration.spec.ts so other spec files
// can reuse it — Playwright forbids importing one test file from another.
import { expect, type Page } from '@playwright/test'
import type {
  OrchestrationAgentReviewRecord,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationMemoryOverview,
  OrchestrationMemorySection,
  OrchestrationProcessRunRecord,
  OrchestrationConversationInvestigation,
  OrchestrationConversationFeedback,
  OrchestrationSteeringLedger,
  OrchestrationSteeringProposal,
  OrchestrationSteeringRequest,
  OrchestrationSupervision,
  OrchestrationSupervisionCondition,
  OrchestrationWarningRecord,
} from '../../src/lib/types'
import { gotoDashboard, selectSeedProject } from './pages/app'

export const goal = {
  id: 'goal-release',
  project_id: 'project-route-value',
  objective: 'Ship an evidence-backed release',
  success_criteria: [
    { key: 'release-ready', description: 'Release is independently validated.' },
    { key: 'secondary-review', description: 'A second review path remains operable.' },
  ],
  constraints: { owned_files: ['frontend/src'] },
  budget: { max_tokens: 20000 },
  status: 'blocked',
  weight: 'substantial',
  weight_overridden_by: 'human:user-1',
  manager_agent_id: 'agent-manager',
  manager_user_id: null,
  authority_model: 'human_approval',
  created_by_user_id: null,
  created_at: '2026-07-15T09:00:00Z',
  updated_at: '2026-07-15T10:00:00Z',
  needs_you_count: 0,
}

export const run = {
  id: 'run-release',
  goal_id: goal.id,
  status: 'blocked',
  event_cursor: 42,
  plan_state: {
    status: 'accepted',
    planning_task_id: 'task-plan',
    accepted_artifact_id: 'artifact-plan',
    expanded_items: [{
      plan_item_id: 'validate-release',
      work_function: 'validation',
      task_id: 'task-validation',
      gate_id: 'gate-release',
    }],
  },
  active_blockers: [
    { kind: 'task_blocked', reason: 'Missing reviewer requirements' },
    { kind: 'warning:warning-blocker', warning_id: 'warning-blocker', reason: 'Blocker: legal approval' },
  ],
  budget_state: {},
  retry_state: {},
  started_at: '2026-07-15T09:00:00Z',
  completed_at: null,
  created_at: '2026-07-15T09:00:00Z',
  updated_at: '2026-07-15T10:00:00Z',
}

export const acceptedSupervision: OrchestrationSupervision = {
  condition: 'needs_you',
  operation: 'Choose deployment direction',
  next_action: 'Answer pending direction',
  rationale: 'The release needs an explicit operator decision.',
  criterion: { key: 'release-ready', description: 'Release is independently validated.', status: 'accepted' },
  verified_progress: [{ criterion_key: 'build', status: 'verified', summary: 'Build and unit checks passed.' }],
  useful_learning: [{ learning_key: 'cache-miss', status: 'accepted', summary: 'A clean cache is required for release evidence.' }],
  accepted_evidence: [{
    id: 'evidence-accepted', run_id: run.id, gate_id: 'gate-release', source_type: 'verification',
    source_id: 'report-accepted', observed_event_id: 'event-accepted', producer_agent_id: 'agent-reviewer',
    verdict: 'accepted', evidence_metadata: { summary: 'Independent validation accepted.', artifact_id: 'artifact-release', secret: 'must-not-render' },
    created_at: '2026-07-15T10:01:00Z', updated_at: '2026-07-15T10:01:00Z',
  }],
  workers: [{
    session_id: 'session-worker', task_id: 'task-validation', agent_id: 'agent-reviewer', runner_id: 'runner-validation',
    task_title: 'Validate the release', session_status: 'running', observed_liveness: 'live', observed_at: '2026-07-15T10:02:00Z',
  }],
  waits: [{
    id: 'wait-release', owner: { type: 'session', id: 'session-worker' }, event: 'session.completed',
    matcher: { session_id: 'session-worker', secret: 'must-not-render' }, due_recheck_at: '2026-07-15T10:10:00Z',
    fallback: { reason: 'Escalate if validation stalls.', action_type: 'attention', expected_result: 'Validation outcome', secret: 'must-not-render' },
  }],
  recovery_history: [{
    session_id: 'session-worker', classification: 'live', disposition: 'adopted', backend_observation: 'active',
    assessed_at: '2026-07-15T10:02:00Z', action_id: 'action-request-verification', wait_id: 'wait-release',
  }],
  pending_direction: {
    id: 'direction-release', goal_id: goal.id, run_id: run.id, decision_key: 'deployment-direction', title: 'Approve deployment direction',
    status: 'pending', authority: 'human', authority_agent_id: null, source_process_run_id: null,
    question: 'Should this release proceed to deployment?', context: 'Independent validation is accepted.',
    options: [{ key: 'approve', label: 'Approve deployment' }], recommendation: 'approve', consequences: 'The deployment decision is durable.',
    selected_option: null, reason: null, decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: 'gate-release', related_action_id: 'action-request-verification',
    asked_at: '2026-07-15T10:03:00Z', decided_at: null, created_at: '2026-07-15T10:03:00Z', updated_at: '2026-07-15T10:03:00Z',
  },
  budget: { consumed: { max_tokens: '30' }, committed: { max_tokens: '2' }, reserved: { max_tokens: '1' }, remaining: { max_tokens: '167' } },
  transition: { key: 'pending-direction:release', kind: 'pending_direction', message: 'Approve deployment direction', occurred_at: '2026-07-15T10:03:00Z' },
}

export const supervisionPresentation: Record<OrchestrationSupervisionCondition, Pick<OrchestrationSupervision, 'operation' | 'next_action' | 'rationale'>> = {
  working: { operation: 'Validation is running', next_action: 'Await worker result', rationale: 'An owned worker is active.' },
  waiting: { operation: 'Awaiting validation result', next_action: 'Await session.completed', rationale: 'A durable wait is open.' },
  needs_you: { operation: 'Choose deployment direction', next_action: 'Answer pending direction', rationale: 'The release needs an explicit operator decision.' },
  needs_attention: { operation: 'Resolve release blocker', next_action: 'Resolve blocker', rationale: 'A durable blocker requires attention.' },
  paused: { operation: 'Paused', next_action: 'Resume goal', rationale: 'Goal is paused.' },
  stopped: { operation: 'Stopped', next_action: 'No action', rationale: 'Continuous policy stopped.' },
  cancelled: { operation: 'Cancelled', next_action: 'No action', rationale: 'Goal cancelled.' },
  completed: { operation: 'Completed', next_action: 'No action', rationale: 'Goal completed.' },
}

export function supervisionFor(
  condition: OrchestrationSupervisionCondition,
  transition: OrchestrationSupervision['transition'] = acceptedSupervision.transition,
): OrchestrationSupervision {
  return {
    ...acceptedSupervision,
    condition,
    ...supervisionPresentation[condition],
    pending_direction: condition === 'needs_you' ? acceptedSupervision.pending_direction : null,
    transition,
  }
}

export function detail(
  goalStatus = goal.status,
  runStatus = run.status,
  supervision: OrchestrationSupervision | null = null,
) {
  return {
    goal: { ...goal, status: goalStatus },
    run: { ...run, status: runStatus },
    decisions_count: 1,
    decisions: [{
      id: 'decision-request-verification', run_id: run.id, decision_type: 'request_verification',
      input_snapshot: {}, llm_output: {}, parsed_decision: {}, validator_status: 'accepted',
      rejection_reason: null, reason: 'Request independent validation',
      created_at: '2026-07-15T09:30:00Z', updated_at: '2026-07-15T09:30:00Z',
    }],
    actions_count: 2,
    actions: [{
      id: 'action-request-verification', run_id: run.id, decision_id: 'decision-request-verification',
      idempotency_key: 'run:phase18:request_verification', action_type: 'request_verification',
      request: { reason: 'Collect fresh proof.' }, target_type: 'task', target_id: 'task-validation',
      status: 'completed', error: null,
      created_at: '2026-07-15T09:31:00Z', updated_at: '2026-07-15T09:31:00Z',
    }, {
      id: 'action-schedule-meeting', run_id: run.id, decision_id: 'decision-request-verification',
      idempotency_key: 'run:phase18:schedule_meeting', action_type: 'schedule_meeting',
      request: { reason: 'Resolve contradictory validation outputs.' },
      target_type: 'meeting', target_id: 'meeting-validation',
      status: 'completed', error: null,
      created_at: '2026-07-15T09:32:00Z', updated_at: '2026-07-15T09:32:00Z',
    }],
    gates_count: 2,
    gates: [{
      id: 'gate-release', run_id: run.id, success_criterion_key: 'release-ready',
      gate_type: 'validation_passed', required_evidence: { min_count: 1 }, status: 'open',
      failure_reason: null, created_at: '2026-07-15T09:10:00Z',
      updated_at: '2026-07-15T09:10:00Z', accepted_at: null, failed_at: null,
    }, {
      id: 'gate-secondary-review', run_id: run.id, success_criterion_key: 'secondary-review',
      gate_type: 'review_complete', required_evidence: { min_count: 1 }, status: 'open',
      failure_reason: null, created_at: '2026-07-15T09:11:00Z',
      updated_at: '2026-07-15T09:11:00Z', accepted_at: null, failed_at: null,
    }],
    evidence_count: 1,
    evidence: [{
      id: 'evidence-review', run_id: run.id, gate_id: 'gate-release', source_type: 'review',
      source_id: 'review-1', observed_event_id: null, producer_agent_id: 'agent-reviewer',
      verdict: 'candidate', evidence_metadata: { summary: 'Review is pending independent confirmation.' },
      created_at: '2026-07-15T09:20:00Z', updated_at: '2026-07-15T09:20:00Z',
    }],
    agent_suggestions_count: 1,
    agent_suggestions: [{
      id: 'suggestion-validation', run_id: run.id, missing_work_function: 'validation',
      reason: 'No independent validator is active.', suggested_role: 'validator',
      suggested_capabilities: ['validation', 'testing'], suggested_adapter_type: 'api',
      suggested_model: 'gpt-4o-mini', suggested_system_prompt_outline: 'Validate and report evidence.',
      status: 'open', created_at: '2026-07-15T09:40:00Z', updated_at: '2026-07-15T09:40:00Z',
    }],
    timeline: [],
    supervision,
  }
}

export type BaselineMutation = 'answer' | 'skip' | 'acknowledge' | 'resolve' | 'run' | 'rerun'
const BASELINE_CHECKPOINT_MAX_QUESTIONS = 5

export function baselineProcess(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-goal-definition', goal_id: goal.id, run_id: run.id, process_type: 'goal_definition',
    process_version: 1, status: 'waiting_decision', trigger_reason: 'A human checkpoint is required.',
    input_snapshot: {}, outputs: {}, skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: '2026-07-28T09:00:00Z', completed_at: null,
    created_at: '2026-07-28T09:00:00Z', updated_at: '2026-07-28T09:00:00Z',
    ...overrides,
  }
}

export function baselineWarning(overrides: Partial<OrchestrationWarningRecord> = {}): OrchestrationWarningRecord {
  return {
    id: 'warning-acknowledge', goal_id: goal.id, run_id: run.id, warning_type: 'coverage_gap', severity: 'warning',
    message: 'Coverage needs attention.', source_process_run_id: null, related_gate_id: null, related_action_id: null,
    related_agent_id: null, source_agent_review_id: null, related_authority_decision_id: null,
    acknowledged_by: null, acknowledged_at: null, active: true, resolved_by: null, resolved_reason: null,
    resolved_at: null, blocks_completion: true, created_at: '2026-07-28T09:00:00Z',
    updated_at: '2026-07-28T09:00:00Z',
    ...overrides,
  }
}

export function baselineDecision(
  overrides: Partial<OrchestrationAuthorityDecisionRecord> = {},
): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'decision-human-object', goal_id: goal.id, run_id: run.id, decision_key: 'release-policy',
    title: 'Choose the release policy', status: 'pending', authority: 'human', authority_agent_id: null,
    source_process_run_id: 'process-goal-definition', question: 'Which policy should govern this release?',
    context: 'The release needs an explicit operator decision.',
    options: [{ key: 'approve_with_conditions', label: 'Approve with conditions' }, { key: 'defer', label: 'Defer' }],
    recommendation: 'defer', consequences: 'The selected policy is recorded for the release.', selected_option: null,
    reason: null, decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: '2026-07-28T09:00:00Z', decided_at: null, created_at: '2026-07-28T09:00:00Z',
    updated_at: '2026-07-28T09:00:00Z',
    ...overrides,
  }
}

export function baselineCheckpoint(decisions: readonly OrchestrationAuthorityDecisionRecord[]) {
  const pendingHuman = decisions
    .filter((item) => item.status === 'pending' && item.authority === 'human')
    .sort((left, right) => left.asked_at.localeCompare(right.asked_at)
      || left.created_at.localeCompare(right.created_at)
      || left.id.localeCompare(right.id))
  return {
    items: pendingHuman.slice(0, BASELINE_CHECKPOINT_MAX_QUESTIONS),
    deferred_count: Math.max(0, pendingHuman.length - BASELINE_CHECKPOINT_MAX_QUESTIONS),
    max_questions: BASELINE_CHECKPOINT_MAX_QUESTIONS,
  }
}

export function baselineDashboardState(): {
  processes: OrchestrationProcessRunRecord[]
  warnings: OrchestrationWarningRecord[]
  decisions: OrchestrationAuthorityDecisionRecord[]
  reviews: OrchestrationAgentReviewRecord[]
  memory: OrchestrationMemoryOverview
  memorySections: Record<string, OrchestrationMemorySection>
} {
  const introduction: OrchestrationMemorySection = {
    id: 'memory-introduction', project_id: goal.project_id, goal_id: goal.id, run_id: run.id,
    section_key: 'introduction', title: 'Introduction', section_type: 'markdown', body: 'Operator preface.',
    summary: 'What the operator needs to know.', always_load: true, toc_order: 1, created_by: 'orchestrator',
    created_from_event_id: null, updated_from_event_id: null, created_at: '2026-07-28T09:00:00Z',
    updated_at: '2026-07-28T10:00:00Z',
  }
  const runbook: OrchestrationMemorySection = {
    id: 'memory-runbook', project_id: goal.project_id, goal_id: goal.id, run_id: run.id,
    section_key: 'runbook', title: 'Runbook', section_type: 'markdown', body: 'Confirm the legal approval before closing out.',
    summary: 'Manual procedure.', always_load: false, toc_order: 2, created_by: 'orchestrator',
    created_from_event_id: null, updated_from_event_id: null, created_at: '2026-07-28T09:00:00Z',
    updated_at: '2026-07-28T11:00:00Z',
  }
  const memory: OrchestrationMemoryOverview = {
    toc: [
      { section_key: 'introduction', title: 'Introduction', summary: introduction.summary, section_type: 'markdown', always_load: true, toc_order: 1, updated_at: introduction.updated_at },
      { section_key: 'runbook', title: 'Runbook', summary: runbook.summary, section_type: 'markdown', always_load: false, toc_order: 2, updated_at: runbook.updated_at },
    ],
    always_loaded: [introduction],
    preface: {
      objective: goal.objective, goal_status: 'blocked', goal_weight: 'substantial', run_status: 'blocked',
      current_process: { process_type: 'team_hierarchy', status: 'running' }, manager: 'agent-manager', hierarchy: 'Proposed',
      constraints: 'No unverified manager assignment.', active_warnings: [{ severity: 'blocker', warning_type: 'legal_approval', message: 'Blocker: legal approval', acknowledged: false }],
      recent_decisions: [], open_blockers: ['Blocker: legal approval'],
      skipped_processes: [{ process_type: 'manager_selection', skipped_by: 'human:user-1' }], introduction: 'Operator preface.',
      always_loaded: [{ section_key: 'introduction', summary: introduction.summary }], toc: [{ section_key: 'runbook', title: 'Runbook' }],
    },
  }
  return {
    processes: [
      baselineProcess(),
      baselineProcess({
        id: 'process-team-hierarchy', process_type: 'team_hierarchy', status: 'running',
        trigger_reason: 'Team fit is under review.',
        outputs: {
          role_to_agent: { researcher: 'agent-researcher' },
          hierarchy: { manager: { kind: 'agent', id: 'agent-manager' } },
          missing_work_functions: ['independent_verifier'], weak_fits: ['agent-researcher'],
          independent_verification: { required: true, possible: false, verifier_agent_ids: [] },
        },
      }),
      baselineProcess({
        id: 'process-manager-selection', process_type: 'manager_selection', status: 'skipped',
        skipped_by: 'human:user-1', override_reason: 'Operator chose a manual manager.', completed_at: '2026-07-28T10:00:00Z',
      }),
      baselineProcess({
        id: 'process-goal-closeout', process_type: 'goal_closeout', status: 'completed',
        outputs: { mode: 'completion', gates: { closeout_completed: false }, completion_authorized: false },
        completed_at: '2026-07-28T10:00:00Z',
      }),
      baselineProcess({ id: 'process-superseded', superseded_by_id: 'process-team-hierarchy' }),
    ],
    warnings: [
      baselineWarning({ id: 'warning-manager-skip', message: 'Manager selection was skipped', source_process_run_id: 'process-manager-selection' }),
      baselineWarning({ id: 'warning-acknowledged', message: 'An acknowledged risk remains active', acknowledged_by: 'human:user-1', acknowledged_at: '2026-07-28T10:00:00Z', blocks_completion: false }),
      baselineWarning({ id: 'warning-blocker', warning_type: 'legal_approval', severity: 'blocker', message: 'Blocker: legal approval', blocks_completion: true }),
      baselineWarning({ id: 'warning-hard-stop', warning_type: 'independent_verification', severity: 'hard_stop', message: 'Hard stop: independent verification unavailable' }),
      baselineWarning({ id: 'warning-resolved', message: 'Resolved concern', active: false, resolved_by: 'human:user-1', resolved_reason: 'Evidence supplied.', resolved_at: '2026-07-28T11:00:00Z', source_agent_review_id: 'review-deleted-agent', blocks_completion: false }),
    ],
    decisions: [
      baselineDecision(),
      baselineDecision({ id: 'decision-human-2', decision_key: 'verification-depth', title: 'Choose verification depth', asked_at: '2026-07-28T10:00:00Z', created_at: '2026-07-28T10:00:00Z' }),
      baselineDecision({ id: 'decision-human-3', decision_key: 'evidence-source', title: 'Choose evidence source', asked_at: '2026-07-28T11:00:00Z', created_at: '2026-07-28T11:00:00Z' }),
      baselineDecision({ id: 'decision-human-4', decision_key: 'closeout-timing', title: 'Choose closeout timing', asked_at: '2026-07-28T12:00:00Z', created_at: '2026-07-28T12:00:00Z' }),
      baselineDecision({ id: 'decision-human-5', decision_key: 'rollout-scope', title: 'Choose rollout scope', asked_at: '2026-07-28T13:00:00Z', created_at: '2026-07-28T13:00:00Z' }),
      baselineDecision({ id: 'decision-human-deferred', decision_key: 'follow-up-owner', title: 'Choose follow-up owner', asked_at: '2026-07-28T14:00:00Z', created_at: '2026-07-28T14:00:00Z' }),
      baselineDecision({
        id: 'decision-manager', decision_key: 'manager-policy', title: 'Manager decision', authority: 'manager',
        authority_agent_id: 'agent-manager', question: 'Choose the review policy.',
      }),
      baselineDecision({
        id: 'decision-answered', decision_key: 'scope-choice', title: 'Scope choice', status: 'answered',
        selected_option: 'manual', reason: 'Operator judgement.', decided_by_user_id: 'human:user-1',
        decided_at: '2026-07-28T11:00:00Z', overrides_recommendation: true,
      }),
    ],
    reviews: [{
      id: 'review-deleted-agent', goal_id: goal.id, run_id: run.id, agent_id: null, source_process_run_id: 'process-team-hierarchy',
      review_context: 'Review hierarchy fit.', proposed_work_functions: [],
      definition_snapshot: {
        description: 'deleted agent',
        adapter_type: 'api',
        cli_runtime: null,
        capabilities: ['review'],
      },
      fit_summary: 'Suitable with an independent verifier.',
      strengths: ['Independent review'], risks: ['Independent verification unavailable'], recommended_changes: ['Assign an independent verifier.'],
      approved_for_work_functions: ['review'], created_at: '2026-07-28T09:00:00Z', updated_at: '2026-07-28T09:00:00Z',
    }],
    memory,
    memorySections: { runbook },
  }
}

export async function interceptOrchestration(
  page: Page,
  {
    deferOverrideRejection = false,
    deferLifecycle = [],
    rejectLifecycle,
    deferBaseline = [],
    activeBlockers,
    decisions,
    investigation = false,
    steering = false,
    steeringEligibility = 'active',
    proposal = false,
    supervision,
    warnings,
  }: {
    deferOverrideRejection?: boolean
    deferLifecycle?: Array<'pause' | 'resume'>
    rejectLifecycle?: 'pause' | 'resume'
    deferBaseline?: BaselineMutation[]
    activeBlockers?: typeof run.active_blockers
    decisions?: OrchestrationAuthorityDecisionRecord[]
    investigation?: boolean
    steering?: boolean
    steeringEligibility?: OrchestrationSteeringLedger['eligibility']
    proposal?: boolean
    supervision?: OrchestrationSupervision | null
    warnings?: OrchestrationWarningRecord[]
  } = {},
) {
  function withActiveBlockers(detailState: ReturnType<typeof detail>): ReturnType<typeof detail> {
    if (activeBlockers === undefined || detailState.run === null) return detailState
    return { ...detailState, run: { ...detailState.run, active_blockers: activeBlockers } }
  }
  let state = withActiveBlockers(detail(goal.status, run.status, supervision))
  let detailGets = 0
  let conversationGets = 0
  const completedConversationResponses = new Set<string>()
  let investigationCompleted = false
  let conversationRequest: { client_request_id: string; content: string } | null = null
  const conversationRequests: Array<{ client_request_id: string; content: string }> = []
  const conversationFeedback = new Map<string, OrchestrationConversationFeedback>()
  let failNextFeedback = false
  let proposalStatus: OrchestrationSteeringProposal['status'] = 'proposed'
  const steeringLedger: OrchestrationSteeringLedger = {
    enabled: steering,
    eligibility: steeringEligibility,
    eligibility_reason: steeringEligibility === 'active' || steeringEligibility === 'paused' ? null : 'Steering is unavailable for this goal state.',
    inbox_version: 0,
    direction_version: 0,
    requests: [],
    proposals: [],
  }
  const scenarioRequests: Array<{ method: string; pathname: string }> = []
  const conversationHistories: Array<{ items: unknown[]; total: number; omitted: number; allowance: { enabled: boolean; limit: number; used: number; remaining: number } }> = []
  const conversationAnswer = 'The release is blocked pending independent validation.'
  const steeringProposal = (): OrchestrationSteeringProposal => ({
    proposal_id: 'proposal-steered', response_id: 'conversation-response-1', status: proposalStatus,
    directive: 'Prioritize validation before other unstarted work.', target_type: 'goal', target_id: goal.id,
    scope: 'run', lifetime: 'remaining_current_run', impact_summary: 'Validation runs before other unstarted work.',
    dismissed_at: proposalStatus === 'dismissed' ? '2026-09-16T09:00:03Z' : null,
    promoted_request_id: proposalStatus === 'promoted' ? 'request-steered' : null,
    created_at: '2026-09-16T09:00:01Z', updated_at: '2026-09-16T09:00:03Z',
  })
  const conversationSourceReference = `huddleroom/services/${'release-validation-source-'.repeat(8)}.py#L10-L14`
  page.on('request', (request) => {
    const url = new URL(request.url())
    if (url.pathname.startsWith('/api/')) scenarioRequests.push({ method: request.method(), pathname: url.pathname })
  })
  const conversationInvestigation = (): OrchestrationConversationInvestigation & { private_provider_payload: string } => ({
    investigation_id: 'conversation-investigation-1',
    status: investigationCompleted ? 'completed' : 'running',
    objective: `Inspect the release validation state ${'release-validation-boundary-'.repeat(16)}`,
    attempt_count: 1, repair_count: 0, retry_count: 0,
    sources: [{
      reference: conversationSourceReference,
      operation: 'read', status: 'included', freshness_at: '2026-09-14T09:00:01Z', truncated: false,
    }],
    report: investigationCompleted ? {
      findings: `The validation gate is still pending. ${'independent-validation-required-'.repeat(14)}`,
      uncertainty: 'No independent validator output was in scope.',
      sources: [conversationSourceReference],
    } : null,
    error: null,
    started_at: '2026-09-14T09:00:01Z', deadline_at: '2026-09-14T09:02:01Z',
    finished_at: investigationCompleted ? '2026-09-14T09:00:03Z' : null,
    created_at: '2026-09-14T09:00:01Z', updated_at: '2026-09-14T09:00:03Z',
    private_provider_payload: 'must-not-render',
  })
  const conversationTurn = (request: NonNullable<typeof conversationRequest>, sequence: number) => {
    const responseId = `conversation-response-${sequence}`
    const turnInvestigation = investigation ? conversationInvestigation() : null
    const completed = turnInvestigation?.status === 'completed' || completedConversationResponses.has(responseId)
    const answer = turnInvestigation?.report?.findings ?? (completed ? conversationAnswer : null)
    const feedback = conversationFeedback.get(responseId) ?? null
    const actorId = 'human:operator'
    return {
      ...(turnInvestigation ? { investigation: turnInvestigation } : {}),
      ...(proposal && conversationRequest ? { proposed_steering: steeringProposal() } : {}),
      message_id: `conversation-message-${sequence}`, response_id: responseId,
      client_request_id: request.client_request_id, sequence, actor_id: actorId,
      content: request.content, message_created_at: '2026-09-14T09:00:00Z',
      status: completed ? 'completed' : 'running', run_id: run.id,
      answer, error: null,
      started_at: '2026-09-14T09:00:01Z', deadline_at: '2026-09-14T09:02:01Z',
      finished_at: completed ? '2026-09-14T09:00:02Z' : null,
      created_at: '2026-09-14T09:00:00Z', updated_at: '2026-09-14T09:00:02Z', context_version: 'safe-context-v1',
      context_manifest: {
        run_id: run.id, truncated: false, excluded_categories: ['provider_payload', 'raw_dossier'],
        sources: [{ source: 'goal', status: 'included', freshness_at: '2026-09-14T09:00:00Z', available: 1, included: 1, omitted: 0, truncated: false, references: ['goal-release'] }],
      },
      feedback,
      feedback_eligible: actorId === 'human:operator' && completed && Boolean(answer?.trim()) && feedback === null,
      dossier: 'must-not-render', provider_payload: 'must-not-render', reservation: { reserved_tokens: 999 },
    }
  }
  const baselineState = baselineDashboardState()
  if (decisions !== undefined) baselineState.decisions = decisions
  if (warnings !== undefined) baselineState.warnings = warnings
  let rejectNextCancel = true
  let rejectNextOverride = true
  const requests = { pause: 0, cancel: 0, override: 0 }
  let releaseCancelRejection = () => {}
  const cancelRejection = new Promise<void>((resolve) => {
    releaseCancelRejection = resolve
  })
  let releaseOverrideRejection = () => {}
  const overrideRejection = new Promise<void>((resolve) => {
    releaseOverrideRejection = resolve
  })
  const lifecycleReleases = {
    pause: () => {},
    resume: () => {},
  }
  const lifecycleResponses = {
    pause: new Promise<void>((resolve) => { lifecycleReleases.pause = resolve }),
    resume: new Promise<void>((resolve) => { lifecycleReleases.resume = resolve }),
  }
  const baselineRequests = { answer: 0, skip: 0, acknowledge: 0, resolve: 0, run: 0, rerun: 0 }
  const baselineReleases: Record<BaselineMutation, () => void> = {
    answer: () => {}, skip: () => {}, acknowledge: () => {}, resolve: () => {}, run: () => {}, rerun: () => {},
  }
  const baselineResponses: Record<BaselineMutation, Promise<void>> = {
    answer: new Promise<void>((resolve) => { baselineReleases.answer = resolve }),
    skip: new Promise<void>((resolve) => { baselineReleases.skip = resolve }),
    acknowledge: new Promise<void>((resolve) => { baselineReleases.acknowledge = resolve }),
    resolve: new Promise<void>((resolve) => { baselineReleases.resolve = resolve }),
    run: new Promise<void>((resolve) => { baselineReleases.run = resolve }),
    rerun: new Promise<void>((resolve) => { baselineReleases.rerun = resolve }),
  }

  async function deferBaselineMutation(kind: BaselineMutation) {
    baselineRequests[kind] += 1
    if (deferBaseline.includes(kind)) await baselineResponses[kind]
  }
  await page.route('**/api/v1/projects/*/tasks**', async (route) => {
    if (!new URL(route.request().url()).pathname.endsWith('/tasks')) {
      await route.fallback()
      return
    }
    await route.fulfill({
      json: {
        items: [{
          id: 'task-validation', project_id: goal.project_id, title: 'Validate the release',
          priority: 1, status: 'in_progress',
          metadata: { orchestration: { run_id: run.id, work_function: 'validation' } },
          created_at: '2026-07-15T09:00:00Z', updated_at: '2026-07-15T09:20:00Z',
        }],
        next_cursor: null,
      },
    })
  })
  await page.route('**/api/v1/projects/*/orchestration/goals**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const suffix = url.pathname.split('/orchestration/goals')[1]
    if (request.method() === 'GET' && (suffix === '' || suffix === '/')) {
      await route.fulfill({ json: { items: [state.goal], next_cursor: null } })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}`) {
      detailGets += 1
      await route.fulfill({ json: state })
      return
    }
    if (suffix === `/${goal.id}/conversation`) {
      if (request.method() === 'GET') {
        conversationGets += 1
        const history = { items: conversationRequests.map((item, index) => conversationTurn(item, index + 1)), total: conversationRequests.length, omitted: 0, allowance: { enabled: true, limit: 500, used: 200, remaining: 300 }, steering: steeringLedger }
        conversationHistories.push(history)
        await route.fulfill({ json: history })
        return
      }
      if (request.method() === 'POST') {
        conversationRequest = await request.postDataJSON() as NonNullable<typeof conversationRequest>
        conversationRequests.push(conversationRequest)
        await route.fulfill({ json: conversationTurn(conversationRequest, conversationRequests.length) })
        return
      }
    }
    const feedback = suffix.match(new RegExp(`^/${goal.id}/conversation/([^/]+)/feedback$`))
    if (request.method() === 'PUT' && feedback) {
      if (failNextFeedback) {
        failNextFeedback = false
        await route.fulfill({ status: 503, json: { detail: 'Feedback was not recorded.' } })
        return
      }
      const payload = await request.postDataJSON() as { rating: OrchestrationConversationFeedback['rating']; reason: OrchestrationConversationFeedback['reason'] }
      const responseId = decodeURIComponent(feedback[1])
      const recordedFeedback: OrchestrationConversationFeedback = {
        feedback_id: `feedback-${responseId}`, rating: payload.rating, reason: payload.reason, created_at: '2026-09-16T09:00:00Z',
      }
      conversationFeedback.set(responseId, recordedFeedback)
      await route.fulfill({ json: recordedFeedback })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/conversation/steering`) {
      const payload = await request.postDataJSON() as {
        client_request_id: string; directive: string; target_type: OrchestrationSteeringRequest['target_type']; target_id: string
        scope: OrchestrationSteeringRequest['scope']; lifetime: OrchestrationSteeringRequest['lifetime']; impact_summary: string
        source_proposal_id: string | null; supersedes_request_id: string | null
      }
      expect(Object.keys(payload)).toEqual(['client_request_id', 'directive', 'target_type', 'target_id', 'scope', 'lifetime', 'impact_summary', 'source_proposal_id', 'supersedes_request_id'])
      expect(payload.client_request_id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
      const applied = payload.supersedes_request_id !== null
      const overlaps = !applied && steeringLedger.requests.some((item) => item.status === 'applied')
      if (applied) {
        const prior = steeringLedger.requests.find((item) => item.request_id === payload.supersedes_request_id)
        if (prior) {
          prior.status = 'superseded'; prior.reason_code = 'superseded_by_explicit_request'; prior.finished_at = '2026-09-16T09:00:04Z'
          prior.transitions.push({ status: 'superseded', reason_code: prior.reason_code, actor: 'human:operator', created_at: prior.finished_at })
        }
      }
      const status: OrchestrationSteeringRequest['status'] = applied ? 'applied' : overlaps ? 'needs_clarification' : 'pending'
      const requestRecord: OrchestrationSteeringRequest = {
        request_id: `request-steered-${steeringLedger.requests.length + 1}`, client_request_id: payload.client_request_id,
        sequence: steeringLedger.requests.length + 1, directive: payload.directive, target_type: payload.target_type, target_id: payload.target_id,
        scope: payload.scope, lifetime: payload.lifetime, impact_summary: payload.impact_summary, source_proposal_id: payload.source_proposal_id,
        supersedes_request_id: payload.supersedes_request_id, status, reason_code: status === 'pending' ? 'submitted' : status,
        submitted_at: '2026-09-16T09:00:02Z', considered_at: status === 'pending' ? null : '2026-09-16T09:00:04Z',
        finished_at: status === 'pending' ? null : '2026-09-16T09:00:04Z', updated_at: '2026-09-16T09:00:04Z',
        transitions: [{ status, reason_code: status === 'pending' ? 'submitted' : status, actor: 'human:operator', created_at: '2026-09-16T09:00:02Z' }], result_action_ids: [],
      }
      steeringLedger.requests.push(requestRecord)
      steeringLedger.inbox_version += 1
      if (status === 'applied') steeringLedger.direction_version += 1
      if (payload.source_proposal_id === 'proposal-steered') proposalStatus = 'promoted'
      await route.fulfill({ json: requestRecord })
      return
    }
    const withdraw = suffix.match(new RegExp(`^/${goal.id}/conversation/steering/([^/]+)/withdraw$`))
    if (request.method() === 'POST' && withdraw) {
      const requestRecord = steeringLedger.requests.find((item) => item.request_id === decodeURIComponent(withdraw[1]))
      if (!requestRecord || requestRecord.status !== 'pending') {
        await route.fulfill({ status: 409, json: { detail: 'Only pending requests may be withdrawn.' } })
        return
      }
      requestRecord.status = 'withdrawn'; requestRecord.reason_code = 'withdrawn_by_operator'; requestRecord.finished_at = '2026-09-16T09:00:03Z'
      requestRecord.transitions.push({ status: 'withdrawn', reason_code: requestRecord.reason_code, actor: 'human:operator', created_at: requestRecord.finished_at })
      steeringLedger.inbox_version += 1
      await route.fulfill({ json: requestRecord })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/conversation/steering/proposals/proposal-steered/dismiss`) {
      proposalStatus = 'dismissed'
      const dismissed = steeringProposal()
      steeringLedger.proposals = [dismissed]
      await route.fulfill({ json: dismissed })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/memory`) {
      await route.fulfill({ json: baselineState.memory })
      return
    }
    const memorySection = suffix.match(new RegExp(`^/${goal.id}/memory/([^/]+)$`))
    if (request.method() === 'GET' && memorySection) {
      const section = baselineState.memorySections[decodeURIComponent(memorySection[1])]
      if (section) await route.fulfill({ json: section })
      else await route.fulfill({ status: 404, json: { detail: 'Memory section not found.' } })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/processes`) {
      await route.fulfill({ json: baselineState.processes })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/agent-reviews`) {
      await route.fulfill({ json: baselineState.reviews })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/warnings`) {
      await route.fulfill({ json: baselineState.warnings })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/decisions`) {
      await route.fulfill({ json: baselineState.decisions })
      return
    }
    if (request.method() === 'GET' && suffix === `/${goal.id}/decisions/checkpoint`) {
      await route.fulfill({
        json: { goal_id: goal.id, ...baselineCheckpoint(baselineState.decisions) },
      })
      return
    }
    const baselineAction = suffix.match(new RegExp(`^/${goal.id}/baseline/(step|rerun)$`))
    if (request.method() === 'POST' && baselineAction) {
      await deferBaselineMutation(baselineAction[1] === 'step' ? 'run' : 'rerun')
      await route.fulfill({ json: {} })
      return
    }
    const answer = suffix.match(new RegExp(`^/${goal.id}/decisions/([^/]+)/answer$`))
    if (request.method() === 'POST' && answer) {
      await deferBaselineMutation('answer')
      const payload = await request.postDataJSON() as { selected_option: string; reason: string | null }
      const index = baselineState.decisions.findIndex((item) => item.id === decodeURIComponent(answer[1]))
      if (index < 0) {
        await route.fulfill({ status: 404, json: { detail: 'Decision not found.' } })
        return
      }
      const decision: OrchestrationAuthorityDecisionRecord = {
        ...baselineState.decisions[index], status: 'answered', selected_option: payload.selected_option,
        reason: payload.reason, decided_by_user_id: 'human:operator', decided_at: '2026-07-29T12:00:00Z',
        overrides_recommendation: payload.selected_option !== baselineState.decisions[index].recommendation,
      }
      baselineState.decisions[index] = decision
      await route.fulfill({ json: { decision } })
      return
    }
    const skip = suffix.match(new RegExp(`^/${goal.id}/processes/([^/]+)/skip$`))
    if (request.method() === 'POST' && skip) {
      await deferBaselineMutation('skip')
      const processType = decodeURIComponent(skip[1])
      const index = baselineState.processes.findIndex((item) => item.process_type === processType && item.superseded_by_id === null)
      if (index < 0) {
        await route.fulfill({ status: 404, json: { detail: 'Process not found.' } })
        return
      }
      const payload = await request.postDataJSON() as { reason: string }
      const process: OrchestrationProcessRunRecord = {
        ...baselineState.processes[index], status: 'skipped', skipped_by: 'human:operator', override_reason: payload.reason,
        completed_at: '2026-07-29T12:00:00Z', updated_at: '2026-07-29T12:00:00Z',
      }
      baselineState.processes[index] = process
      baselineState.warnings.push(baselineWarning({
        id: `warning-${process.id}-skipped`, warning_type: `${process.process_type}_skipped`,
        message: `Process '${process.process_type}' was skipped by ${process.skipped_by}: ${payload.reason}`,
        source_process_run_id: process.id, blocks_completion: true,
      }))
      await route.fulfill({ json: process })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/pause`) {
      requests.pause += 1
      if (deferLifecycle.includes('pause')) await lifecycleResponses.pause
      if (rejectLifecycle === 'pause') {
        await route.fulfill({ status: 409, json: { detail: 'Pause was rejected.' } })
        return
      }
      state = withActiveBlockers(detail('paused', 'paused', supervisionFor('paused')))
      await route.fulfill({ json: state })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/resume`) {
      if (deferLifecycle.includes('resume')) await lifecycleResponses.resume
      if (rejectLifecycle === 'resume') {
        await route.fulfill({ status: 409, json: { detail: 'Resume was rejected.' } })
        return
      }
      state = withActiveBlockers(detail('active', 'running', supervisionFor('working')))
      await route.fulfill({ json: state })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/cancel`) {
      requests.cancel += 1
      if (rejectNextCancel) {
        rejectNextCancel = false
        await cancelRejection
        await route.fulfill({ status: 409, json: { detail: 'Cancellation was rejected.' } })
        return
      }
      state = withActiveBlockers(detail('cancelled', 'cancelled', supervisionFor('cancelled')))
      await route.fulfill({ json: state })
      return
    }
    if (request.method() === 'POST' && suffix === `/${goal.id}/override`) {
      requests.override += 1
      expect(await request.postDataJSON()).toEqual({
        gate_id: 'gate-release',
        decision: 'reject',
        reason: 'Independent proof is still missing.',
        evidence_metadata: {},
      })
      if (rejectNextOverride) {
        rejectNextOverride = false
        if (deferOverrideRejection) await overrideRejection
        await route.fulfill({ status: 409, json: { detail: 'Gate override was rejected.' } })
        return
      }
      await route.fulfill({ json: state })
      return
    }
    await route.fallback()
  })
  await page.route('**/api/v1/projects/*/orchestration/warnings/*/*', async (route) => {
    const request = route.request()
    const match = new URL(request.url()).pathname.match(/\/orchestration\/warnings\/([^/]+)\/(acknowledge|resolve)$/)
    if (request.method() !== 'POST' || !match) {
      await route.fallback()
      return
    }
    const kind = match[2] as 'acknowledge' | 'resolve'
    await deferBaselineMutation(kind)
    const index = baselineState.warnings.findIndex((item) => item.id === decodeURIComponent(match[1]))
    if (index < 0) {
      await route.fulfill({ status: 404, json: { detail: 'Warning not found.' } })
      return
    }
    const payload = await request.postDataJSON() as { reason: string }
    const warning: OrchestrationWarningRecord = kind === 'acknowledge'
      ? {
        ...baselineState.warnings[index], acknowledged_by: 'human:operator', acknowledged_at: '2026-07-29T12:00:00Z',
        blocks_completion: false, updated_at: '2026-07-29T12:00:00Z',
      }
      : {
        ...baselineState.warnings[index], active: false, resolved_by: 'human:operator', resolved_reason: payload.reason,
        resolved_at: '2026-07-29T12:00:00Z', blocks_completion: false, updated_at: '2026-07-29T12:00:00Z',
      }
    baselineState.warnings[index] = warning
    if (kind === 'resolve' && warning.severity === 'blocker') {
      state = {
        ...state,
        run: state.run && {
          ...state.run,
          active_blockers: state.run.active_blockers.filter((blocker) => blocker.kind !== `warning:${warning.id}`),
        },
      }
    }
    await route.fulfill({ json: warning })
  })
  return {
    releaseCancelRejection,
    releaseOverrideRejection,
    releasePause: lifecycleReleases.pause,
    releaseResume: lifecycleReleases.resume,
    releaseAnswer: baselineReleases.answer,
    releaseBaseline: baselineReleases,
    requests,
    baselineRequests,
    detailGets: () => detailGets,
    conversationGets: () => conversationGets,
    scenarioRequests: () => [...scenarioRequests],
    lastConversationHistory: () => conversationHistories.at(-1),
    completeConversation: (responseId = `conversation-response-${conversationRequests.length}`) => { completedConversationResponses.add(responseId) },
    failNextFeedback: () => { failNextFeedback = true },
    completeInvestigation: () => { investigationCompleted = true },
    applySteering: () => {
      const requestRecord = steeringLedger.requests.at(-1)
      if (!requestRecord) return
      requestRecord.status = 'applied'; requestRecord.reason_code = 'applied'; requestRecord.considered_at = '2026-09-16T09:00:04Z'; requestRecord.finished_at = '2026-09-16T09:00:04Z'
      requestRecord.updated_at = requestRecord.finished_at
      requestRecord.transitions.push({ status: 'applied', reason_code: 'applied', actor: 'orchestrator', created_at: requestRecord.finished_at })
      requestRecord.result_action_ids = ['action-steered']
      steeringLedger.direction_version += 1
      state = { ...state, actions: [...state.actions, {
        id: 'action-steered', run_id: run.id, decision_id: 'decision-request-verification', idempotency_key: 'steering:action-steered',
        action_type: 'request_verification', request: { reason: 'Steering direction applied.' }, target_type: 'task', target_id: 'task-validation',
        status: 'completed', error: null, created_at: '2026-09-16T09:00:04Z', updated_at: '2026-09-16T09:00:04Z',
      }] }
    },
    setSteeringEligibility: (eligibility: OrchestrationSteeringLedger['eligibility']) => {
      steeringLedger.eligibility = eligibility
      steeringLedger.eligibility_reason = eligibility === 'active' || eligibility === 'paused' ? null : 'Steering is unavailable for this goal state.'
    },
    steeringLedger,
    setSupervision: (next: OrchestrationSupervision | null) => {
      state = { ...state, supervision: next }
    },
  }
}

export async function openOrchestrationGoal(page: Page) {
  await gotoDashboard(page)
  await selectSeedProject(page)
  const projectId = await page.getByRole('combobox', { name: 'Project switcher' }).inputValue()
  await page.getByRole('link', { name: 'Orchestration' }).click()
  await page.getByText(goal.objective).click()
  return projectId
}
