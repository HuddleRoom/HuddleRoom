// Task statuses
export type TaskStatus = 'backlog' | 'ready' | 'in_progress' | 'blocked' | 'done' | 'cancelled' | 'failed'

// Meeting statuses
export type MeetingStatus = 'scheduled' | 'preparing' | 'active' | 'concluding' | 'concluded' | 'cancelled'

// Hook statuses
export type HookStatus = 'proposed' | 'active' | 'shadow' | 'disabled'

// Optimization statuses
export type OptimizationStatus = 'proposed' | 'requires_approval' | 'approved' | 'active' | 'shadow' | 'disabled' | 'rejected'

// Session statuses
export type SessionStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

// Priority levels (user-facing labels; backend stores as int)
export type TaskPriorityLabel = 'low' | 'medium' | 'high' | 'critical'

// Adapter types
export type AdapterType = 'api' | 'cli' | 'routine'

// Content types
export type ContentType = 'text' | 'markdown' | 'code' | 'json'

// Memory scope
export type MemoryScope = 'project' | 'global'

// Core entities
export interface Project {
  id: string
  name: string
  workspace_path?: string | null
  config: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface Agent {
  id: string
  name: string
  role: string
  provider: string
  model: string
  adapter_type: AdapterType
  cli_runtime: string | null
  system_prompt: string | null
  description: string | null
  capabilities: string[]
  is_active: boolean
  config: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface Task {
  id: string
  project_id: string
  title: string
  description?: string
  priority: number
  assigned_to?: string
  parent_id?: string
  status: TaskStatus
  trigger?: string
  metadata?: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface Session {
  id: string
  project_id: string
  task_id?: string
  agent_id: string
  adapter_type?: string
  status: SessionStatus
  input_context?: Record<string, unknown>
  output?: string | null
  error?: string | null
  runner_task_id?: string | null
  sandbox_path?: string | null
  metadata?: Record<string, unknown>
  origin?: string
  started_at?: string | null
  ended_at?: string | null
  completed_at?: string | null
  resumable?: boolean
  provider_session_id?: string | null
  created_at: string
}

export interface Meeting {
  id: string
  project_id: string
  title: string
  meeting_type: string
  status: MeetingStatus
  participant_agent_ids: string[]
  participant_user_ids?: string[]
  agenda_items: AgendaItem[]
  max_duration_minutes?: number
  turn_strategy?: string
  deadlock_strategy?: string
  organizer_user_id?: string
  organizer_agent_id?: string
  resume_state?: { failed?: boolean; error?: string; speaker_agent_id?: string; agenda_item_id?: string; adapter?: string; resuming?: boolean }
  created_at: string
  updated_at: string
}

export interface MeetingTurn {
  id: string
  meeting_id: string
  agenda_item_id?: string | null
  turn_number?: number
  round_number?: number
  speaker_agent_id?: string | null
  speaker_user_id?: string | null
  is_human_turn: boolean
  is_override?: boolean
  moderator_note?: string | null
  token_count?: number | null
  model_used?: string | null
  latency_ms?: number | null
  prompt_messages?: unknown[] | null
  raw_response?: string | null
  organizer_selection?: Record<string, unknown> | null
  reasoning_content?: string | null
  references?: unknown[]
  content: string
  created_at: string
}

export interface MeetingWSEvent {
  id?: string
  project_id?: string
  meeting_id?: string
  event_type: string
  payload?: Record<string, unknown>
  source?: string
  emitted_at?: string
  turn?: MeetingTurn
}

export interface MeetingDecision {
  id: string
  meeting_id: string
  agenda_item_id?: string | null
  title: string
  question?: string | null
  chosen_option: string
  rationale?: string | null
  alternatives_rejected?: string[]
  participants_agreed?: string[]
  dissent?: string[]
  decided_by: string
  confidence?: number | null
  is_partial?: boolean
  is_vetoed?: boolean
  veto_reason?: string | null
  content?: string
  created_at: string
}

export interface MeetingActionItem {
  id: string
  meeting_id: string
  description: string
  assignee_agent_id?: string
  assignee_user_id?: string
  status: TaskStatus
  due_date?: string
  depends_on_decision_id?: string | null
  created_at: string
}

export interface MeetingFinalReview {
  reviewer_kind: 'organizer_agent' | 'organizer_user' | 'orchestrator'
  reviewer_id: string | null
  decisions_made: boolean
  decisions_clear: boolean
  suggested_action_items: string[]
}

export interface MeetingFinalReviewSubmit {
  decisions_made: boolean
  decisions_clear: boolean
  action_items_needed: boolean
  action_items: string[]
}

export interface AgendaItem {
  id: string
  meeting_id: string
  title: string
  description?: string
  question?: string
  options?: string[]
  max_rounds?: number
  order: number
  status: string
  resolution_summary?: string | null
  resolution_kind?: string | null
}

export interface KnowledgeItem {
  id: string
  project_id: string
  content: string
  content_type: ContentType
  title?: string
  tags: string[]
  provenance?: string
  created_at: string
  updated_at: string
}

export interface KnowledgeSearchResult {
  id: string
  project_id: string | null
  title?: string
  content: string
  content_type: ContentType
  tags: string[] | null
  relevance_score: number
}

export interface MemoryItem {
  id: string
  project_id?: string
  agent_id?: string
  content: string
  tags: string[]
  shared: boolean
  scope: MemoryScope
  created_at: string
}

export interface RoutingRule {
  id: string
  project_id: string
  name: string
  description?: string
  priority: number
  on_event: string
  conditions: Record<string, unknown>
  actions: Record<string, unknown>
  enabled: boolean
  created_at: string
  updated_at: string
}

export interface Hook {
  id: string
  project_id: string
  name: string
  description?: string
  code: string
  status: HookStatus
  trigger_event: string
  execution_count: number
  error_count: number
  created_at: string
  updated_at: string
}

export interface Optimization {
  id: string
  project_id: string
  pattern_id?: string
  type: 'hook' | 'rule' | 'shortcut'
  generated_code: string
  status: OptimizationStatus
  error_rate: number
  fire_count: number
  created_at: string
  updated_at: string
}

export interface Pattern {
  id: string
  project_id: string
  pattern_type: string
  description: string
  confidence: number
  sample_size: number
  context: Record<string, unknown>
  created_at: string
}

export interface User {
  id: string
  email: string
  display_name?: string
  role: string
  is_active: boolean
  created_at: string
}

export interface ApiKey {
  id: string
  user_id: string
  label?: string
  prefix: string
  agent_id?: string
  project_id?: string
  expires_at?: string
  created_at: string
}

export interface CursorPage<T> {
  items: T[]
  next_cursor: string | null
}

export interface MeetingSignal {
  id: string
  meeting_id: string
  signal_type: string
  content: string
  created_at: string
}

export interface Graph {
  id: string
  project_id?: string
  name: string
  version: string
  description?: string
  definition: Record<string, unknown>
  triggers: string[]
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface GraphRun {
  id: string
  graph_id: string
  project_id: string
  linked_task_id?: string | null
  artifact_id?: string | null
  current_node: string
  status: string
  actor_assignments?: Record<string, unknown>
  context?: Record<string, unknown>
  escalation_step?: number | null
  started_at: string
  last_stepped_at?: string | null
  completed_at?: string | null
}

export interface GraphRunStep {
  id: string
  graph_run_id: string
  from_node: string
  to_node: string
  edge_name?: string | null
  trigger_event_id?: string | null
  trigger_reason?: string | null
  actions_executed: unknown[]
  stepped_at: string
}

export interface GraphSummary {
  id: string
  name: string
  version: string
}

export interface GraphRunTimeout {
  id: string
  graph_run_id: string
  node_name: string
  timeout_action: string
  expires_at: string
  resolved: boolean
  resolved_at?: string | null
  retry_count: number
  created_at: string
}

export interface GraphRunSession {
  id: string
  task_id?: string | null
  agent_id?: string | null
  status: string
  origin: string
  graph_run_id?: string | null
  output?: string | null
  error?: string | null
  started_at?: string | null
  ended_at?: string | null
  created_at: string
}

export interface GraphRunTaskSummary {
  id: string
  title: string
  status: string
  parent_id?: string | null
  graph_run_id?: string | null
}

export interface GraphRunDetail {
  run: GraphRun
  graph: GraphSummary
  steps: GraphRunStep[]
  sessions: GraphRunSession[]
  timeouts: GraphRunTimeout[]
  tasks: GraphRunTaskSummary[]
}

export interface HuddleRoomEvent {
  id: string
  project_id: string
  event_type: string
  payload: Record<string, unknown>
  source: string
  emitted_at: string
}

export type OrchestrationGoalStatus = 'active' | 'blocked' | 'paused' | 'completed' | 'cancelled'
export type OrchestrationRunStatus = 'running' | 'blocked' | 'paused' | 'completed' | 'cancelled'
export type OrchestrationConversationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'interrupted_unknown'
export type OrchestrationGoalWeight = 'trivial' | 'standard' | 'substantial'
export type OrchestrationGoalCommand = 'pause' | 'resume' | 'cancel'
export type OrchestrationGateOverrideDecision = 'accept' | 'reject'
export type OrchestrationProcessStatus = 'running' | 'waiting_decision' | 'completed' | 'skipped'
export type OrchestrationWarningSeverity = 'recommendation' | 'warning' | 'blocker' | 'hard_stop'
export type OrchestrationAuthority = 'human' | 'manager' | 'team_lead' | 'agent'
export type OrchestrationAuthorityDecisionStatus = 'pending' | 'answered' | 'cancelled' | 'expired'
export type OrchestrationDecisionOption = string | ({ key: string } & Record<string, unknown>)

export interface OrchestrationSuccessCriterion extends Record<string, unknown> {
  key?: string
  description?: string
}

export interface OrchestrationPlanItemState extends Record<string, unknown> {
  plan_item_id?: string
  work_function?: string
  task_id?: string
  gate_id?: string
}

export interface OrchestrationPlanState extends Record<string, unknown> {
  status?: string
  planning_task_id?: string
  accepted_artifact_id?: string
  revision_requests?: unknown[]
  expanded_items?: OrchestrationPlanItemState[]
}

export interface OrchestrationBlocker extends Record<string, unknown> {
  kind?: string
  reason?: string
  task_id?: string
  gate_id?: string
}

export interface OrchestrationGoal {
  id: string
  project_id: string
  objective: string
  success_criteria: OrchestrationSuccessCriterion[]
  orchestrator_context: Record<string, unknown>
  constraints: Record<string, unknown>
  budget: Record<string, unknown>
  goal_type: 'outcome' | 'roadmap' | 'continuous'
  supersedes_goal_id: string | null
  status: OrchestrationGoalStatus
  weight: OrchestrationGoalWeight
  weight_overridden_by: string | null
  manager_agent_id: string | null
  manager_user_id: string | null
  authority_model: string | null
  created_by_user_id: string | null
  created_at: string
  updated_at: string
  needs_you_count: number
}

export interface OrchestrationConversationError {
  code: string
}

export type OrchestrationConversationFeedbackRating = 'helpful' | 'not_helpful'

export type OrchestrationConversationFeedbackReason =
  | 'unanswered'
  | 'incorrect'
  | 'missing_context'
  | 'stale_context'
  | 'unclear'
  | 'too_limited'
  | 'other'

export interface OrchestrationConversationFeedback {
  feedback_id: string
  rating: OrchestrationConversationFeedbackRating
  reason: OrchestrationConversationFeedbackReason | null
  created_at: string
}

export interface OrchestrationConversationFeedbackInput {
  rating: OrchestrationConversationFeedbackRating
  reason: OrchestrationConversationFeedbackReason | null
}

export type OrchestrationConversationInvestigationStatus =
  | 'pending' | 'running' | 'completed' | 'limited' | 'failed'
  | 'cancelled' | 'unavailable' | 'interrupted_unknown'

export interface OrchestrationConversationInvestigationSource {
  reference: string
  operation: 'list' | 'read' | 'search'
  status: 'included' | 'restricted' | 'unsafe' | 'binary' | 'too_large' | 'changed' | 'omitted_by_limit'
  freshness_at: string | null
  truncated: boolean
}

export interface OrchestrationConversationInvestigationReport {
  findings: string
  uncertainty: string
  sources: string[]
}

export interface OrchestrationConversationInvestigation {
  investigation_id: string
  status: OrchestrationConversationInvestigationStatus
  objective: string
  attempt_count: number
  repair_count: number
  retry_count: number
  sources: OrchestrationConversationInvestigationSource[]
  report: OrchestrationConversationInvestigationReport | null
  error: OrchestrationConversationError | null
  started_at: string | null
  deadline_at: string | null
  finished_at: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationConversationManifestSource {
  source: string
  status: string
  freshness_at: string | null
  available: number
  included: number
  omitted: number
  truncated: boolean
  references: string[]
}

export interface OrchestrationConversationManifest {
  run_id: string | null
  excluded_categories: string[]
  sources: OrchestrationConversationManifestSource[]
  truncated: boolean
}

export interface OrchestrationConversationTurn {
  message_id: string
  response_id: string
  client_request_id: string
  sequence: number
  actor_id: string
  content: string
  message_created_at: string
  status: OrchestrationConversationStatus
  run_id: string | null
  answer: string | null
  error: OrchestrationConversationError | null
  started_at: string | null
  deadline_at: string | null
  finished_at: string | null
  created_at: string
  updated_at: string
  context_version: string
  context_manifest: OrchestrationConversationManifest
  feedback: OrchestrationConversationFeedback | null
  feedback_eligible: boolean
  investigation?: OrchestrationConversationInvestigation | null
  proposed_steering?: OrchestrationSteeringProposal | null
}

export type OrchestrationSteeringScope = 'item' | 'run' | 'goal'
export type OrchestrationSteeringLifetime = 'selected_item' | 'remaining_current_run' | 'future_runs'
export type OrchestrationSteeringTargetType = 'goal' | 'plan_item' | 'task'
export type OrchestrationSteeringStatus = 'pending' | 'being_considered' | 'applied' | 'deferred' | 'rejected' | 'superseded' | 'needs_clarification' | 'withdrawn'

export interface OrchestrationSteeringTransition {
  status: OrchestrationSteeringStatus
  reason_code: string
  actor: string
  created_at: string
}

export interface OrchestrationSteeringProposal {
  proposal_id: string
  response_id: string
  status: 'proposed' | 'dismissed' | 'promoted'
  directive: string
  target_type: OrchestrationSteeringTargetType
  target_id: string
  scope: OrchestrationSteeringScope
  lifetime: OrchestrationSteeringLifetime
  impact_summary: string
  dismissed_at: string | null
  promoted_request_id: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationSteeringRequest {
  request_id: string
  client_request_id: string
  sequence: number
  directive: string
  target_type: OrchestrationSteeringTargetType
  target_id: string
  scope: OrchestrationSteeringScope
  lifetime: OrchestrationSteeringLifetime
  impact_summary: string
  source_proposal_id: string | null
  supersedes_request_id: string | null
  status: OrchestrationSteeringStatus
  reason_code: string
  submitted_at: string
  considered_at: string | null
  finished_at: string | null
  updated_at: string
  transitions: OrchestrationSteeringTransition[]
  result_action_ids: string[]
}

export interface OrchestrationSteeringLedger {
  enabled: boolean
  eligibility: 'active' | 'paused' | 'unstarted' | 'terminal' | 'forbidden'
  eligibility_reason: string | null
  inbox_version: number
  direction_version: number
  requests: OrchestrationSteeringRequest[]
  proposals: OrchestrationSteeringProposal[]
}

export interface OrchestrationConversationAllowance {
  enabled: boolean
  limit: number
  used: number
  remaining: number
}

export interface OrchestrationConversationHistory {
  items: OrchestrationConversationTurn[]
  total: number
  omitted: number
  allowance: OrchestrationConversationAllowance
  steering: OrchestrationSteeringLedger
}

// "Ask the orchestrator" project advisor (read-only, project-scoped Q&A).
export interface ProjectAdvisorCitation {
  type: 'goal' | 'decision' | 'meeting'
  id: string
  label: string
  goal_id?: string | null
}

export interface ProjectAdvisorTurn {
  id: string
  question: string
  answer: string | null
  citations: ProjectAdvisorCitation[]
  off_topic: boolean
  status: string
  created_at: string
}

export interface ProjectAdvisorAllowance {
  enabled: boolean
  unlimited: boolean
  limit: number
  remaining: number
}

export interface ProjectAdvisorHistory {
  items: ProjectAdvisorTurn[]
  allowance: ProjectAdvisorAllowance
}

export interface OrchestrationRun {
  id: string
  goal_id: string
  status: OrchestrationRunStatus
  phase: string
  condition: string
  baseline_authorized: boolean
  event_cursor: number | null
  plan_state: OrchestrationPlanState
  active_blockers: OrchestrationBlocker[]
  budget_state: Record<string, unknown>
  retry_state: Record<string, unknown>
  started_at: string
  completed_at: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationDecision {
  id: string
  run_id: string
  decision_type: string
  input_snapshot: Record<string, unknown>
  llm_output: unknown
  parsed_decision: Record<string, unknown>
  validator_status: 'pending' | 'accepted' | 'rejected'
  rejection_reason: string | null
  reason: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationAction {
  id: string
  run_id: string
  decision_id: string | null
  idempotency_key: string
  action_type: string
  request: Record<string, unknown>
  target_type: string | null
  target_id: string | null
  status: 'reserved' | 'completed' | 'failed'
  error: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationGate {
  id: string
  run_id: string
  success_criterion_key: string
  gate_type: string
  required_evidence: Record<string, unknown>
  status: 'open' | 'accepted' | 'failed'
  failure_reason: string | null
  created_at: string
  updated_at: string
  accepted_at: string | null
  failed_at: string | null
}

export interface OrchestrationEvidence {
  id: string
  run_id: string
  gate_id: string
  source_type: string
  source_id: string | null
  observed_event_id: string | null
  producer_agent_id: string | null
  verdict: 'candidate' | 'accepted' | 'rejected'
  evidence_metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface OrchestrationAgentSuggestion {
  id: string
  run_id: string
  missing_work_function: string
  reason: string
  suggested_role: string | null
  suggested_capabilities: string[]
  suggested_adapter_type: string | null
  suggested_model: string | null
  suggested_system_prompt_outline: string | null
  status: 'open' | 'accepted' | 'dismissed'
  created_at: string
  updated_at: string
}

export interface OrchestrationProcessRunRecord {
  id: string
  goal_id: string
  run_id: string | null
  process_type: string
  process_version: number
  status: OrchestrationProcessStatus
  trigger_reason: string
  input_snapshot: Record<string, unknown>
  outputs: Record<string, unknown> & { lm_retry?: OrchestrationLmRetryDescriptor }
  skipped_by: string | null
  override_reason: string | null
  superseded_by_id: string | null
  started_at: string
  completed_at: string | null
  created_at: string
  updated_at: string
}

export type OrchestrationBaselineProcessType =
  | 'goal_definition'
  | 'manager_selection'
  | 'agent_definition_review'
  | 'team_hierarchy'
  | 'effectiveness_review'
  | 'goal_closeout'

export interface OrchestrationLmRetryDescriptor {
  available: boolean
  kind: string | null
  warning_id: string | null
  model: string | null
  hint: string | null
}

export interface OrchestrationHealth {
  consumers: Record<string, { running: boolean; active_connections?: number | null }>
  active_sessions: number
  event_bus_mode: string
  event_log_total: number
  debug_enabled: boolean
}

export interface OrchestrationDebugActionResult {
  action: 'step' | 'rerun_last' | 'retry'
  goal_id: string
  run_id: string
  process_type: string
  process: Record<string, unknown>
}

export interface OrchestrationAgentReviewRecord {
  id: string
  goal_id: string
  run_id: string | null
  agent_id: string | null
  source_process_run_id: string | null
  review_context: string | null
  proposed_work_functions: unknown[]
  definition_snapshot: Record<string, unknown>
  fit_summary: string
  strengths: unknown[]
  risks: unknown[]
  recommended_changes: unknown[]
  approved_for_work_functions: unknown[]
  created_at: string
  updated_at: string
}

export interface OrchestrationWarningRecord {
  id: string
  goal_id: string
  run_id: string | null
  warning_type: string
  severity: OrchestrationWarningSeverity
  message: string
  source_process_run_id: string | null
  related_gate_id: string | null
  related_action_id: string | null
  related_agent_id: string | null
  source_agent_review_id: string | null
  related_authority_decision_id: string | null
  acknowledged_by: string | null
  acknowledged_at: string | null
  active: boolean
  resolved_by: string | null
  resolved_reason: string | null
  resolved_at: string | null
  blocks_completion: boolean
  created_at: string
  updated_at: string
}

export interface OrchestrationAuthorityDecisionRecord {
  id: string
  goal_id: string
  run_id: string | null
  decision_key: string
  title: string
  status: OrchestrationAuthorityDecisionStatus
  authority: OrchestrationAuthority
  authority_agent_id: string | null
  source_process_run_id: string | null
  question: string
  context: string | null
  options: OrchestrationDecisionOption[]
  recommendation: string | null
  consequences: string | null
  selected_option: string | null
  reason: string | null
  decided_by_user_id: string | null
  decided_by_agent_id: string | null
  overrides_recommendation: boolean
  created_warning_id: string | null
  related_gate_id: string | null
  related_action_id: string | null
  asked_at: string
  decided_at: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationDecisionAnswerResult {
  decision: OrchestrationAuthorityDecisionRecord
  process?: OrchestrationProcessRunRecord | null
}

export interface OrchestrationCheckpoint {
  goal_id: string
  items: OrchestrationAuthorityDecisionRecord[]
  deferred_count: number
  max_questions: number
}

export interface OrchestrationMemorySection {
  id: string
  project_id: string
  goal_id: string
  run_id: string | null
  section_key: string
  title: string
  section_type: string
  body: string
  summary: string | null
  always_load: boolean
  toc_order: number
  created_by: string
  created_from_event_id: string | null
  updated_from_event_id: string | null
  created_at: string
  updated_at: string
}

export interface OrchestrationMemoryTocEntry {
  section_key: string
  title: string
  summary: string | null
  section_type: string
  always_load: boolean
  toc_order: number
  updated_at: string
}

export interface OrchestrationMemoryPreface {
  objective: string | null
  goal_status: OrchestrationGoalStatus
  goal_weight: OrchestrationGoalWeight
  run_status: OrchestrationRunStatus | null
  current_process: { process_type: string; status: OrchestrationProcessStatus } | null
  manager: string | null
  hierarchy: string | null
  constraints: string | null
  active_warnings: {
    severity: OrchestrationWarningSeverity
    warning_type: string
    message: string | null
    acknowledged: boolean
  }[]
  recent_decisions: {
    title: string | null
    authority: OrchestrationAuthority
    selected_option: string | null
    overrides_recommendation: boolean
  }[]
  open_blockers: (string | null)[]
  skipped_processes: { process_type: string; skipped_by: string | null }[]
  introduction: string | null
  always_loaded: { section_key: string; summary: string | null }[]
  toc: { section_key: string; title: string | null }[]
}

export interface OrchestrationMemoryOverview {
  toc: OrchestrationMemoryTocEntry[]
  always_loaded: OrchestrationMemorySection[]
  preface: OrchestrationMemoryPreface
}

export interface OrchestrationGoalDetail {
  goal: OrchestrationGoal
  run: OrchestrationRun | null
  decisions_count: number
  decisions: OrchestrationDecision[]
  actions_count: number
  actions: OrchestrationAction[]
  gates_count: number
  gates: OrchestrationGate[]
  evidence_count: number
  evidence: OrchestrationEvidence[]
  agent_suggestions_count: number
  agent_suggestions: OrchestrationAgentSuggestion[]
  timeline: unknown[]
  supervision: OrchestrationSupervision | null
}

export type OrchestrationSupervisionCondition =
  | 'working' | 'waiting' | 'needs_you' | 'needs_attention'
  | 'paused' | 'stopped' | 'cancelled' | 'completed'

export interface OrchestrationSupervision {
  condition: OrchestrationSupervisionCondition
  operation: string
  next_action: string
  rationale: string
  criterion: Record<string, unknown> | null
  verified_progress: Record<string, unknown>[]
  useful_learning: Record<string, unknown>[]
  accepted_evidence: OrchestrationEvidence[]
  workers: Record<string, unknown>[]
  waits: Record<string, unknown>[]
  recovery_history: Record<string, unknown>[]
  pending_direction: OrchestrationAuthorityDecisionRecord | null
  budget: Record<string, Record<string, string>>
  transition: { key: string; message: string; kind: string; occurred_at?: string | null } | null
}

export interface OrchestrationGateOverrideInput {
  goalId: string
  gateId: string
  decision: OrchestrationGateOverrideDecision
  reason: string
  evidenceMetadata?: Record<string, unknown>
}

export interface OrchestrationGoalCreateInput {
  objective: string
  success_criteria: { description: string }[]
  constraints?: Record<string, unknown>
  budget?: Record<string, unknown>
}
