import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { PlanTab } from '../../src/pages/orchestration/PlanTab'
import { StartWorkDialog } from '../../src/pages/orchestration/StartWorkDialog'
import { ProjectChatPanel } from '../../src/components/layout/ProjectChatPanel'
import type { OrchestrationAuthorityDecisionRecord, OrchestrationGoal, OrchestrationGoalDetail } from '../../src/lib/types'
import '../../src/index.css'

const stale = new URLSearchParams(location.search).get('state') === 'stale'
const goal = { id: 'goal-1', project_id: 'project-1', objective: 'Deliver review', success_criteria: [], constraints: {}, budget: {}, goal_type: 'roadmap', status: 'active', authority_model: 'human_review' } as OrchestrationGoal
const detail = {
  goal, run: null, runs: [{ id: 'run-1', status: 'running', phase: 'planning', cycle_key: null, supersedes_run_id: null, terminal_reason: null, goal_type_snapshot: 'roadmap', planning_mode_snapshot: 'roadmap', goal_contract_fingerprint: 'fp', started_at: '2026-10-19T00:00:00Z', completed_at: null, version_label: 'v1', is_current: true }],
  plan: { status: 'accepted', artifact_id: 'accepted-3', kind: 'roadmap', author_agent_id: null, planning_version: 3, planning_task_id: 'task-3', summary: null, scope: null, items: [], review: { artifact_id: 'artifact-review-4', state: stale ? 'stale' : 'ready', markdown: '# Candidate\n\nOwner: implementation agent\nDependencies: accepted evidence', export_path: null, reason: stale ? 'The plan changed after review.' : null, authority_decision_id: 'decision-4', authority_gate_id: 'gate-4', authority_status: 'Pending', review_url: '/orchestration/goal-1#plan-review-artifact-review-4', identity: { project_id: 'project-1', goal_id: 'goal-1', run_id: 'run-1', artifact_id: 'artifact-review-4', planning_task_id: 'task-4', planning_version: 4, winning_session_id: null, provenance_hash: 'provenance-4', canonical_content_hash: 'canonical-4', renderer_version: 1, rendered_content_hash: 'rendered-4', review_cycle: 2 } } },
  evidence: [], task_reports: [], child_goals: [], gates: [], actions: [], warnings: [], agent_suggestions: [], meeting_commitments: [], needs_you_references: [],
} as unknown as OrchestrationGoalDetail
const decisions = [{ id: 'decision-4', goal_id: 'goal-1', run_id: 'run-1', status: 'pending', authority: 'human', authority_agent_id: null, related_gate_id: 'gate-4' }] as unknown as OrchestrationAuthorityDecisionRecord[]
const authorization = { allowed: true, reason_code: null, required_decision: null, authority_source: 'root_start', run_id: 'run-1', contract_fingerprint: 'fp', model_spend_remaining: {}, external_write_targets: [], live_send_channels: [], workspace_mutation: 'staged', tool_constraints: [] } as const
function Fixture() { const [scope, setScope] = useState('none'); return <main className="mx-auto max-w-3xl p-6"><PlanTab projectId="project-1" detail={detail} decisions={decisions} onViewDecision={() => setScope('reviewed')} /><StartWorkDialog open={false} goal={goal} authorization={authorization} missingRecurringPolicy={false} isPending={false} error={null} onOpenChange={() => undefined} onConfirm={(next) => setScope(next)} /><p aria-live="polite">Selected scope: {scope}</p><ProjectChatPanel projectId="project-1" contextGoalId="goal-1" /></main> }
createRoot(document.getElementById('root')!).render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter><Fixture /></MemoryRouter></QueryClientProvider>)
