// Isolated real Step UI: browser tests intercept the only API query; no backend.
import React, { createRef } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ProcessFocus, type ProcessFocusProps } from '../../src/pages/orchestration/ProcessFocus'
import type { OrchestrationAgentReviewRecord } from '../../src/lib/types'
import '../../src/index.css'

const now = '2026-10-19T08:00:00Z'
const historical = new URLSearchParams(location.search).get('state') === 'historical'
const reviews: OrchestrationAgentReviewRecord[] = Array.from({ length: 5 }, (_, i) => ({
  id: `review-${i}`, goal_id: 'goal-1', run_id: 'run-1', agent_id: `agent-${i}`,
  source_process_run_id: 'process-1', review_context: null,
  definition_snapshot: { name: `Reviewed agent ${i + 1}`, role: 'reviewer', provider: 'historic-provider', model: `historic-model-${i + 1}` },
  proposed_work_functions: ['review'], approved_for_work_functions: ['review'], fit_summary: 'Saved assessment remains available.',
  strengths: [], risks: [], recommended_changes: [], created_at: now, updated_at: now,
}))
const forbidden = () => { throw new Error('Historical fixture attempted mutation') }
const props: ProcessFocusProps = {
  selectedProcessType: 'agent_definition_review',
  process: {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'agent_definition_review', process_version: 4,
    status: 'waiting_decision', trigger_reason: 'Saved review',
    input_snapshot: { fingerprint: 'a'.repeat(64), capture: { source: 'captured', formula_version: 3, original_available: true }, effective_comparison: { fingerprint: 'b'.repeat(64) } },
    outputs: {}, skipped_by: null, override_reason: null, superseded_by_id: historical ? 'replacement-12345678-1234-1234-1234-123456789012' : null,
    started_at: now, completed_at: null, created_at: now, updated_at: now,
  },
  goal: { id: 'goal-1', project_id: 'project-1', objective: 'Provider-free retained G4 review', success_criteria: [], orchestrator_context: {}, constraints: {}, budget: {}, status: 'active', weight: 'standard', weight_overridden_by: null, manager_agent_id: null, manager_user_id: null, authority_model: null, created_by_user_id: null, created_at: now, updated_at: now, needs_you_count: 0 },
  decisions: [], decisionsUnready: false, warnings: [], reviews, debug: true,
  canSkip: historical, canRun: historical, canRerun: historical,
  mutationBusy: false, runPending: false, rerunPending: false, runRerunError: null,
  retryPending: false, retryError: null, openAction: historical ? { kind: 'skip', id: 'process-1' } : null,
  onRun: forbidden, onRerun: forbidden, onRetry: forbidden, onOpenAction: forbidden, onActionTrigger: () => {},
  headingRef: createRef(), renderActionForm: () => <button onClick={forbidden}>Forbidden historical submit</button>,
}

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <main className="mx-auto max-w-4xl p-4"><h1 className="mb-4 text-lg font-semibold">G4 saved review fixture</h1><ProcessFocus {...props} /></main>
  </QueryClientProvider>,
)
