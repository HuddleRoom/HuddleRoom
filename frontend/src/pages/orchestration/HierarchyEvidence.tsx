import type { OrchestrationAgentReviewRecord } from '@/lib/types'
import { hierarchyAgentResolver } from './hierarchyAgentNames'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function text(value: unknown, fallback = 'Not recorded') {
  return typeof value === 'string' && value.trim() ? value : fallback
}

function rows(value: unknown) {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

export function HierarchyEvidence({ proposal, reviews, revisionContext }: {
  proposal: Record<string, unknown>
  reviews: readonly OrchestrationAgentReviewRecord[]
  revisionContext?: Record<string, unknown> | null
}) {
  const resolveAgent = hierarchyAgentResolver(reviews)
  const agentName = (value: unknown) => resolveAgent(value) ?? 'Unknown agent'
  const feedback = rows(proposal.feedback_results)
  const revision = isRecord(revisionContext) ? revisionContext : isRecord(proposal.revision_context) ? proposal.revision_context : null
  const feedbackText = revision ? text(revision.feedback_text, '') : ''
  const assignments = rows(proposal.assignments)
  const reporting = rows(proposal.reporting_lines)
  const coverageRecorded = Array.isArray(proposal.responsibility_coverage)
  const feedbackRecorded = revision !== null || Array.isArray(proposal.feedback_results)
  const coverage = rows(proposal.responsibility_coverage)
  const gaps = coverage.filter((item) => item.status === 'gap')
  const assignedRefs = new Set(assignments.map((item) => item.agent_ref).filter((value): value is string => typeof value === 'string'))

  return <section aria-label="Hierarchy evidence" className="mt-2 min-w-0 space-y-2 text-sm text-huddleroom-text-secondary">
    <div>
      <h4 className="text-xs font-medium text-huddleroom-text-muted">Feedback requested{feedbackRecorded ? '' : ': Not recorded'}</h4>
      {feedbackText && <p className="mt-1 whitespace-pre-wrap break-words">{feedbackText}</p>}
      {!feedbackText && feedback.length === 0 && feedbackRecorded && <p className="mt-1">No requested feedback recorded.</p>}
      {feedback.length > 0 && <><h4 className="mt-1 text-xs font-medium text-huddleroom-text-muted">Outcomes</h4><ul className="mt-1 space-y-1">
        {feedback.map((item, index) => <li key={text(item.item_id, String(index))} className="min-w-0 break-words">
          <span className="font-medium text-huddleroom-text-primary">{text(item.source_excerpt, 'Feedback item')}</span>: {text(item.disposition)} — {text(item.reason)}
        </li>)}
      </ul></>}
    </div>
    <div>
      <p className="text-xs font-medium text-huddleroom-text-muted">Assignments</p>
      {assignments.length > 0 ? <ul className="mt-1 space-y-1">{assignments.map((item, index) => <li key={`${text(item.work_function, String(index))}:${String(item.agent_ref)}`} className="min-w-0 break-words">{text(item.work_function, 'Work function')}: {agentName(item.agent_ref)}</li>)}</ul> : <p className="mt-1">Not recorded</p>}
    </div>
    <div>
      <p className="text-xs font-medium text-huddleroom-text-muted">Reporting</p>
      {reporting.length > 0 ? <ul className="mt-1 space-y-1">{reporting.map((item, index) => <li key={`${String(item.agent_ref)}:${index}`} className="min-w-0 break-words">{agentName(item.agent_ref)} reports to {agentName(item.reports_to)}</li>)}</ul> : <p className="mt-1">Not recorded</p>}
    </div>
    <div>
      <h4 className="text-xs font-medium text-huddleroom-text-muted">Responsibility coverage{coverageRecorded ? '' : ': Not recorded'}</h4>
      {!coverageRecorded ? <p className="mt-1">Not recorded</p> : coverage.length === 0 ? <p className="mt-1">No uncovered responsibilities.</p> : <ul className="mt-1 space-y-1">{coverage.map((item, index) => {
        const status = text(item.status)
        const proposed = status === 'proposed'
        const gap = status === 'gap'
        const domainOnly = !gap && !proposed && typeof item.agent_ref === 'string' && !assignedRefs.has(item.agent_ref)
        return <li key={text(item.responsibility_id, String(index))} className="min-w-0 break-words">
          <span className="font-medium text-huddleroom-text-primary">{text(item.responsibility, 'Responsibility')}</span>: {gap ? 'Gap' : proposed ? 'Proposed, awaiting agent approval' : agentName(item.agent_ref)}{domainOnly ? ' · Domain owner — lacks execution role' : ''}{item.reason ? ` — ${text(item.reason)}` : ''}
        </li>
      })}</ul>}
    </div>
    <div>
      <h4 className="text-xs font-medium text-huddleroom-text-muted">Blocking gaps</h4>
      {!coverageRecorded ? <p className="mt-1">Not recorded</p> : gaps.length === 0 ? <p className="mt-1">No uncovered responsibilities.</p> : <ul className="mt-1 space-y-1">{gaps.map((item, index) => <li key={text(item.responsibility_id, String(index))} className="min-w-0 break-words">{text(item.responsibility, 'Responsibility')}: {text(item.reason)}</li>)}</ul>}
    </div>
  </section>
}
