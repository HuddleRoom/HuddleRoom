import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useProjectChat, useSubmitProjectChat, useRecordOrchestrationConversationFeedback, useOrchestrationGoal } from '@/api/orchestration'
import { Button, Textarea } from '@/components/common/uiPrimitives'
import { Panel } from '@/components/common/Panel'
import { Dialog } from '@/components/common/Dialog'
import { ErrorRecord } from '@/components/common/ErrorRecord'
import { StartWorkDialog } from '@/pages/orchestration/StartWorkDialog'
import type { OrchestrationAuthorization, OrchestrationGoal, OrchestrationConversationFeedbackReason, ProjectChatAction, ProjectChatTurn } from '@/lib/types'

function historicalAllowanceError(code: unknown): string | null {
  return typeof code === 'string' && ['conversation_disabled', 'conversation_exhausted', 'conversation_allowance_exhausted', 'advisor_disabled', 'advisor_allowance_exhausted'].includes(code)
    ? `Historical conversation stopped at a retired conversation allowance limit (${code}). Send a new message to ask again.` : null
}

function reviewEnvelope(action: ProjectChatAction) {
  try { return JSON.parse(action.summary.slice(action.summary.indexOf('\n') + 1)) as {
    contract?: Partial<OrchestrationGoal>; authorization?: OrchestrationAuthorization
  } } catch { return null }
}

export function ProjectChatPanel({ projectId, contextGoalId = null, focusRequest = 0, active = true }: {
  projectId: string; contextGoalId?: string | null; focusRequest?: number; active?: boolean
}) {
  const query = useProjectChat(projectId)
  const mutation = useSubmitProjectChat(projectId)
  const feedback = useRecordOrchestrationConversationFeedback(projectId)
  const goalQuery = useOrchestrationGoal(projectId, contextGoalId ?? undefined)
  const [draft, setDraft] = useState('')
  const [draftGoalId, setDraftGoalId] = useState<string | null>(null)
  const [feedbackReasons, setFeedbackReasons] = useState<Record<string, OrchestrationConversationFeedbackReason>>({})
  const [replyAction, setReplyAction] = useState<ProjectChatAction | null>(null)
  const [review, setReview] = useState<ProjectChatAction | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const reviewButton = useRef<HTMLButtonElement | null>(null)
  const requestRef = useRef<{ content: string; actionId?: string; contextGoalId: string | null; id: string } | null>(null)
  const [announcement, setAnnouncement] = useState('')
  useEffect(() => { if (focusRequest && active) textareaRef.current?.focus() }, [focusRequest, active])
  const unavailable = query.isLoading || query.isError || !query.data
  const send = (content: string, action?: ProjectChatAction, confirming = false) => {
    if (mutation.isPending || !content.trim()) return
    const goalId = action?.goal_id ?? draftGoalId ?? contextGoalId
    const previous = requestRef.current
    const id = previous && previous.content === content && previous.actionId === action?.action_id && previous.contextGoalId === goalId
      ? previous.id : crypto.randomUUID()
    requestRef.current = { content, actionId: action?.action_id, contextGoalId: goalId, id }
    mutation.mutate({ content, clientRequestId: id, contextGoalId: goalId, actionId: action?.action_id }, {
      onSuccess: (turn) => {
        requestRef.current = null
        if (!confirming) { setDraft(''); setReplyAction(null); setDraftGoalId(null) }
        const staleReceipt = turn.receipts.find((receipt) => receipt.error_code === 'chat_stale_review')
        const refreshed = staleReceipt && turn.actions.find((item) => item.action_id === staleReceipt.action_id && item.goal_id === action?.goal_id && item.operation === action?.operation && item.status === 'awaiting_confirmation')
        if (confirming) setReview(refreshed ?? null)
        setAnnouncement(refreshed ? 'Review changed. Check the refreshed target and limits, then confirm again.' : 'Message recorded.')
      },
    })
  }
  const closeFocus = (event: Event) => { event.preventDefault(); reviewButton.current?.focus() }
  const envelope = review ? reviewEnvelope(review) : null
  const start = review && ['start_execution', 'restart_execution'].includes(review.operation)
  const frozenGoal = start && envelope?.contract ? {
    ...envelope.contract, id: review.goal_id, project_id: projectId,
    constraints: envelope.contract.constraints ?? {}, budget: envelope.contract.budget ?? {},
  } as OrchestrationGoal : null
  const terminalActionIds = new Set(query.data?.items.flatMap((turn) => turn.receipts.filter((receipt) => ['completed', 'applied', 'recorded', 'failed', 'interrupted_unknown'].includes(receipt.status)).map((receipt) => receipt.action_id)) ?? [])
  const goalLabel = (id: string, turn?: ProjectChatTurn) => {
    const citation = turn?.citations.find((item) => item.type === 'goal' && item.id === id)
    const frozen = turn?.actions.find((item) => item.goal_id === id)
    return (frozen && reviewEnvelope(frozen)?.contract?.objective) || citation?.label || (id === contextGoalId ? goalQuery.data?.goal.objective : null) || id
  }
  const renderTurn = (turn: ProjectChatTurn) => <li key={turn.id} className="min-w-0 border-b border-huddleroom-border p-3 last:border-0">
    <p className="font-medium">{turn.historical ? 'Human (historical)' : 'You'}</p><p className="whitespace-pre-wrap break-words">{turn.content}</p>
    <p className="mt-2 font-medium">Orchestrator</p><p className="whitespace-pre-wrap break-words">{turn.answer ?? turn.status}</p>
    <p className="mt-1 text-xs text-huddleroom-text-muted">{turn.historical ? 'Historical · ' : ''}{turn.context_goal_id ? `Goal: ${goalLabel(turn.context_goal_id, turn)}` : 'Project overview'} · {turn.status}</p>
    {turn.source_references.map((source) => <p key={`${source.kind}:${source.id}`} className="break-words text-xs">Source: {source.kind.replace(/_/g, ' ')} · {source.id}</p>)}
    {turn.citations.map((citation) => <Link className="mr-2 text-xs underline" key={`${citation.type}:${citation.id}`} to={citation.type === 'meeting' ? `/meetings/${citation.id}` : `/orchestration/${citation.goal_id ?? citation.id}${citation.type === 'decision' ? `#decision-${citation.id}` : ''}`}>{citation.label}</Link>)}
    {turn.error && <p role="alert" className="break-words text-huddleroom-status-red">{historicalAllowanceError(turn.error.code) ?? String(turn.error.message ?? turn.error.code ?? 'Message failed.')}</p>}
    {turn.actions.map((action) => <section key={action.action_id} className="mt-2 rounded border border-huddleroom-border p-2" aria-label={`${action.operation} action`}>
      <p className="font-medium">{action.operation.replace(/_/g, ' ')} · {terminalActionIds.has(action.action_id) ? 'Result recorded' : action.status}</p>
      <p className="break-words text-xs">Goal: {goalLabel(action.goal_id, turn)} ({action.goal_id}) · Run: {action.run_id ?? 'not started'}</p>
      <p className="whitespace-pre-wrap break-words text-xs">{action.requires_confirmation ? action.summary.split('\n')[0] : action.summary}</p>
      {action.status === 'awaiting_confirmation' && !terminalActionIds.has(action.action_id) && <Button type="button" variant="secondary" onClick={(event) => { reviewButton.current = event.currentTarget; setReview(action) }}>Review {action.operation.replace(/_/g, ' ')}</Button>}
      {['suggested', 'prepared'].includes(action.status) && <Button type="button" variant="secondary" onClick={() => { setReplyAction(action); textareaRef.current?.focus() }}>Reply to this action</Button>}
    </section>)}
    {turn.receipts.map((receipt) => <p key={receipt.action_id} className="mt-2 break-words text-xs">{receipt.operation.replace(/_/g, ' ')} · {receipt.status}: {receipt.message}{receipt.error_code ? ` (${receipt.error_code})` : ''}</p>)}
    {turn.goal_turn?.investigation && <section className="mt-2 text-xs" aria-label="Investigation">
      <p>Investigation: {turn.goal_turn.investigation.status}</p>
      {turn.goal_turn.investigation.status === 'limited' && <p>Historical investigation stopped at a retired conversation allowance limit. Send a new message to ask again.</p>}
      {turn.goal_turn.investigation.status === 'interrupted_unknown' && <p>Investigation outcome is unknown. Usage remains held and the orchestrator will not retry it automatically. Send a new message to ask again.</p>}<p>{turn.goal_turn.investigation.objective}</p>
      {turn.goal_turn.investigation.report && <><p>Findings: {turn.goal_turn.investigation.report.findings}</p><p>Uncertainty: {turn.goal_turn.investigation.report.uncertainty}</p></>}
      {turn.goal_turn.investigation.error && <p role="alert">{turn.goal_turn.investigation.error.code}</p>}
      {turn.goal_turn.investigation.sources.map((source, i) => <p className="break-words" key={i}>{source.operation}: {source.reference}</p>)}
    </section>}
    {turn.goal_turn?.feedback && <p className="text-xs">Feedback recorded: {turn.goal_turn.feedback.rating}</p>}
    {turn.goal_turn?.feedback_eligible && !turn.goal_turn.feedback && turn.context_goal_id && <div className="mt-2 flex min-w-0 flex-wrap gap-2" aria-label="Response feedback">
      <Button variant="secondary" disabled={feedback.isPending} onClick={() => feedback.mutate({ goalId: turn.context_goal_id!, responseId: turn.goal_turn!.response_id, rating: 'helpful', reason: null })}>Helpful</Button>
      <select className="max-w-full" aria-label="Feedback reason" value={feedbackReasons[turn.id] ?? ''} onChange={(event) => setFeedbackReasons((current) => ({ ...current, [turn.id]: event.target.value as OrchestrationConversationFeedbackReason }))}>
        <option value="">Choose reason</option>{(['unanswered', 'incorrect', 'missing_context', 'stale_context', 'unclear', 'too_limited', 'other'] as const).map((reason) => <option key={reason} value={reason}>{reason.replace(/_/g, ' ')}</option>)}
      </select>
      <Button variant="secondary" disabled={feedback.isPending || !feedbackReasons[turn.id]} onClick={() => feedback.mutate({ goalId: turn.context_goal_id!, responseId: turn.goal_turn!.response_id, rating: 'not_helpful', reason: feedbackReasons[turn.id] })}>Needs work</Button>
    </div>}
  </li>
  return <Panel className="flex min-w-0 flex-col gap-3 text-sm">
    <h3 className="font-semibold">Conversation with the orchestrator</h3>
    <p className="text-xs">Context: {goalQuery.data?.goal.objective ?? contextGoalId ?? 'project overview'}</p>
    <p className="text-xs">Questions receive answers. Instructions may require review before taking effect.</p>
    <div role="status" aria-live="polite">{announcement}{mutation.isPending ? ' Sending…' : ''}</div>
    {query.isLoading && <p>Loading conversation…</p>}
    {query.isError && <><ErrorRecord error={query.error} entity="conversation" /><Button onClick={() => void query.refetch()}>Retry</Button></>}
    {query.data && <><ol aria-label="Project conversation" className="rounded border border-huddleroom-border">{query.data.items.map(renderTurn)}</ol>
      {query.data.omitted > 0 && <p className="text-xs">{query.data.omitted} older records omitted.</p>}
      {Object.entries(query.data.steering).map(([goalId, ledger]) => <section key={goalId} className="text-xs" aria-label={`Directions for ${goalId}`}>
        {ledger.requests.map((request) => <div key={request.request_id}><p>Direction · {request.status}: {request.directive}</p>
          {['pending', 'being_considered', 'deferred', 'needs_clarification'].includes(request.status) && <Button variant="secondary" onClick={() => { setDraftGoalId(goalId); setDraft(`Withdraw steering request ${request.request_id}`); textareaRef.current?.focus() }}>Withdraw direction</Button>}
        </div>)}
        {ledger.proposals.filter((proposal) => proposal.status === 'proposed').map((proposal) => <div key={proposal.proposal_id}>
          <p>Pending proposal: {proposal.directive}</p><p>Goal: {goalId} · {proposal.scope} · {proposal.lifetime.replace(/_/g, ' ')}</p><p>{proposal.impact_summary}</p>
          <Button variant="secondary" onClick={() => { setDraftGoalId(goalId); setDraft(`Promote steering proposal ${proposal.proposal_id}`); textareaRef.current?.focus() }}>Use proposal</Button>
          <Button variant="secondary" onClick={() => { setDraftGoalId(goalId); setDraft(`Dismiss proposal ${proposal.proposal_id}`); textareaRef.current?.focus() }}>Dismiss proposal</Button>
        </div>)}
      </section>)}
    </>}
    <form className="flex flex-col gap-2" onSubmit={(event) => { event.preventDefault(); if (!unavailable) send(draft, replyAction ?? undefined) }}>
      {draftGoalId && <p className="text-xs break-words">Draft target: {draftGoalId}<Button variant="secondary" onClick={() => setDraftGoalId(null)}>Use current context</Button></p>}
      {replyAction && <p className="break-words text-xs">Replying to {replyAction.operation} · Goal {replyAction.goal_id}<Button variant="secondary" onClick={() => setReplyAction(null)}>Clear reply</Button></p>}
      <Textarea ref={textareaRef} label="Message the orchestrator" rows={3} value={draft} disabled={mutation.isPending || unavailable} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); if (!unavailable) send(draft, replyAction ?? undefined) } }} />
      <Button type="submit" disabled={mutation.isPending || unavailable || !draft.trim()}>Send</Button>
    </form>
    {mutation.isError && <ErrorRecord error={mutation.error} entity="conversation" />}
    {feedback.isError && <ErrorRecord error={feedback.error} entity="feedback" />}
    {review && frozenGoal && <StartWorkDialog open targetSummary={`Goal: ${frozenGoal.objective ?? review.goal_id} (${review.goal_id}) · Run: ${review.run_id ?? 'not started'}`} goal={frozenGoal} authorization={envelope?.authorization ?? null} missingRecurringPolicy={frozenGoal.goal_type === 'continuous' && !frozenGoal.continuous_policy} restartReason={review.operation === 'restart_execution' ? 'Fresh authorization is required' : null} isPending={mutation.isPending} error={mutation.error} onOpenChange={(open) => { if (!open) setReview(null) }} onConfirm={() => send('yes', review, true)} onCloseAutoFocus={closeFocus} />}
    {review && !frozenGoal && <Dialog open title={`Confirm ${review.operation.replace(/_/g, ' ')}?`} role="alertdialog" onOpenChange={(open) => { if (!open) setReview(null) }} onCloseAutoFocus={closeFocus} footer={{ primaryLabel: 'Confirm', isPending: mutation.isPending, onPrimary: () => send('yes', review, true) }}>
      <p className="break-words">Goal: {envelope?.contract?.objective ?? review.goal_id} ({review.goal_id}) · Run: {review.run_id ?? 'not started'}</p><p className="whitespace-pre-wrap break-words">{review.summary.split('\n')[0]}</p>
      {review.operation === 'cancel' && <p>Cancels this goal and its children, tasks and sessions.</p>}
      {review.operation === 'pause' && <p>Pauses work; baseline authorization is cleared during baseline.</p>}
      {mutation.isError && <ErrorRecord error={mutation.error} entity="confirmation" />}
    </Dialog>}
  </Panel>
}
