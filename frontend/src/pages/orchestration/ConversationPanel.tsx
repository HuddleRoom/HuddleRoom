import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type RefObject } from 'react'
import { useDismissOrchestrationSteeringProposal, useOrchestrationConversation, useRecordOrchestrationConversationFeedback, useSubmitOrchestrationConversation, useSubmitOrchestrationSteering, useWithdrawOrchestrationSteering } from '@/api/orchestration'
import { Button, SkeletonRow, StatusBadge, Textarea } from '@/components/common/uiPrimitives'
import { Record } from '@/components/common/Record'
import { Panel } from '@/components/common/Panel'
import { ErrorRecord } from '@/components/common/ErrorRecord'
import { Dialog } from '@/components/common/Dialog'
import type { OrchestrationConversationFeedbackReason, OrchestrationConversationInvestigation, OrchestrationConversationInvestigationStatus, OrchestrationConversationTurn, OrchestrationSteeringLedger, OrchestrationSteeringLifetime, OrchestrationSteeringProposal, OrchestrationSteeringScope, OrchestrationSteeringTargetType } from '@/lib/types'
import { absolute } from '@/lib/time'
import { useGoalAnnouncer } from './goalAnnouncer'

function answerText(turn: OrchestrationConversationTurn) {
  if (turn.answer !== null) return <><span className="font-medium">Answer</span><p className="whitespace-pre-wrap break-words">{turn.answer}</p></>
  if (turn.error) return <><span className="font-medium">Error</span><p className="font-mono text-[13px]">{turn.error.code}</p></>
  if (turn.status === 'pending' || turn.status === 'running') return <p>Awaiting response.</p>
  if (turn.status === 'interrupted_unknown') return <p>Response status is unknown after interruption.</p>
  return <p>No response recorded.</p>
}

const investigationStatusCopy: Record<OrchestrationConversationInvestigationStatus, string> = {
  pending: 'Investigation queued.',
  running: 'Investigation in progress.',
  completed: 'Investigation completed.',
  limited: 'Investigation stopped at the conversation allowance limit. Send a new message when allowance is available.',
  failed: 'Investigation failed safely. No finding was accepted.',
  cancelled: 'Investigation cancelled. Send a new message to investigate again; late output is ignored.',
  unavailable: 'Investigation is not enabled for this project. The conversation remains read-only.',
  interrupted_unknown: 'Investigation outcome is unknown. Usage remains held and the orchestrator will not retry it automatically. Send a new message if allowance remains.',
}

function InvestigationDetails({ investigation, sequence, isLatest }: {
  investigation: OrchestrationConversationInvestigation
  sequence: number
  isLatest: boolean
}) {
  const { announceQueue } = useGoalAnnouncer()
  useEffect(() => {
    if (isLatest) announceQueue(investigationStatusCopy[investigation.status])
  }, [isLatest, investigation.status, announceQueue])
  return <section aria-label={`Investigation for message ${sequence}`} className="mt-3 min-w-0 overflow-hidden text-sm">
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      <span className="font-medium">Investigation</span>
      <StatusBadge status={investigation.status} label={investigationStatusLabel[investigation.status] ?? investigation.status} />
      <span className="text-xs text-huddleroom-text-muted">Advisory · unverified</span>
    </div>
    <p className="mt-2 text-huddleroom-text-secondary">
      {investigationStatusCopy[investigation.status]}
    </p>
    <p className="mt-2 break-words"><span className="font-medium">Scope: </span>{investigation.objective}</p>
    {investigation.report && <div className="mt-2 min-w-0 break-words">
      <p className="break-words"><span className="font-medium">Findings: </span>{investigation.report.findings}</p>
      <p className="break-words"><span className="font-medium">Uncertainty: </span>{investigation.report.uncertainty || 'None reported.'}</p>
      <p className="mt-2 text-xs text-huddleroom-text-muted">This report is advisory and is not accepted evidence.</p>
    </div>}
    {investigation.error && <p className="mt-2 break-all font-mono text-[13px]">{investigation.error.code}</p>}
    <details className="mt-2 min-w-0 text-xs text-huddleroom-text-secondary">
      <summary className="cursor-pointer">Sources ({investigation.sources.length})</summary>
      <ul className="mt-2 min-w-0 space-y-1 overflow-hidden">
        {investigation.sources.map((source, index) => <li key={`${source.operation}:${source.reference}:${index}`} className="min-w-0 break-all font-mono text-[13px]">
          {source.reference}{' · '}{source.operation}{' · '}{source.status}{' · '}freshness {source.freshness_at ?? '—'}{' · '}truncated {String(source.truncated)}
        </li>)}
      </ul>
    </details>
    {investigation.finished_at && <time dateTime={investigation.finished_at} className="mt-2 block text-xs text-huddleroom-text-muted">
      Investigation finished {absolute(investigation.finished_at)}
    </time>}
  </section>
}

function mutationErrorText(error: unknown) {
  if (error && typeof error === 'object') {
    const detail = 'detail' in error && error.detail
    if (detail && typeof detail === 'object' && 'message' in detail && typeof detail.message === 'string') return detail.message
  }
  return 'Message was not sent. Check your connection and try again.'
}

function Context({ turn }: { turn: OrchestrationConversationTurn }) {
  const manifest = turn.context_manifest
  return <details className="min-w-0 text-xs text-huddleroom-text-secondary">
    <summary className="cursor-pointer">Context for message {turn.sequence}</summary>
    <dl className="mt-2 grid gap-1 break-words">
      <div><dt className="inline">Context version: </dt><dd className="inline font-mono text-[13px]">{turn.context_version}</dd></div>
      <div><dt className="inline">Run context: </dt><dd className="inline font-mono text-[13px]">{turn.run_id ?? 'No run attached'}</dd></div>
      <div><dt className="inline">Truncated: </dt><dd className="inline">{String(manifest.truncated)}</dd></div>
      <div><dt className="inline">Excluded categories: </dt><dd className="inline">{manifest.excluded_categories.join(', ') || 'None'}</dd></div>
      {manifest.sources.map((source, sourceIndex) => <div key={`${source.source}:${sourceIndex}`} className="border-t border-huddleroom-border pt-1">
        <span className="font-mono text-[13px]">{source.source}</span>{' · '}{source.status}{' · '}freshness {source.freshness_at ?? '—'}{' · '}available {source.available}{' · '}included {source.included}{' · '}omitted {source.omitted}{' · '}truncated {String(source.truncated)}
        {source.references.map((reference, referenceIndex) => <span key={`${reference}:${referenceIndex}`} className="block break-all font-mono text-[13px]">{reference}</span>)}
      </div>)}
    </dl>
  </details>
}

const feedbackReasons: Array<[string, OrchestrationConversationFeedbackReason]> = [
  ['Did not answer', 'unanswered'], ['Incorrect', 'incorrect'], ['Missing context', 'missing_context'],
  ['Stale context', 'stale_context'], ['Unclear', 'unclear'], ['Too limited', 'too_limited'], ['Other', 'other'],
]

const feedbackReasonCopy: Record<OrchestrationConversationFeedbackReason, string> = Object.fromEntries(feedbackReasons.map(([label, reason]) => [reason, label])) as Record<OrchestrationConversationFeedbackReason, string>

type FeedbackDraft = { responseId: string; reason: OrchestrationConversationFeedbackReason | null; phase: 'choose-reason' | 'pending' | 'error' | 'success'; rating: 'helpful' | 'not_helpful' }

function Ledger({ items, onReview, onDismiss, mutationsPending, canSteer, unavailableReason, feedbackDraft, feedbackPending, onHelpful, onNeedsWork, onReason, onRecord, feedbackErrorRef, feedbackError }: {
  items: OrchestrationConversationTurn[]
  onReview: (proposal: OrchestrationSteeringProposal) => void
  onDismiss: (proposalId: string) => void
  mutationsPending: boolean
  canSteer: boolean
  unavailableReason: string
  feedbackDraft: FeedbackDraft | null
  feedbackPending: boolean
  onHelpful: (responseId: string) => void
  onNeedsWork: (responseId: string) => void
  onReason: (reason: OrchestrationConversationFeedbackReason) => void
  onRecord: () => void
  feedbackErrorRef: RefObject<HTMLDivElement | null>
  feedbackError: unknown
}) {
  let priorRun: string | null | undefined
  const latestInvestigationSequence = Math.max(...items.filter((turn) => turn.investigation).map((turn) => turn.sequence))
  return <ol aria-label="Conversation history" className="overflow-hidden rounded-md border border-huddleroom-border bg-huddleroom-depth">
    {items.map((turn, index) => {
      const boundary = index === 0 || turn.run_id !== priorRun
      priorRun = turn.run_id
      return <li key={turn.response_id} className="min-w-0 border-b border-huddleroom-border p-3 last:border-b-0">
        {boundary && <div role="separator" aria-label={`Run context: ${turn.run_id ?? 'No run attached'}`} className="mb-2 border-b border-huddleroom-border pb-2 text-xs font-mono text-huddleroom-text-muted">Run context: {turn.run_id ?? 'No run attached'}</div>}
        <div className="flex min-w-0 flex-wrap items-center gap-2"><span className="font-medium">Message {turn.sequence}</span><StatusBadge status={turn.status} label={messageStatusLabel[turn.status] ?? turn.status} /><span className="min-w-0 break-all font-mono text-[13px] text-huddleroom-text-muted">{turn.actor_id}</span></div>
        <div className="mt-2 min-w-0 text-sm text-huddleroom-text-primary"><span className="font-medium">Message</span><p className="whitespace-pre-wrap break-words">{turn.content}</p></div>
        <div className="mt-2 min-w-0 text-sm text-huddleroom-text-secondary">{answerText(turn)}</div>
        {(turn.feedback || turn.feedback_eligible === true) && <section aria-label={`Feedback for message ${turn.sequence}`} className={turn.feedback ? 'mt-3 min-w-0 break-words text-sm text-huddleroom-text-secondary' : 'mt-3 min-w-0'}>
          {turn.feedback ? <p>Feedback recorded: {turn.feedback.rating === 'helpful' ? 'Helpful' : `Needs work — ${turn.feedback.reason ? feedbackReasonCopy[turn.feedback.reason] : 'Other'}`}</p>
            : (() => {
              const active = feedbackDraft?.responseId === turn.response_id ? feedbackDraft : null
              const disabled = feedbackPending
              return <>
                <p className="text-sm font-medium text-huddleroom-text-primary">Was this answer useful?</p>
                <div className="mt-2 flex min-w-0 flex-wrap gap-2">
                  <Button type="button" variant="secondary" className="min-h-11" disabled={disabled} onClick={() => onHelpful(turn.response_id)}>{active?.phase === 'pending' && active.rating === 'helpful' ? 'Recording…' : 'Helpful'}</Button>
                  <Button type="button" variant="secondary" className="min-h-11" disabled={disabled} onClick={() => onNeedsWork(turn.response_id)}>Needs work</Button>
                </div>
                {active?.rating === 'not_helpful' && <fieldset className="mt-3 min-w-0" disabled={disabled}><legend className="text-sm font-medium text-huddleroom-text-primary">What needs work?</legend><div className="mt-2 flex min-w-0 flex-col gap-1">
                  {feedbackReasons.map(([label, reason]) => { const id = `feedback-${turn.response_id}-${reason}`; return <label key={reason} htmlFor={id} className="flex min-h-11 min-w-0 items-center gap-2 break-words text-sm"><input ref={(input) => input?.setAttribute('value', reason)} id={id} type="radio" name={`feedback-${turn.response_id}`} value={reason} checked={active.reason === reason} disabled={disabled} onChange={() => onReason(reason)} />{label}</label> })}
                </div>
                {active.reason && <div className="mt-2 flex min-w-0 flex-wrap gap-2"><Button type="button" className="min-h-11" disabled={disabled} onClick={onRecord}>{disabled ? 'Recording…' : 'Record feedback'}</Button></div>}
                </fieldset>}
                {active?.phase === 'error' && <div ref={feedbackErrorRef} tabIndex={-1} className="mt-2 min-w-0"><ErrorRecord error={feedbackError} entity="feedback" action={<Button type="button" variant="secondary" className="min-h-11" onClick={onRecord}>Retry</Button>} /></div>}
              </>
            })()}
        </section>}
        {turn.proposed_steering && <Panel className="mt-3">
          <div aria-label={`Proposed steering for message ${turn.sequence}`}>
            <h5 className="text-sm font-medium">Proposed steering</h5>
            <p className="mt-1 text-xs text-huddleroom-text-secondary">Advisory — not applied</p>
            <Record className="mt-2" keyWidth={72} rows={[
              { key: 'Directive', value: <p className="whitespace-pre-wrap break-words">{turn.proposed_steering.directive}</p> },
              { key: 'Impact', value: <p className="break-words text-huddleroom-text-secondary">{turn.proposed_steering.impact_summary}</p> },
              { key: 'Target', value: `${turn.proposed_steering.target_type} ${turn.proposed_steering.target_id}` },
              { key: 'Scope', value: turn.proposed_steering.scope },
              { key: 'Lifetime', value: turn.proposed_steering.lifetime },
            ]} />
            {turn.proposed_steering.status === 'proposed' && <div className="mt-2 flex flex-wrap gap-2">{canSteer ? <Button type="button" variant="secondary" disabled={mutationsPending} onClick={() => onReview(turn.proposed_steering!)}>Review & apply</Button> : <p className="text-xs text-huddleroom-text-muted">Review unavailable: {unavailableReason}</p>}<Button type="button" variant="secondary" disabled={mutationsPending} onClick={() => onDismiss(turn.proposed_steering!.proposal_id)}>Dismiss</Button></div>}
            {turn.proposed_steering.status === 'dismissed' && <p className="mt-2 text-xs text-huddleroom-text-muted">Dismissed</p>}
          </div>
        </Panel>}
        {turn.investigation && <InvestigationDetails investigation={turn.investigation} sequence={turn.sequence} isLatest={turn.sequence === latestInvestigationSequence} />}
        <time dateTime={turn.message_created_at} className="mt-2 block text-xs text-huddleroom-text-muted">{absolute(turn.message_created_at)}</time>
        {turn.finished_at && <time dateTime={turn.finished_at} className="block text-xs text-huddleroom-text-muted">Finished {absolute(turn.finished_at)}</time>}
        <div className="mt-2"><Context turn={turn} /></div>
      </li>
    })}
  </ol>
}

const steeringStatusCopy: Record<string, string> = {
  pending: 'Submitted — awaiting control plane',
  being_considered: 'Being considered at a safe boundary',
  applied: 'Applied as advisory direction',
  deferred: 'Deferred', rejected: 'Rejected', superseded: 'Superseded',
  needs_clarification: 'Needs clarification', withdrawn: 'Withdrawn',
}

const messageStatusLabel: Record<string, string> = { pending: 'Pending', running: 'Running', completed: 'Answered', failed: 'Failed', interrupted_unknown: 'Interrupted' }
const investigationStatusLabel: Record<string, string> = { pending: 'Pending', running: 'Running', completed: 'Completed', limited: 'Limited', failed: 'Failed', cancelled: 'Cancelled', unavailable: 'Not enabled', interrupted_unknown: 'Interrupted' }

function SteeringLedger({ ledger, onWithdraw, onReplace, mutationsPending }: {
  ledger: OrchestrationSteeringLedger
  onWithdraw: (requestId: string) => void
  onReplace: (request: OrchestrationSteeringLedger['requests'][number]) => void
  mutationsPending: boolean
}) {
  const { announceQueue } = useGoalAnnouncer()
  const requests = ledger.requests ?? []
  const transitions = requests.flatMap((request) => request.transitions.map((transition, index) => ({ request, transition, index })))
    .sort((left, right) => right.transition.created_at.localeCompare(left.transition.created_at) || right.request.sequence - left.request.sequence || right.index - left.index)
  const newest = transitions[0]
  useEffect(() => {
    if (newest) announceQueue(`${steeringStatusCopy[newest.transition.status]} · ${newest.transition.reason_code}`)
  }, [newest?.transition.status, newest?.transition.reason_code, announceQueue])
  if (!ledger.enabled || requests.length === 0) return null
  return <section aria-label="Steering ledger" className="min-w-0">
    <h4 className="text-sm font-medium">Steering</h4>
    <ol aria-label="Steering requests" className="mt-2 space-y-3">
      {requests.map((request) => <li key={request.request_id} className="min-w-0">
        <Panel>
          <div className="flex min-w-0 flex-wrap items-center gap-2"><StatusBadge status={request.status} label={steeringStatusCopy[request.status]} /></div>
          <Record className="mt-2" keyWidth={72} rows={[
            { key: 'Directive', value: <p className="whitespace-pre-wrap break-words">{request.directive}</p> },
            { key: 'Impact', value: <p className="break-words text-huddleroom-text-secondary">{request.impact_summary}</p> },
            { key: 'Target', value: `${request.target_type} ${request.target_id}` },
            { key: 'Scope', value: request.scope },
            { key: 'Lifetime', value: request.lifetime },
          ]} />
          {newest?.request.request_id === request.request_id && <p className="mt-1 break-words text-xs text-huddleroom-text-muted">{steeringStatusCopy[newest.transition.status]} · {newest.transition.reason_code}</p>}
          <details className="mt-2"><summary className="cursor-pointer text-xs text-huddleroom-text-secondary">Transition history ({request.transitions.length})</summary><ol aria-label={`Transitions for steering request ${request.sequence}`} className="mt-2 space-y-1 text-xs text-huddleroom-text-secondary">{request.transitions.map((transition, index) => <li key={`${transition.created_at}:${index}`} className="break-words"><time dateTime={transition.created_at}>{absolute(transition.created_at)}</time>{' · '}{steeringStatusCopy[transition.status] ?? transition.status}{' · '}{transition.reason_code}</li>)}</ol></details>
          {request.result_action_ids.map((actionId) => <a key={actionId} href={`#action-${actionId}`} className="mr-2 text-xs text-huddleroom-primary underline-offset-2 hover:underline">View action in Activity</a>)}
          {request.status === 'pending' && <Button type="button" variant="secondary" className="mt-2 min-h-11" disabled={mutationsPending} onClick={() => onWithdraw(request.request_id)}>Withdraw</Button>}
          {request.status === 'applied' && <Button type="button" variant="secondary" className="mt-2 min-h-11" disabled={mutationsPending} onClick={() => onReplace(request)}>Replace direction</Button>}
        </Panel>
      </li>)}
    </ol>
  </section>
}

export function ConversationPanel({ projectId, goalId }: { projectId: string; goalId: string }) {
  const query = useOrchestrationConversation(projectId, goalId)
  const mutation = useSubmitOrchestrationConversation(projectId)
  const submitSteering = useSubmitOrchestrationSteering(projectId)
  const withdrawSteering = useWithdrawOrchestrationSteering(projectId)
  const dismissProposal = useDismissOrchestrationSteeringProposal(projectId)
  const feedback = useRecordOrchestrationConversationFeedback(projectId)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const steeringDirectiveRef = useRef<HTMLTextAreaElement>(null)
  const errorRef = useRef<HTMLParagraphElement>(null)
  const steeringErrorRef = useRef<HTMLDivElement>(null)
  const feedbackErrorRef = useRef<HTMLDivElement>(null)
  const steeringAttempt = useRef(0)
  const feedbackAttempt = useRef(0)
  const candidateId = useRef<string | null>(null)
  const lastSteeringActionRef = useRef<'submit' | (() => void) | null>(null)
  const [draft, setDraft] = useState('')
  const [sent, setSent] = useState(false)
  const [errorDismissed, setErrorDismissed] = useState(false)
  const [steeringOpen, setSteeringOpen] = useState(false)
  const [steeringDraft, setSteeringDraft] = useState<{ clientRequestId: string; directive: string; impactSummary: string; scope: OrchestrationSteeringScope; targetType: OrchestrationSteeringTargetType; targetId: string; lifetime: OrchestrationSteeringLifetime; sourceProposalId: string | null; supersedesRequestId: string | null }>({ clientRequestId: '', directive: '', impactSummary: '', scope: 'run', targetType: 'goal', targetId: goalId, lifetime: 'remaining_current_run', sourceProposalId: null, supersedesRequestId: null })
  const [steeringErrorDismissed, setSteeringErrorDismissed] = useState(false)
  const [activeSteeringMutation, setActiveSteeringMutation] = useState<'submit' | 'withdraw' | 'dismiss' | null>(null)
  const [steeringMutationPending, setSteeringMutationPending] = useState(false)
  const [feedbackDraft, setFeedbackDraft] = useState<FeedbackDraft | null>(null)
  const [feedbackAnnouncementScope, setFeedbackAnnouncementScope] = useState<{ projectId: string; goalId: string } | null>(null)
  const { announceQueue, announceError } = useGoalAnnouncer()

  useEffect(() => { if (mutation.isError && !errorDismissed) errorRef.current?.focus() }, [mutation.isError, errorDismissed])
  const activeSteeringError = activeSteeringMutation === 'submit' ? submitSteering.error : activeSteeringMutation === 'withdraw' ? withdrawSteering.error : activeSteeringMutation === 'dismiss' ? dismissProposal.error : null
  const activeSteeringFailed = activeSteeringMutation === 'submit' ? submitSteering.isError : activeSteeringMutation === 'withdraw' ? withdrawSteering.isError : activeSteeringMutation === 'dismiss' ? dismissProposal.isError : false
  useEffect(() => { if (activeSteeringFailed && !steeringErrorDismissed) steeringErrorRef.current?.focus() }, [activeSteeringFailed, steeringErrorDismissed])
  useEffect(() => {
    if (feedback.isError && feedbackDraft?.phase === 'pending') setFeedbackDraft((draft) => draft ? { ...draft, phase: 'error' } : draft)
  }, [feedback.isError, feedbackDraft?.phase])
  useEffect(() => { if (feedbackDraft?.phase === 'error') feedbackErrorRef.current?.focus() }, [feedbackDraft?.phase])
  useEffect(() => {
    feedbackAttempt.current += 1
    setFeedbackAnnouncementScope(null)
    setFeedbackDraft(null)
  }, [projectId, goalId])
  useEffect(() => {
    if (activeSteeringFailed) {
      steeringAttempt.current += 1
      setSteeringMutationPending(false)
    }
  }, [activeSteeringFailed])

  const unknown = query.isLoading || query.isError || !query.data
  const allowance = query.data?.allowance
  const disabled = unknown || !allowance?.enabled || allowance.remaining <= 0
  const submit = () => {
    if (disabled || mutation.isPending || !draft.trim()) return
    setSent(false)
    setErrorDismissed(false)
    const clientRequestId = candidateId.current ?? crypto.randomUUID()
    candidateId.current = clientRequestId
    mutation.mutate({ goalId, clientRequestId, content: draft }, {
      onSuccess: () => { candidateId.current = null; setDraft(''); setSent(true); setErrorDismissed(true); textareaRef.current?.focus() },
    })
  }
  const onSubmit = (event: FormEvent) => { event.preventDefault(); submit() }
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() }
  }
  const onChange = (value: string) => { candidateId.current = null; setSent(false); setErrorDismissed(true); setDraft(value) }
  const safeError = mutationErrorText(mutation.error)
  const steering = query.data?.steering
  const canSteer = steering?.enabled && (steering.eligibility === 'active' || steering.eligibility === 'paused')
  const steeringMutationsPending = steeringMutationPending || submitSteering.isPending || withdrawSteering.isPending || dismissProposal.isPending
  const feedbackAnnouncementVisible = feedbackAnnouncementScope?.projectId === projectId && feedbackAnnouncementScope.goalId === goalId
  useEffect(() => { if (query.isLoading) announceQueue('Loading conversation.') }, [query.isLoading, announceQueue])
  useEffect(() => { if (query.isError) announceError('Failed to load conversation. Retry to view messages.') }, [query.isError, announceError])
  useEffect(() => { if (mutation.isError && !errorDismissed) announceError(safeError) }, [mutation.isError, errorDismissed, safeError, announceError])
  useEffect(() => { if (sent) announceQueue('Message sent.') }, [sent, announceQueue])
  useEffect(() => { if (feedbackAnnouncementVisible) announceQueue('Feedback recorded.') }, [feedbackAnnouncementVisible, announceQueue])
  const recordFeedback = (responseId: string, rating: 'helpful' | 'not_helpful', reason: OrchestrationConversationFeedbackReason | null) => {
    setFeedbackAnnouncementScope(null)
    setFeedbackDraft({ responseId, rating, reason, phase: 'pending' })
    const scope = { projectId, goalId }
    const attempt = ++feedbackAttempt.current
    feedback.mutate({ goalId, responseId, rating, reason }, { onSuccess: () => {
      if (feedbackAttempt.current !== attempt) return
      setFeedbackAnnouncementScope(scope)
      setFeedbackDraft((draft) => draft?.responseId === responseId ? { ...draft, phase: 'success' } : draft)
    } })
  }
  useEffect(() => {
    if (feedbackDraft && query.data?.items.some((turn) => turn.response_id === feedbackDraft.responseId && (turn.feedback || !turn.feedback_eligible))) setFeedbackDraft(null)
  }, [feedbackDraft, query.data])
  useEffect(() => {
    if (!canSteer && steeringOpen) {
      setSteeringDraft({ clientRequestId: '', directive: '', impactSummary: '', scope: 'run', targetType: 'goal', targetId: goalId, lifetime: 'remaining_current_run', sourceProposalId: null, supersedesRequestId: null })
      setSteeringOpen(false); setSteeringErrorDismissed(true)
    }
  }, [canSteer, goalId, steeringOpen])
  const discardSteering = () => {
    if (activeSteeringMutation === 'submit') { submitSteering.reset(); setActiveSteeringMutation(null); lastSteeringActionRef.current = null }
    setSteeringDraft({ clientRequestId: '', directive: '', impactSummary: '', scope: 'run', targetType: 'goal', targetId: goalId, lifetime: 'remaining_current_run', sourceProposalId: null, supersedesRequestId: null })
    setSteeringOpen(false); setSteeringErrorDismissed(true)
  }
  const submitDraft = () => {
    if (!canSteer || steeringMutationsPending || !steeringDraft.directive.trim() || !steeringDraft.impactSummary.trim()) return
    lastSteeringActionRef.current = 'submit'
    const clientRequestId = steeringDraft.clientRequestId || crypto.randomUUID()
    if (!steeringDraft.clientRequestId) setSteeringDraft((draft) => ({ ...draft, clientRequestId }))
    setSteeringErrorDismissed(false)
    setActiveSteeringMutation('submit')
    setSteeringMutationPending(true)
    const attempt = ++steeringAttempt.current
    submitSteering.mutate({
      goalId, clientRequestId, directive: steeringDraft.directive.trim(), targetType: steeringDraft.targetType, targetId: steeringDraft.targetId,
      scope: steeringDraft.scope, lifetime: steeringDraft.lifetime,
      impactSummary: steeringDraft.impactSummary.trim(), sourceProposalId: steeringDraft.sourceProposalId, supersedesRequestId: steeringDraft.supersedesRequestId,
    }, {
      onSuccess: () => {
        if (steeringAttempt.current !== attempt) return
        setSteeringMutationPending(false); setActiveSteeringMutation(null); discardSteering()
      },
      onError: () => { if (steeringAttempt.current === attempt) setSteeringMutationPending(false) },
    })
  }

  // ponytail: hide panel entirely when loaded + disabled allowance + zero items; preserves history-with-disabled-composer case
  if (!unknown && !allowance?.enabled && query.data.items.length === 0) return null

  return <section aria-label="Conversation" className="flex flex-col gap-3 rounded-md border border-huddleroom-border bg-huddleroom-surface p-3">
    <h3 className="text-sm font-semibold text-huddleroom-text-primary">Conversation</h3>
    <p className="text-sm text-huddleroom-text-secondary">Messages are advisory. They do not change goal state, plans, agents, or reservations.</p>
    {query.isLoading ? <div aria-busy="true" aria-label="Loading conversation" className="space-y-2"><SkeletonRow /><SkeletonRow /><SkeletonRow /></div>
      : query.isError || !query.data ? <div className="flex flex-wrap items-center gap-2"><span className="text-sm text-huddleroom-status-red">Failed to load conversation. Retry to view messages.</span><Button type="button" variant="secondary" className="min-h-11" onClick={() => { void query.refetch() }}>Retry</Button></div>
        : <>
          {query.data.allowance.enabled && <p className="text-xs text-huddleroom-text-secondary">{`Allowance: ${query.data.allowance.remaining} remaining of ${query.data.allowance.limit}`}</p>}
          {query.data.allowance.enabled && query.data.allowance.remaining <= 0 && <p className="text-sm text-huddleroom-status-red">Conversation allowance is exhausted. New messages are unavailable.</p>}
          {!query.data.allowance.enabled && <p className="text-sm text-huddleroom-text-secondary">Conversation is not enabled for this goal.</p>}
          {query.data.items.length === 0 ? <p className="text-sm text-huddleroom-text-secondary">No conversation messages yet. Send an advisory question to the orchestrator.</p> : <Ledger items={query.data.items} mutationsPending={steeringMutationsPending} canSteer={Boolean(canSteer)} unavailableReason={steering?.eligibility_reason ?? 'steering is not enabled for this goal'} feedbackDraft={feedbackDraft} feedbackPending={feedback.isPending} feedbackErrorRef={feedbackErrorRef} feedbackError={feedback.error} onHelpful={(responseId) => recordFeedback(responseId, 'helpful', null)} onNeedsWork={(responseId) => setFeedbackDraft({ responseId, rating: 'not_helpful', reason: null, phase: 'choose-reason' })} onReason={(reason) => setFeedbackDraft((draft) => draft ? { ...draft, reason, phase: 'choose-reason' } : draft)} onRecord={() => { if (feedbackDraft?.reason) recordFeedback(feedbackDraft.responseId, 'not_helpful', feedbackDraft.reason) }} onReview={(proposal) => { setSteeringDraft({ clientRequestId: '', directive: proposal.directive, impactSummary: proposal.impact_summary, scope: proposal.scope, targetType: proposal.target_type, targetId: proposal.target_id, lifetime: proposal.lifetime, sourceProposalId: proposal.proposal_id, supersedesRequestId: null }); setSteeringOpen(true) }} onDismiss={(proposalId) => { if (steeringMutationsPending) return; const dismissAction = () => { setSteeringErrorDismissed(false); setActiveSteeringMutation('dismiss'); setSteeringMutationPending(true); const attempt = ++steeringAttempt.current; dismissProposal.mutate({ goalId, proposalId }, { onSuccess: () => { if (steeringAttempt.current === attempt) { setSteeringMutationPending(false); setActiveSteeringMutation(null) } }, onError: () => { if (steeringAttempt.current === attempt) setSteeringMutationPending(false) } }) }; lastSteeringActionRef.current = dismissAction; dismissAction() }} />}
        </>}
    <form onSubmit={onSubmit} className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-end">
      <Textarea ref={textareaRef} label="Advisory message" placeholder="Ask about the current goal context…" rows={3} value={draft} disabled={disabled || mutation.isPending} onChange={(event) => onChange(event.target.value)} onKeyDown={onKeyDown} className="min-w-0 flex-1 break-words" />
      <Button type="submit" disabled={disabled || mutation.isPending} className="min-h-11 shrink-0 sm:mb-3 sm:w-auto">{mutation.isPending ? 'Sending…' : 'Send message'}</Button>
      {canSteer && !steeringOpen && <Button type="button" variant="secondary" className="min-h-11 shrink-0 sm:mb-3 sm:w-auto" onClick={() => { setSteeringErrorDismissed(true); setSteeringOpen(true) }}>Steer work</Button>}
    </form>
    <Dialog
      open={steeringOpen}
      onOpenChange={(next) => { if (!next) discardSteering() }}
      size="md"
      title={steeringDraft.sourceProposalId !== null || steeringDraft.supersedesRequestId !== null ? 'Review and apply steering' : 'Steer work'}
      description={`Submitting will ask the control plane to consider this direction at a safe boundary. It will not interrupt running work.${steering?.eligibility === 'paused' ? ' It waits for Resume and will not dispatch.' : ''}`}
      initialFocusRef={steeringDirectiveRef}
      footer={{
        cancelLabel: 'Discard',
        primaryLabel: steeringMutationsPending ? 'Submitting…' : 'Submit steering',
        onPrimary: submitDraft,
        isPending: steeringMutationsPending,
        primaryDisabled: !steeringDraft.directive.trim() || !steeringDraft.impactSummary.trim(),
      }}
    >
      <Textarea ref={steeringDirectiveRef} label="Steering directive" rows={3} value={steeringDraft.directive} onChange={(event) => setSteeringDraft((draft) => ({ ...draft, directive: event.target.value, clientRequestId: '' }))} className="min-w-0 break-words" />
      <Record className="mb-3" keyWidth={72} rows={[{ key: 'Target', value: <span className="break-all">{steeringDraft.targetType} {steeringDraft.targetId}</span> }]} />
      <fieldset className="mb-3 min-w-0"><legend className="text-xs font-medium text-huddleroom-text-secondary">Scope and lifetime</legend><div className="mt-1 flex flex-col gap-2">
        {([['item', 'Selected item — selected item only', 'selected_item'], ['run', 'Current run — remaining unstarted work in current run', 'remaining_current_run'], ['goal', 'Goal — future runs', 'future_runs']] as const).map(([scope, label, lifetime]) => <label key={scope} className="flex items-center gap-2 text-sm"><input type="radio" name="steering-scope" checked={steeringDraft.scope === scope} disabled={steeringDraft.sourceProposalId !== null || scope === 'item'} onChange={() => setSteeringDraft((draft) => ({ ...draft, scope, lifetime, clientRequestId: '' }))} />{label}</label>)}
      </div></fieldset>
      <Textarea label="Expected impact" placeholder="Describe the expected effect on unstarted work." rows={2} value={steeringDraft.impactSummary} onChange={(event) => setSteeringDraft((draft) => ({ ...draft, impactSummary: event.target.value, clientRequestId: '' }))} className="min-w-0 break-words" />
      {activeSteeringMutation === 'submit' && submitSteering.isError && !steeringErrorDismissed && <div ref={steeringErrorRef} tabIndex={-1}><ErrorRecord error={submitSteering.error} entity="steering request" action={<Button type="button" variant="secondary" className="min-h-11" onClick={submitDraft}>Retry</Button>} /></div>}
    </Dialog>
    {query.data && <SteeringLedger ledger={query.data.steering} mutationsPending={steeringMutationsPending} onReplace={(request) => { setSteeringDraft({ clientRequestId: '', directive: request.directive, impactSummary: request.impact_summary, scope: request.scope, targetType: request.target_type, targetId: request.target_id, lifetime: request.lifetime, sourceProposalId: null, supersedesRequestId: request.request_id }); setSteeringOpen(true) }} onWithdraw={(requestId) => {
      if (steeringMutationsPending) return
      const withdrawAction = () => {
        setSteeringErrorDismissed(false); setActiveSteeringMutation('withdraw'); setSteeringMutationPending(true)
        const attempt = ++steeringAttempt.current
        withdrawSteering.mutate({ goalId, requestId }, {
          onSuccess: () => { if (steeringAttempt.current === attempt) { setSteeringMutationPending(false); setActiveSteeringMutation(null) } },
          onError: () => { if (steeringAttempt.current === attempt) setSteeringMutationPending(false) },
        })
      }
      lastSteeringActionRef.current = withdrawAction
      withdrawAction()
    }} />}
    {mutation.isError && !errorDismissed && <p ref={errorRef} tabIndex={-1} className="text-sm text-huddleroom-status-red">{safeError}</p>}
    {activeSteeringFailed && activeSteeringMutation !== 'submit' && !steeringErrorDismissed && <div ref={steeringErrorRef} tabIndex={-1}><ErrorRecord error={activeSteeringError} entity="steering request" action={<Button type="button" variant="secondary" className="min-h-11" onClick={() => lastSteeringActionRef.current instanceof Function && lastSteeringActionRef.current()}>Retry</Button>} /></div>}
  </section>
}
