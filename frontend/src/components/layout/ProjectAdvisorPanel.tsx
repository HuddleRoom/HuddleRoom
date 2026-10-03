import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import { useProjectAdvisorConversation, useSubmitProjectAdvisorTurn } from '@/api/orchestration'
import { Button, Textarea } from '@/components/common/uiPrimitives'
import { Panel } from '@/components/common/Panel'
import { ErrorRecord } from '@/components/common/ErrorRecord'
import { Tag } from '@/components/common/Tag'
import type { ProjectAdvisorCitation, ProjectAdvisorTurn } from '@/lib/types'
import { absolute } from '@/lib/time'
import { useGoalAnnouncer } from '@/pages/orchestration/goalAnnouncer'

const offTopicDeflection = "That's outside what I can help with here — I can answer questions about this project's goals, decisions, and activity."

function citationHref(citation: ProjectAdvisorCitation): string | null {
  switch (citation.type) {
    case 'goal':
      return `/orchestration/${citation.id}`
    case 'meeting':
      return `/meetings/${citation.id}`
    case 'decision':
      return citation.goal_id ? `/orchestration/${citation.goal_id}#decision-${citation.id}` : null
    default:
      return null
  }
}

function TurnRecord({ turn }: { turn: ProjectAdvisorTurn }) {
  return <li className="min-w-0 border-b border-huddleroom-border p-3 last:border-b-0">
    <div className="min-w-0 text-sm text-huddleroom-text-primary">
      <span className="font-medium">Question</span>
      <p className="whitespace-pre-wrap break-words">{turn.question}</p>
    </div>
    <div className="mt-2 min-w-0 text-sm text-huddleroom-text-secondary">
      <span className="font-medium">Answer</span>
      <p className="whitespace-pre-wrap break-words">{turn.off_topic ? offTopicDeflection : turn.answer ?? 'No response recorded.'}</p>
    </div>
    {!turn.off_topic && turn.citations.length > 0 && <div className="mt-2 flex flex-wrap gap-1.5">
      {turn.citations.map((citation) => {
        const href = citationHref(citation)
        const key = `${citation.type}:${citation.id}`
        return href
          ? <Link key={key} to={href}><Tag mono>{citation.label}</Tag></Link>
          : <Tag key={key} mono>{citation.label}</Tag>
      })}
    </div>}
    <time dateTime={turn.created_at} className="mt-2 block text-xs text-huddleroom-text-muted">{absolute(turn.created_at)}</time>
  </li>
}

function mutationErrorText(error: unknown) {
  if (error && typeof error === 'object') {
    const detail = 'detail' in error && error.detail
    if (detail && typeof detail === 'object' && 'message' in detail && typeof detail.message === 'string') return detail.message
  }
  return 'Question was not sent. Check your connection and try again.'
}

export function ProjectAdvisorPanel({ projectId }: { projectId: string }) {
  const query = useProjectAdvisorConversation(projectId)
  const mutation = useSubmitProjectAdvisorTurn(projectId)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const errorRef = useRef<HTMLDivElement>(null)
  const [draft, setDraft] = useState('')
  const [errorDismissed, setErrorDismissed] = useState(false)
  const { announceQueue, announceError } = useGoalAnnouncer()

  const safeError = mutationErrorText(mutation.error)
  useEffect(() => { if (mutation.isError && !errorDismissed) errorRef.current?.focus() }, [mutation.isError, errorDismissed])
  useEffect(() => { if (query.isError) announceError('Failed to load advisor history. Retry to view answers.') }, [query.isError, announceError])
  useEffect(() => { if (mutation.isError && !errorDismissed) announceError(safeError) }, [mutation.isError, errorDismissed, safeError, announceError])

  const allowance = query.data?.allowance
  if (query.data && !allowance?.enabled) return null

  const unknown = query.isLoading || query.isError || !query.data
  const exhausted = !unknown && !allowance!.unlimited && allowance!.remaining <= 0
  const disabled = unknown || exhausted

  const submit = () => {
    if (disabled || mutation.isPending || !draft.trim()) return
    setErrorDismissed(false)
    mutation.mutate(draft, {
      onSuccess: () => { setDraft(''); setErrorDismissed(true); announceQueue('Question sent.'); textareaRef.current?.focus() },
    })
  }
  const onSubmit = (event: FormEvent) => { event.preventDefault(); submit() }
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); submit() }
  }

  return <Panel className="flex flex-col gap-3">
    <h3 className="text-sm font-semibold text-huddleroom-text-primary">Ask the orchestrator</h3>
    <p className="text-xs text-huddleroom-text-secondary">Answers are grounded in project state and explain, never act. They don't create, start, or change goals.</p>
    {query.isLoading ? <p className="text-sm text-huddleroom-text-secondary" aria-busy="true">Loading advisor history…</p>
      : query.isError || !query.data ? <div className="flex flex-wrap items-center gap-2"><span className="text-sm text-huddleroom-status-red">Failed to load advisor history. Retry to view answers.</span><Button type="button" variant="secondary" className="min-h-11" onClick={() => { void query.refetch() }}>Retry</Button></div>
        : <>
          {query.data.items.length === 0
            ? <p className="text-sm text-huddleroom-text-secondary">Ask the orchestrator about this project's goals, decisions, and recent activity.</p>
            : <ol aria-label="Advisor conversation" className="overflow-hidden rounded-md border border-huddleroom-border bg-huddleroom-depth">
              {query.data.items.map((turn) => <TurnRecord key={turn.id} turn={turn} />)}
            </ol>}
        </>}
    <form onSubmit={onSubmit} className="flex min-w-0 flex-col gap-2">
      <Textarea ref={textareaRef} label="Ask the orchestrator" placeholder="Ask about this project…" rows={3} value={draft} disabled={disabled || mutation.isPending} onChange={(event) => setDraft(event.target.value)} onKeyDown={onKeyDown} className="min-w-0 break-words" />
      <Button type="submit" disabled={disabled || mutation.isPending} className="min-h-11 shrink-0 self-start">{mutation.isPending ? 'Sending…' : 'Send question'}</Button>
      {!unknown && !allowance!.unlimited && <p className="text-xs text-huddleroom-text-secondary">{`Allowance: ${allowance!.remaining} remaining of ${allowance!.limit}`}</p>}
      {exhausted && <p className="text-sm text-huddleroom-status-red">Advisor allowance is exhausted. New questions are unavailable.</p>}
    </form>
    {mutation.isError && !errorDismissed && <div ref={errorRef} tabIndex={-1}><ErrorRecord error={mutation.error} entity="advisor" /></div>}
  </Panel>
}
