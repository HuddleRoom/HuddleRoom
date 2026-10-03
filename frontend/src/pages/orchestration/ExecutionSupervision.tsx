import { useEffect, useRef, type ReactNode } from 'react'
import { StatusBadge, type StatusVariant } from '@/components/common/uiPrimitives'
import type { OrchestrationSupervision, OrchestrationSupervisionCondition } from '@/lib/types'
import { useGoalAnnouncer } from './goalAnnouncer'

function entries(value: Record<string, string> | undefined) {
  return Object.entries(value ?? {}).filter(([, amount]) => amount !== '0')
}

function label(value: unknown) {
  return typeof value === 'string' ? value : 'None'
}

function value(input: unknown) {
  return typeof input === 'string' || typeof input === 'number' || typeof input === 'boolean' ? String(input) : 'None'
}

function field(item: Record<string, unknown>, ...keys: string[]) {
  for (const key of keys) {
    const result = value(item[key])
    if (result !== 'None') return result
  }
  return 'None'
}

function objectField(item: Record<string, unknown>, key: string, allowed: readonly string[]) {
  const source = item[key]
  if (!source || typeof source !== 'object' || Array.isArray(source)) return 'None'
  const values = allowed.flatMap((name) => {
    const result = value((source as Record<string, unknown>)[name])
    return result === 'None' ? [] : [`${name} ${result}`]
  })
  return values.join('; ') || 'None'
}

const CONDITION_BADGES: Record<OrchestrationSupervisionCondition, { status: StatusVariant; label: string }> = {
  working: { status: 'running', label: 'Working' }, waiting: { status: 'ready', label: 'Waiting' },
  needs_you: { status: 'needs-you', label: 'Needs you' }, needs_attention: { status: 'blocked', label: 'Needs attention' },
  paused: { status: 'paused', label: 'Paused' }, stopped: { status: 'idle', label: 'Stopped' },
  cancelled: { status: 'cancelled', label: 'Cancelled' }, completed: { status: 'completed', label: 'Completed' },
}

const MATERIAL_TRANSITIONS = new Set([
  'completed', 'cancelled', 'paused', 'stopped', 'failed_gate', 'failed_action',
  'attention_blocker', 'recovery_action', 'recovery_wait', 'accepted_evidence',
])

function Fact({ label: name, children, mono = false }: { label: string; children: string; mono?: boolean }) {
  return <span><span className="text-huddleroom-text-muted">{name}: </span><span className={mono ? 'font-mono break-all text-huddleroom-text-primary' : 'text-huddleroom-text-primary'}>{children}</span></span>
}

function FactList({ title, items, render }: { title: string; items: readonly Record<string, unknown>[]; render: (item: Record<string, unknown>) => ReactNode }) {
  return <div>
    <h3 className="text-xs font-medium text-huddleroom-text-primary">{title}</h3>
    <ul className="mt-1 list-disc pl-5 text-huddleroom-text-secondary">
      {items.length ? items.map((item, index) => <li key={index} className="flex flex-wrap gap-x-2 gap-y-1">{render(item)}</li>) : <li>None</li>}
    </ul>
  </div>
}

export function ExecutionSupervision({ supervision }: { supervision: OrchestrationSupervision | null }) {
  const initialized = useRef(false)
  const baselineKey = useRef<string | null>(null)
  const { announceQueue } = useGoalAnnouncer()
  useEffect(() => {
    const transition = supervision?.transition
    if (!initialized.current) {
      initialized.current = true
      baselineKey.current = transition?.key ?? null
      return
    }
    if (!transition || transition.key === baselineKey.current || transition.kind === 'tick' || transition.kind === 'heartbeat') {
      baselineKey.current = transition?.key ?? baselineKey.current
      return
    }
    baselineKey.current = transition.key
    if (supervision.condition === 'needs_you') announceQueue(`Needs your direction: ${transition.message}`)
    else if (MATERIAL_TRANSITIONS.has(transition.kind)) announceQueue(transition.message)
  }, [supervision?.condition, supervision?.transition, announceQueue])
  if (!supervision) return null
  const budget = supervision.budget
  const badge = CONDITION_BADGES[supervision.condition]
  return (
    <section aria-labelledby="execution-supervision-title" className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="execution-supervision-title" className="text-sm font-bold tracking-[0.05em] text-huddleroom-text-primary">Current operation</h2>
        <StatusBadge status={badge.status} label={badge.label} />
      </div>
      <p className="mt-2 text-sm text-huddleroom-text-secondary">{supervision.operation} · {supervision.next_action}</p>
      <p className="mt-1 text-xs text-huddleroom-text-muted">{supervision.rationale}</p>
      <dl className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <div><dt className="text-xs text-huddleroom-text-muted">Criterion</dt><dd>{label(supervision.criterion?.description ?? supervision.criterion?.key)}</dd></div>
        <div><dt className="text-xs text-huddleroom-text-muted">Criterion status</dt><dd>{label(supervision.criterion?.status)}</dd></div>
      </dl>
      <div className="mt-4 space-y-3 text-sm">
        <FactList title="Verified progress" items={supervision.verified_progress} render={(item) => <><Fact label="key" mono>{field(item, 'key', 'criterion_key')}</Fact><Fact label="status">{field(item, 'status')}</Fact><Fact label="summary">{field(item, 'summary', 'description')}</Fact></>} />
        <FactList title="Useful learning" items={supervision.useful_learning} render={(item) => <><Fact label="key" mono>{field(item, 'key', 'learning_key')}</Fact><Fact label="status">{field(item, 'status')}</Fact><Fact label="summary">{field(item, 'summary', 'description')}</Fact></>} />
        <FactList title="Accepted evidence" items={supervision.accepted_evidence.map((item) => ({ id: item.id, gate_id: item.gate_id, source_type: item.source_type, source_id: item.source_id, observed_event_id: item.observed_event_id, producer_agent_id: item.producer_agent_id, verdict: item.verdict, created_at: item.created_at, updated_at: item.updated_at, evidence_metadata: item.evidence_metadata }))} render={(item) => <><Fact label="id" mono>{field(item, 'id')}</Fact><Fact label="gate" mono>{field(item, 'gate_id')}</Fact><Fact label="source type">{field(item, 'source_type')}</Fact><Fact label="source id" mono>{field(item, 'source_id')}</Fact><Fact label="event" mono>{field(item, 'observed_event_id')}</Fact><Fact label="producer" mono>{field(item, 'producer_agent_id')}</Fact><Fact label="verdict">{field(item, 'verdict')}</Fact><Fact label="time" mono>{field(item, 'created_at', 'updated_at')}</Fact><Fact label="metadata">{objectField(item, 'evidence_metadata', ['summary', 'name', 'artifact_id', 'url', 'version', 'kind'])}</Fact></>} />
        <FactList title="Workers" items={supervision.workers} render={(item) => <><Fact label="session" mono>{field(item, 'session_id')}</Fact><Fact label="task" mono>{field(item, 'task_id')}</Fact><Fact label="agent" mono>{field(item, 'agent_id')}</Fact><Fact label="runner" mono>{field(item, 'runner_id')}</Fact><Fact label="title">{field(item, 'task_title')}</Fact><Fact label="status">{field(item, 'session_status')}</Fact><Fact label="liveness">{field(item, 'observed_liveness')}</Fact><Fact label="time" mono>{field(item, 'observed_at')}</Fact></>} />
        <FactList title="Open waits" items={supervision.waits} render={(item) => <><Fact label="id" mono>{field(item, 'id')}</Fact><Fact label="owner">{objectField(item, 'owner', ['type', 'id'])}</Fact><Fact label="event" mono>{field(item, 'event')}</Fact><Fact label="matcher" mono>{objectField(item, 'matcher', ['task_id', 'session_id', 'run_id', 'decision_id', 'subject_id'])}</Fact><Fact label="due" mono>{field(item, 'due_recheck_at')}</Fact><Fact label="fallback">{objectField(item, 'fallback', ['reason', 'action_type', 'expected_result'])}</Fact></>} />
        <FactList title="Recovery linkage" items={supervision.recovery_history} render={(item) => <><Fact label="session" mono>{field(item, 'session_id')}</Fact><Fact label="classification">{field(item, 'classification')}</Fact><Fact label="disposition">{field(item, 'disposition')}</Fact><Fact label="backend">{field(item, 'backend_observation')}</Fact><Fact label="assessed" mono>{field(item, 'assessed_at')}</Fact><Fact label="action" mono>{field(item, 'action_id')}</Fact><Fact label="wait" mono>{field(item, 'wait_id')}</Fact></>} />
      </div>
      <div className="mt-4">
        <h3 className="text-xs font-medium text-huddleroom-text-primary">Pending question</h3>
        <p className="mt-1 text-sm text-huddleroom-text-secondary">{supervision.pending_direction?.question ?? 'None'}</p>
        {supervision.pending_direction && <p className="mt-1 text-xs text-huddleroom-text-muted">Answer in Needs you below.</p>}
      </div>
      <div className="mt-4 border-t border-huddleroom-border pt-3 text-xs text-huddleroom-text-secondary">
        <span className="font-medium text-huddleroom-text-primary">Budget: </span>
        {(['consumed', 'committed', 'reserved', 'remaining'] as const).map((name) => (
          <span key={name} className="mr-3">{name} {entries(budget[name]).map(([key, value]) => `${key} ${value}`).join(', ') || '—'}</span>
        ))}
      </div>
    </section>
  )
}
