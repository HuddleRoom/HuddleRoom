import { useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode, type RefObject } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { useQueries } from '@tanstack/react-query'
import { Dialog } from '@/components/common/Dialog'
import { useAllTasks } from '@/api/tasks'
import { useAgents } from '@/api/agents'
import { meetingKeys, useMeetingDecisions } from '@/api/meetings'
import { apiFetch } from '@/lib/api-client'
import {
  useBaselineDashboard,
  useCreateOrchestrationGoal,
  useOrchestrationGoal,
  useOrchestrationGoalCommand,
  useOrchestrationGoals,
  useOrchestrationHealth,
  useOverrideOrchestrationGate,
  useRecoverGoalDefinition,
  useResetOrchestrationGoal,
  useStartOrchestrationGoal,
} from '@/api/orchestration'
import {
  Button,
  ConfirmDialog,
  PageHeader,
  QueryState,
  SkeletonRow,
  StatusBadge,
  Textarea,
  NoProjectSelected,
} from '@/components/common/uiPrimitives'
import { ErrorRecord } from '@/components/common/ErrorRecord'
import { Record } from '@/components/common/Record'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import type {
  Agent,
  Meeting,
  MeetingDecision,
  OrchestrationAction,
  OrchestrationBaselineProcessType,
  OrchestrationBlocker,
  OrchestrationEvidence,
  OrchestrationGate,
  OrchestrationGateOverrideDecision,
  OrchestrationGoalDetail,
  Task,
} from '@/lib/types'
import { useUIStore } from '@/stores/ui'
import { BaselineDashboard, currentProcesses, type BaselineMutationOwner } from './BaselineDashboard'
import { DetailsTabs, type DetailsTabsHandle } from './DetailsTabs'
import { errorRecord, gateOverrideDecisionLabel, goalStatusLabel, runStatusLabel } from './humanize'
import { ZONE_TITLE_CLASS } from './zoneTitle'
import { StartWorkDialog } from './StartWorkDialog'
import { ExecutionSupervision } from './ExecutionSupervision'
import { RoomPanel, type RoomPanelAgentRow, type RoomPanelMeetingRow } from './RoomPanel'
import { absolute } from '@/lib/time'
import { GoalAnnouncerProvider, useGoalAnnouncer } from './goalAnnouncer'

const ACTIVE_TASK_STATUSES = new Set(['backlog', 'ready', 'in_progress', 'blocked'])

type MutationOwner = `lifecycle:${'pause' | 'resume'}` | 'start' | 'cancel' | 'reset' | 'recover' | `override:${string}` | BaselineMutationOwner

function errorText(error: unknown) {
  return error instanceof Error ? error.message : 'The request failed. Check your connection and try again.'
}

function jsonText(value: unknown) {
  const encoded = JSON.stringify(value, null, 2)
  return encoded === undefined ? String(value) : encoded
}

export type ActiveDelegation = {
  id: string
  targetType: string
  targetId: string
  actionType: string
  status: string
  title?: string
}

export function buildActiveDelegations(
  detail: OrchestrationGoalDetail,
  tasksById: Map<string, Task>,
): ActiveDelegation[] {
  // Only task targets have a resolvable live status (via useAllTasks), so only
  // tasks can be proven active. Session/meeting/protocol targets have no live
  // sub-status in this read model — the delegating action status only proves the
  // delegation was created, not that the downstream work is still running — so
  // asserting them "active" here would be unreliable. Non-task coordination is
  // inspected through the decision/action timeline instead.
  const latestByTask = new Map<string, OrchestrationAction>()
  for (const action of detail.actions) {
    if (action.target_type !== 'task' || !action.target_id) continue
    latestByTask.set(action.target_id, action)
  }
  const delegations: ActiveDelegation[] = []
  for (const [taskId, action] of latestByTask) {
    const task = tasksById.get(taskId)
    // A task the live query no longer returns cannot be confirmed active, so it
    // is not listed here; it still appears in the timeline with its action.
    if (!task || !ACTIVE_TASK_STATUSES.has(task.status)) continue
    delegations.push({
      id: `task:${taskId}`,
      targetType: 'task',
      targetId: taskId,
      actionType: action.action_type,
      status: task.status,
      title: task.title,
    })
  }
  return delegations
}

// Defensive field() accessor duplicated from ExecutionSupervision.tsx — the
// supervision worker shape is informally typed (Record<string, unknown>[]),
// any field may be absent.
function workerField(item: Record<string, unknown>, key: string) {
  const raw = item[key]
  return typeof raw === 'string' || typeof raw === 'number' ? String(raw) : undefined
}

export function buildAgentRows(
  workers: Record<string, unknown>[],
  agentsById: Map<string, Agent>,
): RoomPanelAgentRow[] {
  const latestByAgent = new Map<string, Record<string, unknown>>()
  for (const worker of workers) {
    const agentId = workerField(worker, 'agent_id')
    if (!agentId) continue
    const existing = latestByAgent.get(agentId)
    if (!existing || (workerField(worker, 'observed_at') ?? '') > (workerField(existing, 'observed_at') ?? '')) {
      latestByAgent.set(agentId, worker)
    }
  }
  return [...latestByAgent.entries()].map(([agentId, worker]) => {
    const agent = agentsById.get(agentId)
    return {
      agentId,
      role: agent?.role ?? 'Unknown agent',
      adapterType: agent?.adapter_type,
      provider: agent?.provider,
      activity: workerField(worker, 'task_title') ?? 'Idle',
      working: workerField(worker, 'session_status') === 'running',
    }
  })
}

export function buildMeetingRow(
  meeting: Meeting | null,
  decisions: MeetingDecision[] | undefined,
): RoomPanelMeetingRow | null {
  if (!meeting || !decisions || decisions.length === 0) return null
  const latest = [...decisions].sort((a, b) => b.created_at.localeCompare(a.created_at))[0]
  return {
    question: latest.question ?? latest.title,
    chosenOption: latest.chosen_option,
    dissentCount: latest.dissent?.length ?? 0,
  }
}

// Extracted from the old header <details> block — now rendered inside the
// Debug tab (DetailsTabs debugContent) instead of the goal header.
function DangerZoneSection({
  resetButton,
  globalBusy,
  onReset,
}: {
  resetButton: RefObject<HTMLButtonElement | null>
  globalBusy: boolean
  onReset: () => void
}) {
  return (
    <section className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
      <h3 className="text-xs font-medium text-huddleroom-text-secondary">Danger zone</h3>
      <div className="mt-2 space-y-2">
        <p className="text-xs text-huddleroom-text-muted">This permanently deletes all decisions, assumptions, warnings, agent reviews, and process history for this goal. The goal returns to its freshly-created state.</p>
        <Button
          ref={resetButton}
          variant="danger"
          className="text-xs min-h-11"
          disabled={globalBusy}
          onClick={onReset}
        >
          Reset goal — deletes all history
        </Button>
      </div>
    </section>
  )
}

export function canSubmitGateOverride(
  decision: OrchestrationGateOverrideDecision | null,
  reason: string,
) {
  return decision !== null && reason.trim().length > 0
}

// `bordered` defaults true (a standalone card: border + surface + its own
// padding). Every caller nested inside DetailsTabs' tabpanel (already one
// bordered/surfaced/padded pane) passes `bordered={false}` — single
// elevation per pane, no border-in-a-border (review round 1, finding 4). The
// one caller left bordered is the standalone "Run" panel, rendered as a
// sibling of DetailsTabs (not nested inside it) when the goal has no run.
function Panel({ title, id, badge, muted, bordered = true, level = 'h2', children }: { title: string; id?: string; badge?: number; muted?: boolean; bordered?: boolean; level?: 'h2' | 'h3'; children: ReactNode }) {
  const badgeEl = badge !== undefined && badge > 0 && (
    <span className="ml-2 inline-flex items-center rounded-full bg-huddleroom-status-red px-2 py-1 text-xs font-medium text-white">
      {badge}
    </span>
  )
  const Heading = level
  if (!bordered) {
    return (
      <section id={id}>
        <Heading className={`text-xs font-medium ${muted ? 'text-huddleroom-text-muted' : 'text-huddleroom-text-primary'}`}>{title}{badgeEl}</Heading>
        <div className="mt-2">{children}</div>
      </section>
    )
  }
  return (
    <section id={id} className="overflow-hidden rounded-md border border-huddleroom-border bg-huddleroom-surface">
      <Heading className={`border-b border-huddleroom-border px-4 py-3 text-sm font-semibold ${muted ? 'text-huddleroom-text-secondary' : 'text-huddleroom-text-primary'}`}>
        {title}{badgeEl}
      </Heading>
      <div className="p-4">{children}</div>
    </section>
  )
}

function isScalarValue(val: unknown): val is string | number | boolean {
  return typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean'
}

function isFlatObject(val: unknown): boolean {
  if (typeof val !== 'object' || val === null || Array.isArray(val)) return false
  for (const v of Object.values(val)) {
    if (typeof v === 'object' && v !== null) return false
  }
  return true
}

function ScalarDetails({ label, value }: { label: string; value: unknown }) {
  if (isFlatObject(value)) {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => isScalarValue(v) || v === null || v === undefined)
    if (entries.length > 0) {
      return (
        <details className="mt-2">
          <summary className="flex min-h-11 cursor-pointer items-center text-xs text-huddleroom-text-secondary">▸ {label}</summary>
          <dl className="mt-2 space-y-1 text-sm">
            {entries.map(([k, v]) => (
              <div key={k} className="flex flex-wrap gap-2">
                <dt className="font-medium text-huddleroom-text-primary">{k}:</dt>
                <dd className="text-huddleroom-text-secondary">{v === null || v === undefined ? '—' : String(v)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )
    }
  }

  if (Array.isArray(value)) {
    const scalars = value.filter((v) => isScalarValue(v) || v === null || v === undefined)
    if (scalars.length === value.length && value.length > 0) {
      return (
        <details className="mt-2">
          <summary className="flex min-h-11 cursor-pointer items-center text-xs text-huddleroom-text-secondary">▸ {label}</summary>
          <ul className="mt-2 space-y-1 text-sm text-huddleroom-text-secondary">
            {scalars.map((v, i) => (
              <li key={i}>{v === null || v === undefined ? '—' : String(v)}</li>
            ))}
          </ul>
        </details>
      )
    }
  }

  return <JsonDetails label={label} value={value} />
}

function ScalarRecord({ label, value }: { label: string; value: unknown }) {
  if (isFlatObject(value)) {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => isScalarValue(v) || v == null)
    if (entries.length > 0) return <Record rows={entries.map(([k, v]) => ({ key: k, value: v == null ? undefined : String(v) }))} />
  }
  return <ScalarDetails label={label} value={value} />
}

function JsonDetails({ label, value }: { label: string; value: unknown }) {
  return (
    <details className="mt-2">
      <summary className="flex min-h-11 cursor-pointer items-center text-xs text-huddleroom-text-secondary">▸ {label}</summary>
      <pre className="mt-2 max-w-full overflow-auto whitespace-pre-wrap break-words rounded bg-huddleroom-depth p-3 font-mono text-xs text-huddleroom-text-primary">
        {jsonText(value)}
      </pre>
    </details>
  )
}

export function GateOverrideForm({
  gateId,
  pending,
  disabled = false,
  error,
  onCancel,
  onSubmit,
}: {
  gateId: string
  pending: boolean
  disabled?: boolean
  error: string | null
  onCancel: () => void
  onSubmit: (decision: OrchestrationGateOverrideDecision, reason: string) => void
}) {
  const [decision, setDecision] = useState<OrchestrationGateOverrideDecision | null>(null)
  const [reason, setReason] = useState('')
  const firstRadio = useRef<HTMLInputElement>(null)
  const applyButton = useRef<HTMLButtonElement>(null)
  const applyFocusIntent = useRef(false)
  const { announceError } = useGoalAnnouncer()

  useEffect(() => firstRadio.current?.focus(), [])
  useEffect(() => {
    if (pending || !error || !applyFocusIntent.current) return
    applyFocusIntent.current = false
    applyButton.current?.focus()
  }, [pending, error])
  useEffect(() => { if (error) announceError(error) }, [error, announceError])

  function submit(event: FormEvent) {
    event.preventDefault()
    if (!canSubmitGateOverride(decision, reason)) return
    applyFocusIntent.current = true
    onSubmit(decision!, reason.trim())
  }

  return (
    <form onSubmit={submit} aria-busy={disabled} className="mt-3 border-t border-huddleroom-border pt-3">
      <fieldset disabled={disabled || pending}>
        <legend className="text-xs font-medium text-huddleroom-text-muted">Override gate</legend>
        <div className="mt-2 flex flex-wrap gap-4">
          {(['accept', 'reject'] as const).map((value, index) => (
            <label key={value} className="flex min-h-11 items-center gap-2 text-sm text-huddleroom-text-primary">
              <input
                ref={index === 0 ? firstRadio : undefined}
                type="radio"
                name={`override-${gateId}`}
                value={value}
                checked={decision === value}
                onChange={() => setDecision(value)}
              />
              {gateOverrideDecisionLabel(value)}
            </label>
          ))}
        </div>
        <Textarea
          label="Reason"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          rows={3}
          required
        />
        {error && <p className="mb-2 text-xs text-huddleroom-status-red">{error}</p>}
        <div className="flex flex-wrap justify-end gap-2">
          <Button type="button" variant="secondary" className="min-h-11" onClick={onCancel}>
            Cancel override
          </Button>
          <Button
            ref={applyButton}
            type="submit"
            className="min-h-11"
            disabled={disabled || pending || !canSubmitGateOverride(decision, reason)}
          >
            {pending ? 'Applying…' : 'Apply override'}
          </Button>
        </div>
      </fieldset>
    </form>
  )
}

function GoalListView({ projectId }: { projectId: string }) {
  const navigate = useNavigate()
  const query = useOrchestrationGoals(projectId)
  const goals = useMemo(
    () => query.data?.pages.flatMap((page) => page.items) ?? [],
    [query.data],
  )
  const [createGoalOpen, setCreateGoalOpen] = useState(false)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <PageHeader title="Orchestration" />
        <Button
          onClick={() => setCreateGoalOpen(true)}
          aria-label="Create new orchestration goal"
          className="min-h-11"
        >
          New goal
        </Button>
      </div>

      <div className="overflow-hidden rounded-md border border-huddleroom-border bg-huddleroom-surface">
        <QueryState
          query={{
            isLoading: query.isLoading,
            isError: query.isError,
            data: query.data ? goals : undefined,
            error: query.error,
            refetch: () => { void query.refetch() },
          }}
          skeleton="table"
          skeletonCount={6}
          errorLabel="Failed to load orchestration goals"
          emptyLabel="No orchestration goals"
          emptyDetail="Goals created through the project-scoped orchestration API appear here."
        >
          {(items) => (
            <ul aria-label="Orchestration goals">
              {items.map((goal) => (
                <li key={goal.id} className="border-b border-huddleroom-depth last:border-b-0">
                  <Link
                    to={`/orchestration/${goal.id}`}
                    className="flex min-h-11 flex-col gap-2 px-4 py-3 text-left no-underline hover:bg-huddleroom-row-hover focus:outline-2 focus:outline-huddleroom-primary focus:outline-offset-[-2px]"
                  >
                    <span className="block w-full min-w-0 break-words text-sm font-medium text-huddleroom-text-primary">
                      <span className="sr-only">Objective: </span>{goal.objective}
                    </span>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                      <span><span className="sr-only">Status: </span><StatusBadge status={goal.status} label={goalStatusLabel(goal.status)} /></span>
                      <span className="text-xs text-huddleroom-text-secondary">
                        <span className="sr-only">Success criteria: </span>
                        {goal.success_criteria.length} success {goal.success_criteria.length === 1 ? 'criterion' : 'criteria'}
                      </span>
                      {goal.needs_you_count > 0 ? (
                        <span className="text-xs font-medium inline-flex items-center gap-1.5">
                          <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ backgroundColor: STATUS_COLORS.blue }} />
                          <span><span className="sr-only">Needs attention: </span>{goal.needs_you_count} need{goal.needs_you_count === 1 ? 's' : ''} you</span>
                        </span>
                      ) : null}
                      <time dateTime={goal.updated_at} className="text-xs text-huddleroom-text-muted">
                        <span className="sr-only">Updated: </span>{absolute(goal.updated_at)}
                      </time>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </QueryState>
      </div>

      {query.hasNextPage && (
        <Button
          variant="secondary"
          className="min-h-11 self-start"
          disabled={query.isFetchingNextPage}
          onClick={() => { void query.fetchNextPage() }}
        >
          {query.isFetchingNextPage ? 'Loading…' : 'Load more'}
        </Button>
      )}

      <CreateGoalModal
        projectId={projectId}
        open={createGoalOpen}
        onClose={() => setCreateGoalOpen(false)}
      />
    </div>
  )
}

// ─── Create Goal Modal ────────────────────────────────────────────────────────

function CreateGoalModal({
  projectId,
  open,
  onClose,
}: {
  projectId: string
  open: boolean
  onClose: () => void
}) {
  const navigate = useNavigate()
  const createGoal = useCreateOrchestrationGoal(projectId)
  const [form, setForm] = useState({
    objective: '',
    successCriteria: '',
  })
  const [validationError, setValidationError] = useState<string | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)
  const { announceError } = useGoalAnnouncer()
  useEffect(() => {
    const message = validationError ?? createError
    if (message) announceError(message)
  }, [validationError, createError, announceError])

  function reset() {
    setForm({ objective: '', successCriteria: '' })
    setValidationError(null)
    setCreateError(null)
  }

  function handleCreate() {
    const trimmedObjective = form.objective.trim()
    if (!trimmedObjective) return

    const criteria = form.successCriteria
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0)

    if (criteria.length === 0) {
      setValidationError('At least one success criterion is required')
      return
    }

    setValidationError(null)
    setCreateError(null)
    createGoal.mutate(
      {
        objective: trimmedObjective,
        success_criteria: criteria.map((description) => ({ description })),
        constraints: {},
        budget: {},
      },
      {
        onSuccess: (detail) => {
          reset()
          onClose()
          navigate(`/orchestration/${detail.goal.id}`)
        },
        onError: (e) => setCreateError(errorRecord(e).what),
      },
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) { reset(); onClose() } }}
      title="New goal"
      description="Create a new orchestration goal with objectives and success criteria."
      size="lg"
      footer={{
        primaryLabel: createGoal.isPending ? 'Creating…' : 'Create goal',
        primaryType: 'submit',
        formId: 'create-goal-form',
        isPending: createGoal.isPending,
        primaryDisabled: !form.objective.trim(),
      }}
    >
      <form id="create-goal-form" onSubmit={(e) => { e.preventDefault(); handleCreate() }} className="space-y-4">
        <div>
          <label className="text-sm font-medium text-huddleroom-text-primary block mb-2">
            Objective *
          </label>
          <Textarea
            value={form.objective}
            onChange={(e) => setForm((p) => ({ ...p, objective: e.target.value }))}
            placeholder="What should the orchestration achieve?"
            rows={3}
          />
        </div>

        <div>
          <label className="text-sm font-medium text-huddleroom-text-primary block mb-2">
            Success criteria *
          </label>
          <p className="text-xs text-huddleroom-text-secondary mb-2">
            One criterion per line. At least one required.
          </p>
          <Textarea
            value={form.successCriteria}
            onChange={(e) => {
              setForm((p) => ({ ...p, successCriteria: e.target.value }))
              setValidationError(null)
            }}
            placeholder="Success criterion 1&#10;Success criterion 2"
            rows={4}
          />
          {validationError && (
            <p className="mt-2 text-xs text-huddleroom-status-red">{validationError}</p>
          )}
          {createError && (
            <p className="mt-2 text-xs text-huddleroom-status-red">{createError}</p>
          )}
        </div>
      </form>
    </Dialog>
  )
}

function GatesPanel({
  gates,
  evidence,
  criteriaByKey,
  terminal,
  debug,
  overrideGateId,
  overrideError,
  mutationOwner,
  mutationOwnerRef,
  override,
  acquireMutation,
  releaseMutation,
  setOverrideError,
  setOverrideGateId,
  closeOverride,
  setMessage,
  setLastMessageOwner,
  overrideButtons,
  gateOverrideBusy,
  goalId,
  message,
  lastMessageOwner,
  badge,
  showEmptyState,
}: {
  gates: OrchestrationGate[]
  evidence: OrchestrationEvidence[]
  criteriaByKey: Map<string, string | undefined>
  terminal: boolean
  debug: boolean
  overrideGateId: string | null
  overrideError: string | null
  mutationOwner: MutationOwner | null
  mutationOwnerRef: React.MutableRefObject<MutationOwner | null>
  override: ReturnType<typeof useOverrideOrchestrationGate>
  acquireMutation: (owner: MutationOwner) => boolean
  releaseMutation: (owner: MutationOwner) => void
  setOverrideError: (error: string | null) => void
  setOverrideGateId: (id: string | null) => void
  closeOverride: (gateId: string) => void
  setMessage: (msg: string) => void
  setLastMessageOwner: (owner: MutationOwner | null) => void
  overrideButtons: React.MutableRefObject<Record<string, HTMLButtonElement | null>>
  gateOverrideBusy: (gateId: string) => boolean
  goalId: string
  message: string
  lastMessageOwner: MutationOwner | null
  badge?: number
  showEmptyState: boolean
}) {
  const { announceQueue } = useGoalAnnouncer()
  useEffect(() => {
    if (lastMessageOwner?.startsWith('override:') && message) announceQueue(message)
  }, [lastMessageOwner, message, announceQueue])
  return (
    <Panel title="Gates and evidence" id="panel-gates" badge={badge} bordered={false}>
      <p className="mb-2 text-xs text-huddleroom-text-muted">A gate is a checkpoint tied to one success criterion; it needs evidence before the orchestrator accepts it.</p>
      {showEmptyState && gates.length === 0 ? (
        <p className="text-sm text-huddleroom-text-secondary">No gates recorded.</p>
      ) : (
        <Record
          keyWidth={140}
          rows={gates.map((gate) => {
            const rows = evidence.filter((row) => row.gate_id === gate.id)
            const gateIsBusy = gateOverrideBusy(gate.id)
            const humanDescription = criteriaByKey.get(gate.success_criterion_key)
            return {
              key: gate.success_criterion_key,
              id: gate.id,
              value: (
              <article key={gate.id} data-testid={`gate-${gate.success_criterion_key}`} className="min-w-0" aria-busy={gateIsBusy}>
                {humanDescription && (
                  <p className="break-words text-sm text-huddleroom-text-primary" title={`${gate.success_criterion_key} · ${gate.gate_type}`}>
                    {humanDescription}
                  </p>
                )}
                <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
                  {!humanDescription && (
                    <>
                      <span className="min-w-0 break-all font-mono text-huddleroom-text-muted">{gate.success_criterion_key}</span>
                      <span className="min-w-0 break-all font-mono text-huddleroom-text-muted">{gate.gate_type}</span>
                    </>
                  )}
                  <StatusBadge status={gate.status} />
                </div>
                {gate.failure_reason && <p className="mt-1 break-words text-sm text-huddleroom-status-red">{gate.failure_reason}</p>}
                {debug ? (
                  <JsonDetails label="Required evidence" value={gate.required_evidence} />
                ) : (
                  <ScalarDetails label="Required evidence" value={gate.required_evidence} />
                )}
                {rows.length === 0 ? (
                  <p className="mt-2 text-sm text-huddleroom-text-secondary">No evidence recorded.</p>
                ) : (
                  <ul data-testid="orchestration-evidence" className="mt-2 min-w-0 divide-y divide-huddleroom-border rounded bg-huddleroom-depth px-3">
                    {rows.map((row) => (
                      <li key={row.id} className="min-w-0 py-2 text-xs">
                        <div className="flex min-w-0 flex-wrap gap-2">
                          <span className="min-w-0 break-all font-mono text-huddleroom-text-primary">{row.source_type}:{row.source_id ?? 'none'}</span>
                          <StatusBadge status={row.verdict} />
                          <time dateTime={row.created_at} className="ml-auto font-mono text-huddleroom-text-secondary">{absolute(row.created_at)}</time>
                        </div>
                        <ScalarDetails label="Evidence metadata" value={row.evidence_metadata} />
                      </li>
                    ))}
                  </ul>
                )}
                {!terminal && (
                  overrideGateId === gate.id ? (
                    <GateOverrideForm
                      gateId={gate.id}
                      pending={mutationOwner === `override:${gate.id}` || override.isPending}
                      disabled={gateIsBusy}
                      error={overrideError}
                      onCancel={() => closeOverride(gate.id)}
                      onSubmit={(decision, reason) => {
                        const owner: MutationOwner = `override:${gate.id}`
                        if (!acquireMutation(owner)) return
                        setOverrideError(null)
                        override.mutate({ goalId, gateId: gate.id, decision, reason }, {
                          onSuccess: () => {
                            setMessage(`Gate ${gate.success_criterion_key} override applied.`)
                            setLastMessageOwner(owner)
                          },
                          onError: (error) => setOverrideError(errorText(error)),
                          onSettled: (_data, error) => {
                            releaseMutation(owner)
                            if (!error) closeOverride(gate.id)
                          },
                        })
                      }}
                    />
                  ) : (
                    <Button
                      ref={(element) => { overrideButtons.current[gate.id] = element }}
                      variant="secondary"
                      className="mt-3 min-h-11"
                      disabled={gateIsBusy}
                      aria-label={`Override gate ${gate.success_criterion_key}`}
                      onClick={() => {
                        if (mutationOwnerRef.current !== null) return
                        setOverrideError(null)
                        setOverrideGateId(gate.id)
                      }}
                    >
                      Override gate
                    </Button>
                  )
                )}
                {lastMessageOwner === `override:${gate.id}` && message && <p className="mt-2 text-xs text-huddleroom-text-secondary">{message}</p>}
              </article>
              ),
            }
          })}
        />
      )}
    </Panel>
  )
}

function GoalDetailView({ projectId, goalId }: { projectId: string; goalId: string }) {
  const location = useLocation()
  const navigate = useNavigate()
  const query = useOrchestrationGoal(projectId, goalId)
  const tasksQuery = useAllTasks(projectId)
  const baseline = useBaselineDashboard(projectId, goalId)
  const health = useOrchestrationHealth()
  const command = useOrchestrationGoalCommand(projectId)
  const override = useOverrideOrchestrationGate(projectId)
  const recover = useRecoverGoalDefinition(projectId)
  const reset = useResetOrchestrationGoal(projectId)
  const start = useStartOrchestrationGoal(projectId)
  const [startOpen, setStartOpen] = useState(false)
  const [cancelOpen, setCancelOpen] = useState(false)
  const [resetOpen, setResetOpen] = useState(false)
  const [overrideGateId, setOverrideGateId] = useState<string | null>(null)
  const [recoverError, setRecoverError] = useState<string | null>(null)
  const [recoverPendingMode, setRecoverPendingMode] = useState<'proceed' | 'another_round' | null>(null)
  const [message, setMessage] = useState('')
  const [lastMessageOwner, setLastMessageOwner] = useState<MutationOwner | null>(null)
  const [commandError, setCommandError] = useState<string | null>(null)
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [startError, setStartError] = useState<unknown>(null)
  const [resetError, setResetError] = useState<string | null>(null)
  const [overrideError, setOverrideError] = useState<string | null>(null)
  const [mutationOwner, setMutationOwner] = useState<MutationOwner | null>(null)
  const mutationOwnerRef = useRef<MutationOwner | null>(null)
  const cancelButton = useRef<HTMLButtonElement>(null)
  const startButton = useRef<HTMLButtonElement>(null)
  const resetButton = useRef<HTMLButtonElement>(null)
  const lifecycleButton = useRef<HTMLButtonElement>(null)
  const lifecycleFocusIntent = useRef(false)
  const goalHeader = useRef<HTMLElement>(null)
  const overrideButtons = useRef<Record<string, HTMLButtonElement | null>>({})
  const overrideFocusGateId = useRef<string | null>(null)
  const cancelFocusTarget = useRef<'trigger' | 'header'>('trigger')
  const detailsTabsRef = useRef<DetailsTabsHandle>(null)
  const [activityStepType, setActivityStepType] = useState<OrchestrationBaselineProcessType | null>(null)
  const [pauseConfirming, setPauseConfirming] = useState(false)
  const { announceQueue, announceError } = useGoalAnnouncer()

  // Override-owned messages announce from GatesPanel itself (it owns the
  // per-gate text); every other header message announces here.
  useEffect(() => {
    if (!lastMessageOwner?.startsWith('override:') && message) announceQueue(message)
  }, [message, lastMessageOwner, announceQueue])
  useEffect(() => { if (commandError) announceError(commandError) }, [commandError, announceError])
  useEffect(() => { if (recoverError) announceError(recoverError) }, [recoverError, announceError])
  useEffect(() => { if (tasksQuery.isError) announceError('Failed to load active delegations.') }, [tasksQuery.isError, announceError])
  useEffect(() => {
    if (tasksQuery.isLoading || tasksQuery.isFetchingNextPage) announceQueue('Loading active delegations.')
  }, [tasksQuery.isLoading, tasksQuery.isFetchingNextPage, announceQueue])

  // Room panel data wiring — hooks called unconditionally, ahead of the
  // isLoading/isError early returns below.
  const agentsQuery = useAgents()
  const meetingActionIds = useMemo(() => {
    const ids = new Set<string>()
    for (const action of query.data?.actions ?? []) {
      if (action.target_type === 'meeting' && action.target_id) ids.add(action.target_id)
    }
    return [...ids]
  }, [query.data?.actions])
  const meetingQueries = useQueries({
    queries: meetingActionIds.map((meetingId) => ({
      queryKey: meetingKeys.detail(projectId, meetingId),
      queryFn: () => apiFetch<Meeting>(`/api/v1/meetings/${meetingId}`),
    })),
  })
  const latestMeeting = meetingQueries
    .map((meetingQuery) => meetingQuery.data)
    .filter((meeting): meeting is Meeting => !!meeting && meeting.status === 'concluded')
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0] ?? null
  const meetingDecisionsQuery = useMeetingDecisions(projectId, latestMeeting?.id)
  const mutationOwned = mutationOwner !== null
  const globalBusy = mutationOwned || command.isPending || reset.isPending || recover.isPending
  // ponytail: Fix 1 - disable state honest: all mutation controls disable when ANY mutation in flight
  const lifecycleBusy = globalBusy
  const gateOverrideBusy = (_gateId: string) => globalBusy
  const baselineBusy = globalBusy

  const lifecycleCommand = query.data?.goal.status === 'paused'
    ? 'resume'
    : query.data?.goal.status === 'active' || query.data?.goal.status === 'blocked'
      ? 'pause'
      : null

  useEffect(() => {
    if (lifecycleBusy || !lifecycleFocusIntent.current) return
    lifecycleFocusIntent.current = false
    lifecycleButton.current?.focus()
  }, [lifecycleBusy, lifecycleCommand])

  useEffect(() => {
    const gateId = overrideFocusGateId.current
    if (override.isPending || overrideGateId !== null || !gateId) return
    overrideFocusGateId.current = null
    overrideButtons.current[gateId]?.focus()
  }, [override.isPending, overrideGateId])

  useEffect(() => {
    if (location.hash === '#needs-you-queue-heading') {
      document.getElementById('needs-you-queue-heading')?.focus()
    }
    // query.data added: on cold nav the heading mounts only once goal data
    // loads (GoalDetailView shows a skeleton until then), so retry then.
  }, [location.hash, location.key, location.pathname, query.data])

  if (query.isLoading) {
    return <QueryState query={{ isLoading: true, isError: false, data: undefined }} skeleton="cards" children={() => null} />
  }
  if ((query.isError && !query.dataUpdatedAt) || !query.data) {
    const isRetryable = errorRecord(query.error).errorClass !== 'not_found'
    return (
      <div className="flex flex-col gap-3">
        <PageHeader title="Orchestration goal unavailable" />
        <ErrorRecord
          error={query.error}
          entity="goal"
          action={<Button variant="ghost" className="min-h-11" onClick={() => navigate('/orchestration')}>Back to goals</Button>}
        />
        {isRetryable && (
          <Button variant="secondary" className="min-h-11" onClick={() => { void query.refetch() }}>Retry</Button>
        )}
      </div>
    )
  }

  const detail = query.data
  const decisionFocusId = location.hash.match(/^#decision-(.+)$/)?.[1] ?? null
  const decisionNavigation = decisionFocusId
    && !baseline.processes.isLoading
    && !baseline.decisions.isLoading
    && !baseline.warnings.isLoading
    ? `${location.key}:${location.pathname}${location.hash}`
    : null
  const { goal, run } = detail
  const terminal = goal.status === 'completed' || goal.status === 'cancelled'
  const canStart = run?.phase === 'ready'
    && (goal.status === 'active' || goal.status === 'blocked')
    && goal.goal_type === 'outcome'
  const gateCounts = detail.gates.reduce(
    (counts, gate) => {
      if (gate.status === 'open' || gate.status === 'accepted' || gate.status === 'failed') {
        return { ...counts, [gate.status]: counts[gate.status] + 1 }
      }
      return counts
    },
    { open: 0, accepted: 0, failed: 0 },
  )
  const tasksById = new Map((tasksQuery.items ?? []).map((task) => [task.id, task] as const))
  const activeDelegations = buildActiveDelegations(detail, tasksById)
  const debug = health.data?.debug_enabled === true
  const criteriaByKey = new Map(goal.success_criteria.filter((c) => c.key).map((c) => [c.key as string, c.description]))

  const agentsById = new Map((agentsQuery.data?.items ?? []).map((agent) => [agent.id, agent] as const))
  const agentRows = buildAgentRows(detail.supervision?.workers ?? [], agentsById)
  const meetingRow = buildMeetingRow(latestMeeting, meetingDecisionsQuery.data)
  const roomGates = detail.gates_count > 0
    ? { total: detail.gates_count, accepted: gateCounts.accepted, open: gateCounts.open, failed: gateCounts.failed }
    : null

  function acquireMutation(owner: MutationOwner) {
    if (mutationOwnerRef.current !== null) return false
    mutationOwnerRef.current = owner
    setMutationOwner(owner)
    return true
  }

  function releaseMutation(owner: MutationOwner) {
    if (mutationOwnerRef.current !== owner) return
    mutationOwnerRef.current = null
    setMutationOwner(null)
  }

  function runCommand(nextCommand: 'pause' | 'resume') {
    const owner: MutationOwner = `lifecycle:${nextCommand}`
    if (!acquireMutation(owner)) return
    setCommandError(null)
    setMessage(nextCommand === 'pause' ? 'Pausing goal…' : 'Resuming goal…')
    setLastMessageOwner(owner)
    lifecycleFocusIntent.current = true
    command.mutate({ goalId, command: nextCommand }, {
      onSuccess: (nextDetail) => {
        setMessage(`Goal is now ${nextDetail.goal.status}.`)
        setLastMessageOwner(owner)
      },
      onError: (error) => {
        setMessage('')
        setLastMessageOwner(null)
        setCommandError(errorText(error))
      },
      onSettled: () => releaseMutation(owner),
    })
  }

  function closeOverride(gateId: string) {
    if (mutationOwnerRef.current !== null) return
    setOverrideError(null)
    overrideFocusGateId.current = gateId
    setOverrideGateId(null)
  }

  function setCancelDialogOpen(open: boolean, preferHeader = false) {
    if (!open) {
      setCancelError(null)
      cancelFocusTarget.current = preferHeader ? 'header' : 'trigger'
    }
    setCancelOpen(open)
  }

  function setStartDialogOpen(open: boolean) {
    if (!open) setStartError(null)
    setStartOpen(open)
  }

  // The goal_definition_clarification_limit recovery action, rendered by the
  // Needs-you queue as that blocker row's primary action (Kill list #2). Only
  // this blocker kind has a recovery flow today — NeedsYouQueue falls back to
  // reference-only text for every other kind.
  function renderBlockerRecovery(
    blocker: OrchestrationBlocker,
    register: (element: HTMLButtonElement | null) => void,
    onSuccessFocus?: () => void,
  ) {
    if (blocker.kind !== 'goal_definition_clarification_limit') return null
    return (
      <>
        <p className="text-sm text-huddleroom-text-secondary">
          Two rounds of clarifying questions didn't fully resolve the ambiguity in this goal. Every answer and assumption you've given so far is saved and will be used — none of it is lost.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          <Button
            ref={register}
            variant="primary"
            className="min-h-11"
            disabled={globalBusy}
            onClick={() => {
              const owner: MutationOwner = 'recover'
              if (!acquireMutation(owner)) return
              setRecoverError(null)
              setRecoverPendingMode('proceed')
              recover.mutate({ goalId, mode: 'proceed' }, {
                onSuccess: () => {
                  setMessage('Recovery started — proceeding with current understanding.')
                  setLastMessageOwner(owner)
                  onSuccessFocus?.()
                },
                onError: (error) => {
                  setRecoverError(errorText(error))
                  setRecoverPendingMode(null)
                },
                onSettled: (_data, error) => {
                  releaseMutation(owner)
                  if (!error) setRecoverPendingMode(null)
                },
              })
            }}
          >
            {recoverPendingMode === 'proceed' ? 'Proceeding…' : 'Proceed with current understanding'}
          </Button>
          <Button
            variant="secondary"
            className="min-h-11"
            disabled={globalBusy}
            onClick={() => {
              const owner: MutationOwner = 'recover'
              if (!acquireMutation(owner)) return
              setRecoverError(null)
              setRecoverPendingMode('another_round')
              recover.mutate({ goalId, mode: 'another_round' }, {
                onSuccess: () => {
                  setMessage('Recovery started — sending another round of clarifying questions.')
                  setLastMessageOwner(owner)
                  onSuccessFocus?.()
                },
                onError: (error) => {
                  setRecoverError(errorText(error))
                  setRecoverPendingMode(null)
                },
                onSettled: (_data, error) => {
                  releaseMutation(owner)
                  if (!error) setRecoverPendingMode(null)
                },
              })
            }}
          >
            {recoverPendingMode === 'another_round' ? 'Sending…' : 'Ask another round'}
          </Button>
        </div>
        <div className="mt-2 flex flex-col gap-1">
          <p className="text-xs text-huddleroom-text-muted">
            Move forward using the answers and assumptions already recorded. Anything still unresolved is logged as an open assumption you can revisit later.
          </p>
          <p className="text-xs text-huddleroom-text-muted">
            Send one more round of clarifying questions before proceeding. Use this only if the remaining ambiguity would likely cause real rework.
          </p>
        </div>
        {recoverError && <p className="mt-2 text-xs text-huddleroom-status-red">{recoverError}</p>}
      </>
    )
  }

  const debugContent = (
    <>
      <ExecutionSupervision supervision={detail.supervision} />
      {!terminal && (
        <DangerZoneSection
          resetButton={resetButton}
          globalBusy={globalBusy}
          onReset={() => {
            if (mutationOwnerRef.current !== null) return
            setResetError(null)
            setResetOpen(true)
          }}
        />
      )}
    </>
  )

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <button
        type="button"
        className="min-h-11 self-start text-sm text-huddleroom-text-secondary hover:text-huddleroom-text-primary focus:outline-2 focus:outline-huddleroom-primary"
        onClick={() => navigate('/orchestration')}
      >
        ← Orchestration goals
      </button>

      <RoomPanel goalTitle={goal.objective} agentRows={agentRows} meetingRow={meetingRow} gates={roomGates} />

      <header
        ref={goalHeader}
        tabIndex={-1}
        data-testid="orchestration-goal-header"
        className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4"
      >
        <div className="min-w-0">
          <PageHeader title={goal.objective} className="sr-only" />

          {/* PRIMARY: goal status + current phase */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <StatusBadge status={goal.status} label={goalStatusLabel(goal.status)} className="text-sm font-semibold" />
            {run && run.status !== goal.status && (
              <span data-testid="orchestration-run-status" className="flex items-center gap-1 text-sm text-huddleroom-text-secondary">
                <span aria-hidden="true">·</span>
                <StatusBadge status={run.status} label={runStatusLabel(run.status)} />
              </span>
            )}
          </div>

          {/* SECONDARY: compact reference cluster; gate counts grouped */}
          <div className="mt-2 flex flex-wrap gap-x-6 gap-y-2 text-xs">
            <div className="flex flex-col gap-1">
              <span className="text-huddleroom-text-muted">Plan</span>
              <StatusBadge status={String(run?.plan_state.status ?? 'pending')} label={runStatusLabel(String(run?.plan_state.status ?? 'pending'))} />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-huddleroom-text-muted">Gates</span>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-huddleroom-text-secondary">{gateCounts.open} open</span>
                <span className={gateCounts.failed > 0 ? 'font-semibold text-huddleroom-status-red' : 'text-huddleroom-text-secondary'}>
                  {gateCounts.failed} failed
                </span>
                <span className="text-huddleroom-text-secondary">{gateCounts.accepted} accepted</span>
              </div>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-huddleroom-text-muted">Blockers</span>
              <span className="text-huddleroom-text-primary">{run?.active_blockers.length ?? 0}</span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-huddleroom-text-muted">Updated</span>
              <time dateTime={goal.updated_at} className="font-mono text-huddleroom-text-primary">{absolute(goal.updated_at)}</time>
            </div>
          </div>

          {/* Freshness cue (reuses react-query dataUpdatedAt/isFetching) */}
          {query.dataUpdatedAt ? (
            <p className={`mt-1 text-[11px] ${query.isError ? 'text-huddleroom-status-amber' : 'text-huddleroom-text-muted'}`}>
              Data as of {absolute(query.dataUpdatedAt)}
              {query.isError ? ' · not updating' : query.isFetching ? ' · refreshing…' : ''}
            </p>
          ) : null}
        </div>
        <div className="mt-2 min-h-5 text-xs text-huddleroom-text-secondary">{lastMessageOwner?.startsWith('override:') ? '' : message}</div>
        {commandError && <p className="text-xs text-huddleroom-status-red">{commandError}</p>}
      </header>

      {!terminal && (
        <div className="mt-3 flex items-center justify-between">
          <div aria-busy={lifecycleBusy} className="flex flex-wrap items-center gap-2">
            {lifecycleCommand && (
              lifecycleCommand === 'pause' && pauseConfirming ? (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs text-huddleroom-text-secondary">Pause goal? Running agents finish their current step, then stop.</span>
                  <Button
                    size="sm"
                    variant="secondary"
                    className="min-h-11"
                    disabled={lifecycleBusy}
                    onClick={() => { setPauseConfirming(false); runCommand(lifecycleCommand) }}
                  >
                    {mutationOwner === 'lifecycle:pause' ? 'Pausing…' : 'Confirm pause'}
                  </Button>
                  <Button size="sm" variant="ghost" className="min-h-11" disabled={lifecycleBusy} onClick={() => setPauseConfirming(false)}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  ref={lifecycleButton}
                  size="sm"
                  variant="secondary"
                  className="min-h-11"
                  disabled={lifecycleBusy}
                  onClick={() => (lifecycleCommand === 'pause' ? setPauseConfirming(true) : runCommand(lifecycleCommand))}
                >
                  {mutationOwner === 'lifecycle:pause'
                    ? 'Pausing…'
                    : mutationOwner === 'lifecycle:resume'
                      ? 'Resuming…'
                      : lifecycleCommand === 'pause' ? 'Pause goal' : 'Resume goal'}
                </Button>
              )
            )}
            {canStart && (
              <Button
                ref={startButton}
                size="sm"
                variant="secondary"
                className="min-h-11"
                disabled={globalBusy}
                onClick={() => {
                  if (mutationOwnerRef.current !== null) return
                  setStartError(null)
                  setStartOpen(true)
                }}
              >
                Start work
              </Button>
            )}
            <Button
              ref={cancelButton}
              size="sm"
              variant="secondary"
              className="min-h-11"
              disabled={globalBusy}
              onClick={() => {
                if (mutationOwnerRef.current !== null) return
                setCancelError(null)
                setCancelOpen(true)
              }}
            >
              Cancel goal
            </Button>
          </div>
          {debug && (
            <button
              type="button"
              className="text-xs text-huddleroom-text-muted underline-offset-2 hover:underline"
              onClick={() => detailsTabsRef.current?.showDebugTab()}
            >
              Danger zone
            </button>
          )}
        </div>
      )}

      <BaselineDashboard
        projectId={projectId}
        goalId={goalId}
        goal={goal}
        run={run}
        gates={detail.gates}
        mutationBusy={baselineBusy}
        acquireMutation={acquireMutation}
        releaseMutation={releaseMutation}
        onMessage={(msg) => { setMessage(msg); setLastMessageOwner(null) }}
        onSelectedProcessTypeChange={setActivityStepType}
        onViewActivity={() => detailsTabsRef.current?.showStepActivity()}
        renderBlockerRecovery={renderBlockerRecovery}
        suppressQueueAnnouncement
      />

      <DetailsTabs
          ref={detailsTabsRef}
          projectId={projectId}
          goalId={goalId}
          detail={detail}
          processes={currentProcesses(baseline.processes.data)}
          decisions={baseline.decisions.data ?? []}
          warnings={baseline.warnings.data ?? []}
          selectedStepType={activityStepType}
          memory={baseline.memory}
          debug={debug}
          decisionFocusId={decisionFocusId}
          decisionNavigation={decisionNavigation}
          goalPlanContent={<>
            {/* The tab itself is the disclosure (item 4: unwrap double-
                disclosure) — these no longer need their own collapsible. */}
            <section>
              <h3 className="text-xs font-medium text-huddleroom-text-muted">Goal context</h3>
              <p className="mb-2 mt-1 text-xs text-huddleroom-text-muted">The success criteria and constraints this goal was created with.</p>
              <ol className="list-decimal space-y-2 pl-5 text-sm text-huddleroom-text-primary">
                {goal.success_criteria.map((criterion, index) => (
                  <li key={String(criterion.key ?? index)} className="min-w-0 break-words">
                    {criterion.description && <span className="break-words">{criterion.description}</span>}
                    <span className="ml-2 break-all font-mono text-xs text-huddleroom-text-muted">{String(criterion.key ?? `criterion-${index + 1}`)}</span>
                  </li>
                ))}
              </ol>
              <ScalarRecord label="Constraints" value={goal.constraints} />
              <ScalarRecord label="Budget" value={goal.budget} />
            </section>

            {run ? <section className="mt-4 border-t border-huddleroom-border pt-4">
              <h3 className="text-xs font-medium text-huddleroom-text-muted">{run.plan_state.status === 'accepted' ? 'Accepted plan' : 'Plan state'}</h3>
              <div className="mt-2 grid gap-2 text-sm sm:grid-cols-2">
                <span>Status <StatusBadge status={String(run.plan_state.status ?? 'pending')} /></span>
                <span title={`Planning task ${run.plan_state.planning_task_id ?? '—'}`}>Planning task</span>
                <span title={`Artifact ${run.plan_state.accepted_artifact_id ?? '—'}`}>Artifact</span>
                <span>{run.plan_state.revision_requests?.length ?? 0} revisions</span>
              </div>
              {(run.plan_state.expanded_items ?? []).length > 0 && (
                <ul className="mt-3 divide-y divide-huddleroom-border border-t border-huddleroom-border">
                  {(run.plan_state.expanded_items ?? []).map((item, index) => (
                    <li key={item.plan_item_id ?? index} className="py-2 text-xs" title={`work function ${item.work_function ?? 'unspecified'} · task ${item.task_id ?? '—'} · gate ${item.gate_id ?? '—'}`}>
                      <span className="font-medium text-huddleroom-text-primary">{item.plan_item_id ?? `item-${index + 1}`}</span>
                    </li>
                  ))}
                </ul>
              )}
              {debug && <JsonDetails label="Raw plan state" value={run.plan_state} />}
            </section> : <section className="mt-4 border-t border-huddleroom-border pt-4">
              <h3 className="text-xs font-medium text-huddleroom-text-muted">Plan state</h3>
              <p className="mt-2 text-sm text-huddleroom-text-secondary">No orchestration run is attached to this goal.</p>
            </section>}
          </>}
          gatesContent={
            <GatesPanel
              gates={detail.gates}
              evidence={detail.evidence}
              criteriaByKey={criteriaByKey}
              terminal={terminal}
              debug={debug}
              overrideGateId={overrideGateId}
              overrideError={overrideError}
              mutationOwner={mutationOwner}
              mutationOwnerRef={mutationOwnerRef}
              override={override}
              acquireMutation={acquireMutation}
              releaseMutation={releaseMutation}
              setOverrideError={setOverrideError}
              setOverrideGateId={setOverrideGateId}
              closeOverride={closeOverride}
              setMessage={setMessage}
              setLastMessageOwner={setLastMessageOwner}
              overrideButtons={overrideButtons}
              gateOverrideBusy={gateOverrideBusy}
              goalId={goalId}
              message={message}
              lastMessageOwner={lastMessageOwner}
              badge={gateCounts.failed}
              showEmptyState={true}
            />
          }
          delegationsContent={
            <Panel title="Active delegations" muted={true} bordered={false}>
              {tasksQuery.isError ? (
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm text-huddleroom-status-red">Failed to load active delegations.</p>
                  <Button variant="ghost" className="min-h-11" onClick={() => { void tasksQuery.refetch() }}>Retry</Button>
                </div>
              ) : tasksQuery.isLoading || tasksQuery.hasNextPage || tasksQuery.isFetchingNextPage ? (
                <div aria-busy className="space-y-2">
                  <SkeletonRow />
                  <SkeletonRow />
                </div>
              ) : activeDelegations.length === 0 ? (
                <p className="text-xs text-huddleroom-text-muted">No active delegations — session, meeting, and protocol targets only appear in the timeline.</p>
              ) : (
                <div className="divide-y divide-huddleroom-border">
                  {activeDelegations.map((delegation) => (
                    <div key={delegation.id} title={`${delegation.targetType}:${delegation.targetId}`}>
                      <Record
                        rows={[
                          { key: 'Delegation', value: delegation.title ?? delegation.actionType },
                          { key: 'Status', value: <StatusBadge status={delegation.status} /> },
                        ]}
                      />
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          }
          suggestionsContent={
            <Panel title="Agent suggestions" muted={true} bordered={false} level="h3">
              {detail.agent_suggestions.length === 0 ? (
                <p className="text-sm text-huddleroom-text-secondary">No agent suggestions — the orchestrator proposes a new agent here only when no existing one fits a required work function.</p>
              ) : (
                <ul className="divide-y divide-huddleroom-border">
                  {detail.agent_suggestions.map((suggestion) => (
                    <li key={suggestion.id} className="min-w-0 py-2 first:pt-0 last:pb-0">
                      <div className="flex min-w-0 flex-wrap gap-2">
                        <span className="min-w-0 break-all font-mono text-xs font-semibold text-huddleroom-text-primary">{suggestion.missing_work_function}</span>
                        <StatusBadge status={suggestion.status} />
                      </div>
                      <p className="mt-1 break-words text-sm text-huddleroom-text-secondary">{suggestion.reason}</p>
                      <p className="mt-1 break-words text-xs text-huddleroom-text-secondary">
                        {suggestion.suggested_role ?? 'unspecified role'} · {suggestion.suggested_model ?? 'unspecified model'}
                      </p>
                      <p className="mt-1 min-w-0 break-words font-mono text-xs text-huddleroom-text-muted">
                        {suggestion.suggested_capabilities.join(', ') || 'no capabilities listed'}
                      </p>
                      {suggestion.suggested_system_prompt_outline && (
                        typeof suggestion.suggested_system_prompt_outline === 'string' ? (
                          <p className="mt-2 text-sm text-huddleroom-text-secondary">{suggestion.suggested_system_prompt_outline}</p>
                        ) : (
                          <JsonDetails label="System prompt outline" value={suggestion.suggested_system_prompt_outline} />
                        )
                      )}
                      <Link to="/agents" className="mt-2 inline-flex min-h-11 items-center text-xs text-huddleroom-primary">Open Agents</Link>
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          }
          debugContent={debug ? debugContent : undefined}
      />

      <ConfirmDialog
        open={cancelOpen}
        onOpenChange={setCancelDialogOpen}
        title="Cancel this orchestration goal?"
        consequence="The orchestrator stops coordinating this goal and cancels its unfinished delegated tasks and their active sessions. Late output from cancelled work is ignored."
        confirmLabel="Confirm cancel"
        isPending={mutationOwner === 'cancel' || command.isPending}
        error={cancelError}
        onCloseAutoFocus={(event) => {
          event.preventDefault()
          const focusTarget = cancelFocusTarget.current
          cancelFocusTarget.current = 'trigger'
          const target = focusTarget === 'header'
            ? goalHeader.current
            : cancelButton.current ?? goalHeader.current
          target?.focus()
        }}
        onConfirm={() => {
          const owner: MutationOwner = 'cancel'
          if (!acquireMutation(owner)) return
          setCancelError(null)
          command.mutate({ goalId, command: 'cancel' }, {
            onSuccess: () => {
              setCancelDialogOpen(false, true)
              setMessage('Goal is now cancelled.')
            },
            onError: (error) => setCancelError(errorText(error)),
            onSettled: () => releaseMutation(owner),
          })
        }}
      />

      <StartWorkDialog
        open={startOpen}
        goal={goal}
        isPending={mutationOwner === 'start' || start.isPending}
        error={startError}
        onOpenChange={setStartDialogOpen}
        onCloseAutoFocus={(event) => {
          event.preventDefault()
          const focusTarget = startButton.current ?? goalHeader.current
          focusTarget?.focus()
        }}
        onConfirm={() => {
          const owner: MutationOwner = 'start'
          if (!acquireMutation(owner)) return
          setStartError(null)
          start.mutate({ goalId }, {
            onSuccess: () => {
              setStartDialogOpen(false)
              setMessage('Work started.')
              setLastMessageOwner(owner)
            },
            onError: (error) => setStartError(error),
            onSettled: () => releaseMutation(owner),
          })
        }}
      />

      <ConfirmDialog
        open={resetOpen}
        onOpenChange={setResetOpen}
        title="Reset this goal?"
        consequence="This permanently deletes all decisions, assumptions, warnings, agent reviews, and process history for this goal. The goal returns to its freshly-created state."
        confirmLabel="Confirm reset"
        isPending={mutationOwner === 'reset' || reset.isPending}
        error={resetError}
        onCloseAutoFocus={(event) => {
          event.preventDefault()
          resetButton.current?.focus()
        }}
        onConfirm={() => {
          const owner: MutationOwner = 'reset'
          if (!acquireMutation(owner)) return
          setResetError(null)
          reset.mutate({ goalId }, {
            onSuccess: () => {
              setResetOpen(false)
              setMessage('Goal has been reset.')
            },
            onError: (error) => setResetError(errorText(error)),
            onSettled: () => releaseMutation(owner),
          })
        }}
      />
    </div>
  )
}

export function OrchestrationPage() {
  useDocumentTitle('Orchestration')
  const navigate = useNavigate()
  const projectId = useUIStore((value) => value.activeProjectId)
  const setCreateProjectOpen = useUIStore((value) => value.setCreateProjectOpen)
  const { goalId } = useParams<{ goalId?: string }>()

  if (!projectId) {
    return (
      <NoProjectSelected
        pageTitle="Orchestration"
        title="Select a project to inspect orchestration"
        detail="Goal runs, gates, evidence, blockers, and control actions are scoped to one project. Choose a workspace to load its active and past goals."
        primaryAction={{ label: 'New project', onClick: () => setCreateProjectOpen(true) }}
        secondaryAction={{ label: 'Open dashboard', onClick: () => navigate('/') }}
        prerequisites={[
          'Your account can list at least one HuddleRoom project.',
          'The project switcher in the top bar shows the workspace you want to inspect.',
          'That workspace has at least one goal, or is ready for one to be started.',
        ]}
        alternatePaths={[
          'If the switcher is empty, confirm you are in the correct environment or ask an administrator to create a project.',
          'If no goals exist yet, start one from Meetings or Protocols before returning here.',
        ]}
      />
    )
  }

  return (
    <GoalAnnouncerProvider key={goalId ?? 'list'}>
      {goalId ? <GoalDetailView projectId={projectId} goalId={goalId} /> : <GoalListView projectId={projectId} />}
    </GoalAnnouncerProvider>
  )
}
