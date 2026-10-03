import { useEffect, useRef, useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { Button, STATUS_COLOR_MAP, SkeletonCard, StatusBadge } from '@/components/common/uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationGoal,
  OrchestrationProcessRunRecord,
  OrchestrationRun,
} from '@/lib/types'
import { BASELINE_PROCESS_TYPES, nowLine, statusPhrase, stepDisplayLabel } from './humanize'
import { NowLine } from './NowLine'
import { ZONE_TITLE_CLASS } from './zoneTitle'
import { useGoalAnnouncer } from './goalAnnouncer'

export const isTerminalProcess = (process: OrchestrationProcessRunRecord | undefined) =>
  process?.status === 'completed' || process?.status === 'skipped'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

// Combined visual "needs attention" signal for the stepper dot: an LM request
// that can be retried, or a process-level error the operator already knows is
// retryable (surfaced via the frontier Run affordance). Do not invent new
// backend fields — these are the two signals the dashboard already reads.
function needsRetrySignal(process: OrchestrationProcessRunRecord | undefined) {
  if (!process) return false
  const outputs = isRecord(process.outputs) ? process.outputs : null
  if (!outputs) return false
  const lmRetry = isRecord(outputs.lm_retry) ? outputs.lm_retry : null
  if (lmRetry?.available === true) return true
  return typeof outputs.error === 'string' && outputs.error.trim().length > 0 && outputs.retryable === true
}

function processByType(processes: readonly OrchestrationProcessRunRecord[]) {
  const byType = new Map<OrchestrationBaselineProcessType, OrchestrationProcessRunRecord>()
  for (const process of processes) {
    if (process.superseded_by_id !== null || !BASELINE_PROCESS_TYPES.includes(process.process_type as OrchestrationBaselineProcessType)) continue
    const type = process.process_type as OrchestrationBaselineProcessType
    const previous = byType.get(type)
    if (!previous || process.updated_at.localeCompare(previous.updated_at) > 0) byType.set(type, process)
  }
  return byType
}

function closeoutReady(
  rows: Map<OrchestrationBaselineProcessType, OrchestrationProcessRunRecord>,
  goal: OrchestrationGoal,
  run: OrchestrationRun | null,
) {
  return isTerminalProcess(rows.get('agent_definition_review'))
    && isTerminalProcess(rows.get('team_hierarchy'))
    && rows.get('effectiveness_review')?.status !== 'waiting_decision'
    && run?.active_blockers.length === 0
    && (goal.status === 'active' || goal.status === 'blocked')
}

function canRun(
  type: OrchestrationBaselineProcessType,
  process: OrchestrationProcessRunRecord | undefined,
  rows: Map<OrchestrationBaselineProcessType, OrchestrationProcessRunRecord>,
  goal: OrchestrationGoal,
  run: OrchestrationRun | null,
) {
  if (process?.status === 'waiting_decision' || (process && process.status !== 'running')) return false
  if (type === 'goal_definition') return true
  if (type === 'manager_selection') return isTerminalProcess(rows.get('goal_definition'))
  if (type === 'agent_definition_review') return isTerminalProcess(rows.get('manager_selection'))
  if (type === 'team_hierarchy') return isTerminalProcess(rows.get('agent_definition_review'))
  if (type === 'effectiveness_review') return isTerminalProcess(rows.get('agent_definition_review')) && isTerminalProcess(rows.get('team_hierarchy'))
  return closeoutReady(rows, goal, run)
}

// The single non-terminal step whose predecessors are all terminal — the only
// step the header offers a Run control for. A step stuck on something else
// (e.g. an earlier decision) doesn't block a later step whose own
// prerequisites are already met — canRun already encodes each step's real
// predecessors, so this just returns the first step it clears for
// (ponytail: effectiveness_review and goal_closeout can run in parallel on
// the backend, but this picks effectiveness_review first when both are
// eligible; revisit if operators need to jump straight to closeout).
export function frontierProcessType(
  processes: readonly OrchestrationProcessRunRecord[],
  goal: OrchestrationGoal,
  run: OrchestrationRun | null,
): OrchestrationBaselineProcessType | null {
  const rows = processByType(processes)
  for (const type of BASELINE_PROCESS_TYPES) {
    const process = rows.get(type)
    if (isTerminalProcess(process)) continue
    if (canRun(type, process, rows, goal, run)) return type
  }
  return null
}

export interface ProcessChainProps {
  processes: readonly OrchestrationProcessRunRecord[]
  processState: 'ready' | 'loading' | 'error'
  selectedProcessType: OrchestrationBaselineProcessType
  onSelect: (type: OrchestrationBaselineProcessType) => void
  // Feeds the now-line composer (spec: narrative now line). Optional so
  // callers/tests that only exercise the stepper itself keep working.
  goal?: OrchestrationGoal
  checkpointItems?: readonly OrchestrationAuthorityDecisionRecord[]
  // Baseline-gate button (Task 12b) — "Start baseline" / "Continue baseline",
  // visible only while run.phase === 'baseline' && !run.baseline_authorized.
  // Optional so callers/tests that don't exercise the gate keep working.
  run?: OrchestrationRun | null
  onAuthorize?: (trigger: HTMLButtonElement | null) => void
  authorizePending?: boolean
  authorizeError?: string | null
}

export function ProcessChain({ processes, processState, selectedProcessType, onSelect, goal, checkpointItems = [], run, onAuthorize, authorizePending = false, authorizeError = null }: ProcessChainProps) {
  const [focusedType, setFocusedType] = useState(selectedProcessType)
  const selectButtonRefs = useRef<Partial<Record<OrchestrationBaselineProcessType, HTMLButtonElement | null>>>({})
  const authorizeButtonRef = useRef<HTMLButtonElement | null>(null)
  const rows = processByType(processes)
  const { announceQueue, announceError } = useGoalAnnouncer()
  const showAuthorize = !!(onAuthorize && run && run.phase === 'baseline' && !run.baseline_authorized)
  const authorizeLabel = rows.size === 0 ? 'Start baseline' : 'Continue baseline'

  useEffect(() => {
    setFocusedType(selectedProcessType)
  }, [selectedProcessType])
  useEffect(() => { if (processState === 'loading') announceQueue('Loading orchestrator steps.') }, [processState, announceQueue])
  useEffect(() => { if (processState === 'error') announceError('Orchestrator steps are unavailable.') }, [processState, announceError])

  function handleChainKeyDown(e: React.KeyboardEvent, index: number) {
    const processTypes = BASELINE_PROCESS_TYPES
    let nextIndex = index

    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      nextIndex = Math.min(index + 1, processTypes.length - 1)
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      nextIndex = Math.max(index - 1, 0)
    } else if (e.key === 'Home') {
      nextIndex = 0
    } else if (e.key === 'End') {
      nextIndex = processTypes.length - 1
    } else {
      return
    }

    e.preventDefault()
    const nextType = processTypes[nextIndex]
    setFocusedType(nextType)
    requestAnimationFrame(() => selectButtonRefs.current[nextType]?.focus())
  }

  return (
    <section aria-labelledby="baseline-process-chain-heading" className="rounded-md border border-huddleroom-border bg-huddleroom-surface p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 id="baseline-process-chain-heading" className={ZONE_TITLE_CLASS}>Orchestrator steps</h2>
        {showAuthorize && (
          <Button
            ref={authorizeButtonRef}
            type="button"
            size="sm"
            className="min-h-11"
            disabled={authorizePending}
            onClick={() => onAuthorize?.(authorizeButtonRef.current)}
          >
            {authorizePending ? 'Starting…' : authorizeLabel}
          </Button>
        )}
      </div>
      {showAuthorize && authorizeError && <p className="mt-1 text-xs text-huddleroom-status-red">{authorizeError}</p>}
      {processState === 'loading' ? (
        <div aria-busy="true" aria-label="Loading orchestrator steps" className="mt-3 flex gap-2 overflow-x-auto">
          {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      ) : processState === 'error' ? (
        <p className="mt-3 text-sm text-huddleroom-status-red">Orchestrator steps are unavailable.</p>
      ) : (
      <>
        <p className="sr-only">Use arrow keys to move between processes. Press Enter or Space to select the focused process.</p>
        {/* `overflow-x-auto` without an explicit overflow-y forces overflow-y
            to compute to `auto` too (CSS overflow spec: one axis non-visible
            drags the other off `visible`) — so the selected/hover ring's
            box-shadow, which extends ~1px past each node's border box, was
            being clipped at the scroll container's top edge. `py-1` gives it
            room on both edges instead of just the bottom (review round 4,
            finding U1). */}
        <ol role="group" aria-label="Orchestrator steps" className="mt-3 flex items-start overflow-x-auto py-1 snap-x snap-proximity max-[899px]:[mask-image:linear-gradient(to_right,transparent,black_16px,black_calc(100%-16px),transparent)] max-[899px]:[-webkit-mask-image:linear-gradient(to_right,transparent,black_16px,black_calc(100%-16px),transparent)]">
          {BASELINE_PROCESS_TYPES.map((type, index) => {
            const process = rows.get(type)
            const phrase = process ? statusPhrase(process.status) : { label: 'Not started', tone: 'idle' as const }
            const color = STATUS_COLOR_MAP[phrase.tone] ?? STATUS_COLORS.neutral
            const running = process?.status === 'running'
            const needsRetry = needsRetrySignal(process)
            const predecessorTerminal = index === 0 || isTerminalProcess(rows.get(BASELINE_PROCESS_TYPES[index - 1]))
            const connectorColor = predecessorTerminal ? STATUS_COLORS.green : STATUS_COLORS.neutral
            const selected = selectedProcessType === type
            return (
              <li key={type} className="flex min-w-0 flex-none items-start snap-start">
                <button
                  ref={(element) => {
                    selectButtonRefs.current[type] = element
                  }}
                  type="button"
                  aria-label={`Select ${stepDisplayLabel(type)}`}
                  aria-pressed={selected}
                  data-selected={selected}
                  tabIndex={focusedType === type ? 0 : -1}
                  className={`flex min-h-11 w-24 flex-none cursor-pointer flex-col items-center gap-1 rounded px-1 py-1 text-center ${selected ? 'bg-huddleroom-surface text-huddleroom-primary ring-1 ring-huddleroom-primary' : 'text-huddleroom-text-primary hover:bg-huddleroom-surface hover:ring-1 hover:ring-huddleroom-border'}`}
                  onClick={() => onSelect(type)}
                  onFocus={() => setFocusedType(type)}
                  onKeyDown={(e) => handleChainKeyDown(e, index)}
                >
                  <span aria-hidden="true" className="relative flex h-6 w-6 items-center justify-center rounded-full" style={{ backgroundColor: color }}>
                    {running && <span className="absolute inset-0 rounded-full motion-safe:animate-pulse" style={{ backgroundColor: STATUS_COLORS.amber }} />}
                    {needsRetry && (
                      <span className="absolute -right-1 -top-1 flex h-3.5 w-3.5 items-center justify-center rounded-full ring-2 ring-huddleroom-surface" style={{ backgroundColor: STATUS_COLORS.red }}>
                        <RotateCcw className="h-2 w-2 text-white" strokeWidth={3} />
                      </span>
                    )}
                  </span>
                  <span className="break-words text-xs font-medium leading-tight">{stepDisplayLabel(type)}</span>
                  <StatusBadge key={phrase.tone} status={phrase.tone} label={phrase.label} className="motion-safe:animate-[badge-fade-in_150ms_ease-out]" />
                  {needsRetry && <span className="text-xs font-medium text-huddleroom-status-red">Needs retry</span>}
                </button>
                {index < BASELINE_PROCESS_TYPES.length - 1 && (
                  <span aria-hidden="true" className="mt-3 h-0.5 w-6 shrink-0 self-start motion-safe:transition-colors motion-safe:duration-200 motion-safe:ease-out" style={{ backgroundColor: connectorColor }} />
                )}
              </li>
            )
          })}
        </ol>
        {goal && <NowLine segments={nowLine({ goal, processes, checkpointItems, run })} />}
      </>
      )}
    </section>
  )
}
