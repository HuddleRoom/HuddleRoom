import { useEffect, useRef, useState } from 'react'
import type * as RadixDialog from '@radix-ui/react-dialog'
import { Dialog } from '@/components/common/Dialog'
import { ApiError } from '@/lib/api-client'
import { Callout } from '@/components/common/Callout'
import { Record } from '@/components/common/Record'
import { Tag } from '@/components/common/Tag'
import type { OrchestrationAuthorization, OrchestrationGoal } from '@/lib/types'
import { authorizationReasonLabel, budgetLeftText, goalTypeLabel, requiredDecisionLabel, toolConstraintLabel, workspaceMutationLabel } from './humanize'

const CONFLICT_COPY: Record<string, string> = {
  baseline_not_ready: 'Baseline is not ready yet. Finish baseline before starting work.',
  already_started: 'This goal has already been started.',
  goal_not_runnable: 'This goal cannot be started right now.',
}

function conflictText(error: unknown) {
  if (error instanceof ApiError && error.detail && typeof error.detail === 'object') {
    const conflict = (error.detail as { conflict?: unknown }).conflict
    if (typeof conflict === 'string' && CONFLICT_COPY[conflict]) return CONFLICT_COPY[conflict]
  }
  return error instanceof Error ? error.message : 'The request failed. Check your connection and try again.'
}

function authority(goal: OrchestrationGoal) {
  return goal.authority_model?.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) || 'No manager'
}

function scope(goal: OrchestrationGoal) {
  const ownedFiles = goal.constraints.owned_files
  if (Object.keys(goal.constraints).length === 0) return 'No explicit scope'
  return Object.keys(goal.constraints).length === 1
    && Array.isArray(ownedFiles) && ownedFiles.every((value) => typeof value === 'string') && ownedFiles.length
    ? ownedFiles.join(', ')
    : summary(goal.constraints) || 'No explicit scope'
}

function budget(goal: OrchestrationGoal) {
  const entries = Object.entries(goal.budget)
  return entries.length
    ? entries.map(([key, value]) => `${key.replace(/_/g, ' ')}: ${summary(value)}`).join(', ')
    : 'No explicit budget'
}

function summary(value: unknown) {
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function allowedList(items: string[]) {
  return items.length === 0
    ? 'Not allowed'
    : <>Allowed: <span className="flex flex-wrap gap-1">{items.map((item) => <Tag key={item} className="font-mono">{item}</Tag>)}</span></>
}

// Start itself resolves a "start" decision (first Start or re-Start), so only
// other denials, a missing policy, or an unavailable projection block it.
function blockedText(authorization: OrchestrationAuthorization | null, missingRecurringPolicy: boolean) {
  if (missingRecurringPolicy) {
    return 'This Continuous goal has no accepted recurring policy yet. Accept one in the baseline, then start.'
  }
  if (!authorization) return 'Authorization details are unavailable. Refresh and try again.'
  if (authorization.allowed || authorization.required_decision === 'start') return null
  const reason = authorization.reason_code ? authorizationReasonLabel(authorization.reason_code) : 'it is not allowed right now'
  const needed = authorization.required_decision ? ` Needed: ${requiredDecisionLabel(authorization.required_decision)}.` : ''
  return `Start isn't available: ${reason}.${needed}`
}

interface StartWorkDialogProps {
  open: boolean
  targetSummary?: string
  goal: OrchestrationGoal
  authorization: OrchestrationAuthorization | null
  missingRecurringPolicy: boolean
  isPending: boolean
  error: unknown
  onOpenChange: (open: boolean) => void
  onConfirm: (scope: 'planning_only' | 'execution') => void
  initialScope?: 'planning_only' | 'execution'
  // Why the goal needs a fresh Start (re-Start); null/undefined for a first Start.
  restartReason?: string | null
  onCloseAutoFocus?: React.ComponentPropsWithoutRef<typeof RadixDialog.Content>['onCloseAutoFocus']
}

export function StartWorkDialog({
  open, goal, targetSummary, authorization, missingRecurringPolicy, isPending, error, onOpenChange, onConfirm, initialScope = 'execution', restartReason, onCloseAutoFocus,
}: StartWorkDialogProps) {
  const confirmFocusIntent = useRef(false)
  const primaryRef = useRef<HTMLButtonElement>(null)
  const [scopeChoice, setScopeChoice] = useState<'planning_only' | 'execution'>(initialScope)
  const message = error ? conflictText(error) : null
  const blocked = blockedText(authorization, missingRecurringPolicy)

  useEffect(() => {
    if (isPending || !message || !confirmFocusIntent.current) return
    confirmFocusIntent.current = false
    primaryRef.current?.focus()
  }, [isPending, message])
  useEffect(() => { if (open) setScopeChoice(initialScope) }, [open, initialScope])

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => { if (!isPending) onOpenChange(nextOpen) }}
      title={restartReason ? 'Start work again?' : 'Start work on this goal?'}
      description={restartReason
        ? `${restartReason.charAt(0).toUpperCase()}${restartReason.slice(1).replace(/\.$/, '')}. Starting again re-authorizes the orchestrator with the limits below.`
        : 'The baseline is complete. Starting authorizes the orchestrator to release work for this goal.'}
      size="md"
      role="alertdialog"
      onCloseAutoFocus={onCloseAutoFocus}
      footer={{
        primaryLabel: restartReason ? 'Start again' : 'Start work',
        pendingLabel: 'Starting…',
        primaryRef,
        onPrimary: () => { confirmFocusIntent.current = true; onConfirm(scopeChoice) },
        isPending,
        primaryDisabled: blocked !== null,
      }}
    >
      {targetSummary && <p className="mb-4 break-words text-sm">{targetSummary}</p>}
      {blocked && <Callout variant="warning" className="mb-4">{blocked}</Callout>}
      <fieldset className="mb-4 flex flex-col gap-2" disabled={isPending || blocked !== null}>
        <legend className="text-xs font-medium text-huddleroom-text-muted">Start scope</legend>
        <label className="flex min-h-11 cursor-pointer items-center gap-2 rounded border border-huddleroom-border px-3 text-sm">
          <input type="radio" name="start-scope" checked={scopeChoice === 'planning_only'} onChange={() => setScopeChoice('planning_only')} />
          Plan only — review and verify the candidate. Implementation needs a later Start.
        </label>
        <label className="flex min-h-11 cursor-pointer items-center gap-2 rounded border border-huddleroom-border px-3 text-sm">
          <input type="radio" name="start-scope" checked={scopeChoice === 'execution'} onChange={() => setScopeChoice('execution')} />
          Implementation — authorize eligible work after the reviewed plan is accepted.
        </label>
      </fieldset>
      <Record
        keyWidth={120}
        rows={[
          { key: 'Goal type', value: goalTypeLabel(goal.goal_type) },
          { key: 'Authority', value: authority(goal) },
          { key: 'Scope', value: <span className="break-words">{scope(goal)}</span> },
          { key: 'Budget', value: <span className="break-words">{budget(goal)}</span> },
        ]}
      />
      {authorization && (
        <section className="mt-4" aria-label="What Start allows">
          <h3 className="mb-2 text-xs font-medium text-huddleroom-text-muted">What Start allows</h3>
          <Record
            keyWidth={120}
            rows={[
              { key: 'Model spend left', value: budgetLeftText(authorization.model_spend_remaining) },
              { key: 'External writes', value: allowedList(authorization.external_write_targets) },
              { key: 'Live sends', value: allowedList(authorization.live_send_channels) },
              { key: 'Workspace', value: workspaceMutationLabel(authorization.workspace_mutation) },
              { key: 'Tool limits', value: authorization.tool_constraints.length ? authorization.tool_constraints.map(toolConstraintLabel).join(', ') : 'None' },
            ]}
          />
        </section>
      )}
      {message && <p role="alert" aria-live="assertive" className="mt-4 text-xs text-huddleroom-status-red">{message}</p>}
    </Dialog>
  )
}
