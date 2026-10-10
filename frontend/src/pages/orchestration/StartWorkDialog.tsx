import { useEffect, useRef } from 'react'
import type * as RadixDialog from '@radix-ui/react-dialog'
import { Dialog } from '@/components/common/Dialog'
import { ApiError } from '@/lib/api-client'
import { Callout } from '@/components/common/Callout'
import { Record } from '@/components/common/Record'
import { Tag } from '@/components/common/Tag'
import type { OrchestrationAuthorization, OrchestrationGoal } from '@/lib/types'
import { authorizationReasonLabel, goalTypeLabel, workspaceMutationLabel } from './humanize'

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

function spendText(remaining: Record<string, string>) {
  const entries = Object.entries(remaining)
  if (entries.length === 0) return 'No measured budget'
  return entries
    .map(([key, value]) => `${key.replace(/_/g, ' ')} ${/^\d+$/.test(value) ? Number(value).toLocaleString('en-US') : value}`)
    .join(' · ')
}

function allowedList(items: string[]) {
  return items.length === 0
    ? 'Not allowed'
    : <>Allowed: {items.map((item) => <Tag key={item} className="mr-1 font-mono">{item}</Tag>)}</>
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
  const needed = authorization.required_decision ? ` Needed: ${authorization.required_decision.replace(/_/g, ' ')}.` : ''
  return `Start isn't available: ${reason}.${needed}`
}

interface StartWorkDialogProps {
  open: boolean
  goal: OrchestrationGoal
  authorization: OrchestrationAuthorization | null
  missingRecurringPolicy: boolean
  isPending: boolean
  error: unknown
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  onCloseAutoFocus?: React.ComponentPropsWithoutRef<typeof RadixDialog.Content>['onCloseAutoFocus']
}

export function StartWorkDialog({
  open, goal, authorization, missingRecurringPolicy, isPending, error, onOpenChange, onConfirm, onCloseAutoFocus,
}: StartWorkDialogProps) {
  const confirmFocusIntent = useRef(false)
  const contentRef = useRef<HTMLDivElement>(null)
  const message = error ? conflictText(error) : null
  const blocked = blockedText(authorization, missingRecurringPolicy)

  useEffect(() => {
    if (isPending || !message || !confirmFocusIntent.current) return
    confirmFocusIntent.current = false
    // ponytail: scoped to this dialog's own content via contentRef, so we
    // focus the primary footer button by DOM position rather than by ref.
    // Ceiling: relies on the primary button being the last button in the footer.
    const buttons = contentRef.current?.querySelectorAll('button')
    const primary = buttons?.[buttons.length - 1] as HTMLButtonElement | undefined
    primary?.focus()
  }, [isPending, message])

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => { if (!isPending) onOpenChange(nextOpen) }}
      title="Start work on this goal?"
      description="The baseline is complete. Starting authorizes the orchestrator to release work for this goal."
      size="md"
      role="alertdialog"
      onCloseAutoFocus={onCloseAutoFocus}
      contentRef={contentRef}
      footer={{
        primaryLabel: isPending ? 'Starting…' : 'Start work',
        onPrimary: () => { confirmFocusIntent.current = true; onConfirm() },
        isPending,
        primaryDisabled: blocked !== null,
      }}
    >
      {blocked && <Callout variant="warning" className="mb-4">{blocked}</Callout>}
      <dl className="space-y-2 text-sm">
        <div><dt className="text-huddleroom-text-muted">Goal type</dt><dd className="text-huddleroom-text-primary">{goalTypeLabel(goal.goal_type)}</dd></div>
        <div><dt className="text-huddleroom-text-muted">Authority</dt><dd className="break-words text-huddleroom-text-primary">{authority(goal)}</dd></div>
        <div><dt className="text-huddleroom-text-muted">Scope</dt><dd className="break-words text-huddleroom-text-primary">{scope(goal)}</dd></div>
        <div><dt className="text-huddleroom-text-muted">Budget</dt><dd className="break-words text-huddleroom-text-primary">{budget(goal)}</dd></div>
      </dl>
      {authorization && (
        <section className="mt-4" aria-label="What Start allows">
          <h3 className="mb-2 text-xs font-medium text-huddleroom-text-muted">What Start allows</h3>
          <Record
            keyWidth={120}
            rows={[
              { key: 'Model spend left', value: spendText(authorization.model_spend_remaining) },
              { key: 'External writes', value: allowedList(authorization.external_write_targets) },
              { key: 'Live sends', value: allowedList(authorization.live_send_channels) },
              { key: 'Workspace', value: workspaceMutationLabel(authorization.workspace_mutation) },
              { key: 'Tool limits', value: authorization.tool_constraints.length ? authorization.tool_constraints.join(', ') : 'None' },
            ]}
          />
        </section>
      )}
      {message && <p role="alert" aria-live="assertive" className="mt-4 text-xs text-huddleroom-status-red">{message}</p>}
    </Dialog>
  )
}
