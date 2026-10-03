import { useEffect, useRef } from 'react'
import type * as RadixDialog from '@radix-ui/react-dialog'
import { Dialog } from '@/components/common/Dialog'
import { ApiError } from '@/lib/api-client'
import type { OrchestrationGoal } from '@/lib/types'

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

interface StartWorkDialogProps {
  open: boolean
  goal: OrchestrationGoal
  isPending: boolean
  error: unknown
  onOpenChange: (open: boolean) => void
  onConfirm: () => void
  onCloseAutoFocus?: React.ComponentPropsWithoutRef<typeof RadixDialog.Content>['onCloseAutoFocus']
}

export function StartWorkDialog({
  open, goal, isPending, error, onOpenChange, onConfirm, onCloseAutoFocus,
}: StartWorkDialogProps) {
  const confirmFocusIntent = useRef(false)
  const contentRef = useRef<HTMLDivElement>(null)
  const message = error ? conflictText(error) : null

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
      size="sm"
      role="alertdialog"
      onCloseAutoFocus={onCloseAutoFocus}
      contentRef={contentRef}
      footer={{
        primaryLabel: isPending ? 'Starting…' : 'Start work',
        onPrimary: () => { confirmFocusIntent.current = true; onConfirm() },
        isPending,
      }}
    >
      <dl className="space-y-2 text-sm">
        <div><dt className="text-huddleroom-text-muted">Goal type</dt><dd className="text-huddleroom-text-primary">Outcome</dd></div>
        <div><dt className="text-huddleroom-text-muted">Authority</dt><dd className="break-words text-huddleroom-text-primary">{authority(goal)}</dd></div>
        <div><dt className="text-huddleroom-text-muted">Scope</dt><dd className="break-words text-huddleroom-text-primary">{scope(goal)}</dd></div>
        <div><dt className="text-huddleroom-text-muted">Budget</dt><dd className="break-words text-huddleroom-text-primary">{budget(goal)}</dd></div>
      </dl>
      {message && <p role="alert" aria-live="assertive" className="mt-4 text-xs text-huddleroom-status-red">{message}</p>}
    </Dialog>
  )
}
