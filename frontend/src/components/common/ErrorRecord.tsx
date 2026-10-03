import React, { useEffect } from 'react'
import { errorRecord } from '@/pages/orchestration/humanize'
import { useGoalAnnouncer } from '@/pages/orchestration/goalAnnouncer'
import { cn } from './uiPrimitives'

export interface ErrorRecordProps {
  error: unknown
  entity?: string
  action?: React.ReactNode
  className?: string
}

/**
 * ErrorRecord displays a classified error as a record with structured fields.
 * Internally classifies the error via errorRecord(). Shows three dt/dd rows
 * ("What happened", "Why", "Do this"), optionally renders an action slot under
 * the "Do this" value, and puts raw details inside a native <details> disclosure.
 * Used for API errors, mutations, warnings, and banners.
 */
export function ErrorRecord({ error, entity, action, className }: ErrorRecordProps) {
  const data = errorRecord(error, { entity })
  const { active, announceError } = useGoalAnnouncer()
  // On the goal-detail page (inside GoalAnnouncerProvider) this announces
  // through the shared assertive region instead of its own role="alert" —
  // avoids double-announcing. Everywhere else (no provider) it keeps its
  // original local role="alert" so those pages don't lose the alert.
  useEffect(() => { if (active) announceError(data.what) }, [active, data.what, announceError])

  return (
    <div role={active ? undefined : 'alert'} className={cn('flex flex-col gap-3', className)}>
      {/* Record rows using semantic dl/dt/dd */}
      <dl className="space-y-2">
        <div className="flex gap-4">
          <dt className="w-16 flex-shrink-0 text-xs font-medium uppercase text-huddleroom-text-muted">What happened</dt>
          <dd className="flex-1 text-sm text-huddleroom-text-primary m-0">{data.what}</dd>
        </div>
        <div className="flex gap-4">
          <dt className="w-16 flex-shrink-0 text-xs font-medium uppercase text-huddleroom-text-muted">Why</dt>
          <dd className="flex-1 text-sm text-huddleroom-text-primary m-0">{data.why}</dd>
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="flex gap-4">
            <dt className="w-16 flex-shrink-0 text-xs font-medium uppercase text-huddleroom-text-muted">Do this</dt>
            <dd className="flex-1 text-sm text-huddleroom-text-primary m-0">{data.doThis}</dd>
          </div>
          {action && <dd className="pl-20 m-0">{action}</dd>}
        </div>
      </dl>

      {/* Details disclosure using native <details> */}
      <details className="border-t border-huddleroom-border pt-2">
        <summary className={cn(
          'text-sm font-medium text-huddleroom-text-primary cursor-pointer',
          'hover:text-huddleroom-text-secondary transition-colors',
          'focus:outline-none focus-visible:outline-2 focus-visible:outline-huddleroom-primary focus-visible:outline-offset-1',
        )}>
          Details
        </summary>
        <pre className="mt-2 rounded-md bg-huddleroom-bg-2 border border-huddleroom-border p-3 text-xs text-huddleroom-text-muted overflow-auto whitespace-pre-wrap break-words font-mono">
          {data.details}
        </pre>
      </details>
    </div>
  )
}
