import type { ReactNode } from 'react'
import { cn } from './uiPrimitives'
import { timeOfDay } from '@/lib/time'

export interface LedgerRowProps {
  iso: string          // → 44px mono muted time-of-day column
  action: ReactNode    // 168px mono, font-weight 600, blue
  detail: ReactNode    // muted, one line, wraps under action on narrow widths
  children?: ReactNode // optional extra content rendered BELOW the summary line
  className?: string
}

/**
 * Single-row ledger summary. Presentational only.
 * Time column fixed 44px, action column fixed 168px, detail flexes.
 * Detail wraps below on narrow widths.
 * Date-group headers are consumer's responsibility, not this component's.
 */
export function LedgerRow({ iso, action, detail, children, className }: LedgerRowProps) {
  return (
    <div className={cn('', className)}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
        <span
          style={{ width: 44 }}
          className="shrink-0 font-mono text-sm text-huddleroom-text-muted"
        >
          {timeOfDay(iso)}
        </span>
        <span
          style={{ width: 168 }}
          className="shrink-0 truncate font-mono text-sm font-semibold text-huddleroom-primary"
        >
          {action}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm text-huddleroom-text-muted">
          {detail}
        </span>
      </div>
      {children && <div className="mt-1">{children}</div>}
    </div>
  )
}
