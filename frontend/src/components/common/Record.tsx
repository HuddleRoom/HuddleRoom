import type { ReactNode } from 'react'
import { cn } from './uiPrimitives'

export interface RecordRow {
  key: string
  value: ReactNode
  // Optional stable identity for React reconciliation when `key` (the
  // visible label) is not unique across rows — e.g. gates sharing a
  // success_criterion_key. Falls back to `key` when omitted.
  id?: string
}

export interface RecordProps {
  rows: RecordRow[]
  // Fixed key-column width in px — 64 default, 84 for decisions (design spec
  // Primitives #2).
  keyWidth?: number
  className?: string
}

/**
 * `<dl>`-based key/value list. Key column: 11px uppercase muted, fixed width.
 * Value: 14px ink. A row whose value is null/undefined/'' is not rendered —
 * this is the core rule callers rely on to keep None/{}/null out of the UI.
 */
export function Record({ rows, keyWidth = 64, className }: RecordProps) {
  const visible = rows.filter((row) => row.value !== null && row.value !== undefined && row.value !== '')
  if (visible.length === 0) return null
  return (
    <dl className={cn('space-y-1', className)}>
      {visible.map((row) => (
        <div key={row.id ?? row.key} className="flex items-baseline gap-2">
          <dt
            className="shrink-0 text-[11px] font-medium uppercase tracking-[0.1em] text-huddleroom-text-muted"
            style={{ width: keyWidth }}
          >
            {row.key}
          </dt>
          <dd className="m-0 min-w-0 flex-1 text-sm text-huddleroom-text-primary">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}
