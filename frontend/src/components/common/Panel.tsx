import type { ReactNode } from 'react'
import { cn } from './uiPrimitives'

export interface PanelProps {
  header?: { keyText?: string; title: string }
  children: ReactNode
  className?: string
}

export function Panel({ header, children, className }: PanelProps) {
  return (
    <section
      className={cn(
        'rounded-lg border border-huddleroom-border bg-white shadow-[0_1px_2px_rgba(31,34,38,.06),0_12px_32px_rgba(31,34,38,.06)]',
        className,
      )}
    >
      {header && (
        <div className="bg-huddleroom-bg-2 border-b border-huddleroom-border px-4 py-3">
          {header.keyText && (
            <div className="text-xs text-huddleroom-text-muted">{header.keyText}</div>
          )}
          <h3 className="font-semibold text-huddleroom-text-primary">{header.title}</h3>
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}
