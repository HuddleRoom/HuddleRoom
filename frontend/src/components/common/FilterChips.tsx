import React from 'react'
import { cn } from './uiPrimitives'

interface FilterChipsProps {
  options: { id: string; label: string }[]
  activeId?: string
  onChange?: (id: string) => void
  ariaLabel: string
  className?: string
  multi?: boolean
  activeIds?: Set<string>
  onToggle?: (id: string) => void
}

export function FilterChips({
  options,
  activeId,
  onChange,
  ariaLabel,
  className,
  multi,
  activeIds,
  onToggle,
}: FilterChipsProps) {
  return (
    <div
      role={multi ? 'group' : 'radiogroup'}
      aria-label={ariaLabel}
      className={cn('flex flex-wrap gap-1.5', className)}
    >
      {options.map((o) => {
        const checked = multi ? Boolean(activeIds?.has(o.id)) : o.id === activeId
        return (
          <button
            key={o.id}
            type="button"
            role={multi ? 'button' : 'radio'}
            {...(multi ? { 'aria-pressed': checked } : { 'aria-checked': checked })}
            onClick={() => (multi ? onToggle?.(o.id) : onChange?.(o.id))}
            className={cn(
              'rounded-md px-2.5 py-1 text-xs font-medium transition-colors duration-150',
              checked
                ? 'bg-huddleroom-primary text-white'
                : 'border border-huddleroom-border bg-white text-huddleroom-text-secondary hover:bg-huddleroom-depth hover:border-huddleroom-border-strong',
            )}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
