import React from 'react'
import { Link } from 'react-router-dom'
import { cn } from './uiPrimitives'

interface StatTileProps {
  label: string
  value: number | undefined
  color?: string
  to?: string
  className?: string
  layout?: 'tile' | 'chip'
  onClick?: () => void
}

export function StatTile({ label, value, color, to, className, layout = 'tile', onClick }: StatTileProps) {
  // value undefined = loading/error — show em-dash, never fabricate a zero or a color.
  const countColor = value !== undefined && value > 0 && color ? color : '#5a6270'

  if (layout === 'chip') {
    const chipBase = 'flex items-center gap-2 px-3 py-1.5 rounded-md border border-huddleroom-border bg-white transition-colors duration-150 hover:border-huddleroom-primary'
    const chipBody = (
      <>
        <span className="text-sm font-semibold tabular-nums" style={{ color: countColor }}>{value ?? '—'}</span>
        <span className="text-xs text-huddleroom-text-muted whitespace-nowrap">{label}</span>
      </>
    )
    if (onClick && !to) {
      return (
        <button type="button" onClick={onClick} aria-label={`${value ?? '—'} ${label}`} className={cn(chipBase, 'text-left', className)}>
          {chipBody}
        </button>
      )
    }
    if (to) {
      return (
        <Link to={to} className={cn(chipBase, 'no-underline', className)}>
          {chipBody}
        </Link>
      )
    }
    return <div className={cn(chipBase, className)}>{chipBody}</div>
  }

  const body = (
    <>
      <span className="text-xs text-huddleroom-text-secondary">{label}</span>
      <span className="text-lg font-semibold leading-tight" style={{ color: countColor }}>{value ?? '—'}</span>
    </>
  )
  const base = 'flex flex-col gap-0.5 rounded-md border border-huddleroom-border bg-white px-3 py-2'
  if (to) {
    return (
      <Link to={to} className={cn(base, 'no-underline transition-colors duration-150 hover:border-huddleroom-primary', className)}>
        {body}
      </Link>
    )
  }
  if (onClick) {
    return (
      <button type="button" onClick={onClick} aria-label={`${value ?? '—'} ${label}`} className={cn(base, 'text-left', className)}>
        {body}
      </button>
    )
  }
  return <div className={cn(base, className)}>{body}</div>
}
