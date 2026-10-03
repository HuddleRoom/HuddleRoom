import React from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft } from 'lucide-react'
import { StatusBadge, cn } from './uiPrimitives'

interface DetailHeaderProps {
  backTo: string
  backLabel: string
  title: string
  titleClassName?: string
  status?: string
  statusLabel?: string
  actions?: React.ReactNode
  className?: string
}

export function DetailHeader({ backTo, backLabel, title, titleClassName, status, statusLabel, actions, className }: DetailHeaderProps) {
  return (
    <div className={cn('flex flex-wrap items-center gap-3', className)}>
      <Link
        to={backTo}
        className="inline-flex items-center gap-0.5 text-xs font-medium text-huddleroom-text-muted no-underline hover:text-huddleroom-text-primary"
      >
        <ChevronLeft size={14} aria-hidden="true" />
        {backLabel}
      </Link>
      <h1 className={cn('m-0 text-[18px] font-semibold leading-snug text-huddleroom-text-primary', titleClassName)}>{title}</h1>
      {status && <StatusBadge status={status} label={statusLabel} />}
      {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
    </div>
  )
}
