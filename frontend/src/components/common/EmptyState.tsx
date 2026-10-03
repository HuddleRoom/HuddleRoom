import React from 'react'
import { Button, cn } from './uiPrimitives'

interface EmptyStateAction { label: string; onClick: () => void }
interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  body: string
  action?: EmptyStateAction
  secondaryAction?: EmptyStateAction
  className?: string
}

export function EmptyState({ icon, title, body, action, secondaryAction, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-start gap-2 py-6', className)}>
      {icon && <div aria-hidden="true" className="text-huddleroom-faint">{icon}</div>}
      <h3 className="m-0 text-base font-semibold text-huddleroom-text-primary">{title}</h3>
      <p className="m-0 max-w-[560px] text-sm leading-relaxed text-huddleroom-text-secondary">{body}</p>
      {(action || secondaryAction) && (
        <div className="mt-2 flex flex-wrap gap-2.5">
          {action && <Button variant="primary" onClick={action.onClick}>{action.label}</Button>}
          {secondaryAction && (
            <Button variant="secondary" onClick={secondaryAction.onClick}>{secondaryAction.label}</Button>
          )}
        </div>
      )}
    </div>
  )
}
