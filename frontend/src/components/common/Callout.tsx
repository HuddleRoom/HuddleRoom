import React from 'react'
import { Info, TriangleAlert } from 'lucide-react'
import { cn } from './uiPrimitives'

interface CalloutProps {
  variant: 'info' | 'warning'
  children: React.ReactNode
  className?: string
}

const styles = {
  warning: 'bg-huddleroom-warning-bg border-huddleroom-warning-border text-huddleroom-warning-text',
  info: 'bg-white border-huddleroom-info-border text-huddleroom-text-primary',
}

export function Callout({ variant, children, className }: CalloutProps) {
  const Icon = variant === 'warning' ? TriangleAlert : Info
  return (
    <div role="status"
      className={cn('flex items-start gap-2 rounded-md border px-3 py-2.5 text-sm', styles[variant], className)}>
      <Icon size={15} aria-hidden="true" className="mt-0.5 shrink-0" />
      <div>{children}</div>
    </div>
  )
}
