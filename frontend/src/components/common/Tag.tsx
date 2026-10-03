import React from 'react'
import { cn } from './uiPrimitives'

interface TagProps {
  children: React.ReactNode
  mono?: boolean
  className?: string
}

export function Tag({ children, mono, className }: TagProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center bg-huddleroom-depth text-huddleroom-text-secondary rounded-[3px] px-1.5 py-0.5 text-xs',
        mono && 'font-mono text-[11px]',
        className,
      )}
    >
      {children}
    </span>
  )
}
