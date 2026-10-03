import React from 'react'
import { cn } from './uiPrimitives'

export interface TabDef { id: string; label: string }

interface TabsProps {
  tabs: TabDef[]
  activeId: string
  onChange: (id: string) => void
  idPrefix: string
  ariaLabel?: string
  withPanels?: boolean
  className?: string
}

export function Tabs({ tabs, activeId, onChange, idPrefix, ariaLabel, withPanels = true, className }: TabsProps) {
  const focusTab = (id: string) => {
    onChange(id)
    if (typeof document !== 'undefined') {
      document.getElementById(`${idPrefix}-tab-${id}`)?.focus()
    }
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    const idx = tabs.findIndex((t) => t.id === activeId)
    if (e.key === 'ArrowRight') focusTab(tabs[(idx + 1) % tabs.length].id)
    else if (e.key === 'ArrowLeft') focusTab(tabs[(idx - 1 + tabs.length) % tabs.length].id)
    else if (e.key === 'Home') focusTab(tabs[0].id)
    else if (e.key === 'End') focusTab(tabs[tabs.length - 1].id)
    else return
    e.preventDefault()
  }
  return (
    <div role="tablist" aria-label={ariaLabel} onKeyDown={onKeyDown}
      className={cn('flex gap-1 border-b border-huddleroom-border', className)}>
      {tabs.map((t) => {
        const active = t.id === activeId
        return (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`${idPrefix}-tab-${t.id}`}
            aria-selected={active}
            {...(withPanels && { 'aria-controls': `${idPrefix}-panel-${t.id}` })}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(t.id)}
            className={cn(
              'inline-flex min-h-11 items-center px-3 py-2 text-sm font-medium -mb-px border-b-2 transition-colors duration-150',
              active
                ? 'text-huddleroom-text-primary border-huddleroom-primary'
                : 'text-huddleroom-text-muted border-transparent hover:text-huddleroom-text-primary',
            )}
          >
            {t.label}
          </button>
        )
      })}
    </div>
  )
}

interface TabPanelProps {
  tabId: string
  activeId: string
  idPrefix: string
  children: React.ReactNode
  className?: string
}

export function TabPanel({ tabId, activeId, idPrefix, children, className }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      id={`${idPrefix}-panel-${tabId}`}
      aria-labelledby={`${idPrefix}-tab-${tabId}`}
      hidden={tabId !== activeId}
      className={className}
    >
      {children}
    </div>
  )
}
