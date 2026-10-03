import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { LedgerRow } from './LedgerRow'

describe('LedgerRow', () => {
  // Date-group headers are the consumer's responsibility.
  // This component renders only a single summary row.

  it('renders time, action, and detail columns', () => {
    const html = renderToStaticMarkup(
      <LedgerRow
        iso="2026-09-16T14:27:00Z"
        action="Deploy"
        detail="Production cluster update"
      />,
    )
    // Time format depends on user timezone, just verify a time was rendered
    expect(html).toMatch(/\d{1,2}:\d{2}\s(?:AM|PM)/)
    expect(html).toContain('Deploy')
    expect(html).toContain('Production cluster update')
  })

  it('renders children below the summary line', () => {
    const html = renderToStaticMarkup(
      <LedgerRow
        iso="2026-09-16T14:27:00Z"
        action="Deploy"
        detail="Production cluster update"
      >
        <span>Additional info</span>
      </LedgerRow>,
    )
    expect(html).toContain('Additional info')
  })

  it('does not render children block when children is undefined', () => {
    const html = renderToStaticMarkup(
      <LedgerRow
        iso="2026-09-16T14:27:00Z"
        action="Deploy"
        detail="Production cluster update"
      />,
    )
    expect(html).not.toContain('mt-1')
  })

  it('applies custom className to root', () => {
    const html = renderToStaticMarkup(
      <LedgerRow
        iso="2026-09-16T14:27:00Z"
        action="Deploy"
        detail="Production cluster update"
        className="custom-class"
      />,
    )
    expect(html).toContain('custom-class')
  })
})
