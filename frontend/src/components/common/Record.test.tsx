import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Record } from './Record'

describe('Record', () => {
  it('does not render rows with null, undefined, or empty-string values', () => {
    const html = renderToStaticMarkup(<Record rows={[
      { key: 'Role', value: 'Reviewer' },
      { key: 'Config', value: null },
      { key: 'CLI runtime', value: undefined },
      { key: 'Notes', value: '' },
    ]} />)
    expect(html).toContain('Role')
    expect(html).toContain('Reviewer')
    expect(html).not.toContain('Config')
    expect(html).not.toContain('CLI runtime')
    expect(html).not.toContain('Notes')
  })

  it('renders nothing when every row is empty', () => {
    const html = renderToStaticMarkup(<Record rows={[{ key: 'Config', value: null }]} />)
    expect(html).toBe('')
  })

  it('applies keyWidth to the key column', () => {
    const html = renderToStaticMarkup(<Record rows={[{ key: 'Role', value: 'Reviewer' }]} keyWidth={84} />)
    expect(html).toContain('width:84px')
  })

  it('defaults keyWidth to 64px', () => {
    const html = renderToStaticMarkup(<Record rows={[{ key: 'Role', value: 'Reviewer' }]} />)
    expect(html).toContain('width:64px')
  })

  it('lets long unbroken keys wrap inside the key column instead of overflowing', () => {
    const html = renderToStaticMarkup(<Record rows={[{ key: 'plan_item:implement_booking_app', value: 'Pending' }]} keyWidth={140} />)
    const dtClass = /<dt class="([^"]*)"/.exec(html)?.[1] ?? ''
    expect(dtClass).toContain('min-w-0')
    expect(dtClass).toContain('[overflow-wrap:anywhere]')
  })

  it('uses the minimum label typography token for record keys', () => {
    const html = renderToStaticMarkup(<Record rows={[{ key: 'Role', value: 'Reviewer' }]} />)
    expect(html).toContain('text-label')
    expect(html).not.toContain('text-[11px]')
  })
})
