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
})
