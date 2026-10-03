import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { StatTile } from './StatTile'

describe('StatTile', () => {
  it('neutral when zero even with color', () => {
    const html = renderToStaticMarkup(<StatTile label="Blocked" value={0} color="#7E22CE" />)
    expect(html).toContain('#5a6270')
    expect(html).not.toContain('#7E22CE')
  })
  it('status color when count > 0', () => {
    const html = renderToStaticMarkup(<StatTile label="Blocked" value={3} color="#7E22CE" />)
    expect(html).toContain('#7E22CE')
  })
  it('neutral when no color even with count > 0', () => {
    const html = renderToStaticMarkup(<StatTile label="Blocked" value={3} />)
    expect(html).toContain('#5a6270')
  })
  it('renders as Link when to is provided', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <StatTile label="Blocked" value={3} to="/goals?filter=blocked" />
      </MemoryRouter>,
    )
    expect(html).toContain('href="/goals?filter=blocked"')
  })
  it('renders as plain div when to is omitted', () => {
    const html = renderToStaticMarkup(<StatTile label="Blocked" value={3} />)
    expect(html).not.toContain('<a ')
  })
  it('renders em-dash — never a fabricated zero or a color — when value is undefined', () => {
    const html = renderToStaticMarkup(<StatTile label="Blocked" value={undefined} color="#7E22CE" />)
    expect(html).toContain('—')
    expect(html).toContain('#5a6270')
    expect(html).not.toContain('#7E22CE')
  })
})
