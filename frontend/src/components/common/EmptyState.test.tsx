import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { EmptyState } from './EmptyState'

describe('EmptyState', () => {
  it('renders title, body, CTA', () => {
    const html = renderToStaticMarkup(
      <EmptyState title="No meetings yet" body="Meetings appear here once scheduled or convened."
        action={{ label: 'New meeting', onClick: () => {} }} />,
    )
    expect(html).toContain('No meetings yet')
    expect(html).toContain('Meetings appear here')
    expect(html).toContain('New meeting')
  })
  it('renders without CTA', () => {
    const html = renderToStaticMarkup(<EmptyState title="T" body="B" />)
    expect(html).not.toContain('<button')
  })
})
