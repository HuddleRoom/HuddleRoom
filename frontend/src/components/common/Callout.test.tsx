import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Callout } from './Callout'

describe('Callout', () => {
  it('warning uses warning tokens, sans-serif, status role', () => {
    const html = renderToStaticMarkup(<Callout variant="warning">Hook execution is not yet active.</Callout>)
    expect(html).toContain('bg-huddleroom-warning-bg')
    expect(html).toContain('border-huddleroom-warning-border')
    expect(html).toContain('text-huddleroom-warning-text')
    expect(html).toContain('role="status"')
    expect(html).not.toContain('font-mono')
  })
  it('info uses info border', () => {
    const html = renderToStaticMarkup(<Callout variant="info">FYI</Callout>)
    expect(html).toContain('border-huddleroom-info-border')
  })
})
