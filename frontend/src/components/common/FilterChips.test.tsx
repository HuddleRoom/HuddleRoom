import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { FilterChips } from './FilterChips'

describe('FilterChips', () => {
  it('radiogroup with checked chip filled', () => {
    const html = renderToStaticMarkup(
      <FilterChips
        ariaLabel="Meeting status"
        activeId="all"
        options={[
          { id: 'all', label: 'All' },
          { id: 'active', label: 'Active' },
        ]}
        onChange={() => {}}
      />,
    )
    expect(html).toContain('role="radiogroup"')
    expect(html).toContain('aria-label="Meeting status"')
    expect(html).toContain('aria-checked="true"')
    expect(html).toContain('aria-checked="false"')
    expect(html).toContain('bg-huddleroom-primary')
  })
  it('renders one radio per option', () => {
    const html = renderToStaticMarkup(
      <FilterChips
        ariaLabel="Meeting status"
        activeId="all"
        options={[
          { id: 'all', label: 'All' },
          { id: 'active', label: 'Active' },
          { id: 'done', label: 'Done' },
        ]}
        onChange={() => {}}
      />,
    )
    expect(html.match(/role="radio"/g)?.length).toBe(3)
  })
  it('multi mode renders group with pressable chips', () => {
    const html = renderToStaticMarkup(
      <FilterChips
        ariaLabel="Meeting status"
        activeId="all"
        onChange={() => {}}
        multi
        activeIds={new Set(['all', 'active'])}
        onToggle={() => {}}
        options={[
          { id: 'all', label: 'All' },
          { id: 'active', label: 'Active' },
          { id: 'done', label: 'Done' },
        ]}
      />,
    )
    expect(html).toContain('role="group"')
    expect(html.match(/aria-pressed="true"/g)?.length).toBe(2)
  })
})
