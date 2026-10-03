import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Tag } from './Tag'

describe('Tag', () => {
  it('renders quiet depth-bg tag', () => {
    const html = renderToStaticMarkup(<Tag>interaction design</Tag>)
    expect(html).toContain('bg-huddleroom-depth')
    expect(html).toContain('text-huddleroom-text-secondary')
    expect(html).toContain('rounded-[3px]')
    expect(html).not.toContain('font-mono')
  })
  it('mono variant for system vocabulary', () => {
    const html = renderToStaticMarkup(<Tag mono>task.created</Tag>)
    expect(html).toContain('font-mono')
  })
})
