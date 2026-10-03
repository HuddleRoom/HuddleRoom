import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Panel } from './Panel'

describe('Panel', () => {
  it('renders border and shadow classes', () => {
    const html = renderToStaticMarkup(<Panel>Content</Panel>)
    expect(html).toContain('border-huddleroom-border')
    expect(html).toContain('shadow-[0_1px_2px_rgba(31,34,38,.06),0_12px_32px_rgba(31,34,38,.06)]')
    expect(html).toContain('rounded-lg')
    expect(html).toContain('bg-white')
  })

  it('renders header row only when header prop is provided', () => {
    const html = renderToStaticMarkup(<Panel header={{ title: 'Test Title' }}>Body</Panel>)
    expect(html).toContain('Test Title')
    expect(html).toContain('bg-huddleroom-bg-2')
    expect(html).toContain('border-b')

    const htmlNoHeader = renderToStaticMarkup(<Panel>Body</Panel>)
    expect(htmlNoHeader).not.toContain('bg-huddleroom-bg-2')
  })

  it('renders keyText when provided in header', () => {
    const html = renderToStaticMarkup(
      <Panel header={{ keyText: 'KEY', title: 'Title' }}>Body</Panel>,
    )
    expect(html).toContain('KEY')
    expect(html).toContain('Title')
  })

  it('renders children always', () => {
    const html = renderToStaticMarkup(<Panel>Test Content</Panel>)
    expect(html).toContain('Test Content')

    const html2 = renderToStaticMarkup(
      <Panel header={{ title: 'Header' }}>Another Content</Panel>,
    )
    expect(html2).toContain('Another Content')
  })

  it('applies body padding p-4', () => {
    const html = renderToStaticMarkup(<Panel>Body</Panel>)
    expect(html).toContain('p-4')
  })

  it('merges custom className onto section', () => {
    const html = renderToStaticMarkup(<Panel className="w-full">Content</Panel>)
    expect(html).toContain('w-full')
    expect(html).toContain('rounded-lg')
  })
})
