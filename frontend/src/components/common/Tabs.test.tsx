import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Tabs, TabPanel } from './Tabs'

const tabs = [
  { id: 'sessions', label: 'Sessions' },
  { id: 'tasks', label: 'Tasks' },
]

describe('Tabs', () => {
  it('renders WAI-ARIA tablist with roving tabindex', () => {
    const html = renderToStaticMarkup(
      <Tabs tabs={tabs} activeId="sessions" onChange={() => {}} idPrefix="agent" />,
    )
    expect(html).toContain('role="tablist"')
    expect(html).toContain('role="tab"')
    expect(html).toContain('aria-selected="true"')
    expect(html).toContain('tabindex="-1"')
    expect(html).toContain('id="agent-tab-sessions"')
    expect(html).toContain('aria-controls="agent-panel-sessions"')
  })
  it('keeps inactive panels mounted but hidden', () => {
    const html = renderToStaticMarkup(
      <TabPanel tabId="tasks" activeId="sessions" idPrefix="agent">content</TabPanel>,
    )
    expect(html).toContain('hidden')
    expect(html).toContain('role="tabpanel"')
    expect(html).toContain('content')
  })
})
