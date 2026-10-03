import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DetailHeader } from './DetailHeader'

describe('DetailHeader', () => {
  it('renders breadcrumb link, title, status, actions', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <DetailHeader
          backTo="/meetings"
          backLabel="Meetings"
          title="Greetings"
          status="concluded"
          actions={<button>Copy meeting</button>}
        />
      </MemoryRouter>,
    )
    expect(html).toContain('href="/meetings"')
    expect(html).toContain('Meetings')
    expect(html).toContain('Greetings')
    expect(html).toContain('concluded')
    expect(html).toContain('Copy meeting')
  })
  it('omits status badge when status is not provided', () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <DetailHeader backTo="/meetings" backLabel="Meetings" title="Greetings" />
      </MemoryRouter>,
    )
    expect(html).toContain('Greetings')
    expect(html).not.toContain('●')
  })
})
