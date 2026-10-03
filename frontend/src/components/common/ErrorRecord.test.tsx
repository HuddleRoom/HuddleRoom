import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { ApiError } from '@/lib/api-client'
import { ErrorRecord } from './ErrorRecord'

describe('ErrorRecord', () => {
  it('renders the three record keys and a native Details disclosure from a 401 error', () => {
    const error = new ApiError(401, 'Unauthorized')
    const html = renderToStaticMarkup(<ErrorRecord error={error} />)
    expect(html).toContain('What happened')
    // HTML escapes the apostrophe, so match flexibly
    expect(html).toMatch(/The orchestrator[&#x27;']+s language-model key was rejected\./)
    expect(html).toContain('Why')
    expect(html).toContain('The provider returned an authentication error')
    expect(html).toContain('Do this')
    expect(html).toContain('Update the key in Settings.')
    expect(html).toContain('<details')
    expect(html).toMatch(/<summary[^>]*>Details<\/summary>/)
    // Root has role="alert"
    expect(html).toContain('role="alert"')
  })

  it('uses semantic dl/dt/dd markup', () => {
    const error = new ApiError(401, 'Unauthorized')
    const html = renderToStaticMarkup(<ErrorRecord error={error} />)
    expect(html).toContain('<dl')
    expect(html).toContain('<dt')
    expect(html).toContain('<dd')
  })

  it('distinguishes 404 from 500 errors with spec copy', () => {
    const error404 = new ApiError(404, 'Not Found')
    const html404 = renderToStaticMarkup(<ErrorRecord error={error404} />)
    const error500 = new ApiError(500, 'Internal Server Error')
    const html500 = renderToStaticMarkup(<ErrorRecord error={error500} />)
    // HTML escapes the apostrophe, so match flexibly
    expect(html404).toMatch(/This item doesn[&#x27;']+t exist\./)
    expect(html404).toContain('It may have been deleted, or the link is stale.')
    expect(html404).toContain('Go back to the list.')
    expect(html500).toContain('The orchestrator could not answer.')
    expect(html500).toContain('The server returned an unexpected error.')
  })

  it('renders custom entity in 404 message', () => {
    const error = new ApiError(404, 'Not Found')
    const html = renderToStaticMarkup(<ErrorRecord error={error} entity="goal" />)
    expect(html).toMatch(/This goal doesn[&#x27;']+t exist\./)
    expect(html).toContain('Go back to goals.')
  })

  it('renders a 409 unknown error with message as what', () => {
    const error = new ApiError(409, 'An active goal already exists')
    const html = renderToStaticMarkup(<ErrorRecord error={error} />)
    // For unknown class, message becomes what
    expect(html).toContain('An active goal already exists')
    expect(html).toContain('No further detail is available.')
    expect(html).toContain('Check Details, or try again.')
  })

  it('classifies string errors with auth pattern', () => {
    const error = 'unauthorized key'
    const html = renderToStaticMarkup(<ErrorRecord error={error} />)
    // HTML escapes the apostrophe
    expect(html).toMatch(/The orchestrator[&#x27;']+s language-model key was rejected\./)
  })

  it('renders the action slot under Do this', () => {
    const error = new ApiError(401, 'Unauthorized')
    const action = <button>Update key</button>
    const html = renderToStaticMarkup(<ErrorRecord error={error} action={action} />)
    expect(html).toContain('Update the key in Settings.')
    expect(html).toContain('Update key')
  })

  it('includes HTTP status in details format', () => {
    const error = new ApiError(401, 'Unauthorized')
    const html = renderToStaticMarkup(<ErrorRecord error={error} />)
    // renderToStaticMarkup renders the details content inside <details>,
    // so the HTTP status is in the HTML (in a closed state in browser)
    expect(html).toContain('HTTP 401')
  })

  it('accepts optional className for styling', () => {
    const error = new ApiError(500, 'Test')
    const html = renderToStaticMarkup(<ErrorRecord error={error} className="custom-class" />)
    expect(html).toContain('custom-class')
  })
})
