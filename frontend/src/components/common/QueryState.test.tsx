/**
 * QueryState unit tests
 *
 * Covers the five required branches:
 *   1. loading  – renders skeleton, not children
 *   2. error    – renders errorLabel + Retry button, not children
 *   3. empty    – data is [] → emptyLabel shown, not children
 *   4. data     – children(data) called with the resolved data
 *   5. undefined-data guard – data===undefined, isLoading=false, isError=false
 *                              → children must NOT be called (crash guard)
 */
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { QueryState } from './uiPrimitives'

// ──────────────────────────────────────────────────────────────────────────────
// 1. Loading branch
// ──────────────────────────────────────────────────────────────────────────────
describe('QueryState – loading branch', () => {
  it('renders skeleton markup and not children', () => {
    const children = vi.fn(() => <span>rendered data</span>)

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: true, isError: false, data: undefined }}
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
    expect(markup).not.toContain('rendered data')
    expect(markup).toContain('role="status"')
    expect(markup).toContain('class="skeleton"')
  })

  it('still does not call children even when data is present during loading', () => {
    const children = vi.fn((items: string[]) => <ul>{items.map(i => <li key={i}>{i}</li>)}</ul>)

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: true, isError: false, data: ['alpha', 'beta'] }}
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
    expect(markup).not.toContain('alpha')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 2. Error branch
// ──────────────────────────────────────────────────────────────────────────────
describe('QueryState – error branch', () => {
  it('renders errorLabel with role="alert" and does not call children', () => {
    const children = vi.fn(() => <span>rendered data</span>)

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: true, data: undefined }}
        errorLabel="Could not load items"
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
    expect(markup).not.toContain('rendered data')
    expect(markup).toContain('role="alert"')
    expect(markup).toContain('Could not load items')
  })

  it('renders a Retry button when refetch is provided', () => {
    const refetch = vi.fn()

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: true, data: undefined, refetch }}
        children={() => null}
      />,
    )

    expect(markup).toContain('Retry')
  })

  it('omits the Retry button when no refetch is given', () => {
    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: true, data: undefined }}
        children={() => null}
      />,
    )

    expect(markup).not.toContain('Retry')
  })

  it('error branch wins over data — does not call children even with non-empty data', () => {
    const children = vi.fn(() => <span>rendered data</span>)

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: true, data: ['x', 'y'] }}
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
    expect(markup).not.toContain('rendered data')
    expect(markup).toContain('role="alert"')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 3. Empty branch
// ──────────────────────────────────────────────────────────────────────────────
describe('QueryState – empty branch', () => {
  it('renders emptyLabel when data is an empty array and does not call children', () => {
    const children = vi.fn(() => <span>rendered data</span>)

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: false, data: [] }}
        emptyLabel="Nothing here yet"
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
    expect(markup).not.toContain('rendered data')
    expect(markup).toContain('Nothing here yet')
    expect(markup).toContain('<h3')
  })

  it('uses the default emptyLabel when none is supplied', () => {
    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: false, data: [] }}
        children={() => null}
      />,
    )

    expect(markup).toContain('Nothing here yet')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 4. Data branch
// ──────────────────────────────────────────────────────────────────────────────
describe('QueryState – data branch', () => {
  it('calls children with the resolved data and renders the result', () => {
    const data = ['alpha', 'beta']
    const children = vi.fn((items: string[]) => (
      <ul>
        {items.map(i => (
          <li key={i}>{i}</li>
        ))}
      </ul>
    ))

    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: false, data }}
        children={children}
      />,
    )

    expect(children).toHaveBeenCalledOnce()
    expect(children).toHaveBeenCalledWith(data)
    expect(markup).toContain('alpha')
    expect(markup).toContain('beta')
    expect(markup).not.toContain('Nothing here yet')
  })
})

// ──────────────────────────────────────────────────────────────────────────────
// 5. Undefined-data guard (critical crash-prevention contract)
// ──────────────────────────────────────────────────────────────────────────────
describe('QueryState – undefined-data guard', () => {
  /**
   * When query.data is undefined but the query is not loading and has no error,
   * calling children(undefined) would be a runtime crash (T is non-undefined in
   * the callers' type signatures). The component must NOT invoke children.
   */
  it('does NOT call children when data is undefined, isLoading=false, isError=false', () => {
    const children = vi.fn(() => <span>should not appear</span>)

    renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: false, data: undefined }}
        children={children}
      />,
    )

    expect(children).not.toHaveBeenCalled()
  })

  it('shows emptyLabel (not a crash) when data is undefined', () => {
    const markup = renderToStaticMarkup(
      <QueryState
        query={{ isLoading: false, isError: false, data: undefined }}
        emptyLabel="No data available"
        children={() => <span>should not appear</span>}
      />,
    )

    expect(markup).toContain('No data available')
    expect(markup).not.toContain('should not appear')
    expect(markup).toContain('<h3')
  })
})
