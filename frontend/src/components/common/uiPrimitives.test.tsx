import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { Button, QueryState, SectionLabel, Select, StatusBadge, StructuralLabel, type QueryStateProps } from './uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'

// Helper: call QueryState as a plain function to inspect its React element output
// without needing a DOM environment.
function renderQueryState<T>(props: QueryStateProps<T>): React.ReactElement {
  return (QueryState as (p: QueryStateProps<T>) => React.ReactElement)(props)
}

describe('Button focus visibility', () => {
  it.each(['primary', 'secondary', 'danger', 'ghost'] as const)('uses a primary 2px focus-visible outline for %s buttons', (variant) => {
    const markup = renderToStaticMarkup(<Button variant={variant}>Action</Button>)
    expect(markup).toContain('focus-visible:outline-2')
    expect(markup).toContain('focus-visible:outline-[#1a5cff]')
    expect(markup).toContain('focus-visible:outline-offset-2')
    expect(markup).toContain('focus:outline-none')
  })
})

describe('QueryState', () => {
  // ---------------------------------------------------------------------------
  // Loading branch
  // ---------------------------------------------------------------------------
  describe('loading branch', () => {
    it('renders role="status" with aria-busy="true"', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: true, isError: false, data: undefined }}
          children={() => null}
        />,
      )
      expect(markup).toContain('role="status"')
      expect(markup).toContain('aria-busy="true"')
    })

    it('renders skeleton rows (default skeleton type)', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: true, isError: false, data: undefined }}
          skeletonCount={3}
          children={() => null}
        />,
      )
      // SkeletonRow emits elements with className="skeleton"
      const skeletonOccurrences = markup.split('class="skeleton"').length - 1
      expect(skeletonOccurrences).toBeGreaterThanOrEqual(3)
    })

    it('renders skeleton cards when skeleton="cards"', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: true, isError: false, data: undefined }}
          skeleton="cards"
          skeletonCount={2}
          children={() => null}
        />,
      )
      expect(markup).toContain('role="status"')
      const skeletonOccurrences = markup.split('class="skeleton"').length - 1
      expect(skeletonOccurrences).toBeGreaterThanOrEqual(2)
    })

    it('does not call children during loading', () => {
      const children = vi.fn(() => <span>data rendered</span>)
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: true, isError: false, data: ['item'] }}
          children={children}
        />,
      )
      expect(children).not.toHaveBeenCalled()
      expect(markup).not.toContain('data rendered')
    })
  })

  // ---------------------------------------------------------------------------
  // Error branch
  // ---------------------------------------------------------------------------
  describe('error branch', () => {
    it('renders role="alert" with the error label', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: true, data: undefined }}
          errorLabel="Failed to fetch items"
          children={() => null}
        />,
      )
      expect(markup).toContain('role="alert"')
      expect(markup).toContain('Failed to fetch items')
    })

    it('uses the default error label when errorLabel is omitted', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: true, data: undefined }}
          children={() => null}
        />,
      )
      expect(markup).toContain('Failed to load data')
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

    it('omits the Retry button when refetch is not provided', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: true, data: undefined }}
          children={() => null}
        />,
      )
      expect(markup).not.toContain('Retry')
    })

    it('calls refetch when the Retry onClick fires', () => {
      const refetch = vi.fn()
      const element = renderQueryState({
        query: { isLoading: false, isError: true, data: undefined as never, refetch },
        children: (_: never) => null,
      })

      // element = <div role="alert"> [span, Button] </div>
      const children = React.Children.toArray(element.props.children)
      const buttonEl = children.find(
        (c): c is React.ReactElement =>
          React.isValidElement(c) &&
          typeof (c as React.ReactElement).props.onClick === 'function',
      ) as React.ReactElement

      expect(buttonEl).toBeDefined()
      expect(buttonEl.props.className).toContain('min-h-11')
      buttonEl.props.onClick()
      expect(refetch).toHaveBeenCalledTimes(1)
    })

    it('does not render loading or data when in error state', () => {
      const children = vi.fn(() => <span>data rendered</span>)
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: true, data: ['item'] }}
          children={children}
        />,
      )
      expect(markup).not.toContain('role="status"')
      expect(children).not.toHaveBeenCalled()
      expect(markup).not.toContain('data rendered')
    })
  })

  // ---------------------------------------------------------------------------
  // Empty branch
  // ---------------------------------------------------------------------------
  describe('empty branch', () => {
    it('shows emptyLabel when data is an empty array (default isEmpty)', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: [] }}
          emptyLabel="No items yet"
          children={() => <span>has data</span>}
        />,
      )
      expect(markup).toContain('No items yet')
      expect(markup).not.toContain('has data')
    })

    it('shows emptyLabel when custom isEmpty returns true', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: { items: [] as string[] } }}
          emptyLabel="Nothing to show"
          isEmpty={(d) => d.items.length === 0}
          children={() => <span>has data</span>}
        />,
      )
      expect(markup).toContain('Nothing to show')
      expect(markup).not.toContain('has data')
    })

    it('renders the empty branch when data is undefined (no crash)', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: undefined }}
          emptyLabel="Nothing available"
          children={() => <span>has data</span>}
        />,
      )
      expect(markup).toContain('Nothing available')
      expect(markup).not.toContain('has data')
    })

    it('uses the default emptyLabel when omitted', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: [] }}
          children={() => null}
        />,
      )
      expect(markup).toContain('Nothing here yet')
    })

    it('shows optional emptyDetail copy when provided', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: [] }}
          emptyLabel="No items"
          emptyDetail="Create one to get started"
          children={() => null}
        />,
      )
      expect(markup).toContain('No items')
      expect(markup).toContain('Create one to get started')
    })

    it('does not show emptyDetail when it is omitted', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: [] }}
          emptyLabel="No items"
          children={() => null}
        />,
      )
      // There should be no extra explanation text
      expect(markup).not.toContain('Create one to get started')
    })
  })

  // ---------------------------------------------------------------------------
  // Data branch
  // ---------------------------------------------------------------------------
  describe('data branch', () => {
    it('renders children with the data when array is non-empty', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: ['alpha', 'beta'] }}
          children={(items) => (
            <ul>
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
        />,
      )
      expect(markup).toContain('alpha')
      expect(markup).toContain('beta')
      expect(markup).not.toContain('Nothing here yet')
    })

    it('does not render the empty state when custom isEmpty returns false', () => {
      const markup = renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: { items: ['x'] } }}
          emptyLabel="Nothing here"
          isEmpty={(d) => d.items.length === 0}
          children={(d) => <span>{d.items[0]}</span>}
        />,
      )
      expect(markup).not.toContain('Nothing here')
      expect(markup).toContain('>x<')
    })

    it('passes the typed data to children so callers can use it without narrowing', () => {
      // Verify TypeScript narrowing: children receives T, not T | undefined
      const received: string[][] = []
      renderToStaticMarkup(
        <QueryState
          query={{ isLoading: false, isError: false, data: ['a', 'b', 'c'] }}
          children={(items) => {
            received.push(items)
            return null
          }}
        />,
      )
      expect(received).toHaveLength(1)
      expect(received[0]).toEqual(['a', 'b', 'c'])
    })
  })
})

describe('StatusBadge orchestration states', () => {
  it('renders raw orchestration vocabulary with canonical semantic colors', () => {
    const markup = renderToStaticMarkup(
      <div>
        <StatusBadge status="accepted" />
        <StatusBadge status="rejected" />
        <StatusBadge status="open" />
        <StatusBadge status="in_progress" />
      </div>,
    )

    expect(markup).toContain('accepted')
    expect(markup).toContain('rejected')
    expect(markup).toContain('open')
    expect(markup).toContain('in_progress')
    expect(markup).not.toContain('in progress')
    expect(markup).toContain(STATUS_COLORS.green)
    expect(markup).toContain(STATUS_COLORS.red)
    expect(markup).toContain(STATUS_COLORS.amber)
  })
})

describe('StatusBadge dot', () => {
  it('renders an aria-hidden dot by default', () => {
    const html = renderToStaticMarkup(<StatusBadge status="active" />)
    expect(html).toContain('aria-hidden')
    expect(html).toContain('●')
  })
  it('omits the dot when dot={false}', () => {
    const html = renderToStaticMarkup(<StatusBadge status="active" dot={false} />)
    expect(html).not.toContain('●')
  })
})

describe('Select', () => {
  it('suppresses the native arrow and renders a custom chevron', () => {
    const html = renderToStaticMarkup(
      <Select aria-label="Test"><option value="">All</option></Select>,
    )
    expect(html).toContain('appearance-none')
    expect(html).toContain('pr-8') // room for chevron
    expect(html).toContain('data-select-chevron')
  })
})

describe('labels', () => {
  it('SectionLabel is slate, not teal', () => {
    const html = renderToStaticMarkup(<SectionLabel>Description</SectionLabel>)
    expect(html).toContain('text-[#33373d]')
    expect(html).not.toContain('#0F766E')
  })
  it('StructuralLabel is muted uppercase tracked, not teal', () => {
    const html = renderToStaticMarkup(<StructuralLabel>Agenda</StructuralLabel>)
    expect(html).toContain('text-[#5d6470]')
    expect(html).not.toContain('#0F766E')
    expect(html).toContain('uppercase')
    expect(html).toContain('tracking-wide')
  })
})
