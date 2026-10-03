import React, { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { descendants, findElement, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'

// The custom DOM harness runs under vitest's node environment, so
// @radix-ui/react-use-layout-effect's SSR-safety check resolves to a no-op at
// module import time and Radix's real Portal never mounts here (see
// ConversationPanel.test.tsx for the same finding). Mock Radix's primitives
// the same way every page test that adopted the Dialog does: Root mounts
// Content only while `open`, and Content fires onOpenAutoFocus once per
// mount, mirroring Radix's real open-focus behavior closely enough to
// exercise Dialog's own initialFocusRef wiring.
const rootProps = vi.hoisted(() => ({ onOpenChange: undefined as ((open: boolean) => void) | undefined }))

vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ open, onOpenChange, children }: { open?: boolean; onOpenChange?: (open: boolean) => void; children: React.ReactNode }) => {
    rootProps.onOpenChange = onOpenChange
    return open ? <>{children}</> : null
  },
  Portal: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Overlay: (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />,
  Content: ({ children, onOpenAutoFocus, onCloseAutoFocus, ...props }: React.HTMLAttributes<HTMLDivElement> & { onOpenAutoFocus?: (event: Event) => void; onCloseAutoFocus?: (event: Event) => void }) => {
    // eslint-disable-next-line react-hooks/exhaustive-deps
    React.useEffect(() => { onOpenAutoFocus?.(new Event('focus') as Event) }, [])
    return <div {...props}>{children}</div>
  },
  Title: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => <h2 {...props}>{children}</h2>,
  Description: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props}>{children}</p>,
  Close: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

import { Dialog } from './Dialog'

describe('Dialog', () => {
  it('renders title + description with resolvable aria ids', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog open onOpenChange={vi.fn()} title="New project" description="Create a new project.">
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      const content = descendants(view.container).find((node) => node.getAttribute('role') === 'dialog')
      expect(content).toBeDefined()
      const labelledBy = content?.getAttribute('aria-labelledby')
      const describedBy = content?.getAttribute('aria-describedby')
      expect(labelledBy).toBeTruthy()
      expect(describedBy).toBeTruthy()
      const titleNode = findElement(view.container, 'h2', ['id', labelledBy!])
      const descNode = findElement(view.container, 'p', ['id', describedBy!])
      expect(titleNode && textOf(titleNode)).toBe('New project')
      expect(descNode && textOf(descNode)).toBe('Create a new project.')
    } finally { view.cleanup() }
  })

  it('falls back to a screen-reader-only description built from the title when none is given', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog open onOpenChange={vi.fn()} title="Delete agent">
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      expect(textOf(view.container)).toContain('Dialog: Delete agent')
    } finally { view.cleanup() }
  })

  it.each([
    ['sm', 400],
    ['md', 520],
    ['lg', 720],
  ] as const)('maps size=%s to a %spx width', async (size, width) => {
    const view = await mountWithTestDom(() => (
      <Dialog open onOpenChange={vi.fn()} title="Sized" size={size}>
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      const content = descendants(view.container).find((node) => node.getAttribute('role') === 'dialog')
      expect(content?.style.width).toBe(`${width}px`)
    } finally { view.cleanup() }
  })

  it('defaults to the md size when none is given', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog open onOpenChange={vi.fn()} title="Default size">
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      const content = descendants(view.container).find((node) => node.getAttribute('role') === 'dialog')
      expect(content?.style.width).toBe('520px')
    } finally { view.cleanup() }
  })

  it('disables the primary footer button via primaryDisabled', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog
        open
        onOpenChange={vi.fn()}
        title="Form"
        footer={{ primaryLabel: 'Save', onPrimary: vi.fn(), primaryDisabled: true }}
      >
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      expect(getButton(view.container, 'Save').getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.container, 'Cancel').getAttribute('disabled')).toBeNull()
    } finally { view.cleanup() }
  })

  it('shows a pending label and disables both footer buttons while isPending', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog
        open
        onOpenChange={vi.fn()}
        title="Form"
        footer={{ primaryLabel: 'Save', onPrimary: vi.fn(), isPending: true }}
      >
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      expect(getButton(view.container, 'Working…').getAttribute('disabled')).not.toBeNull()
      expect(getButton(view.container, 'Cancel').getAttribute('disabled')).not.toBeNull()
    } finally { view.cleanup() }
  })

  it('blocks onOpenChange(false) while a footer mutation is pending', async () => {
    const onOpenChange = vi.fn()
    const view = await mountWithTestDom(() => (
      <Dialog
        open
        onOpenChange={onOpenChange}
        title="Form"
        footer={{ primaryLabel: 'Save', onPrimary: vi.fn(), isPending: true }}
      >
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      // Dialog wraps the onOpenChange it hands to Radix's Root so a close
      // triggered while a footer mutation is pending is swallowed.
      rootProps.onOpenChange?.(false)
      expect(onOpenChange).not.toHaveBeenCalled()
    } finally { view.cleanup() }
  })

  it('renders a corner Close control with an accessible name', async () => {
    const view = await mountWithTestDom(() => (
      <Dialog open onOpenChange={vi.fn()} title="Form">
        <p>Body</p>
      </Dialog>
    ), act)
    try {
      expect(getButton(view.container, '✕').getAttribute('aria-label')).toBe('Close')
    } finally { view.cleanup() }
  })

  it('focuses the initialFocusRef element on open', async () => {
    function Fixture() {
      const ref = React.useRef<HTMLInputElement>(null)
      return (
        <Dialog open onOpenChange={vi.fn()} title="Form" initialFocusRef={ref}>
          <input ref={ref} aria-label="First field" />
        </Dialog>
      )
    }
    const view = await mountWithTestDom(() => <Fixture />, act)
    try {
      expect(view.activeElement?.getAttribute('aria-label')).toBe('First field')
    } finally { view.cleanup() }
  })
})
