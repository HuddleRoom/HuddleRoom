import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { Banner } from './Banner'

describe('Banner component', () => {
  describe('variant rendering', () => {
    it('renders info variant with title and message', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="info" title="Info Title" message="Info message content" />,
      )
      expect(markup).toContain('Info Title')
      expect(markup).toContain('Info message content')
      expect(markup).toContain('#EFF6FF') // info bg
      expect(markup).toContain('#0369A1') // info border
    })

    it('renders warning variant with title and message', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="warning" title="Warning Title" message="Warning message content" />,
      )
      expect(markup).toContain('Warning Title')
      expect(markup).toContain('Warning message content')
      expect(markup).toContain('#F5F6F8') // warning bg
      expect(markup).toContain('#B45309') // warning border
    })

    it('renders offline variant with title and message', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="offline" title="Offline Title" message="Offline message content" />,
      )
      expect(markup).toContain('Offline Title')
      expect(markup).toContain('Offline message content')
      expect(markup).toContain('#F5F6F8') // offline bg
      expect(markup).toContain('#B45309') // offline border
    })
  })

  describe('action button', () => {
    it('renders action button when action is provided', () => {
      const handleClick = vi.fn()
      const markup = renderToStaticMarkup(
        <Banner
          variant="info"
          title="Title"
          message="Message"
          action={{ label: 'Click me', onClick: handleClick }}
        />,
      )
      expect(markup).toContain('Click me')
    })

    it('does not render action button when action is omitted', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="info" title="Title" message="Message" />,
      )
      expect(markup).not.toContain('<button')
    })

    it('calls onClick when action button is clicked', () => {
      const handleClick = vi.fn()
      const element = Banner({
        variant: 'info',
        title: 'Title',
        message: 'Message',
        action: { label: 'Action', onClick: handleClick },
      }) as React.ReactElement

      // Navigate to the button element in the tree
      const div = element.props.children as React.ReactElement[]
      const buttonContainer = div[1] as React.ReactElement
      const button = buttonContainer.props.children as React.ReactElement

      expect(button.props.onClick).toBeDefined()
      button.props.onClick()
      expect(handleClick).toHaveBeenCalledTimes(1)
    })
  })

  describe('styling', () => {
    it('applies custom className when provided', () => {
      const markup = renderToStaticMarkup(
        <Banner
          variant="info"
          title="Title"
          message="Message"
          className="custom-class"
        />,
      )
      expect(markup).toContain('custom-class')
    })

    it('renders title with bold font weight', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="info" title="Bold Title" message="Message" />,
      )
      expect(markup).toContain('font-weight:600')
    })

    it('renders with proper flex layout for title and message', () => {
      const markup = renderToStaticMarkup(
        <Banner variant="info" title="Title" message="Message" />,
      )
      expect(markup).toContain('flex')
      expect(markup).toContain('flex-col')
    })
  })
})
