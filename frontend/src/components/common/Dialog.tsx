import React from 'react'
import * as RadixDialog from '@radix-ui/react-dialog'
import { UI_COLORS, Z, Button } from './uiPrimitives'

const SIZE_WIDTH: Record<'sm' | 'md' | 'lg', number> = { sm: 400, md: 520, lg: 720 }

const SR_ONLY_STYLE: React.CSSProperties = {
  position: 'absolute',
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: 'hidden',
  clip: 'rect(0,0,0,0)',
  whiteSpace: 'nowrap',
  border: 0,
}

export interface DialogProps<T extends HTMLElement = HTMLElement> {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description?: string
  size?: 'sm' | 'md' | 'lg'
  role?: 'dialog' | 'alertdialog'
  children: React.ReactNode
  footer?: {
    cancelLabel?: string
    primaryLabel: string
    onPrimary?: () => void
    primaryVariant?: 'primary' | 'danger'
    isPending?: boolean
    primaryDisabled?: boolean
    primaryType?: 'button' | 'submit'
    formId?: string
  }
  initialFocusRef?: React.RefObject<T | null>
  onCloseAutoFocus?: React.ComponentPropsWithoutRef<typeof RadixDialog.Content>['onCloseAutoFocus']
  contentRef?: React.Ref<HTMLDivElement>
}

export function Dialog<T extends HTMLElement = HTMLElement>({
  open,
  onOpenChange,
  title,
  description,
  size = 'md',
  role = 'dialog',
  children,
  footer,
  initialFocusRef,
  onCloseAutoFocus,
  contentRef,
}: DialogProps<T>) {
  const id = React.useId()
  const titleId = `${id}-title`
  const descId = `${id}-desc`

  return (
    <RadixDialog.Root
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && footer?.isPending) return
        onOpenChange(nextOpen)
      }}
    >
      <RadixDialog.Portal>
        <RadixDialog.Overlay
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.7)',
            zIndex: Z.modalBackdrop,
          }}
        />
        <RadixDialog.Content
          ref={contentRef}
          role={role}
          aria-labelledby={titleId}
          aria-describedby={descId}
          onOpenAutoFocus={
            initialFocusRef
              ? (e) => {
                  e.preventDefault()
                  initialFocusRef.current?.focus()
                }
              : undefined
          }
          onCloseAutoFocus={onCloseAutoFocus}
          style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%,-50%)',
            zIndex: Z.modal,
            background: UI_COLORS.surface,
            border: `1px solid ${UI_COLORS.border}`,
            // radius 8: dialog is a control-scale transient surface, not a page Panel (which is 12) — do not "fix" to 12
            borderRadius: 8,
            width: SIZE_WIDTH[size],
            maxWidth: 'calc(100vw - 32px)',
            maxHeight: 'calc(100vh - 60px)',
            display: 'flex',
            flexDirection: 'column',
            padding: 24,
          }}
        >
          <div style={{ flexShrink: 0, position: 'relative' }}>
            <RadixDialog.Title
              id={titleId}
              style={{ fontSize: 15, fontWeight: 600, color: UI_COLORS.textPrimary, margin: '0 0 8px' }}
            >
              {title}
            </RadixDialog.Title>
            {description ? (
              <RadixDialog.Description
                id={descId}
                style={{ fontSize: 13, color: UI_COLORS.textSecondary, lineHeight: 1.5, margin: '0 0 16px' }}
              >
                {description}
              </RadixDialog.Description>
            ) : (
              <RadixDialog.Description id={descId} style={SR_ONLY_STYLE}>
                {`Dialog: ${title}`}
              </RadixDialog.Description>
            )}
            <RadixDialog.Close asChild>
              <button
                type="button"
                aria-label="Close"
                style={{
                  position: 'absolute',
                  top: 0,
                  right: 0,
                  background: 'transparent',
                  border: 0,
                  cursor: 'pointer',
                  color: UI_COLORS.textSecondary,
                  fontSize: 16,
                  lineHeight: 1,
                  padding: 4,
                }}
              >
                ✕
              </button>
            </RadixDialog.Close>
          </div>

          <div style={{ flex: 1, overflowY: 'auto' }}>{children}</div>

          {footer && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 24, flexShrink: 0 }}>
              <RadixDialog.Close asChild>
                <Button variant="secondary" size="sm" className="min-h-11" disabled={footer.isPending}>
                  {footer.cancelLabel ?? 'Cancel'}
                </Button>
              </RadixDialog.Close>
              <Button
                type={footer.primaryType === 'submit' ? 'submit' : 'button'}
                form={footer.primaryType === 'submit' ? footer.formId : undefined}
                variant={footer.primaryVariant ?? 'primary'}
                size="sm"
                className="min-h-11"
                disabled={footer.isPending || footer.primaryDisabled}
                onClick={footer.primaryType === 'submit' ? undefined : footer.onPrimary}
              >
                {footer.isPending ? 'Working…' : footer.primaryLabel}
              </Button>
            </div>
          )}
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}
