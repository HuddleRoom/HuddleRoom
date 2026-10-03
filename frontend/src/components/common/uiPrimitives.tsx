import React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import * as Dialog from '@radix-ui/react-dialog'
import { STATUS_COLORS } from '@/lib/statusColors'
import { EmptyState } from './EmptyState'
import { Dialog as ModalDialog } from './Dialog'

export const UI_FONT_FAMILY = "'Inter', system-ui, sans-serif"

export const UI_COLORS = {
  primary: '#1a5cff',
  primaryHover: '#1D4ED8',
  primaryForeground: '#FFFFFF',
  surface: '#FFFFFF',
  depth: '#F1F5F9',
  border: '#E2E8F0',
  borderStrong: '#CBD5E1',
  textPrimary: '#1f2226',
  textSecondary: '#33373d',
  textMuted: '#5d6470',
  danger: '#DC2626',
  overlay: 'rgba(0, 0, 0, 0.7)',
  // Sidebar stays on its own dark palette, independent of the page-ground ink/muted tokens above.
  sidebarBg: '#0F172A',
  sidebarSurface: '#1E293B',
  sidebarText: '#CBD5E1',
  sidebarMuted: '#64748B',
  sidebarActive: '#60A5FA',
  sidebarActiveBg: 'rgba(26, 92, 255, 0.2)',
  sidebarHoverText: '#F8FAFC',
  sidebarHoverBorder: '#94A3B8',
  sidebarGroupLabel: '#94A3B8',
  appBg: '#FFFFFF',
  bg2: '#F5F6F8',
  surfaceBlue: '#EFF6FF',
} as const

// Z-index scale (mirrors --z-* custom properties in index.css)
export const Z = { dropdown: 10, sticky: 20, modalBackdrop: 40, modal: 50, toast: 60, tooltip: 70 } as const

export function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label className="block text-[#33373d] text-xs font-medium mb-1">{label}</label>
      {children}
    </div>
  )
}

// ============================================================================
// NEW COMPONENT EXPORTS (Phase 1)
// ============================================================================

/**
 * cn utility: merges classNames and resolves Tailwind conflicts
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

/**
 * Button component with variants
 * Variants: primary | secondary | danger | ghost
 * Sizes: sm | md | lg
 */
const buttonVariants = cva(
  'inline-flex items-center justify-center gap-1.5 font-medium transition-colors duration-120 rounded-md focus:outline-none focus-visible:outline-2 focus-visible:outline-[#1a5cff] focus-visible:outline-offset-2',
  {
    variants: {
      variant: {
        primary: cn(
          'bg-[#1D4ED8] text-white enabled:hover:bg-[#1E40AF]',
          'disabled:bg-[#E2E8F0] disabled:text-[#33373d] disabled:cursor-not-allowed',
        ),
        secondary: cn(
          'bg-white text-[#1f2226] border border-[#CBD5E1]',
          'hover:bg-[#F1F5F9] hover:border-[#CBD5E1]',
          'disabled:opacity-50 disabled:cursor-not-allowed',
        ),
        danger: cn(
          'bg-white text-[#DC2626] border border-[#E2E8F0]',
          'hover:bg-[#FEF2F2] hover:border-[#DC2626]',
          'disabled:opacity-50 disabled:cursor-not-allowed',
        ),
        ghost: cn(
          'bg-transparent text-[#33373d] border border-[#E2E8F0]',
          'hover:border-[#1a5cff] hover:text-[#1f2226]',
          'disabled:opacity-50 disabled:cursor-not-allowed',
        ),
      },
      size: {
        sm: 'px-2.5 py-1 text-xs',
        md: 'px-3.5 py-2 text-sm',
        lg: 'px-4 py-2.5 text-sm min-h-11',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  ),
)
Button.displayName = 'Button'

/**
 * Card component
 * interactive?: boolean — adds hover border/shadow effect
 * padding?: 'sm' | 'md' | 'lg'
 */
interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  interactive?: boolean
  padding?: 'sm' | 'md' | 'lg'
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, interactive, padding = 'md', ...props }, ref) => {
    const paddingMap = {
      sm: 'p-3',
      md: 'p-4',
      lg: 'p-5',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'bg-white border border-[#E2E8F0] rounded-md',
          paddingMap[padding],
          interactive && 'hover:border-[#1a5cff] hover:shadow-[0_0_0_1px_rgba(26, 92, 255,0.1)] transition-all duration-120',
          className,
        )}
        {...props}
      />
    )
  },
)
Card.displayName = 'Card'

/**
 * Input component
 * Full wrapper: label + input + optional error message
 */
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, id, type = 'text', 'aria-describedby': describedBy, ...props }, ref) => {
    const generatedId = React.useId()
    const inputId = id || generatedId
    const errorId = `${inputId}-error`
    const ariaDescribedBy = [describedBy, error && errorId].filter(Boolean).join(' ') || undefined

    return (
      <div className="mb-3">
        {label && (
          <label
            htmlFor={inputId}
            className="block text-[#33373d] text-xs font-medium mb-1"
          >
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          type={type}
          aria-describedby={ariaDescribedBy}
          className={cn(
            'w-full bg-[#F1F5F9] text-[#1f2226] border border-[#E2E8F0] rounded-md px-2 py-1.5 text-sm',
            'focus:outline-none focus:border-[#1a5cff] transition-colors duration-120',
            'placeholder:text-[#33373d]',
            error && 'border-[#DC2626]',
            className,
          )}
          {...props}
        />
        {error && (
          <p id={errorId} role="alert" className="text-[#DC2626] text-xs mt-1">{error}</p>
        )}
      </div>
    )
  },
)
Input.displayName = 'Input'

/**
 * Section component
 * Renders a white card with a header bar
 */
interface SectionProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  subtitle?: string
}

export const Section = React.forwardRef<HTMLDivElement, SectionProps>(
  ({ className, title, subtitle, children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('bg-white border border-[#E2E8F0] rounded-md overflow-hidden', className)}
      {...props}
    >
      <div className="px-4 py-3 border-b border-[#E2E8F0]">
        <h3 className="text-sm font-semibold text-[#1f2226]">{title}</h3>
        {subtitle && (
          <p className="text-xs text-[#5d6470] mt-1">{subtitle}</p>
        )}
      </div>
      <div>{children}</div>
    </div>
  ),
)
Section.displayName = 'Section'

/**
 * Textarea component
 * Same visual language as Input
 */
interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, label, error, id, ...props }, ref) => {
    const generatedId = React.useId()
    const textareaId = id || generatedId

    return (
      <div className="mb-3">
        {label && (
          <label
            htmlFor={textareaId}
            className="block text-[#33373d] text-xs font-medium mb-1"
          >
            {label}
          </label>
        )}
        <textarea
          ref={ref}
          id={textareaId}
          className={cn(
            'w-full bg-[#F1F5F9] text-[#1f2226] border border-[#E2E8F0] rounded-md px-2 py-1.5 text-sm',
            'focus:outline-none focus:border-[#1a5cff] transition-colors duration-120',
            'placeholder:text-[#33373d] resize-vertical',
            error && 'border-[#DC2626]',
            className,
          )}
          {...props}
        />
        {error && (
          <p className="text-[#DC2626] text-xs mt-1">{error}</p>
        )}
      </div>
    )
  },
)
Textarea.displayName = 'Textarea'

/**
 * Select component
 * Same visual language as Input. label + select + optional error.
 * containerClassName overrides the outer wrapper (defaults to mb-3 when label present).
 */
interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  containerClassName?: string
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, label, error, id, containerClassName, children, ...props }, ref) => {
    const generatedId = React.useId()
    const selectId = id || generatedId

    return (
      <div className={cn(label && 'mb-3', containerClassName)}>
        {label && (
          <label htmlFor={selectId} className="block text-[#33373d] text-xs font-medium mb-1">
            {label}
          </label>
        )}
        <div className="relative">
          <select
            ref={ref}
            id={selectId}
            className={cn(
              'w-full appearance-none bg-[#F1F5F9] text-[#1f2226] border border-[#E2E8F0] rounded-md pl-2 pr-8 py-1.5 text-sm cursor-pointer',
              'focus:outline-none focus:border-[#1a5cff] transition-colors duration-150',
              error && 'border-[#DC2626]',
              className,
            )}
            {...props}
          >
            {children}
          </select>
          <svg
            data-select-chevron
            aria-hidden="true"
            className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2"
            width="12" height="12" viewBox="0 0 12 12" fill="none"
          >
            <path d="M3 4.5L6 7.5L9 4.5" stroke="#5d6470" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        {error && <p role="alert" className="text-[#DC2626] text-xs mt-1">{error}</p>}
      </div>
    )
  },
)
Select.displayName = 'Select'

/**
 * SectionLabel component
 * Slate form/inline label used inside panels and detail rails (#33373d).
 * See StructuralLabel for uppercase structural section labels (no teal).
 * Replaces local uiLabelStyle / inline label spans across pages.
 */
interface SectionLabelProps {
  children: React.ReactNode
  className?: string
}

export function SectionLabel({ children, className }: SectionLabelProps) {
  return (
    <span className={cn('block text-[#33373d] text-xs font-medium mb-1.5', className)}>
      {children}
    </span>
  )
}

/** Muted structural section header. ≤2-word labels only (AGENDA, INSTANCES). No structural teal (removed). */
export function StructuralLabel({ children, className }: SectionLabelProps) {
  return (
    <span className={cn('block text-[#5d6470] text-xs font-bold uppercase tracking-wide', className)}>
      {children}
    </span>
  )
}

/**
 * StatusBadge component
 * Inline status label with semantic color. Always renders color + text (accessible).
 * Uses the a11y-corrected status palette from lib/statusColors.ts
 */
export type StatusVariant =
  | 'pending' | 'backlog' | 'ready' | 'in_progress' | 'running'
  | 'blocked' | 'done' | 'completed' | 'active' | 'failed'
  | 'cancelled' | 'scheduled' | 'preparing' | 'concluding' | 'concluded'
  | 'proposed' | 'shadow' | 'disabled' | 'paused'
  | 'open' | 'accepted' | 'rejected' | 'candidate' | 'reserved' | 'dismissed'
  | 'needs-you' | 'idle' | 'skipped'
  | 'applied' | 'withdrawn' | 'superseded'
  | 'being_considered' | 'deferred' | 'needs_clarification'

export const STATUS_COLOR_MAP: Record<StatusVariant, string> = {
  pending:     STATUS_COLORS.neutral,
  backlog:     STATUS_COLORS.neutral,
  ready:       STATUS_COLORS.blue,
  in_progress: STATUS_COLORS.amber,
  running:     STATUS_COLORS.amber,
  blocked:     STATUS_COLORS.amber,
  done:        STATUS_COLORS.green,
  completed:   STATUS_COLORS.green,
  active:      STATUS_COLORS.green,
  failed:      STATUS_COLORS.red,
  cancelled:   STATUS_COLORS.neutral,
  scheduled:   STATUS_COLORS.blue,
  preparing:   STATUS_COLORS.amber,
  concluding:  STATUS_COLORS.amber,
  concluded:   STATUS_COLORS.green,
  proposed:    STATUS_COLORS.blue,
  shadow:      STATUS_COLORS.neutral,
  disabled:    STATUS_COLORS.neutral,
  paused:      STATUS_COLORS.amber,
  open:        STATUS_COLORS.amber,
  accepted:    STATUS_COLORS.green,
  rejected:    STATUS_COLORS.red,
  candidate:   STATUS_COLORS.blue,
  reserved:    STATUS_COLORS.blue,
  dismissed:   STATUS_COLORS.neutral,
  'needs-you': STATUS_COLORS.blue,
  idle:        STATUS_COLORS.neutral,
  skipped:     STATUS_COLORS.neutral,
  applied:     STATUS_COLORS.green,
  withdrawn:   STATUS_COLORS.neutral,
  superseded:  STATUS_COLORS.neutral,
  being_considered: STATUS_COLORS.amber,
  deferred:    STATUS_COLORS.blue,
  needs_clarification: STATUS_COLORS.blue,
}

interface StatusBadgeProps {
  status: string
  label?: string
  dot?: boolean
  className?: string
}

export function StatusBadge({ status, label, dot = true, className }: StatusBadgeProps) {
  const color = STATUS_COLOR_MAP[status as StatusVariant] ?? STATUS_COLORS.neutral
  const displayLabel = label ?? status
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs font-medium', className)} style={{ color }}>
      {dot && <span aria-hidden="true" className="text-[8px] leading-none">●</span>}
      {displayLabel}
    </span>
  )
}

/**
 * PageHeader component
 * Consistent page-level heading: 18px/600, text-[#1f2226]
 */
interface PageHeaderProps {
  title: string
  className?: string
}

export function PageHeader({ title, className }: PageHeaderProps) {
  return (
    <h1 className={cn('text-[18px] font-semibold text-[#1f2226] leading-snug m-0', className)}>
      {title}
    </h1>
  )
}

/**
 * Skeleton shape components
 * Used for loading states; require .skeleton CSS class (defined in index.css)
 */

export function SkeletonRow() {
  return (
    <div
      className="skeleton"
      style={{
        height: '13px',
        borderRadius: '3px',
        width: '100%',
      }}
    />
  )
}

export function SkeletonCard() {
  return (
    <div
      className="skeleton"
      style={{
        height: '120px',
        borderRadius: '6px',
        display: 'block',
      }}
    />
  )
}

export function SkeletonList({ count = 4 }: { count?: number }) {
  return (
    <div>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
          <div
            className="skeleton"
            style={{
              width: '70%',
              height: '13px',
              borderRadius: '3px',
            }}
          />
          <div
            className="skeleton"
            style={{
              width: '30%',
              height: '13px',
              borderRadius: '3px',
            }}
          />
        </div>
      ))}
    </div>
  )
}

export function SkeletonTable({ count = 4 }: { count?: number }) {
  return (
    <div>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          style={{
            display: 'flex',
            padding: '10px 16px',
            borderBottom: '1px solid #E2E8F0',
            gap: '8px',
          }}
        >
          <div
            className="skeleton"
            style={{
              flex: 1,
              height: '13px',
              borderRadius: '3px',
            }}
          />
          <div
            className="skeleton"
            style={{
              width: '48px',
              height: '13px',
              borderRadius: '3px',
            }}
          />
        </div>
      ))}
    </div>
  )
}

/**
 * QueryState: generic async state primitive
 * Renders loading skeleton, error state, empty state, or data.
 */

export interface QueryStateProps<T> {
  query: {
    isLoading: boolean
    isError: boolean
    data: T | undefined
    error?: unknown
    refetch?: () => void
  }
  skeleton?: 'rows' | 'cards' | 'list' | 'table'
  skeletonCount?: number
  errorLabel?: string
  emptyLabel?: string
  emptyDetail?: string
  isEmpty?: (data: T) => boolean
  children: (data: T) => React.ReactNode
}

export function QueryState<T>({
  query,
  skeleton = 'rows',
  skeletonCount = 4,
  errorLabel = 'Failed to load data',
  emptyLabel = 'Nothing here yet',
  emptyDetail,
  isEmpty: isEmptyFn,
  children,
}: QueryStateProps<T>) {
  // Loading state
  if (query.isLoading) {
    return (
      <div role="status" aria-busy="true" aria-label="Loading">
        {skeleton === 'rows' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Array.from({ length: skeletonCount }).map((_, i) => (
              <SkeletonRow key={i} />
            ))}
          </div>
        )}
        {skeleton === 'cards' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {Array.from({ length: skeletonCount }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        )}
        {skeleton === 'list' && <SkeletonList count={skeletonCount} />}
        {skeleton === 'table' && <SkeletonTable count={skeletonCount} />}
      </div>
    )
  }

  // Error state
  if (query.isError) {
    return (
      <div role="alert" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span className="text-[#B91C1C] text-sm">{errorLabel}</span>
        {query.refetch && (
          <Button variant="ghost" size="sm" className="min-h-11" onClick={() => query.refetch?.()}>
            Retry
          </Button>
        )}
      </div>
    )
  }

  // Undefined guard
  if (query.data === undefined) {
    return <EmptyState title={emptyLabel ?? 'Nothing here yet'} body={emptyDetail ?? 'Nothing here yet.'} />
  }

  // Compute emptiness
  const isEmpty = isEmptyFn
    ? isEmptyFn(query.data)
    : Array.isArray(query.data)
      ? query.data.length === 0
      : !query.data

  // Empty state
  if (isEmpty) {
    return <EmptyState title={emptyLabel} body={emptyDetail ?? 'Nothing here yet.'} />
  }

  // Render children with data
  return <>{children(query.data)}</>
}

/**
 * NoProjectSelected: Empty-state component for pages with no active project.
 * Displays guided onboarding copy with prerequisites and alternate paths.
 */
export interface NoProjectSelectedAction {
  label: string
  onClick: () => void
}

export interface NoProjectSelectedProps {
  pageTitle: string
  pageIntro?: string
  title: string
  detail: string
  primaryAction: NoProjectSelectedAction
  secondaryAction?: NoProjectSelectedAction
  prerequisites?: string[]
  alternatePaths?: string[]
}

export function NoProjectSelected({
  pageTitle,
  pageIntro,
  title,
  detail,
  primaryAction,
  secondaryAction,
  prerequisites,
  alternatePaths,
}: NoProjectSelectedProps) {
  const showDisclosure = !!prerequisites?.length && !!alternatePaths?.length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <PageHeader title={pageTitle} className="mb-0.5" />
        {pageIntro && (
          <p style={{ margin: 0, color: UI_COLORS.textSecondary, fontSize: 14, maxWidth: 680 }}>
            {pageIntro}
          </p>
        )}
      </div>

      <div style={{
        background: UI_COLORS.surface, border: `1px solid ${UI_COLORS.border}`,
        borderRadius: 6, padding: 20, display: 'flex', flexDirection: 'column', gap: 16,
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 640 }}>
          <h2 style={{ margin: 0, color: UI_COLORS.textPrimary, fontSize: 20, fontWeight: 600, lineHeight: 1.2 }}>
            {title}
          </h2>
          <p style={{ margin: 0, color: UI_COLORS.textSecondary, fontSize: 14, lineHeight: 1.6 }}>
            {detail}
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          <Button variant="primary" onClick={primaryAction.onClick}>{primaryAction.label}</Button>
          {secondaryAction && (
            <Button variant="secondary" onClick={secondaryAction.onClick}>{secondaryAction.label}</Button>
          )}
        </div>

        {showDisclosure && (
          <details style={{ marginTop: 4 }}>
            <summary style={{ color: UI_COLORS.textMuted, fontSize: 12, cursor: 'pointer', userSelect: 'none', paddingLeft: 4 }}>
              Why is this empty?
            </summary>
            <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
              <div style={{ background: UI_COLORS.depth, border: `1px solid ${UI_COLORS.border}`, borderRadius: 6, padding: '14px 16px' }}>
                <h3 style={{ margin: '0 0 8px', color: UI_COLORS.textPrimary, fontSize: 14, fontWeight: 600 }}>Expected prerequisites</h3>
                <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {prerequisites!.map((item) => (
                    <li key={item} style={{ color: UI_COLORS.textSecondary, fontSize: 13, lineHeight: 1.5 }}>{item}</li>
                  ))}
                </ul>
              </div>
              <div style={{ background: UI_COLORS.depth, border: `1px solid ${UI_COLORS.border}`, borderRadius: 6, padding: '14px 16px' }}>
                <h3 style={{ margin: '0 0 8px', color: UI_COLORS.textPrimary, fontSize: 14, fontWeight: 600 }}>Alternate paths</h3>
                <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {alternatePaths!.map((item) => (
                    <li key={item} style={{ color: UI_COLORS.textSecondary, fontSize: 13, lineHeight: 1.5 }}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

/**
 * ConfirmDialog: accessible destructive-action confirmation modal.
 * Uses role="alertdialog" for screen readers. Esc closes via Radix.
 */
export interface ConfirmDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  consequence?: string
  confirmLabel?: string
  onConfirm: () => void
  isPending?: boolean
  error?: string | null
  onCloseAutoFocus?: React.ComponentPropsWithoutRef<typeof Dialog.Content>['onCloseAutoFocus']
}

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  consequence,
  confirmLabel = 'Confirm',
  onConfirm,
  isPending,
  error,
  onCloseAutoFocus,
}: ConfirmDialogProps) {
  const contentRef = React.useRef<HTMLDivElement>(null)
  const confirmFocusIntent = React.useRef(false)

  React.useEffect(() => {
    if (isPending || !error || !confirmFocusIntent.current) return
    confirmFocusIntent.current = false
    const btns = contentRef.current?.querySelectorAll('button')
    ;(btns?.[btns.length - 1] as HTMLButtonElement | undefined)?.focus()
  }, [isPending, error])

  return (
    <ModalDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && isPending) return
        onOpenChange(nextOpen)
      }}
      title={title}
      description={consequence ?? 'This action cannot be undone.'}
      size="sm"
      role="alertdialog"
      onCloseAutoFocus={onCloseAutoFocus}
      contentRef={contentRef}
      footer={{
        primaryLabel: confirmLabel,
        primaryVariant: 'danger',
        isPending,
        onPrimary: () => {
          confirmFocusIntent.current = true
          onConfirm()
        },
      }}
    >
      {error && <p role="alert" className="mb-2 text-xs text-huddleroom-status-red">{error}</p>}
    </ModalDialog>
  )
}
