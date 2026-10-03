import React from 'react'
import { Button, cn, UI_COLORS } from './uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'

interface BannerAction {
  label: string
  onClick: () => void
}

interface BannerProps {
  variant: 'info' | 'warning' | 'offline'
  title: string
  message?: React.ReactNode
  action?: BannerAction
  className?: string
}

export function Banner({ variant, title, message, action, className }: BannerProps) {
  const variantStyles = {
    info: {
      bg: UI_COLORS.surfaceBlue,
      borderColor: STATUS_COLORS.blue,
      text: UI_COLORS.primaryHover,
    },
    warning: {
      bg: UI_COLORS.bg2,
      borderColor: STATUS_COLORS.amber,
      text: STATUS_COLORS.amber,
    },
    offline: {
      bg: UI_COLORS.bg2,
      borderColor: STATUS_COLORS.amber,
      text: STATUS_COLORS.amber,
    },
  }

  const styles = variantStyles[variant]

  return (
    <div
      data-testid="banner"
      className={cn('flex flex-col gap-2 rounded-md border p-3', className)}
      style={{
        backgroundColor: styles.bg,
        borderColor: styles.borderColor,
        color: styles.text,
        fontSize: '12px',
        lineHeight: '1.4',
        fontFamily: "'Inter', system-ui, sans-serif",
      }}
    >
      <div className="flex flex-col gap-1">
        <strong style={{ color: UI_COLORS.textPrimary, fontWeight: 600 }}>{title}</strong>
        {message && <span>{message}</span>}
      </div>
      {action && (
        <div className="mt-1">
          <Button
            variant="secondary"
            size="sm"
            onClick={action.onClick}
            style={{ fontSize: '12px' }}
          >
            {action.label}
          </Button>
        </div>
      )}
    </div>
  )
}
