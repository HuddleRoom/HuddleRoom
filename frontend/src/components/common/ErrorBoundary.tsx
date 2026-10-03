import React from 'react'
import { UI_COLORS } from '@/components/common/uiPrimitives'

interface Props {
  children: React.ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error) {
    console.error('ErrorBoundary caught:', error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, color: UI_COLORS.danger, fontFamily: 'monospace', fontSize: 14 }}>
          <p>Something went wrong loading this page.</p>
          <p style={{ color: UI_COLORS.textMuted, fontSize: 12, marginTop: 8 }}>Reload to try again. If this keeps happening, check the console for details.</p>
          <button
            onClick={() => window.location.reload()}
            style={{
              marginTop: 12,
              padding: '6px 12px',
              background: 'none',
              border: `1px solid ${UI_COLORS.danger}`,
              color: UI_COLORS.danger,
              borderRadius: 3,
              cursor: 'pointer',
              fontFamily: 'monospace',
              fontSize: 12,
              transition: 'all 120ms',
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget
              el.style.background = UI_COLORS.danger
              el.style.color = UI_COLORS.surface
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget
              el.style.background = 'none'
              el.style.color = UI_COLORS.danger
            }}
          >
            reload
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
