import type { CSSProperties, FormEvent } from 'react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { login, getConfig } from '@/api/auth'
import { isAuthDisabled } from '@/lib/api-client'
import {
  UI_COLORS,
  UI_FONT_FAMILY,
  Button,
  Input,
} from '@/components/common/uiPrimitives'

const pageStyle: CSSProperties = {
  minHeight: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: UI_COLORS.appBg,
  color: UI_COLORS.textPrimary,
  padding: 24,
}

const panelStyle: CSSProperties = {
  width: 'min(100%, 420px)',
  backgroundColor: UI_COLORS.surface,
  border: `1px solid ${UI_COLORS.border}`,
  borderRadius: 8,
  padding: 24,
  display: 'flex',
  flexDirection: 'column',
  gap: 20,
}

const brandBlockStyle: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
}

const headingStyle: CSSProperties = {
  margin: 0,
  color: UI_COLORS.textPrimary,
  fontSize: 24,
  lineHeight: 1.15,
  fontWeight: 600,
  fontFamily: UI_FONT_FAMILY,
  textWrap: 'balance',
}

const bodyStyle: CSSProperties = {
  margin: 0,
  color: UI_COLORS.textSecondary,
  fontSize: 14,
  lineHeight: 1.5,
  fontFamily: UI_FONT_FAMILY,
}

const statusPanelStyle: CSSProperties = {
  backgroundColor: UI_COLORS.sidebarHoverText,
  border: `1px solid ${UI_COLORS.border}`,
  borderRadius: 6,
  padding: '12px 14px',
  display: 'grid',
  gap: 10,
}

const statusRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '96px minmax(0, 1fr)',
  gap: 12,
  alignItems: 'baseline',
}

const statusKeyStyle: CSSProperties = {
  color: UI_COLORS.textMuted,
  fontSize: 12,
  fontFamily: UI_FONT_FAMILY,
}

const statusValueStyle: CSSProperties = {
  color: UI_COLORS.textPrimary,
  fontSize: 13,
  lineHeight: 1.45,
  fontFamily: UI_FONT_FAMILY,
}

const primaryButtonBaseStyle: CSSProperties = {
  width: '100%',
  minHeight: 44,
  border: 'none',
  borderRadius: 6,
  padding: '10px 14px',
  fontSize: 13,
  fontWeight: 600,
  fontFamily: UI_FONT_FAMILY,
  transition: 'background-color 160ms cubic-bezier(0.4, 0, 0.2, 1), opacity 160ms cubic-bezier(0.4, 0, 0.2, 1)',
}

const inputDisabledStyle: CSSProperties = {
  opacity: 0.7,
  cursor: 'not-allowed',
}

export function LoginPage() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [localError, setLocalError] = useState('')

  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
  })

  useEffect(() => {
    if (isAuthDisabled()) navigate('/')
  }, [navigate])

  useEffect(() => {
    if (config?.auth_enabled === false) navigate('/')
  }, [config, navigate])

  const loginMutation = useMutation({
    mutationFn: () => login(email, password),
    onSuccess: () => {
      navigate('/')
    },
    onError: (err: unknown) => {
      const message = err instanceof Error ? err.message : 'Login failed'
      setLocalError(message)
    },
  })

  // If auth is disabled, redirect to dashboard
  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLocalError('')
    loginMutation.mutate()
  }

  const canSubmit = email.trim().length > 0 && password.length > 0 && !loginMutation.isPending
  const helperMessage = loginMutation.isPending
    ? 'Verifying credentials and opening the operator workspace.'
    : 'Use the credentials issued for this HuddleRoom server.'

  if (isAuthDisabled() || config?.auth_enabled === false) {
    return (
      <div style={pageStyle}>
        <div style={panelStyle}>
          <div style={brandBlockStyle}>
            <h1 style={headingStyle}>Authentication is not enabled on this server.</h1>
            <p style={bodyStyle}>
              Continue directly to the operator workspace. This environment does not require credentials before access.
            </p>
          </div>

          <div style={statusPanelStyle}>
            <div style={statusRowStyle}>
              <span style={statusKeyStyle}>Server policy</span>
              <span style={statusValueStyle}>Authentication disabled</span>
            </div>
            <div style={statusRowStyle}>
              <span style={statusKeyStyle}>Destination</span>
              <span style={statusValueStyle}>Operator workspace</span>
            </div>
          </div>

          <Button
            variant="primary"
            size="lg"
            type="button"
            onClick={() => navigate('/')}
            style={{ width: '100%' }}
          >
            Open operator workspace
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div style={pageStyle}>
      <form
        onSubmit={handleSubmit}
        aria-describedby={localError ? 'login-error' : 'login-helper'}
        style={panelStyle}
      >
        <div style={brandBlockStyle}>
          <h1 style={headingStyle}>Sign in to HuddleRoom</h1>
          <p style={bodyStyle}>
            Authenticate to monitor sessions, review protocols, and steer work in this environment.
          </p>
        </div>

        <div style={statusPanelStyle}>
          <div style={statusRowStyle}>
            <span style={statusKeyStyle}>Server policy</span>
            <span style={statusValueStyle}>Authentication required</span>
          </div>
          <div style={statusRowStyle}>
            <span style={statusKeyStyle}>Access scope</span>
            <span style={statusValueStyle}>Workspace navigation, live feeds, and operator actions</span>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <Input
            label="Email"
            id="login-email"
            type="email"
            autoComplete="email"
            autoFocus
            value={email}
            disabled={loginMutation.isPending}
            onChange={(e) => {
              if (localError) setLocalError('')
              setEmail(e.target.value)
            }}
          />

          <Input
            label="Password"
            id="login-password"
            type="password"
            autoComplete="current-password"
            value={password}
            disabled={loginMutation.isPending}
            onChange={(e) => {
              if (localError) setLocalError('')
              setPassword(e.target.value)
            }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <p
            id="login-helper"
            style={{
              margin: 0,
              color: loginMutation.isPending ? UI_COLORS.textSecondary : UI_COLORS.textMuted,
              fontSize: 12,
              lineHeight: 1.45,
              fontFamily: UI_FONT_FAMILY,
            }}
          >
            {helperMessage}
          </p>

          <Button
            variant="primary"
            size="lg"
            type="submit"
            disabled={!canSubmit}
            aria-busy={loginMutation.isPending}
            style={{ width: '100%' }}
          >
            {loginMutation.isPending ? 'Signing in...' : 'Sign in'}
          </Button>
        </div>

        {localError && (
          <p
            id="login-error"
            role="alert"
            style={{
              margin: 0,
              color: UI_COLORS.danger,
              fontSize: 12,
              lineHeight: 1.45,
              fontFamily: UI_FONT_FAMILY,
            }}
          >
            {localError}
          </p>
        )}
      </form>
    </div>
  )
}
