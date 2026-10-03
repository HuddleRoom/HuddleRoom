import { describe, expect, it } from 'vitest'

import { DASHBOARD_BASENAME, dashboardPath, shouldNavigate } from './navigation'

describe('dashboardPath', () => {
  it('returns the dashboard basename for the root path', () => {
    expect(dashboardPath('/')).toBe(DASHBOARD_BASENAME)
  })

  it('prefixes dashboard paths with the dashboard basename', () => {
    expect(dashboardPath('/login')).toBe('/dashboard/login')
  })

  it('normalizes trailing slashes before prefixing', () => {
    expect(dashboardPath('/login/')).toBe('/dashboard/login')
  })
})

describe('shouldNavigate', () => {
  it('allows navigation when the current path is different', () => {
    expect(shouldNavigate('/unknown', '/')).toBe(true)
  })

  it('does not navigate when the current path already matches', () => {
    expect(shouldNavigate('/', '/')).toBe(false)
  })

  it('treats trailing slashes as the same path', () => {
    expect(shouldNavigate('/login/', '/login')).toBe(false)
  })
})
