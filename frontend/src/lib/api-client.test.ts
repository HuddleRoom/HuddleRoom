import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiFetch, ApiError, authDisabledFromEnv } from './api-client'
import { useAuthStore } from '@/stores/auth'

describe('apiFetch', () => {
  afterEach(() => {
    useAuthStore.getState().clearToken()
    vi.unstubAllGlobals()
  })

  it.skip('clears the token without reloading on unauthorized responses', async () => {
    const assign = vi.fn()

    useAuthStore.getState().setToken('expired-token')
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 401 })))
    vi.stubGlobal('window', { location: { assign }, dispatchEvent: vi.fn() })

    await expect(apiFetch('/api/v1/auth/me')).rejects.toThrow('Unauthorized')

    expect(useAuthStore.getState().token).toBeNull()
    expect(assign).not.toHaveBeenCalled()
  })

  it('extracts validation error messages from FastAPI detail array', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        detail: [
          { loc: ['body', 'x'], msg: 'field required', type: 'missing' }
        ]
      }), { status: 422 })
    ))

    await expect(apiFetch('/api/v1/test')).rejects.toThrow('field required')
  })
})

describe.skip('authDisabledFromEnv', () => {
  it('uses the HuddleRoom setting before the legacy Rally alias', () => {
    expect(authDisabledFromEnv({ VITE_HUDDLEROOM_AUTH_DISABLED: 'false', VITE_RALLY_AUTH_DISABLED: 'true' })).toBe(false)
    expect(authDisabledFromEnv({ VITE_HUDDLEROOM_AUTH_DISABLED: 'true', VITE_RALLY_AUTH_DISABLED: 'false' })).toBe(true)
  })

  it('accepts the legacy alias while it remains supported', () => {
    expect(authDisabledFromEnv({ VITE_RALLY_AUTH_DISABLED: 'true' })).toBe(true)
  })
})
