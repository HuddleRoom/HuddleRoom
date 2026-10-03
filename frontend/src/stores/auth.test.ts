import { describe, expect, it } from 'vitest'
import { AUTH_TOKEN_KEY, LEGACY_AUTH_TOKEN_KEY, migrateLegacyAuthToken } from './auth'

function storage(values: Record<string, string> = {}) {
  const data = new Map(Object.entries(values))
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
    removeItem: (key: string) => data.delete(key),
  }
}

describe('migrateLegacyAuthToken', () => {
  it('migrates the legacy token once', () => {
    const sessionStorage = storage({ [LEGACY_AUTH_TOKEN_KEY]: 'token' })

    expect(migrateLegacyAuthToken(sessionStorage)).toBe('token')
    expect(sessionStorage.getItem(AUTH_TOKEN_KEY)).toBe('token')
    expect(sessionStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBeNull()
  })

  it('keeps the HuddleRoom token when both versions exist', () => {
    const sessionStorage = storage({ [AUTH_TOKEN_KEY]: 'current', [LEGACY_AUTH_TOKEN_KEY]: 'legacy' })

    expect(migrateLegacyAuthToken(sessionStorage)).toBe('current')
    expect(sessionStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe('legacy')
  })

  it('returns a readable legacy token when storage writes fail', () => {
    const sessionStorage = {
      getItem: (key: string) => key === LEGACY_AUTH_TOKEN_KEY ? 'legacy' : null,
      setItem: () => { throw new Error('read-only') },
      removeItem: () => { throw new Error('read-only') },
    }

    expect(migrateLegacyAuthToken(sessionStorage)).toBe('legacy')
  })

  it('keeps the legacy token when copying it fails', () => {
    const sessionStorage = storage({ [LEGACY_AUTH_TOKEN_KEY]: 'legacy' })
    sessionStorage.setItem = () => { throw new Error('read-only') }

    expect(migrateLegacyAuthToken(sessionStorage)).toBe('legacy')
    expect(sessionStorage.getItem(LEGACY_AUTH_TOKEN_KEY)).toBe('legacy')
  })
})
