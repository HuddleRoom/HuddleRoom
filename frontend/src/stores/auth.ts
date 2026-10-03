import { create } from 'zustand'

interface AuthState {
  token: string | null
  setToken: (token: string) => boolean
  clearToken: () => void
}

const storage = typeof sessionStorage !== 'undefined' ? sessionStorage : null
export const AUTH_TOKEN_KEY = 'huddleroom_token'
export const LEGACY_AUTH_TOKEN_KEY = 'rally_token'

const safeSet = (key: string, value: string): void => {
  try {
    storage?.setItem(key, value)
  } catch {
    // Swallow error (e.g., Safari private mode)
  }
}

const safeRemove = (key: string): void => {
  try {
    storage?.removeItem(key)
  } catch {
    // Swallow error (e.g., Safari private mode)
  }
}

export function migrateLegacyAuthToken(storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>): string | null {
  const current = storage.getItem(AUTH_TOKEN_KEY)
  if (current !== null) return current
  const legacy = storage.getItem(LEGACY_AUTH_TOKEN_KEY)
  if (legacy !== null) {
    try {
      storage.setItem(AUTH_TOKEN_KEY, legacy)
      storage.removeItem(LEGACY_AUTH_TOKEN_KEY)
    } catch {
      // Keep the readable legacy token when session storage is unwritable.
    }
  }
  return legacy
}

const readToken = (): string | null => {
  try {
    return storage ? migrateLegacyAuthToken(storage) : null
  } catch {
    return null
  }
}

export const useAuthStore = create<AuthState>()((set) => ({
  token: readToken()?.trim() || null,
  setToken: (token: string) => {
    const trimmed = token.trim()
    if (!trimmed) return false
    safeSet(AUTH_TOKEN_KEY, trimmed)
    set({ token: trimmed })
    return true
  },
  clearToken: () => {
    safeRemove(AUTH_TOKEN_KEY)
    safeRemove(LEGACY_AUTH_TOKEN_KEY)
    set({ token: null })
  },
}))
