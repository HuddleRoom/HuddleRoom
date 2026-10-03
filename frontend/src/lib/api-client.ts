import { useAuthStore } from '@/stores/auth'

const API_BASE = import.meta.env.VITE_API_URL ?? ''
export function authDisabledFromEnv(env: {
  VITE_HUDDLEROOM_AUTH_DISABLED?: string
  VITE_RALLY_AUTH_DISABLED?: string
}): boolean {
  return (env.VITE_HUDDLEROOM_AUTH_DISABLED ?? env.VITE_RALLY_AUTH_DISABLED) === 'true'
}

let authDisabled = authDisabledFromEnv(import.meta.env as {
  VITE_HUDDLEROOM_AUTH_DISABLED?: string
  VITE_RALLY_AUTH_DISABLED?: string
})
let runtimeConfigPromise: Promise<void> | null = null

export function isAuthDisabled(): boolean {
  return authDisabled
}

export async function loadRuntimeConfig(): Promise<void> {
  if (runtimeConfigPromise) return runtimeConfigPromise
  runtimeConfigPromise = (async () => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 3000)
    try {
      const res = await fetch(`${API_BASE}/api/v1/config`, { signal: controller.signal })
      if (!res.ok) return
      const data = (await res.json()) as { auth_enabled?: unknown }
      if (typeof data.auth_enabled === 'boolean') {
        authDisabled = !data.auth_enabled
      }
    } catch {
      // authDisabled keeps the build-time HuddleRoom or legacy Rally setting; defaults to false (auth on).
    } finally {
      clearTimeout(timer)
    }
  })()
  return runtimeConfigPromise
}

export function getToken(): string | null {
  return useAuthStore.getState().token
}

export function setToken(token: string): boolean {
  return useAuthStore.getState().setToken(token)
}

export function clearToken(): void {
  useAuthStore.getState().clearToken()
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message)
  }
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken()
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  }
  if (token) headers['Authorization'] = `Bearer ${token}`

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers })

  if (res.status === 401) {
    clearToken()
    window.dispatchEvent(new Event('auth:expired'))
    throw new ApiError(401, 'Unauthorized')
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const detail = body.detail
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail) && detail.length > 0
        ? detail.map((d) => (d && typeof d === 'object' && typeof d.msg === 'string') ? d.msg : JSON.stringify(d)).join('; ')
        : `HTTP ${res.status}`
    throw new ApiError(res.status, message, detail)
  }

  if (res.status === 204) return undefined as T
  return res.json()
}

export async function fetchAllPages<T>(
  basePath: string,
  extraParams: Record<string, string> = {},
  pageSize = 200,
): Promise<{ items: T[]; next_cursor: null }> {
  const items: T[] = []
  let cursor: string | null = null
  do {
    const qs = new URLSearchParams({ limit: String(pageSize), ...extraParams })
    if (cursor) qs.set('cursor', cursor)
    const page = await apiFetch<{ items: T[]; next_cursor: string | null }>(
      `${basePath}?${qs}`,
    )
    items.push(...page.items)
    cursor = page.next_cursor
  } while (cursor)
  return { items, next_cursor: null }
}
