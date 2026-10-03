import { apiFetch, setToken } from '@/lib/api-client'
import type { User } from '@/lib/types'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { ApiKey } from '@/lib/types'

export async function login(email: string, password: string): Promise<string> {
  const data = await apiFetch<{ access_token: string }>('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })
  if (!setToken(data.access_token)) throw new Error('Server returned an invalid token')
  return data.access_token
}

export async function getMe(): Promise<User> {
  return apiFetch<User>('/api/v1/auth/me')
}

export async function getConfig(): Promise<{ auth_enabled: boolean }> {
  return apiFetch<{ auth_enabled: boolean }>('/api/v1/config')
}

export function useApiKeys() {
  return useQuery({
    queryKey: ['api-keys'],
    queryFn: () => apiFetch<ApiKey[]>('/api/v1/auth/api-keys'),
  })
}

export function useCreateApiKey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (data: { label?: string; agent_id?: string; project_id?: string; expires_at?: string }) =>
      apiFetch<{ key: string; api_key: ApiKey }>('/api/v1/auth/api-keys', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['api-keys'] }),
  })
}

export function useRevokeApiKey() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (keyId: string) =>
      apiFetch<void>(`/api/v1/auth/api-keys/${keyId}`, { method: 'DELETE' }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['api-keys'] }),
  })
}

export function useCurrentUser() {
  return useQuery({
    queryKey: ['me'],
    queryFn: () => apiFetch<User>('/api/v1/auth/me'),
  })
}
