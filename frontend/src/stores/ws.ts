import { create } from 'zustand'
import { useAuthStore } from '@/stores/auth'
import { isAuthDisabled } from '@/lib/api-client'
import { useUIStore } from '@/stores/ui'
import { isAgentResponseEvent } from '@/lib/agentResponse'
import { useAgentResponseStore } from '@/stores/agent-response'

interface WSEvent {
  id: string
  event_type: string
  payload: Record<string, unknown>
  emitted_at: string
}

interface WSState {
  connected: boolean
  connecting: boolean
  retrying: boolean
  hasConnectedOnce: boolean
  gaveUp: boolean
  events: WSEvent[]
  connect: (projectId: string, token: string | null) => void
  disconnect: () => void
  manualReconnect: () => void
  resetReplayCursor: (projectId: string) => void
  addEvent: (event: WSEvent) => void
}

const HEARTBEAT_CHECK_MS = 2000
const HEARTBEAT_TIMEOUT_MS = 10000
const MAX_RECONNECT_ATTEMPTS = 8

export const useWSStore = create<WSState>()((set, get) => {
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null
  let backoff = 1000
  let lastMessageAt = 0
  let attempts = 0
  const replayCursors = new Map<string, string>()
  let currentProjectId: string | null = null
  let currentToken: string | null = null
  let activeSocketId = 0  // incremented on each new socket; used to detect stale closures
  let closedByAuth = false

  function clearHeartbeat() {
    if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null }
  }

  function scheduleReconnect() {
    attempts++
    if (attempts >= MAX_RECONNECT_ATTEMPTS) {
      set({ retrying: false, gaveUp: true })
      return
    }
    set({ retrying: true })
    reconnectTimer = setTimeout(() => {
      backoff = Math.min(backoff * 2, 30000)
      const freshToken = useAuthStore.getState().token
      if (!currentProjectId || (!freshToken && !isAuthDisabled())) {
        // Can't reconnect (no target/token) and nothing will re-arm this timer —
        // surface the give-up state so the manual Reconnect banner is offered
        // instead of stranding the client in retrying:true forever.
        set({ retrying: false, gaveUp: true })
        return
      }
      currentToken = freshToken
      connect(currentProjectId, freshToken)
    }, backoff)
  }

  // Reset closedByAuth when a new token is set (re-login after auth expiry)
  useAuthStore.subscribe((state, prev) => {
    if (!prev.token && state.token) closedByAuth = false
  })

  function connect(projectId: string, token: string | null) {
    if (token) closedByAuth = false
    if (closedByAuth) return

    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)
        && projectId === currentProjectId && token === currentToken) return

    // A genuinely new connection target (not an internal reconnect retry) starts fresh
    if (projectId !== currentProjectId || token !== currentToken) {
      attempts = 0
      set({ gaveUp: false })
    }

    // Clear agent response state on project change
    if (projectId !== currentProjectId) {
      useAgentResponseStore.getState().setProject(projectId)
    }

    currentProjectId = projectId
    currentToken = token

    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    clearHeartbeat()
    if (ws) { ws.close(); ws = null }

    const socketId = ++activeSocketId
    set({ connecting: true })

    const params = new URLSearchParams()
    if (token) params.set('token', token)
    const replayCursor = replayCursors.get(projectId)
    if (replayCursor) params.set('replay_since', replayCursor)

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const base = import.meta.env.VITE_API_URL?.replace(/^https?/, protocol) ?? `${protocol}://${window.location.host}`
    const url = `${base}/ws/projects/${projectId}/events?${params}`

    ws = new WebSocket(url)

    ws.onopen = () => {
      if (socketId !== activeSocketId) return
      set({ connected: true, connecting: false, retrying: false, hasConnectedOnce: true })
      backoff = 1000
      attempts = 0
      lastMessageAt = Date.now()
      clearHeartbeat()
      heartbeatTimer = setInterval(() => {
        if (socketId !== activeSocketId) return
        if (Date.now() - lastMessageAt <= HEARTBEAT_TIMEOUT_MS) return
        // Zombie socket: no message (including pings) within the deadline.
        // Invalidate it and force our own reconnect path — don't wait for a real onclose.
        clearHeartbeat()
        activeSocketId++
        const staleWs = ws
        ws = null
        set({ connected: false, connecting: false })
        staleWs?.close()
        if (!currentProjectId) return
        scheduleReconnect()
      }, HEARTBEAT_CHECK_MS)
    }

    ws.onclose = (event) => {
      if (socketId !== activeSocketId) return  // stale socket — disconnect() was called
      clearHeartbeat()
      set({ connected: false, connecting: false })
      if (!currentProjectId) return
      if (event.code === 4001) { closedByAuth = true; if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null } ws = null; currentProjectId = null; currentToken = null; useUIStore.getState().setActiveProject(null); useAuthStore.getState().clearToken(); window.dispatchEvent(new Event('auth:expired')); return }  // auth failure — clears state and signals RequireAuth to redirect
      scheduleReconnect()
    }

    ws.onmessage = (e) => {
      if (socketId !== activeSocketId) return
      try {
        const msg = JSON.parse(e.data) as Record<string, unknown>
        lastMessageAt = Date.now()
        if (!msg.event_type || !msg.emitted_at) return  // pings and malformed messages

        // Route response events before durable cursor mutation
        if (isAgentResponseEvent(msg)) {
          useAgentResponseStore.getState().ingest(msg)
          return
        }

        // Route durable events
        const data = msg as unknown as WSEvent
        if (currentProjectId) replayCursors.set(currentProjectId, data.emitted_at)
        get().addEvent(data)
      } catch {}
    }

    ws.onerror = (e) => {
      if (import.meta.env.DEV) console.error('[ws] socket error', e)
    }
  }

  function disconnect() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    clearHeartbeat()
    activeSocketId++  // invalidate any pending onclose/onmessage from old socket
    if (ws) { ws.close(); ws = null }
    currentProjectId = null
    currentToken = null
    backoff = 1000
    attempts = 0
    replayCursors.clear()
    set({ connected: false, connecting: false, retrying: false, hasConnectedOnce: false, gaveUp: false })
  }

  function manualReconnect() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    attempts = 0
    backoff = 1000
    closedByAuth = false  // an explicit user retry clears a prior auth-close latch
    set({ gaveUp: false })
    if (!currentProjectId) return
    connect(currentProjectId, currentToken)
  }

  return {
    connected: false,
    connecting: false,
    retrying: false,
    hasConnectedOnce: false,
    gaveUp: false,
    events: [],
    connect,
    disconnect,
    manualReconnect,
    resetReplayCursor: (projectId) => { replayCursors.delete(projectId) },
    addEvent: (event) => set((s) => {
      if (s.events.some((e) => e.id === event.id)) return s
      return { events: [event, ...s.events].slice(0, 200) }
    }),
  }
})
