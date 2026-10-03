import { create } from 'zustand'
import type { MeetingTurn, MeetingWSEvent } from '@/lib/types'

const MAX_EVENTS = 200
const REPLAY_FROM_START = '1970-01-01T00:00:00+00:00'

interface MeetingWSState {
  connected: boolean
  meetingId: string | null
  turns: MeetingTurn[]
  events: MeetingWSEvent[]
  connect: (meetingId: string, token: string | null) => void
  disconnect: () => void
  appendTurn: (turn: MeetingTurn) => void
  appendEvent: (event: MeetingWSEvent) => void
  clearTurns: () => void
  clearEvents: () => void
}

export const useMeetingWSStore = create<MeetingWSState>()((set, get) => {
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let backoff = 1000
  let lastEventAt: string | null = null
  let currentMeetingId: string | null = null
  let currentToken: string | null = null
  let activeSocketId = 0

  function connect(meetingId: string, token: string | null) {
    if (meetingId !== currentMeetingId) {
      lastEventAt = null
      set({ turns: [], events: [], meetingId })
    }
    currentMeetingId = meetingId
    currentToken = token

    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }

    const socketId = ++activeSocketId

    const params = new URLSearchParams()
    if (token) params.set('token', token)
    params.set('replay_since', lastEventAt ?? REPLAY_FROM_START)

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const base =
      import.meta.env.VITE_API_URL?.replace(/^https?/, protocol) ??
      `${protocol}://${window.location.host}`
    const url = `${base}/ws/meetings/${meetingId}?${params}`

    ws = new WebSocket(url)

    ws.onopen = () => {
      if (socketId !== activeSocketId) return
      set({ connected: true })
      backoff = 1000
    }

    ws.onclose = (event) => {
      if (socketId !== activeSocketId) return
      set({ connected: false })
      if (!currentMeetingId || event.code === 4001) return
      reconnectTimer = setTimeout(() => {
        backoff = Math.min(backoff * 2, 30000)
        connect(currentMeetingId!, currentToken)
      }, backoff)
    }

    ws.onmessage = (e) => {
      if (socketId !== activeSocketId) return
      try {
        const msg = JSON.parse(e.data) as Record<string, unknown>
        if (!msg.event_type) return

        const eventType = msg.event_type as string
        get().appendEvent(msg as unknown as MeetingWSEvent)
        if (msg.emitted_at) {
          lastEventAt = msg.emitted_at as string
        }

        if (eventType === 'meeting.turn_complete' || eventType === 'meeting.human_turn') {
          const payload = msg.payload as Record<string, unknown> | undefined
          const turn = (msg.turn ?? payload?.turn) as MeetingTurn | undefined
          if (turn) {
            get().appendTurn(turn)
          }
        }
      } catch {}
    }
  }

  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    activeSocketId++
    if (ws) {
      ws.close()
      ws = null
    }
    currentMeetingId = null
    set({ connected: false, meetingId: null })
  }

  return {
    connected: false,
    meetingId: null,
    turns: [],
    events: [],
    connect,
    disconnect,
    appendTurn: (turn) =>
      set((s) => {
        if (s.turns.some((t) => t.id === turn.id)) return s
        return { turns: [turn, ...s.turns] }
      }),
    appendEvent: (event) =>
      set((s) => {
        if (event.id && s.events.some((e) => e.id === event.id)) return s
        return { events: [event, ...s.events].slice(0, MAX_EVENTS) }
      }),
    clearTurns: () => set({ turns: [] }),
    clearEvents: () => set({ events: [] }),
  }
})
