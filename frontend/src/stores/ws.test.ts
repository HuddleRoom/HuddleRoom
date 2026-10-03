import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { useWSStore } from './ws'
import { useAuthStore } from './auth'
import { loadRuntimeConfig } from '@/lib/api-client'

interface WSEvent {
  id: string
  event_type: string
  payload: Record<string, unknown>
  emitted_at: string
}

class MockWebSocket {
  static OPEN = 1
  static CONNECTING = 0
  url: string
  readyState: number = 0
  onopen: (() => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null
  onerror: ((e: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
  }

  close() { this.readyState = 3 }

  simulateOpen() {
    this.readyState = 1
    this.onopen?.()
  }

  simulateMessage(data: Record<string, unknown>) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }

  simulateClose(code = 1000) {
    this.readyState = 3
    this.onclose?.(new CloseEvent('close', { code }))
  }
}

let mockWebSocketInstance: MockWebSocket | null = null

const evt = (id: string, eventType = 'test', emittedAt = '2024-01-01T00:00:00'): WSEvent => ({
  id,
  event_type: eventType,
  payload: {},
  emitted_at: emittedAt,
})

describe('useWSStore', () => {
  beforeEach(() => {
    mockWebSocketInstance = null

    global.WebSocket = Object.assign(
      class MockWS {
        constructor(url: string) {
          const instance = new MockWebSocket(url)
          mockWebSocketInstance = instance
          return instance as any
        }
      },
      { OPEN: 1, CONNECTING: 0 }
    ) as any

    if (!global.window) {
      global.window = {} as any
    }

    global.window.location = {
      protocol: 'http:',
      host: 'localhost:5173',
    } as any

    global.import = {
      meta: {
        env: {},
      },
    } as any
  })

  afterEach(() => {
    useWSStore.getState().disconnect()
    useWSStore.setState({ connected: false, events: [] })
    useAuthStore.setState({ token: null })
    vi.unstubAllEnvs()
    vi.clearAllMocks()
  })

  describe('addEvent', () => {
    it('prepends new events', () => {
      useWSStore.getState().addEvent(evt('a'))
      const state = useWSStore.getState()
      expect(state.events).toHaveLength(1)
      expect(state.events[0].id).toBe('a')
    })

    it('ignores duplicate ids', () => {
      useWSStore.getState().addEvent(evt('a'))
      useWSStore.getState().addEvent(evt('a'))
      const state = useWSStore.getState()
      expect(state.events).toHaveLength(1)
    })

    it('allows events with different ids at same timestamp', () => {
      useWSStore.getState().addEvent(evt('a', 'test', '2024-01-01T00:00:00'))
      useWSStore.getState().addEvent(evt('b', 'test', '2024-01-01T00:00:00'))
      const state = useWSStore.getState()
      expect(state.events).toHaveLength(2)
    })

    it('prepends (newest first)', () => {
      useWSStore.getState().addEvent(evt('a'))
      useWSStore.getState().addEvent(evt('b'))
      const state = useWSStore.getState()
      expect(state.events[0].id).toBe('b')
      expect(state.events[1].id).toBe('a')
    })

    it('caps at 200 events', () => {
      for (let i = 0; i < 200; i++) {
        useWSStore.getState().addEvent(evt(String(i)))
      }
      let state = useWSStore.getState()
      expect(state.events).toHaveLength(200)

      useWSStore.getState().addEvent(evt('overflow'))
      state = useWSStore.getState()
      expect(state.events).toHaveLength(200)
      expect(state.events[0].id).toBe('overflow')
    })
  })

  describe('connect', () => {
    it('resets only the target project replay cursor', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('a1', 'test', '2024-01-01T12:00:00'))
      mockWebSocketInstance?.close()

      useWSStore.getState().connect('projectB', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('b1', 'test', '2024-01-01T13:00:00'))
      mockWebSocketInstance?.close()

      useWSStore.getState().resetReplayCursor('projectA')
      useWSStore.getState().connect('projectA', null)
      expect(mockWebSocketInstance!.url).not.toContain('replay_since')
      mockWebSocketInstance?.close()

      useWSStore.getState().connect('projectB', null)
      expect(mockWebSocketInstance!.url).toContain('replay_since=2024-01-01T13%3A00%3A00')
    })

    it('resets replay cursor (lastEventAt) when project changes', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('e1', 'test', '2024-01-01T12:00:00'))

      useWSStore.getState().connect('projectB', null)
      mockWebSocketInstance?.simulateOpen()

      const secondUrl = mockWebSocketInstance!.url
      expect(secondUrl).toContain('projectB')
      expect(secondUrl).not.toContain('replay_since')
    })

    it('includes replay_since in URL when reconnecting to same project', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('e1', 'test', '2024-01-01T12:00:00'))

      mockWebSocketInstance?.close()  // drop to CLOSED so dedup guard doesn't block reconnect
      useWSStore.getState().connect('projectA', null)
      const secondUrl = mockWebSocketInstance!.url
      expect(secondUrl).toContain('replay_since')
      expect(secondUrl).toContain('2024-01-01T12%3A00%3A00')
    })

    it('sets connected to true on open', () => {
      expect(useWSStore.getState().connected).toBe(false)

      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      expect(useWSStore.getState().connected).toBe(true)
    })

    it.skip('includes token in query params when provided', () => {
      useWSStore.getState().connect('projectA', 'test-token')

      expect(mockWebSocketInstance!.url).toContain('token=test-token')
    })

    it('uses wss protocol when on https', () => {
      global.window.location = {
        protocol: 'https:',
        host: 'example.com',
      } as any

      useWSStore.getState().connect('projectA', null)

      expect(mockWebSocketInstance!.url).toMatch(/^wss:/)
    })

    it('uses ws protocol when on http', () => {
      global.window.location = {
        protocol: 'http:',
        host: 'localhost',
      } as any

      useWSStore.getState().connect('projectA', null)

      expect(mockWebSocketInstance!.url).toMatch(/^ws:/)
    })
  })

  describe('stale socket suppression', () => {
    it('ignores messages after disconnect', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('e1'))

      expect(useWSStore.getState().events).toHaveLength(1)

      useWSStore.getState().disconnect()

      const closedSocket = mockWebSocketInstance
      closedSocket?.simulateMessage(evt('e2'))

      expect(useWSStore.getState().events).toHaveLength(1)
    })

    it('ignores onclose from stale socket after reconnect', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      const firstSocket = mockWebSocketInstance
      expect(useWSStore.getState().connected).toBe(true)

      firstSocket?.close()  // drop to CLOSED so dedup guard doesn't block reconnect
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      expect(useWSStore.getState().connected).toBe(true)

      firstSocket?.simulateClose()

      expect(useWSStore.getState().connected).toBe(true)
    })
  })

  describe('ping filtering', () => {
    it('ignores messages without event_type', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage({ emitted_at: '2024-01-01T00:00:00' })

      expect(useWSStore.getState().events).toHaveLength(0)
    })

    it('ignores messages without emitted_at', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage({
        id: 'e1',
        event_type: 'test',
        payload: {},
      })

      expect(useWSStore.getState().events).toHaveLength(0)
    })

    it('ignores malformed JSON', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      mockWebSocketInstance?.onmessage?.(
        new MessageEvent('message', { data: 'not json' })
      )

      expect(useWSStore.getState().events).toHaveLength(0)
    })
  })

  describe('disconnect', () => {
    it('sets connected to false', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      expect(useWSStore.getState().connected).toBe(true)

      useWSStore.getState().disconnect()
      expect(useWSStore.getState().connected).toBe(false)
    })

    it('prevents reconnection', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      const firstSocket = mockWebSocketInstance

      useWSStore.getState().disconnect()
      firstSocket?.simulateClose()

      expect(useWSStore.getState().connected).toBe(false)

      const waitMs = 100
      return new Promise((resolve) => {
        setTimeout(() => {
          expect(useWSStore.getState().connected).toBe(false)
          resolve(undefined)
        }, waitMs)
      })
    })
  })

  describe('message processing', () => {
    it('updates lastEventAt from received messages', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('e1', 'test', '2024-01-01T12:34:56'))

      mockWebSocketInstance?.close()  // drop to CLOSED so dedup guard doesn't block reconnect
      useWSStore.getState().connect('projectA', null)

      expect(mockWebSocketInstance!.url).toContain('replay_since')
      expect(mockWebSocketInstance!.url).toContain('2024-01-01T12%3A34%3A56')
    })

    it('processes valid events through addEvent', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      mockWebSocketInstance?.simulateMessage(evt('e1'))
      mockWebSocketInstance?.simulateMessage(evt('e2'))

      const state = useWSStore.getState()
      expect(state.events).toHaveLength(2)
      expect(state.events[0].id).toBe('e2')
      expect(state.events[1].id).toBe('e1')
    })
  })

  describe('heartbeat / zombie socket detection', () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('force-reconnects when no message (including pings) arrives within 10s', async () => {
      useAuthStore.setState({ token: null })
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ auth_enabled: false }))))
      await loadRuntimeConfig()
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()
      const firstSocket = mockWebSocketInstance
      expect(useWSStore.getState().connected).toBe(true)

      // The deadline is 10s but the check runs every 2s, so detection lands on the
      // first tick past the deadline (12s) — within the plan's "flips within 12s" bar.
      vi.advanceTimersByTime(12_000)

      // Zombie detected: old socket considered disconnected, reconnect scheduled
      expect(useWSStore.getState().connected).toBe(false)
      expect(useWSStore.getState().retrying).toBe(true)

      // The stale socket's real onclose (if it ever fires) must not resurrect old state
      firstSocket?.simulateClose()
      expect(useWSStore.getState().connected).toBe(false)

      vi.advanceTimersByTime(1000)
      const secondSocket = mockWebSocketInstance
      expect(secondSocket).not.toBe(firstSocket)
      secondSocket?.simulateOpen()
      expect(useWSStore.getState().connected).toBe(true)
    })

    it('a ping-shaped message (no event_type) resets the heartbeat deadline', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      vi.advanceTimersByTime(8000)
      mockWebSocketInstance?.simulateMessage({ ping: true } as any)
      vi.advanceTimersByTime(8000)

      expect(useWSStore.getState().connected).toBe(true)
    })

    it('gives up after repeated reconnect failures and stops scheduling retries', () => {
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      // Each close schedules a reconnect (attempts++); the follow-on socket never
      // opens, so advancing past the backoff spawns the next attempt's socket.
      for (let i = 0; i < 8; i++) {
        mockWebSocketInstance?.simulateClose()
        vi.advanceTimersByTime(30_001)
      }

      expect(useWSStore.getState().gaveUp).toBe(true)
      expect(useWSStore.getState().retrying).toBe(false)
    })

    it('manualReconnect resets attempts/gaveUp and reconnects using the last project', async () => {
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ auth_enabled: false }))))
      await loadRuntimeConfig()
      useWSStore.getState().connect('projectA', null)
      mockWebSocketInstance?.simulateOpen()

      for (let i = 0; i < 8; i++) {
        mockWebSocketInstance?.simulateClose()
        vi.advanceTimersByTime(30_001)
      }
      expect(useWSStore.getState().gaveUp).toBe(true)
      const previousSocket = mockWebSocketInstance

      useWSStore.getState().manualReconnect()

      expect(useWSStore.getState().gaveUp).toBe(false)
      expect(mockWebSocketInstance).not.toBe(previousSocket)
      expect(mockWebSocketInstance!.url).toContain('projectA')

      mockWebSocketInstance?.simulateOpen()
      expect(useWSStore.getState().connected).toBe(true)
    })
  })
})
