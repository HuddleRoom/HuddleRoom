import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { useMeetingWSStore } from './meeting-ws'
import type { MeetingTurn } from '@/lib/types'

class MockWebSocket {
  url: string
  onopen: (() => void) | null = null
  onclose: ((e: CloseEvent) => void) | null = null
  onmessage: ((e: MessageEvent) => void) | null = null

  constructor(url: string) {
    this.url = url
  }

  close() {}

  simulateMessage(data: Record<string, unknown>) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
  }
}

let mockWebSocketInstance: MockWebSocket | null = null

const turn = (id: string): MeetingTurn => ({
  id,
  meeting_id: 'meeting-1',
  is_human_turn: false,
  content: 'real content',
  created_at: '2024-01-01T00:00:00Z',
})

describe('useMeetingWSStore', () => {
  beforeEach(() => {
    mockWebSocketInstance = null
    global.WebSocket = class MockWS {
      constructor(url: string) {
        const instance = new MockWebSocket(url)
        mockWebSocketInstance = instance
        return instance as any
      }
    } as any

    if (!global.window) {
      global.window = {} as any
    }
    global.window.location = { protocol: 'http:', host: 'localhost:5173' } as any
  })

  afterEach(() => {
    useMeetingWSStore.getState().disconnect()
    useMeetingWSStore.setState({ connected: false, turns: [], events: [] })
  })

  it('appends enriched top-level meeting turns', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)
    mockWebSocketInstance?.simulateMessage({
      event_type: 'meeting.turn_complete',
      emitted_at: '2024-01-01T00:00:00Z',
      payload: {},
      turn: turn('turn-1'),
    })

    expect(useMeetingWSStore.getState().turns).toEqual([turn('turn-1')])
  })

  it('keeps non-turn websocket events for verbose mode', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)
    mockWebSocketInstance?.simulateMessage({ type: 'ping' })
    mockWebSocketInstance?.simulateMessage({
      event_type: 'meeting.trace',
      emitted_at: '2024-01-01T00:00:01Z',
      source: 'system',
      payload: {
        meeting_id: 'meeting-1',
        trace: { kind: 'participant_turn', stage: 'response', raw_response: 'full output' },
      },
    })

    expect(useMeetingWSStore.getState().events).toEqual([
      {
        event_type: 'meeting.trace',
        emitted_at: '2024-01-01T00:00:01Z',
        source: 'system',
        payload: {
          meeting_id: 'meeting-1',
          trace: { kind: 'participant_turn', stage: 'response', raw_response: 'full output' },
        },
      },
    ])
    expect(useMeetingWSStore.getState().turns).toEqual([])
  })

  it('requests replay on the first meeting websocket connection', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)

    expect(mockWebSocketInstance?.url).toContain('replay_since')
  })

  it('replays from the latest event timestamp when reconnecting', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)
    mockWebSocketInstance?.simulateMessage({
      event_type: 'meeting.trace',
      emitted_at: '2024-01-01T00:00:01Z',
      payload: { i: 1 },
    })

    useMeetingWSStore.getState().connect('meeting-1', null)

    expect(mockWebSocketInstance?.url).toContain(
      `replay_since=${encodeURIComponent('2024-01-01T00:00:01Z')}`,
    )
  })

  it('dedupes replayed websocket events by id', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)
    const event = {
      id: 'event-1',
      event_type: 'meeting.trace',
      emitted_at: '2024-01-01T00:00:01Z',
      payload: { i: 1 },
    }

    mockWebSocketInstance?.simulateMessage(event)
    mockWebSocketInstance?.simulateMessage(event)

    expect(useMeetingWSStore.getState().events).toEqual([event])
  })

  it('bounds verbose websocket events to the latest 200', () => {
    useMeetingWSStore.getState().connect('meeting-1', null)
    for (let i = 0; i < 201; i++) {
      mockWebSocketInstance?.simulateMessage({
        id: `event-${i}`,
        event_type: 'meeting.trace',
        emitted_at: '2024-01-01T00:00:01Z',
        payload: { i },
      })
    }

    const events = useMeetingWSStore.getState().events
    expect(events).toHaveLength(200)
    expect(events[0].id).toBe('event-200')
    expect(events[199].id).toBe('event-1')
  })
})
