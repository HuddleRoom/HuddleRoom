import { describe, expect, it } from 'vitest'
import { humanizeEvent, collapseRuns } from './humanizeEvents'

describe('humanizeEvent', () => {
  it('maps known event types', () => {
    expect(humanizeEvent('meeting.concluded')).toBe('Meeting concluded')
    expect(humanizeEvent('meeting.turn_complete')).toBe('Agent turn completed')
    expect(humanizeEvent('meeting.trace')).toBe('Meeting trace')
    expect(humanizeEvent('task.created')).toBe('Task created')
    expect(humanizeEvent('task.status_changed')).toBe('Task status changed')
  })
  it('cleans unknown types, never raw-only', () => {
    expect(humanizeEvent('protocol.step_failed')).toBe('Protocol step failed')
    expect(humanizeEvent('weird')).toBe('Weird')
  })
})

describe('collapseRuns', () => {
  it('collapses consecutive same-type events', () => {
    const evts = [
      { type: 'meeting.trace' }, { type: 'meeting.trace' }, { type: 'meeting.trace' },
      { type: 'meeting.concluded' }, { type: 'meeting.trace' },
    ]
    const out = collapseRuns(evts)
    expect(out.map((c) => [c.event.type, c.count])).toEqual([
      ['meeting.trace', 3], ['meeting.concluded', 1], ['meeting.trace', 1],
    ])
  })
  it('keeps the newest event of each run as representative', () => {
    const evts = [{ type: 'a', id: 1 }, { type: 'a', id: 2 }]
    expect(collapseRuns(evts)[0].event.id).toBe(1) // list is newest-first; keep first seen
  })
})
