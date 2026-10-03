import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { relative, absolute, full, timeOfDay, dateHeading } from './time'

describe('time formatters', () => {
  const fixedIso = '2026-09-16T08:27:00Z'

  // Mock Date.now() to a fixed point for relative() tests
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-09-16T10:30:00Z'))
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('relative: formats time under 24 hours', () => {
    const result = relative(fixedIso)
    expect(result).toMatch(/\d+[mh]\s*ago/)
  })

  it('relative: returns "just now" for very recent timestamps', () => {
    const now = new Date('2026-09-16T10:30:00Z').toISOString()
    const result = relative(now)
    expect(result).toBe('just now')
  })

  it('relative: falls back to absolute for timestamps ≥24h old', () => {
    const oldIso = '2026-09-15T08:27:00Z' // More than 24h old
    const result = relative(oldIso)
    // Should use absolute format, not "1d ago"
    expect(result).toMatch(/\w+\s+\d+,\s+\d+,\s+\d+:\d+\s*[AP]M/)
  })

  it('relative: handles invalid/empty input gracefully', () => {
    expect(relative('')).toBe('')
    expect(relative('invalid-date')).toBe('invalid-date')
    expect(relative('not-a-date')).toBe('not-a-date')
  })

  it('absolute: formats as "Sep 16, 2026, 8:27 AM" style', () => {
    const result = absolute(fixedIso)
    expect(result).toMatch(/\w+\s+\d+,\s+\d+,\s+\d+:\d+\s*[AP]M/)
  })

  it('absolute: handles invalid/empty input gracefully', () => {
    expect(absolute('')).toBe('')
    expect(absolute('invalid-date')).toBe('invalid-date')
  })

  it('full: returns raw ISO string', () => {
    const result = full(fixedIso)
    expect(result).toBe(fixedIso)
  })

  it('full: returns any input as-is', () => {
    expect(full('')).toBe('')
    expect(full('anything')).toBe('anything')
  })

  it('timeOfDay: formats as "8:27 AM" style', () => {
    const result = timeOfDay(fixedIso)
    expect(result).toMatch(/\d+:\d+\s*[AP]M/)
  })

  it('timeOfDay: handles invalid/empty input gracefully', () => {
    expect(timeOfDay('')).toBe('')
    expect(timeOfDay('invalid-date')).toBe('invalid-date')
  })

  it('dateHeading: formats as "Sep 16, 2026" style', () => {
    const result = dateHeading(fixedIso)
    expect(result).toMatch(/\w+\s+\d+,\s+\d+/)
  })

  it('dateHeading: handles invalid/empty input gracefully', () => {
    expect(dateHeading('')).toBe('')
    expect(dateHeading('invalid-date')).toBe('invalid-date')
  })
})
