import { describe, it, expect } from 'vitest'
import { priorityLabel, priorityColor, PRIORITY_INT } from './priority'

describe('priorityLabel', () => {
  it('maps boundary values to correct labels', () => {
    expect(priorityLabel(0)).toBe('low')
    expect(priorityLabel(25)).toBe('low')
    expect(priorityLabel(39)).toBe('low')
    expect(priorityLabel(40)).toBe('medium')
    expect(priorityLabel(50)).toBe('medium')
    expect(priorityLabel(64)).toBe('medium')
    expect(priorityLabel(65)).toBe('high')
    expect(priorityLabel(75)).toBe('high')
    expect(priorityLabel(89)).toBe('high')
    expect(priorityLabel(90)).toBe('critical')
    expect(priorityLabel(100)).toBe('critical')
  })

  it('PRIORITY_INT round-trips through priorityLabel', () => {
    for (const [label, int] of Object.entries(PRIORITY_INT)) {
      expect(priorityLabel(int)).toBe(label)
    }
  })
})

describe('priorityColor', () => {
  it('returns distinct colors for each level', () => {
    const colors = ['low', 'medium', 'high', 'critical'].map((_, i) =>
      priorityColor([25, 50, 75, 100][i])
    )
    expect(new Set(colors).size).toBe(4)
  })
})
