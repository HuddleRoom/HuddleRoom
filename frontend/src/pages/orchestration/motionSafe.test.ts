import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

// Grep-level check (spec: motion — reduced-motion verification) that every
// Tailwind animation/transition utility touched by this phase's motion work
// is `motion-safe:`-prefixed, so it's a no-op under prefers-reduced-motion
// without relying solely on the global `* { animation-duration: 0ms }`
// override in index.css.
const DIR = dirname(fileURLToPath(import.meta.url))
const FILES = ['ProcessChain.tsx', 'NeedsYouQueue.tsx', 'BaselineDashboard.tsx']

function bareMotionClasses(source: string) {
  const matches = [...source.matchAll(/(?<![\w-])(animate-[\w[\]().,%_'"-]+|transition-\w+)/g)]
  return matches
    .filter((match) => {
      const precedingContext = source.slice(Math.max(0, match.index - 40), match.index)
      return !precedingContext.endsWith('motion-safe:')
    })
    .map((match) => match[0])
}

describe('reduced-motion: motion-safe prefixing', () => {
  it.each(FILES)('every animate-*/transition-* utility in %s is motion-safe:-prefixed', (file) => {
    const source = readFileSync(join(DIR, file), 'utf-8')
    expect(bareMotionClasses(source)).toEqual([])
  })
})
