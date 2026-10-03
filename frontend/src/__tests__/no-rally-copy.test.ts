import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'fs'
import { join, relative } from 'path'

// Guard (dashboard Phase 0): the product-facing name is "HuddleRoom" and
// actor phrasing is "The orchestrator" — the old "Rally" copy must never
// reappear in rendered UI strings/JSX text. Identifiers (HuddleRoomEvent),
// huddleroom-* CSS/Tailwind classes, the legacy VITE_RALLY_* env var, and import paths are
// not user-facing copy, so \bRally\b's own word-boundary already excludes
// them (case-sensitive, and boundaries don't split mid-identifier).
const SRC_ROOT = join(__dirname, '..')
const RALLY_WORD = /\bRally\b|\bRALLY\b/

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) return walk(full)
    if (!/\.(ts|tsx)$/.test(entry.name)) return []
    if (/\.(test|spec)\.(ts|tsx)$/.test(entry.name)) return []
    return [full]
  })
}

// Strips comments and import/export-from lines so only rendered strings/JSX
// text remain — a plain \bRally\b scan of those is a "string literal or JSX
// text" check without needing a full parser.
function stripNonCopy(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
    .split('\n')
    .filter((line) => !/^\s*(import|export)\b.*from\s+['"]/.test(line) && !/^\s*import\s+['"]/.test(line))
    .join('\n')
}

describe('no leftover "Rally" product copy', () => {
  it('finds no bare "Rally" word in source (excluding identifiers, huddleroom-* classes, VITE_RALLY_* env vars, and imports)', () => {
    const offenders: string[] = []
    for (const file of walk(SRC_ROOT)) {
      const copy = stripNonCopy(readFileSync(file, 'utf8'))
      copy.split('\n').forEach((line, index) => {
        if (RALLY_WORD.test(line)) offenders.push(`${relative(SRC_ROOT, file)}:${index + 1}: ${line.trim()}`)
      })
    }
    expect(offenders).toEqual([])
  })
})
