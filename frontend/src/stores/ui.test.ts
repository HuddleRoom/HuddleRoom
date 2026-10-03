import { describe, expect, it } from 'vitest'
import { LEGACY_UI_STORAGE_KEY, migrateLegacyUIStorage, UI_STORAGE_KEY } from './ui'

function storage(values: Record<string, string> = {}) {
  const data = new Map(Object.entries(values))
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
    removeItem: (key: string) => data.delete(key),
  }
}

describe('migrateLegacyUIStorage', () => {
  it('copies the legacy persisted preferences on first read and removes the old key', () => {
    const legacy = JSON.stringify({ state: { activeProjectId: 'project-1', sidebarCollapsed: true }, version: 1 })
    const localStorage = storage({ [LEGACY_UI_STORAGE_KEY]: legacy })

    expect(migrateLegacyUIStorage(localStorage)).toBe(legacy)
    expect(localStorage.getItem(UI_STORAGE_KEY)).toBe(legacy)
    expect(localStorage.getItem(LEGACY_UI_STORAGE_KEY)).toBeNull()

    localStorage.removeItem(UI_STORAGE_KEY)
    expect(migrateLegacyUIStorage(localStorage)).toBeNull()
  })

  it('keeps the HuddleRoom key when both versions exist', () => {
    const current = JSON.stringify({ state: { activeProjectId: 'new', sidebarCollapsed: false }, version: 1 })
    const localStorage = storage({ [UI_STORAGE_KEY]: current, [LEGACY_UI_STORAGE_KEY]: 'legacy' })

    expect(migrateLegacyUIStorage(localStorage)).toBe(current)
    expect(localStorage.getItem(LEGACY_UI_STORAGE_KEY)).toBe('legacy')
  })
})
