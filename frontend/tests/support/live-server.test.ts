import { existsSync, mkdtempSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, describe, expect, test } from 'vitest'
import { removeSqliteFiles, summarizeServerLog } from './live-server'

const tempRoots: string[] = []

afterEach(() => {
  for (const root of tempRoots.splice(0)) {
    void removeSqliteFiles(join(root, 'huddleroom-ui.db'))
  }
})

describe('live server helpers', () => {
  test('summarizeServerLog returns the tail of the log', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'huddleroom-ui-log-'))
    tempRoots.push(dir)
    const logPath = join(dir, 'server.log')
    writeFileSync(logPath, Array.from({ length: 100 }, (_, i) => `line-${i}`).join('\n'))

    const summary = await summarizeServerLog(logPath)

    expect(summary).not.toContain('line-0')
    expect(summary).toContain('line-99')
  })

  test('removeSqliteFiles deletes the database and sqlite sidecars', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'huddleroom-ui-db-'))
    tempRoots.push(dir)
    const dbPath = join(dir, 'huddleroom-ui.db')
    writeFileSync(dbPath, '')
    writeFileSync(`${dbPath}-wal`, '')
    writeFileSync(`${dbPath}-shm`, '')

    await removeSqliteFiles(dbPath)

    expect(existsSync(dbPath)).toBe(false)
    expect(existsSync(`${dbPath}-wal`)).toBe(false)
    expect(existsSync(`${dbPath}-shm`)).toBe(false)
  })
})
