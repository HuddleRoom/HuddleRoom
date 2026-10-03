import { rm } from 'node:fs/promises'
import type { FullConfig } from '@playwright/test'
import { readRunState } from './support/run-state'
import { removeSqliteFiles, summarizeServerLog, stopServerPid } from './support/live-server'

export default async function globalTeardown(_config: FullConfig): Promise<void> {
  const state = readRunState()
  await stopServerPid(state.serverPid)

  const keepDb = (process.env.HUDDLEROOM_UI_TEST_KEEP_DB ?? process.env.RALLY_UI_TEST_KEEP_DB) === '1' || state.failed === true
  if (!keepDb) {
    await removeSqliteFiles(state.databasePath)
    await rm(state.workspaceDir, { recursive: true, force: true })
  }

  const logTail = await summarizeServerLog(state.serverLogPath)
  const dbNote = keepDb ? `\nIsolated DB preserved: ${state.databasePath}` : ''
  if (logTail.trim()) {
    console.log(`\nHuddleRoom live UI server log: ${state.serverLogPath}${dbNote}\n${logTail}`)
    return
  }
  console.log(`\nHuddleRoom live UI server log: ${state.serverLogPath}${dbNote}`)
}
