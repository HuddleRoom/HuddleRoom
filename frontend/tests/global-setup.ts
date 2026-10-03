import { cp, mkdir, rm } from 'node:fs/promises'
import { join } from 'node:path'
import type { FullConfig } from '@playwright/test'
import { dashboardStaticDir, distDir, repoRoot, runMetadataPath } from './support/paths'
import { startHuddleRoomServer } from './support/live-server'
import { runCommand } from './support/process'
import { buildLiveEnv, createTempRunState, ensureRunDirectories, livePort, writeRunState } from './support/run-state'

async function copyDashboardAssets(): Promise<void> {
  await rm(dashboardStaticDir, { recursive: true, force: true })
  await mkdir(dashboardStaticDir, { recursive: true })
  await cp(distDir, dashboardStaticDir, { recursive: true })
}

export default async function globalSetup(_config: FullConfig): Promise<void> {
  const state = await createTempRunState({ port: livePort() })
  ensureRunDirectories(state)
  const env = buildLiveEnv(state)

  await copyDashboardAssets()
  await runCommand({
    command: join(repoRoot, '.venv/bin/alembic'),
    args: ['upgrade', 'head'],
    cwd: repoRoot,
    env,
    label: 'alembic upgrade for live UI DB',
    logPath: state.serverLogPath,
  })
  await runCommand({
    command: join(repoRoot, '.venv/bin/python'),
    args: ['tests/ui/seed_live_ui.py', '--database-url', state.databaseUrl, '--workspace-dir', state.workspaceDir],
    cwd: repoRoot,
    env,
    label: 'seed live UI DB',
    logPath: state.serverLogPath,
  })

  writeRunState(state)
  const serverPid = await startHuddleRoomServer(state, env)
  writeRunState({ ...state, serverPid })
  process.env.HUDDLEROOM_UI_RUN_STATE = runMetadataPath
}
