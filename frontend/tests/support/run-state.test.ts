import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { afterEach, describe, expect, test } from 'vitest'
import { buildLiveEnv, createRunState } from './run-state'

const tempRoots: string[] = []

afterEach(() => {
  for (const root of tempRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true })
  }
})

describe('live UI run state', () => {
  test('creates isolated database, workspace, artifact, and log paths', () => {
    const root = mkdtempSync(join(tmpdir(), 'huddleroom-ui-state-'))
    tempRoots.push(root)

    const state = createRunState({ repoRoot: root, port: 9234 })

    expect(state.port).toBe(9234)
    expect(state.databaseUrl).toContain('sqlite+aiosqlite:///')
    expect(state.databaseUrl).toContain('/huddleroom-ui.db')
    expect(state.workspaceDir).toContain('/workspace')
    expect(state.serverLogPath).toContain('/server.log')
    expect(state.metadataPath).toContain('/run-state.json')
  })

  test('buildLiveEnv disables auth and points HuddleRoom at the isolated DB', () => {
    const root = mkdtempSync(join(tmpdir(), 'huddleroom-ui-env-'))
    tempRoots.push(root)
    const state = createRunState({ repoRoot: root, port: 9345 })
    const env = buildLiveEnv(state)

    expect(env.HUDDLEROOM_AUTH_ENABLED).toBe('false')
    expect(env.HUDDLEROOM_DATABASE_URL).toBe(state.databaseUrl)
    expect(env.HUDDLEROOM_WORKSPACE_DIR).toBe(state.workspaceDir)
    expect(env.HUDDLEROOM_API_BASE_URL).toBe(`http://127.0.0.1:${state.port}`)
    expect(env.VITE_HUDDLEROOM_AUTH_DISABLED).toBe('true')
  })
})
