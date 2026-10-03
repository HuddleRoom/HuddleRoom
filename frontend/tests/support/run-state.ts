import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { mkdtemp, realpath } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { tmpdir } from 'node:os'
import { runMetadataPath } from './paths'

export interface LiveRunState {
  runRoot: string
  databasePath: string
  databaseUrl: string
  workspaceDir: string
  artifactsDir: string
  serverLogPath: string
  metadataPath: string
  port: number
  baseURL: string
  serverPid?: number
  failed?: boolean
}

export interface LiveEnvOptions {
  authEnabled?: boolean
}

export function livePort(): number {
  const raw = process.env.HUDDLEROOM_UI_TEST_PORT ?? process.env.RALLY_UI_TEST_PORT
  const value = raw ? Number(raw) : 39123
  if (!Number.isInteger(value) || value < 1 || value > 65535) {
    throw new Error(`HUDDLEROOM_UI_TEST_PORT must be an integer TCP port, got ${raw}`)
  }
  return value
}

function buildState(runRoot: string, port: number): LiveRunState {
  const databasePath = join(runRoot, 'huddleroom-ui.db')
  return {
    runRoot,
    databasePath,
    databaseUrl: `sqlite+aiosqlite:///${databasePath}`,
    workspaceDir: join(runRoot, 'workspace'),
    artifactsDir: join(runRoot, 'artifacts'),
    serverLogPath: join(runRoot, 'server.log'),
    metadataPath: runMetadataPath,
    port,
    baseURL: `http://127.0.0.1:${port}`,
  }
}

export function createRunState({ repoRoot, port }: { repoRoot: string; port: number }): LiveRunState {
  return buildState(join(repoRoot, '.tmp/live-ui', `${Date.now()}-${process.pid}`), port)
}

export async function createTempRunState({ port }: { port: number }): Promise<LiveRunState> {
  const runRoot = await realpath(await mkdtemp(join(tmpdir(), 'huddleroom-ui-')))
  return buildState(runRoot, port)
}

export function buildLiveEnv(state: LiveRunState, options: LiveEnvOptions = {}): NodeJS.ProcessEnv {
  const authEnabled = options.authEnabled ?? false
  return {
    ...process.env,
    HUDDLEROOM_DATABASE_URL: state.databaseUrl,
    HUDDLEROOM_AUTH_ENABLED: authEnabled ? 'true' : 'false',
    HUDDLEROOM_WORKSPACE_DIR: state.workspaceDir,
    HUDDLEROOM_API_BASE_URL: state.baseURL,
    VITE_HUDDLEROOM_AUTH_DISABLED: authEnabled ? 'false' : 'true',
    HUDDLEROOM_DEBUG: process.env.HUDDLEROOM_DEBUG ?? process.env.RALLY_DEBUG ?? 'false',
  }
}

export function ensureRunDirectories(state: LiveRunState): void {
  mkdirSync(state.runRoot, { recursive: true })
  mkdirSync(state.workspaceDir, { recursive: true })
  mkdirSync(state.artifactsDir, { recursive: true })
  mkdirSync(dirname(state.metadataPath), { recursive: true })
}

export function writeRunState(state: LiveRunState): void {
  ensureRunDirectories(state)
  writeFileSync(state.metadataPath, JSON.stringify(state, null, 2))
}

export function readRunState(): LiveRunState {
  const metadataPath = process.env.HUDDLEROOM_UI_RUN_STATE ?? process.env.RALLY_UI_RUN_STATE ?? runMetadataPath
  return JSON.parse(readFileSync(metadataPath, 'utf8')) as LiveRunState
}
