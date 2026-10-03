import { spawn } from 'node:child_process'
import { createWriteStream, existsSync } from 'node:fs'
import { readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { repoRoot } from './paths'
import type { LiveRunState } from './run-state'

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== 'ESRCH'
  }
}

async function waitForProcessExit(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (!processAlive(pid)) {
      return true
    }
    await sleep(100)
  }
  return !processAlive(pid)
}

async function terminateChild(child: ReturnType<typeof spawn>, timeoutMs: number): Promise<void> {
  if (!child.pid) {
    return
  }

  child.kill('SIGTERM')
  const exited = await new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), timeoutMs)
    child.once('exit', () => {
      clearTimeout(timer)
      resolve(true)
    })
  })

  if (!exited && child.pid) {
    child.kill('SIGKILL')
    await new Promise<void>((resolve) => {
      child.once('exit', () => resolve())
      setTimeout(() => resolve(), 1_000)
    })
  }
}

export async function waitForReady(url: string, timeoutMs = 45_000): Promise<void> {
  const started = Date.now()
  let lastError = ''

  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(url)
      if (response.ok || response.status === 404) {
        return
      }
      lastError = `${response.status} ${response.statusText}`
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error)
    }
    await sleep(500)
  }

  throw new Error(`HuddleRoom server was not ready at ${url} within ${timeoutMs}ms. Last error: ${lastError}`)
}

export async function startHuddleRoomServer(state: LiveRunState, env: NodeJS.ProcessEnv): Promise<number> {
  const huddleroomBin = join(repoRoot, '.venv/bin/huddleroom')
  if (!existsSync(huddleroomBin)) {
    throw new Error(`Expected HuddleRoom executable at ${huddleroomBin}. Run local setup before live UI tests.`)
  }

  const log = createWriteStream(state.serverLogPath, { flags: 'a' })
  const child = spawn(huddleroomBin, ['serve', '--host', '127.0.0.1', '--port', String(state.port)], {
    cwd: repoRoot,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  child.stdout.pipe(log)
  child.stderr.pipe(log)

  if (!child.pid) {
    throw new Error('HuddleRoom server did not expose a PID')
  }

  try {
    await waitForReady(`${state.baseURL}/dashboard/login`)
    return child.pid
  } catch (error) {
    await terminateChild(child, 3_000)
    throw error
  }
}

export async function stopServerPid(pid: number | undefined): Promise<void> {
  if (!pid) {
    return
  }
  try {
    process.kill(pid, 'SIGTERM')
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code !== 'ESRCH') {
      throw error
    }
    return
  }

  const exited = await waitForProcessExit(pid, 5_000)
  if (exited) {
    return
  }

  try {
    process.kill(pid, 'SIGKILL')
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code
    if (code !== 'ESRCH') {
      throw error
    }
  }

  await waitForProcessExit(pid, 1_000)
}

export async function summarizeServerLog(path: string): Promise<string> {
  const text = await readFile(path, 'utf8').catch(() => '')
  return text.split('\n').slice(-80).join('\n')
}

export async function removeSqliteFiles(databasePath: string): Promise<void> {
  await Promise.all([
    rm(databasePath, { force: true }).catch(() => undefined),
    rm(`${databasePath}-wal`, { force: true }).catch(() => undefined),
    rm(`${databasePath}-shm`, { force: true }).catch(() => undefined),
  ])
}
