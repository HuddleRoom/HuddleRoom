import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'

export interface RunCommandOptions {
  command: string
  args: string[]
  cwd: string
  env?: NodeJS.ProcessEnv
  label: string
  logPath?: string
  timeoutMs?: number
}

export async function runCommand(options: RunCommandOptions): Promise<void> {
  const { command, args, cwd, env, label, logPath, timeoutMs = 120_000 } = options
  const chunks: string[] = []
  const log = logPath ? createWriteStream(logPath, { flags: 'a' }) : null

  await new Promise<void>((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const timer = setTimeout(() => {
      child.kill('SIGTERM')
      reject(new Error(`${label} timed out after ${timeoutMs}ms`))
    }, timeoutMs)

    child.stdout.on('data', (data: Buffer) => {
      const text = data.toString()
      chunks.push(text)
      log?.write(text)
    })
    child.stderr.on('data', (data: Buffer) => {
      const text = data.toString()
      chunks.push(text)
      log?.write(text)
    })
    child.on('error', (error) => {
      clearTimeout(timer)
      log?.end()
      reject(error)
    })
    child.on('exit', (code) => {
      clearTimeout(timer)
      log?.end()
      if (code === 0) {
        resolve()
        return
      }
      reject(new Error(`${label} failed with exit code ${code}\n${chunks.join('').slice(-4000)}`))
    })
  })
}
