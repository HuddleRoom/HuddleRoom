import { cp, mkdir, rm, stat } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(here, '../..')
const repoRoot = resolve(frontendRoot, '..')
const distDir = resolve(frontendRoot, 'dist')
const dashboardDir = resolve(repoRoot, 'huddleroom/static/dashboard')

async function assertDirectory(path, label) {
  const info = await stat(path).catch(() => null)
  if (!info?.isDirectory()) {
    throw new Error(`${label} directory does not exist: ${path}`)
  }
}

await assertDirectory(distDir, 'frontend dist')
await mkdir(dashboardDir, { recursive: true })
await rm(dashboardDir, { recursive: true, force: true })
await mkdir(dashboardDir, { recursive: true })
await cp(distDir, dashboardDir, { recursive: true })
console.log(`Copied ${distDir} to ${dashboardDir}`)
