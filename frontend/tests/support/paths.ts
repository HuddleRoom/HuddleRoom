import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const testsRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
export const frontendRoot = resolve(testsRoot, '..')
export const repoRoot = resolve(frontendRoot, '..')
export const distDir = resolve(frontendRoot, 'dist')
export const dashboardStaticDir = resolve(repoRoot, 'huddleroom/static/dashboard')
export const liveResultsRoot = resolve(frontendRoot, 'test-results/live-ui')
export const runMetadataPath = resolve(liveResultsRoot, 'run-state.json')
