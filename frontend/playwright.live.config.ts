import { defineConfig } from '@playwright/test'

// Live acceptance: runs against an already-running HuddleRoom (LIVE_ACCEPTANCE_BASE_URL).
// No webServer and no stub-seeding globalSetup on purpose.
// The helper selfcheck is its own project: `--project selfcheck` (needs no base URL).
const baseURL = process.env.LIVE_ACCEPTANCE_BASE_URL
// Workers are forked after this runs, so the flag is inherited (their argv lacks --project).
if (process.argv.some((a, i, all) => a === '--project=selfcheck' || (a === '--project' && all[i + 1] === 'selfcheck'))) process.env.LIVE_SELFCHECK_ONLY = '1'
if (!baseURL && !process.env.LIVE_SELFCHECK_ONLY) {
  throw new Error('LIVE_ACCEPTANCE_BASE_URL must be set (URL of the running HuddleRoom)')
}

export default defineConfig({
  fullyParallel: false,
  workers: 1,
  outputDir: process.env.LIVE_ACCEPTANCE_OUTPUT || 'test-results-live',
  use: { baseURL, trace: 'on', screenshot: 'on', video: 'off' },
  projects: [
    { name: 'live', testDir: './tests-live', testIgnore: ['selfcheck/**', 'helpers/**'] },
    { name: 'selfcheck', testDir: './tests-live/selfcheck' },
  ],
})
