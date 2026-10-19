import { defineConfig } from '@playwright/test'

const port = 39261
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests', testMatch: ['orchestration-g3-local.spec.ts'],
  timeout: 30_000, workers: 1, reporter: 'list', outputDir: './test-results/g3-local',
  use: { baseURL, trace: 'retain-on-failure' },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: `${baseURL}/dashboard/`, reuseExistingServer: false,
  },
})
