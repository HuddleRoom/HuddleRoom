import { defineConfig } from '@playwright/test'

const port = 39251
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests', testMatch: ['orchestration-g4-local.spec.ts'],
  timeout: 30_000, workers: 1, reporter: 'list', outputDir: './test-results/g4-local',
  use: { baseURL, trace: 'retain-on-failure' },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: `${baseURL}/dashboard/`, reuseExistingServer: false,
  },
})
