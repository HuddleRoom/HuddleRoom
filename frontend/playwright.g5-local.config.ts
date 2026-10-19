import { defineConfig } from '@playwright/test'

// Provider-free G5 checks: local Vite frontend + mocked API/WebSocket only.
const port = Number(process.env.HUDDLEROOM_G5_FIXTURE_PORT ?? 39251)
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests',
  testMatch: ['orchestration-g5-local.spec.ts'],
  timeout: 45_000,
  workers: 1,
  reporter: 'list',
  outputDir: './test-results/g5',
  use: { baseURL, trace: 'retain-on-failure' },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: `${baseURL}/dashboard/`,
    reuseExistingServer: false,
  },
})
