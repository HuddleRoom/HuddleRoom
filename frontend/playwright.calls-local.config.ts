import { defineConfig } from '@playwright/test'

// Provider-free Calls checks use only a local frontend and mocked API/WebSocket.
// Run with: npx playwright test --config=playwright.calls-local.config.ts
const port = Number(process.env.HUDDLEROOM_CALLS_FIXTURE_PORT ?? 39249)
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests',
  testMatch: ['calls-readable-followup.spec.ts'],
  timeout: 45_000,
  workers: 1,
  reporter: 'list',
  outputDir: './test-results/readable-followup',
  use: { baseURL, trace: 'retain-on-failure' },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: `${baseURL}/dashboard/`,
    reuseExistingServer: false,
  },
})
