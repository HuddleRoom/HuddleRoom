import { defineConfig, devices } from '@playwright/test'

const port = Number(process.env.HUDDLEROOM_UI_TEST_PORT ?? process.env.RALLY_UI_TEST_PORT ?? 39123)
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests',
  testIgnore: ['tests/support/**/*.test.ts'],
  timeout: 45_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ['./tests/support/failure-reporter.ts'],
    ['html', { open: 'never', outputFolder: './playwright-report' }],
    ['list'],
  ],
  outputDir: './test-results/live-ui/artifacts',
  globalSetup: './tests/global-setup.ts',
  globalTeardown: './tests/global-teardown.ts',
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: process.env.CI || (process.env.HUDDLEROOM_UI_TEST_VIDEO ?? process.env.RALLY_UI_TEST_VIDEO) === '1' ? 'retain-on-failure' : 'off',
    actionTimeout: 15_000,
    navigationTimeout: 20_000,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
})
