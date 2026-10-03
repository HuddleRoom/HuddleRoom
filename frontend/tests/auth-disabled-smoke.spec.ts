import { expect, test } from '@playwright/test'
import { expectDashboardReady, gotoDashboard, selectSeedProject } from './support/pages/app'

test('auth-disabled login route lands on the dashboard', async ({ page }) => {
  const navigations: string[] = []
  page.on('framenavigated', (frame) => {
    if (frame === page.mainFrame()) {
      navigations.push(new URL(frame.url()).pathname)
    }
  })

  await page.goto('/dashboard/login')
  await expect(page).toHaveURL(/\/dashboard\/?$/)
  await expect(page.getByRole('combobox', { name: 'Project switcher' })).toBeVisible()
  expect(navigations.length).toBeLessThanOrEqual(8)
})

test('dashboard loads directly with the seeded project in auth-disabled mode', async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
  await expectDashboardReady(page)
})
